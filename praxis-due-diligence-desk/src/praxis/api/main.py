"""FastAPI surface for the dossier desk.

Phase 1: synchronous ``POST /dossiers`` + a corpus API (ingest / search / stats).
Phase 2: the dossier graph runs async end to end — these routes call
``arun_dossier`` directly (never the sync ``run_dossier`` wrapper, which would
deadlock inside a running event loop). Every dossier run is now persisted as a
``DossierRun`` row (``GET /dossiers[/{id}]``) and checkpointed via a per-call
``AsyncSqliteSaver``. SSE streaming and moving ingestion to a worker are still
to come.
"""

from __future__ import annotations

import datetime as dt

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from praxis import __version__
from praxis.api.deps import corpus_dep, db_session_dep
from praxis.config import get_settings
from praxis.db import DossierRun
from praxis.graph import arun_dossier
from praxis.rag import Corpus, RetrievedChunk, ingest_source
from praxis.rag.models import CorpusStats, IngestResult
from praxis.render import to_markdown
from praxis.schemas import DossierRequest, DossierResponse, DossierRunRecord

app = FastAPI(title="Praxis — Due-Diligence Research Desk", version=__version__)


class IngestBody(BaseModel):
    text: str | None = Field(default=None, description="Inline document text")
    path: str | None = Field(default=None, description="Server-side path to a .pdf/.md/.txt/.html")
    title: str | None = None


class SearchBody(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(default=6, ge=1, le=50)


async def _create_and_persist(
    request: DossierRequest, corpus: Corpus, session: AsyncSession
) -> DossierResponse:
    """Run a dossier, recording it as a ``DossierRun`` from creation through its
    outcome, checkpointed under its own id so the run is technically resumable."""
    run = DossierRun(subject=request.subject, depth=request.depth, status="running")
    session.add(run)
    await session.commit()

    try:
        async with AsyncSqliteSaver.from_conn_string(get_settings().checkpoint_db_path) as saver:
            await saver.setup()
            response = await arun_dossier(
                request, corpus=corpus, checkpointer=saver, thread_id=run.id
            )
    except Exception as exc:
        run.status = "failed"
        run.error = str(exc)
        run.finished_at = dt.datetime.now(dt.UTC)
        await session.commit()
        raise

    run.status = "succeeded"
    run.memo = response.memo.model_dump(mode="json")
    run.verification = response.verification.model_dump(mode="json")
    run.sources = response.sources
    run.finished_at = dt.datetime.now(dt.UTC)
    await session.commit()
    return response


@app.get("/")
def root() -> dict:
    return {"service": "praxis", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/ready")
def ready(corpus: Corpus = Depends(corpus_dep)) -> dict:
    s = get_settings()
    return {
        "ok": True,
        "llm_provider": s.llm_provider,
        "embedding_provider": s.embedding_provider,
        "corpus": corpus.stats().model_dump(),
    }


@app.post("/dossiers", response_model=DossierResponse)
async def create_dossier(
    request: DossierRequest,
    corpus: Corpus = Depends(corpus_dep),
    session: AsyncSession = Depends(db_session_dep),
) -> DossierResponse:
    return await _create_and_persist(request, corpus, session)


@app.post("/dossiers.md", response_class=PlainTextResponse)
async def create_dossier_markdown(
    request: DossierRequest,
    corpus: Corpus = Depends(corpus_dep),
    session: AsyncSession = Depends(db_session_dep),
) -> str:
    return to_markdown(await _create_and_persist(request, corpus, session))


@app.get("/dossiers", response_model=list[DossierRunRecord])
async def list_dossiers(
    limit: int = Query(default=20, ge=1, le=100),
    session: AsyncSession = Depends(db_session_dep),
) -> list[DossierRunRecord]:
    rows = (
        (
            await session.execute(
                select(DossierRun)
                .order_by(DossierRun.created_at.desc(), DossierRun.id.desc())
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [DossierRunRecord.model_validate(row) for row in rows]


@app.get("/dossiers/{run_id}", response_model=DossierRunRecord)
async def get_dossier_run(
    run_id: str, session: AsyncSession = Depends(db_session_dep)
) -> DossierRunRecord:
    run = await session.get(DossierRun, run_id)
    if run is None:
        raise HTTPException(404, f"no such run: {run_id}")
    return DossierRunRecord.model_validate(run)


@app.post("/corpus/documents", response_model=IngestResult)
def ingest_document(body: IngestBody, corpus: Corpus = Depends(corpus_dep)) -> IngestResult:
    if not body.text and not body.path:
        raise HTTPException(422, "provide `text` or `path`")
    try:
        return ingest_source(text=body.text, path=body.path, title=body.title, corpus=corpus)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"file not found: {exc}") from exc


@app.get("/corpus/stats", response_model=CorpusStats)
def corpus_stats(corpus: Corpus = Depends(corpus_dep)) -> CorpusStats:
    return corpus.stats()


@app.post("/corpus/search", response_model=list[RetrievedChunk])
def corpus_search(body: SearchBody, corpus: Corpus = Depends(corpus_dep)) -> list[RetrievedChunk]:
    return corpus.search(body.query, k=body.k)
