"""FastAPI surface for the dossier desk.

Phase 1: synchronous ``POST /dossiers`` + a corpus API (ingest / search / stats).
Phase 2: the dossier graph runs async end to end — these routes call
``arun_dossier`` directly (never the sync ``run_dossier`` wrapper, which would
deadlock inside a running event loop). Every dossier run is persisted as a
``DossierRun`` row (``GET /dossiers[/{id}]``) and checkpointed. ``GET
/dossiers/stream`` streams the same run node by node over SSE. Ingestion can
also run on the ``arq`` worker instead of blocking the request
(``POST /corpus/jobs`` / ``GET /corpus/jobs/{id}``).
"""

from __future__ import annotations

import datetime as dt

from arq.connections import ArqRedis
from arq.jobs import Job, JobStatus
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from praxis import __version__
from praxis.api.deps import corpus_dep, db_session_dep, redis_dep
from praxis.api.sse import stream_dossier
from praxis.config import get_settings
from praxis.db import DossierRun, RunEvent
from praxis.graph import arun_dossier
from praxis.graph.checkpoint import open_checkpointer
from praxis.rag import Corpus, RetrievedChunk, ingest_source
from praxis.rag.models import CorpusStats, IngestResult
from praxis.render import to_markdown
from praxis.schemas import (
    Depth,
    DossierRequest,
    DossierResponse,
    DossierRunRecord,
    JobEnqueued,
    JobStatusOut,
    RunEventOut,
)

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
        async with open_checkpointer(get_settings().checkpoint_db_path) as checkpointer:
            response = await arun_dossier(
                request, corpus=corpus, checkpointer=checkpointer, thread_id=run.id
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


@app.get("/dossiers/stream")
async def stream_dossier_route(
    subject: str = Query(min_length=2),
    depth: Depth = "standard",
    corpus: Corpus = Depends(corpus_dep),
    session: AsyncSession = Depends(db_session_dep),
) -> StreamingResponse:
    request = DossierRequest(subject=subject, depth=depth)
    return StreamingResponse(
        stream_dossier(request, corpus, session), media_type="text/event-stream"
    )


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


@app.get("/dossiers/{run_id}/events", response_model=list[RunEventOut])
async def get_dossier_run_events(
    run_id: str, session: AsyncSession = Depends(db_session_dep)
) -> list[RunEventOut]:
    rows = (
        (
            await session.execute(
                select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)
            )
        )
        .scalars()
        .all()
    )
    return [RunEventOut.model_validate(row) for row in rows]


@app.post("/corpus/documents", response_model=IngestResult)
def ingest_document(body: IngestBody, corpus: Corpus = Depends(corpus_dep)) -> IngestResult:
    if not body.text and not body.path:
        raise HTTPException(422, "provide `text` or `path`")
    try:
        return ingest_source(text=body.text, path=body.path, title=body.title, corpus=corpus)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"file not found: {exc}") from exc


@app.post("/corpus/jobs", response_model=JobEnqueued)
async def enqueue_ingest_job(body: IngestBody, redis: ArqRedis = Depends(redis_dep)) -> JobEnqueued:
    if not body.text and not body.path:
        raise HTTPException(422, "provide `text` or `path`")
    job = await redis.enqueue_job("ingest_task", text=body.text, path=body.path, title=body.title)
    assert job is not None
    return JobEnqueued(job_id=job.job_id)


@app.get("/corpus/jobs/{job_id}", response_model=JobStatusOut)
async def get_ingest_job(job_id: str, redis: ArqRedis = Depends(redis_dep)) -> JobStatusOut:
    job = Job(job_id, redis)
    status = await job.status()
    if status == JobStatus.not_found:
        raise HTTPException(404, f"no such job: {job_id}")

    info = await job.result_info()
    if info is None:
        return JobStatusOut(job_id=job_id, status=status.value)  # type: ignore[arg-type]
    if info.success:
        return JobStatusOut(job_id=job_id, status=status.value, success=True, result=info.result)  # type: ignore[arg-type]
    return JobStatusOut(
        job_id=job_id,
        status=status.value,
        success=False,
        error=str(info.result),  # type: ignore[arg-type]
    )


@app.get("/corpus/stats", response_model=CorpusStats)
def corpus_stats(corpus: Corpus = Depends(corpus_dep)) -> CorpusStats:
    return corpus.stats()


@app.post("/corpus/search", response_model=list[RetrievedChunk])
def corpus_search(body: SearchBody, corpus: Corpus = Depends(corpus_dep)) -> list[RetrievedChunk]:
    return corpus.search(body.query, k=body.k)
