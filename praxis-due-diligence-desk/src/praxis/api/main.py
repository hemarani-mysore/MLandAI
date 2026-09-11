"""FastAPI surface for the dossier desk.

Phase 1: synchronous ``POST /dossiers`` + a corpus API (ingest / search / stats).
Phase 2: the dossier graph runs async end to end — these routes call
``arun_dossier`` directly (never the sync ``run_dossier`` wrapper, which would
deadlock inside a running event loop). SSE streaming and moving ingestion to a
worker are still to come.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from praxis import __version__
from praxis.api.deps import corpus_dep
from praxis.config import get_settings
from praxis.graph import arun_dossier
from praxis.rag import Corpus, RetrievedChunk, ingest_source
from praxis.rag.models import CorpusStats, IngestResult
from praxis.render import to_markdown
from praxis.schemas import DossierRequest, DossierResponse

app = FastAPI(title="Praxis — Due-Diligence Research Desk", version=__version__)


class IngestBody(BaseModel):
    text: str | None = Field(default=None, description="Inline document text")
    path: str | None = Field(default=None, description="Server-side path to a .pdf/.md/.txt/.html")
    title: str | None = None


class SearchBody(BaseModel):
    query: str = Field(min_length=2)
    k: int = Field(default=6, ge=1, le=50)


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
    request: DossierRequest, corpus: Corpus = Depends(corpus_dep)
) -> DossierResponse:
    return await arun_dossier(request, corpus=corpus)


@app.post("/dossiers.md", response_class=PlainTextResponse)
async def create_dossier_markdown(
    request: DossierRequest, corpus: Corpus = Depends(corpus_dep)
) -> str:
    return to_markdown(await arun_dossier(request, corpus=corpus))


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
