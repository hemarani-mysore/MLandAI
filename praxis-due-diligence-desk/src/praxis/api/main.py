"""FastAPI surface for the dossier desk.

Phase 0: synchronous ``POST /dossiers``. Phase 2 adds SSE streaming of node
events; Phase 1 makes ``/ready`` check Qdrant + Postgres.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from praxis import __version__
from praxis.config import get_settings
from praxis.graph import run_dossier
from praxis.render import to_markdown
from praxis.schemas import DossierRequest, DossierResponse

app = FastAPI(title="Praxis — Due-Diligence Research Desk", version=__version__)


@app.get("/")
def root() -> dict:
    return {
        "service": "praxis",
        "version": __version__,
        "llm_provider": get_settings().llm_provider,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/ready")
def ready() -> dict:
    # Phase 1: also ping Qdrant + Postgres.
    return {"ok": True, "llm_provider": get_settings().llm_provider}


@app.post("/dossiers", response_model=DossierResponse)
def create_dossier(request: DossierRequest) -> DossierResponse:
    return run_dossier(request)


@app.post("/dossiers.md", response_class=PlainTextResponse)
def create_dossier_markdown(request: DossierRequest) -> str:
    return to_markdown(run_dossier(request))
