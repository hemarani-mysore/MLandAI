"""``praxis-mcp`` — the desk as MCP tools/resources/prompts, for any MCP host
(Claude Code, Cursor, ...). Calls the same library code the API and CLI do —
no logic fork. Runs as its own process with its own corpus singleton, same as
the CLI and the ingestion worker: sharing a corpus with the API needs a real
``PRAXIS_QDRANT_URL``, not the in-memory default.

``mcp`` 1.x's class is ``FastMCP`` (``mcp.server.fastmcp``) — a later major
version of the ``mcp`` package renames it to ``MCPServer``
(``mcp.server.mcpserver``); this project is pinned to the 1.x surface.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from praxis.config import get_settings
from praxis.db.models import DossierRun
from praxis.db.session import get_sessionmaker
from praxis.graph import arun_dossier
from praxis.graph.prompts import EVIDENCE_LOOKUP
from praxis.llm import get_llm
from praxis.obs import setup_tracing
from praxis.rag import RetrievedChunk, get_corpus, ingest_source, search_corpus
from praxis.rag.models import CorpusStats, IngestResult
from praxis.schemas import (
    Depth,
    DossierRequest,
    DossierResponse,
    DossierRunRecord,
    EvidenceIndices,
    EvidenceLookup,
)

_settings = get_settings()
setup_tracing()
# host/port only matter for the networked transports (sse, streamable-http) —
# a stdio host (Claude Code, Cursor, the test suite) ignores them entirely.
mcp = FastMCP("praxis", host=_settings.mcp_host, port=_settings.mcp_port)


@mcp.tool()
async def create_dossier(subject: str, depth: Depth = "standard") -> DossierResponse:
    """Run the full due-diligence graph on `subject` and return the cited memo."""
    return await arun_dossier(DossierRequest(subject=subject, depth=depth))


@mcp.tool(name="search_corpus")
def search_corpus_tool(query: str, k: int = 6) -> list[RetrievedChunk]:
    """Hybrid-search the ingested corpus; each hit carries a grounded citation."""
    return search_corpus(query, k=k)


@mcp.tool()
def ingest_document(
    text: str | None = None, path: str | None = None, title: str | None = None
) -> IngestResult:
    """Ingest a document (inline `text` or a server-side `path`) into the corpus."""
    return ingest_source(text=text, path=path, title=title)


def _render_hits(hits: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] source_id={h.chunk.source_id} | {h.chunk.title} {h.chunk.locator}\n{h.chunk.raw_text}"
        for i, h in enumerate(hits)
    )


@mcp.tool()
async def get_evidence(claim: str, k: int = 6) -> EvidenceLookup:
    """Check the corpus for chunks that support or refute `claim`."""
    hits = search_corpus(claim, k=k)
    if not hits:
        return EvidenceLookup(claim=claim)

    llm = get_llm()
    indices = await llm.agenerate(
        system=EVIDENCE_LOOKUP,
        user=f"Claim: {claim}\n\nRetrieved context:\n{_render_hits(hits)}",
        schema=EvidenceIndices,
        role="evidence_lookup",
    )
    in_range = range(len(hits))
    return EvidenceLookup(
        claim=claim,
        supporting=[hits[i].citation for i in indices.supporting_indices if i in in_range],
        refuting=[hits[i].citation for i in indices.refuting_indices if i in in_range],
    )


async def _get_run(run_id: str) -> DossierRun | None:
    async with get_sessionmaker()() as session:
        return await session.get(DossierRun, run_id)


@mcp.resource("dossier://{run_id}")
async def dossier_resource(run_id: str) -> DossierRunRecord:
    """A persisted dossier run — created via the API's `POST /dossiers`, not
    this server (MCP tools don't write to the run history; see `create_dossier`)."""
    run = await _get_run(run_id)
    if run is None:
        raise ValueError(f"no such run: {run_id}")
    return DossierRunRecord.model_validate(run)


@mcp.resource("dossier://{run_id}/citations")
async def dossier_citations_resource(run_id: str) -> list[dict]:
    """Every citation in a persisted dossier's memo, flattened."""
    run = await _get_run(run_id)
    if run is None:
        raise ValueError(f"no such run: {run_id}")
    record = DossierRunRecord.model_validate(run)
    if record.memo is None:
        return []
    return [
        c.model_dump(mode="json") for section in record.memo.sections for c in section.citations
    ]


@mcp.resource("corpus://stats")
def corpus_stats_resource() -> CorpusStats:
    """Document/chunk/vector counts for the process-wide corpus."""
    return get_corpus().stats()


@mcp.prompt(name="due-diligence-brief")
def due_diligence_brief(subject: str) -> str:
    """A prompt guiding the host's own model through a due-diligence brief."""
    return (
        f"You are conducting due diligence on {subject} using the `praxis` MCP "
        "tools. First call `ingest_document` for any source material you have. "
        "Then call `create_dossier` with the subject. Present the resulting memo "
        "section by section, preserving every citation, and end with the "
        "recommendation and open questions verbatim."
    )


@mcp.prompt(name="investment-memo")
def investment_memo(subject: str, thesis: str = "") -> str:
    """A prompt guiding the host's own model through an investment-memo pass."""
    thesis_line = f" The investment thesis under consideration: {thesis}." if thesis else ""
    return (
        f"Write an investment memo for {subject}.{thesis_line} Use `search_corpus` "
        "to find supporting material and `get_evidence` to check specific claims "
        "before including them. Every claim must carry a citation from one of "
        "these tools — never state a number or fact you haven't looked up."
    )


# Referenced by tests and __main__ so `python -m praxis.mcp` and `praxis mcp`
# both start the same server via the same entrypoint. Transport defaults to
# stdio (PRAXIS_MCP_TRANSPORT=streamable-http for the Docker Compose service).
def main() -> None:
    mcp.run(_settings.mcp_transport)
