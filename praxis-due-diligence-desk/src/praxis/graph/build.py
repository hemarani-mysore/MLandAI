"""Wire the nodes into a graph and expose a one-call runner."""

from __future__ import annotations

import asyncio
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from praxis.config import get_settings
from praxis.graph.nodes import (
    ResearchTask,
    bear,
    bull,
    dispatch_research,
    editor,
    planner,
    red_team,
    research_one,
    route_after_red_team,
    verifier,
)
from praxis.graph.state import DossierState, initial_state
from praxis.llm import StructuredLLM, get_llm
from praxis.schemas import PLACEHOLDER_SOURCE_IDS, DossierRequest, DossierResponse

if TYPE_CHECKING:
    from praxis.rag import Corpus

# The real graph nodes — used by api/sse.py to isolate genuine node execution
# from LangGraph's internal wrapping of the conditional-edge router functions
# (dispatch_research, route_after_red_team), which also emit on_chain_start/
# on_chain_end events, attributed to the *calling* node's metadata.
NODE_NAMES = frozenset(
    {"planner", "research_one", "bull", "bear", "red_team", "editor", "verifier"}
)


def to_response(request: DossierRequest, final: DossierState) -> DossierResponse:
    """Build the API response from a finished run's state — shared by
    ``arun_dossier`` (the `ainvoke` return value) and the SSE streaming path
    (``graph.aget_state(...).values``, once the checkpointed run reaches END)."""
    assert final["plan"] is not None
    assert final["memo"] is not None
    assert final["verification"] is not None
    sources = sorted(
        {
            ev.citation.source_id
            for ev in final["evidence"]
            if ev.citation.source_id not in PLACEHOLDER_SOURCE_IDS
        }
    )
    return DossierResponse(
        request=request,
        plan=final["plan"],
        memo=final["memo"],
        verification=final["verification"],
        evidence_count=len(final["evidence"]),
        iterations=final["iteration"],
        sources=sources,
    )


@lru_cache
def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Compile the dossier graph.

    planner --Send-->> research_one (fan-out, N in parallel) --(fan-in)--> {bull, bear}
        --> red_team --(gaps? Send-->> research_one)--> ...
                     \\--(clean / cap hit)--> editor -> verifier -> END

    ``research_one`` runs once per sub-question, dispatched via ``Send`` from
    ``dispatch_research`` (the initial plan) or ``dispatch_gap_fill`` (inside
    ``route_after_red_team``, for the bounded gap-fill loop). Concurrency is
    capped by ``max_concurrency`` on the run config (``PRAXIS_RESEARCH_CONCURRENCY``).

    All the LLM-calling nodes (``planner``, ``research_one``, ``bull``, ``bear``,
    ``red_team``, ``editor``) are ``async def`` — concurrent branches overlap on
    the event loop, not just in a thread pool. ``verifier`` does no I/O and stays
    sync; LangGraph runs sync nodes fine inside an async invocation. Invoke with
    ``graph.ainvoke(...)`` (see ``arun_dossier`` below) — ``.invoke()`` would fail
    on the async node functions.

    ``checkpointer`` is optional (``None`` for the CLI and most tests — a plain,
    non-resumable run). When given, the caller must also pass a ``thread_id``
    in the run config (``arun_dossier`` handles this). ``@lru_cache`` still
    works here: distinct checkpointer objects (identity-hashed) get distinct
    compiled graphs, and repeated calls with the *same* checkpointer instance
    hit the cache.
    """
    g: StateGraph = StateGraph(DossierState)

    g.add_node("planner", planner)
    # `research_one` is a Send fan-out target with its own input schema
    # (ResearchTask), not the graph's DossierState — mypy's overload resolution
    # for add_node doesn't cleanly cover that shape; verified correct at runtime
    # (tests/test_graph_fanout.py) and against LangGraph's own documented pattern.
    g.add_node("research_one", research_one, input_schema=ResearchTask)  # type: ignore[arg-type,call-overload]
    g.add_node("bull", bull)
    g.add_node("bear", bear)
    g.add_node("red_team", red_team)
    g.add_node("editor", editor)
    g.add_node("verifier", verifier)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", dispatch_research, ["research_one"])
    g.add_edge("research_one", "bull")
    g.add_edge("research_one", "bear")
    g.add_edge("bull", "red_team")
    g.add_edge("bear", "red_team")
    g.add_conditional_edges("red_team", route_after_red_team, ["research_one", "editor"])
    g.add_edge("editor", "verifier")
    g.add_edge("verifier", END)

    return g.compile(checkpointer=checkpointer)


async def arun_dossier(
    request: DossierRequest,
    *,
    llm: StructuredLLM | None = None,
    corpus: Corpus | None = None,
    max_gap_loops: int | None = None,
    research_concurrency: int | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> DossierResponse:
    """Run a dossier end to end. The primary entry point — call this directly
    from async code (e.g. a FastAPI ``async def`` handler); use ``run_dossier``
    from sync code instead.

    Pass ``checkpointer`` to make the run resumable (each superstep is saved);
    ``thread_id`` then identifies it in the checkpoint store, defaulting to a
    fresh uuid4 if a checkpointer is given but no id — the API uses the
    persisted ``DossierRun.id`` instead, so a run's checkpoint history and its
    DB row share the same id.
    """
    graph = build_graph(checkpointer)
    configurable: dict = {"llm": llm or get_llm()}
    if corpus is not None:
        configurable["corpus"] = corpus
    if max_gap_loops is not None:
        configurable["max_gap_loops"] = max_gap_loops
    if checkpointer is not None:
        configurable["thread_id"] = thread_id or str(uuid.uuid4())
    concurrency = (
        research_concurrency
        if research_concurrency is not None
        else get_settings().research_concurrency
    )

    final = await graph.ainvoke(
        initial_state(request.subject, request.depth),
        config={"configurable": configurable, "max_concurrency": concurrency},
    )
    return to_response(request, final)


def run_dossier(
    request: DossierRequest,
    *,
    llm: StructuredLLM | None = None,
    corpus: Corpus | None = None,
    max_gap_loops: int | None = None,
    research_concurrency: int | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    thread_id: str | None = None,
) -> DossierResponse:
    """Sync wrapper over ``arun_dossier``, for the CLI and other sync callers.

    Do NOT call this from inside an already-running event loop (e.g. an async
    FastAPI handler) — ``asyncio.run`` raises there. Call ``arun_dossier``
    directly in async code instead.
    """
    return asyncio.run(
        arun_dossier(
            request,
            llm=llm,
            corpus=corpus,
            max_gap_loops=max_gap_loops,
            research_concurrency=research_concurrency,
            checkpointer=checkpointer,
            thread_id=thread_id,
        )
    )
