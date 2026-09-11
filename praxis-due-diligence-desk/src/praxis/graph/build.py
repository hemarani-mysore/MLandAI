"""Wire the nodes into a graph and expose a one-call runner."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

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


@lru_cache
def build_graph():
    """Compile the dossier graph.

    planner --Send-->> research_one (fan-out, N in parallel) --(fan-in)--> {bull, bear}
        --> red_team --(gaps? Send-->> research_one)--> ...
                     \\--(clean / cap hit)--> editor -> verifier -> END

    ``research_one`` runs once per sub-question, dispatched via ``Send`` from
    ``dispatch_research`` (the initial plan) or ``dispatch_gap_fill`` (inside
    ``route_after_red_team``, for the bounded gap-fill loop). Concurrency is
    capped by ``max_concurrency`` on the run config (``PRAXIS_RESEARCH_CONCURRENCY``).
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

    return g.compile()


def run_dossier(
    request: DossierRequest,
    *,
    llm: StructuredLLM | None = None,
    corpus: Corpus | None = None,
    max_gap_loops: int | None = None,
    research_concurrency: int | None = None,
) -> DossierResponse:
    graph = build_graph()
    configurable: dict = {"llm": llm or get_llm()}
    if corpus is not None:
        configurable["corpus"] = corpus
    if max_gap_loops is not None:
        configurable["max_gap_loops"] = max_gap_loops
    concurrency = (
        research_concurrency
        if research_concurrency is not None
        else get_settings().research_concurrency
    )

    final = graph.invoke(
        initial_state(request.subject, request.depth),
        config={"configurable": configurable, "max_concurrency": concurrency},
    )

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
