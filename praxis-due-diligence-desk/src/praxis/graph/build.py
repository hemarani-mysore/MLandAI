"""Wire the nodes into a graph and expose a one-call runner."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from praxis.graph.nodes import (
    bear,
    bull,
    editor,
    planner,
    red_team,
    researcher,
    route_after_red_team,
    verifier,
)
from praxis.graph.state import DossierState, initial_state
from praxis.llm import StructuredLLM, get_llm
from praxis.schemas import DossierRequest, DossierResponse

if TYPE_CHECKING:
    from praxis.rag import Corpus


@lru_cache
def build_graph():
    """Compile the dossier graph.

    planner -> researcher -> {bull, bear} -> red_team --(gaps?)--> researcher
                                                      \\--(done)--> editor -> verifier -> END
    """
    g: StateGraph = StateGraph(DossierState)

    g.add_node("planner", planner)
    g.add_node("researcher", researcher)
    g.add_node("bull", bull)
    g.add_node("bear", bear)
    g.add_node("red_team", red_team)
    g.add_node("editor", editor)
    g.add_node("verifier", verifier)

    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "bull")
    g.add_edge("researcher", "bear")
    g.add_edge("bull", "red_team")
    g.add_edge("bear", "red_team")
    g.add_conditional_edges(
        "red_team",
        route_after_red_team,
        {"researcher": "researcher", "editor": "editor"},
    )
    g.add_edge("editor", "verifier")
    g.add_edge("verifier", END)

    return g.compile()


def run_dossier(
    request: DossierRequest,
    *,
    llm: StructuredLLM | None = None,
    corpus: Corpus | None = None,
    max_gap_loops: int | None = None,
) -> DossierResponse:
    graph = build_graph()
    configurable: dict = {"llm": llm or get_llm()}
    if corpus is not None:
        configurable["corpus"] = corpus
    if max_gap_loops is not None:
        configurable["max_gap_loops"] = max_gap_loops

    final = graph.invoke(
        initial_state(request.subject, request.depth),
        config={"configurable": configurable},
    )

    sources = sorted(
        {
            ev.citation.source_id
            for ev in final["evidence"]
            if ev.citation.source_id not in ("no-source", "stub-001")
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
