"""Researcher — answer each sub-question with a corpus-grounded piece of evidence.

Fanned out via LangGraph `Send`: one concurrent branch per sub-question, capped
by `PRAXIS_RESEARCH_CONCURRENCY` (the run's `max_concurrency`). ``dispatch_research``
(the initial plan) and ``dispatch_gap_fill`` (the gap-fill loop) are the two
fan-out points; ``research_one`` handles exactly one question — hybrid retrieval
over the corpus, then one *grounded* ``Evidence`` (its citation snapped to a
retrieved chunk if the model cites anything else) — and appends it via the
``evidence`` reducer.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from praxis.config import get_settings
from praxis.graph.context import corpus_from, llm_from
from praxis.graph.prompts import RESEARCHER
from praxis.graph.state import DossierState
from praxis.rag import RetrievedChunk, search_corpus
from praxis.schemas import Citation, Evidence

_NO_SOURCE = Citation(
    source_id="no-source", title="(no matching source in corpus)", locator="-", quote="-"
)


class ResearchTask(TypedDict):
    """The ``Send`` payload for one ``research_one`` branch — deliberately not
    the full ``DossierState``; a branch only needs its own question."""

    subject: str
    section: str
    question: str


def _dispatch(subject: str, items: list[tuple[str, str]]) -> list[Send]:
    return [
        Send("research_one", ResearchTask(subject=subject, section=section, question=question))
        for section, question in items
    ]


def dispatch_research(state: DossierState) -> list[Send]:
    """Fan out one branch per rubric sub-question from the plan."""
    plan = state["plan"]
    assert plan is not None, "planner must run before research"
    return _dispatch(state["subject"], [(sq.section, sq.question) for sq in plan.sub_questions])


def dispatch_gap_fill(state: DossierState) -> list[Send]:
    """Fan out one branch per red-team follow-up question."""
    report = state["redteam"]
    assert report is not None, "red_team must run before a gap-fill dispatch"
    return _dispatch(state["subject"], [("Gap-fill", q) for q in report.gap_questions])


def _render_hits(hits: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] source_id={h.chunk.source_id} | {h.chunk.title} {h.chunk.locator}\n{h.chunk.raw_text}"
        for i, h in enumerate(hits)
    )


def _ground(evidence: Evidence, hits: list[RetrievedChunk]) -> Evidence:
    valid = {h.chunk.source_id: h for h in hits}
    match = valid.get(evidence.citation.source_id)
    if match is None:
        top = hits[0]
        evidence.citation = top.citation
        evidence.confidence = min(evidence.confidence, 0.6)
    else:
        evidence.citation = match.citation
    return evidence


def research_one(task: ResearchTask, config: RunnableConfig) -> dict[str, Any]:
    """Answer exactly one sub-question. Runs as a fanned-out ``Send`` branch, so
    its input is a single ``ResearchTask``, not the full ``DossierState``."""
    llm = llm_from(config)
    corpus = corpus_from(config)
    k = get_settings().retrieval_k
    subject, section, question = task["subject"], task["section"], task["question"]

    hits = search_corpus(question, k=k, corpus=corpus)
    if hits:
        evidence = llm.generate(
            system=RESEARCHER,
            user=(
                f"Subject: {subject}\n"
                f"Section: {section}\n"
                f"Question: {question}\n\n"
                f"Retrieved context:\n{_render_hits(hits)}"
            ),
            schema=Evidence,
            role="researcher",
        )
        evidence = _ground(evidence, hits)
    else:
        evidence = Evidence(
            question=question,
            section=section,
            claim=f"No corpus evidence found for: {question}",
            stance="neutral",
            citation=_NO_SOURCE,
            confidence=0.0,
        )
    evidence.question = question
    evidence.section = section
    return {"evidence": [evidence]}
