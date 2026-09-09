"""Researcher — answer each sub-question with a corpus-grounded piece of evidence.

For each question: hybrid retrieval over the corpus -> feed the top chunks to the
LLM -> emit one ``Evidence`` whose citation is *grounded* (snapped to a retrieved
chunk if the model cites anything else). Phase 2 fans this out with ``Send``.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from praxis.config import get_settings
from praxis.graph.context import corpus_from, llm_from
from praxis.graph.prompts import RESEARCHER
from praxis.graph.state import DossierState
from praxis.rag import RetrievedChunk, search_corpus
from praxis.schemas import Citation, Evidence

_NO_SOURCE = Citation(
    source_id="no-source", title="(no matching source in corpus)", locator="-", quote="-"
)


def _questions(state: DossierState) -> list[tuple[str, str]]:
    if state["iteration"] == 0 or not state["redteam"]:
        plan = state["plan"]
        assert plan is not None, "planner must run before researcher"
        return [(sq.section, sq.question) for sq in plan.sub_questions]
    return [("Gap-fill", q) for q in state["redteam"].gap_questions]


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


def researcher(state: DossierState, config: RunnableConfig) -> dict:
    llm = llm_from(config)
    corpus = corpus_from(config)
    k = get_settings().retrieval_k
    gathered: list[Evidence] = []

    for section, question in _questions(state):
        hits = search_corpus(question, k=k, corpus=corpus)
        if hits:
            evidence = llm.generate(
                system=RESEARCHER,
                user=(
                    f"Subject: {state['subject']}\n"
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
        gathered.append(evidence)

    return {"evidence": gathered}
