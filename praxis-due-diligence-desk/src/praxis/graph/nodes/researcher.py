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

import asyncio
import json
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from praxis.config import get_settings
from praxis.graph.context import corpus_from, external_tools_from, llm_from
from praxis.graph.prompts import RESEARCHER
from praxis.graph.state import DossierState
from praxis.obs import span
from praxis.rag import RetrievedChunk, search_corpus
from praxis.schemas import Citation, Evidence

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

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


def _parse_web_result(item: Any) -> dict | None:
    """A tool result item may be a raw dict, or (as `langchain-mcp-adapters`
    returns them) a content block whose ``text`` is a JSON-encoded string."""
    if isinstance(item, dict) and "url" in item:
        return item
    text = item.get("text") if isinstance(item, dict) else item
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def _web_search_evidence(question: str, section: str, tool: BaseTool) -> Evidence | None:
    """Best-effort: call an external `web_search`-shaped MCP tool and turn its
    first usable result into a URL-cited Evidence. Never raises — a flaky
    external tool degrades to the no-source path, same as an empty corpus."""
    try:
        raw = await tool.ainvoke({"query": question})
    except Exception:  # noqa: BLE001 - an external tool's failure isn't ours
        return None

    for item in raw if isinstance(raw, list) else [raw]:
        parsed = _parse_web_result(item)
        if parsed is None:
            continue
        url = parsed.get("url") or parsed.get("link")
        if not url:
            continue
        snippet = (parsed.get("snippet") or parsed.get("content") or "").strip()
        title = parsed.get("title") or url
        return Evidence(
            question=question,
            section=section,
            claim=snippet or title,
            stance="neutral",
            citation=Citation(source_id=url, title=title, locator=url, quote=snippet[:280]),
            confidence=0.3,  # unverified external source — deliberately capped
        )
    return None


async def research_one(task: ResearchTask, config: RunnableConfig) -> dict[str, Any]:
    """Answer exactly one sub-question. Runs as a fanned-out ``Send`` branch, so
    its input is a single ``ResearchTask``, not the full ``DossierState``."""
    subject, section, question = task["subject"], task["section"], task["question"]
    with span("research_one", role="research_one", section=section):
        llm = llm_from(config)
        corpus = corpus_from(config)
        k = get_settings().retrieval_k

        # search_corpus is sync (a real embedder makes a blocking HTTP call inside
        # it) — run it off the event loop so concurrent branches don't serialize.
        hits = await asyncio.to_thread(search_corpus, question, k=k, corpus=corpus)
        evidence: Evidence | None
        if hits:
            evidence = await llm.agenerate(
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
            evidence = None
            web_search = external_tools_from(config).get("web_search")
            if web_search is not None:
                evidence = await _web_search_evidence(question, section, web_search)
            if evidence is None:
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
