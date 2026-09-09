"""Verifier — deterministic gate on the memo.

No LLM call. Checks that every citation in the memo resolves to a source we
actually gathered evidence from, and that section coverage is reasonable.
Phase 4 extends this with NLI claim<->span checks and turns the flags into
hard eval metrics.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from praxis.graph.state import DossierState
from praxis.schemas import VerificationReport


def verifier(state: DossierState, config: RunnableConfig) -> dict:
    memo = state["memo"]
    plan = state["plan"]
    assert memo is not None and plan is not None

    known_sources = {ev.citation.source_id for ev in state["evidence"]}
    cited = [c for section in memo.sections for c in section.citations]
    resolved = sum(1 for c in cited if c.source_id in known_sources)
    checked = len(cited)
    hallucination_rate = 0.0 if checked == 0 else (checked - resolved) / checked

    covered = {s.heading.strip().lower() for s in memo.sections}
    expected = {sq.section.strip().lower() for sq in plan.sub_questions}
    coverage = len(covered & expected) / len(expected) if expected else 0.0

    flags: list[str] = []
    if checked == 0:
        flags.append("memo contains no citations")
    if hallucination_rate > 0:
        flags.append(
            f"{checked - resolved}/{checked} citations do not resolve to gathered evidence"
        )
    if coverage < 0.5:
        flags.append(f"low rubric coverage ({coverage:.0%})")

    return {
        "verification": VerificationReport(
            citations_checked=checked,
            citations_resolved=resolved,
            hallucinated_citation_rate=round(hallucination_rate, 3),
            coverage_score=round(coverage, 3),
            flags=flags,
        )
    }
