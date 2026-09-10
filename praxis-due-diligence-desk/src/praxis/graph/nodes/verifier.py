"""Verifier — deterministic gate on the memo.

No LLM call. Checks that every citation resolves to a source we actually gathered
evidence from (placeholder ids don't count), that the memo covers the rubric,
and that every section is cited. Phase 4 extends this with NLI claim<->span
checks and turns the flags into hard eval metrics.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from praxis.graph.state import DossierState
from praxis.schemas import PLACEHOLDER_SOURCE_IDS, RUBRIC_SECTIONS, VerificationReport


def verifier(state: DossierState, config: RunnableConfig) -> dict:
    memo = state["memo"]
    plan = state["plan"]
    assert memo is not None and plan is not None

    known_real = {ev.citation.source_id for ev in state["evidence"]} - PLACEHOLDER_SOURCE_IDS
    cited = [c for section in memo.sections for c in section.citations]
    checked = len(cited)
    resolved = sum(1 for c in cited if c.source_id in known_real)
    placeholder = sum(1 for c in cited if c.source_id in PLACEHOLDER_SOURCE_IDS)
    hallucinated = checked - resolved - placeholder
    hallucination_rate = 0.0 if checked == 0 else hallucinated / checked

    covered = {s.heading.strip().lower() for s in memo.sections}
    expected = [s.lower() for s in RUBRIC_SECTIONS]
    missing = [RUBRIC_SECTIONS[i] for i, s in enumerate(expected) if s not in covered]
    coverage = (len(expected) - len(missing)) / len(expected)

    flags: list[str] = []
    if checked == 0:
        flags.append("memo contains no citations")
    if hallucinated > 0:
        flags.append(f"{hallucinated}/{checked} citations do not resolve to gathered evidence")
    if checked > 0 and resolved == 0 and placeholder > 0:
        flags.append("no grounded sources — memo rests on placeholder citations")
    for name in missing:
        flags.append(f"rubric section not covered: {name}")
    for section in memo.sections:
        if not section.citations:
            flags.append(f"section has no citation: {section.heading}")

    return {
        "verification": VerificationReport(
            citations_checked=checked,
            citations_resolved=resolved,
            hallucinated_citation_rate=round(hallucination_rate, 3),
            coverage_score=round(coverage, 3),
            flags=flags,
        )
    }
