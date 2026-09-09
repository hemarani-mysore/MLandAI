"""Render a ``DossierResponse`` as a Markdown memo."""

from __future__ import annotations

from praxis.schemas import DossierResponse


def to_markdown(resp: DossierResponse) -> str:
    memo = resp.memo
    v = resp.verification
    out: list[str] = [
        f"# Due-Diligence Memo — {memo.subject}",
        "",
        f"**Recommendation:** `{memo.recommendation}`  ·  "
        f"**Confidence:** {memo.confidence:.0%}  ·  "
        f"**Evidence:** {resp.evidence_count} items  ·  "
        f"**Gap-fill loops:** {resp.iterations - 1}",
        "",
        "## Summary",
        memo.summary,
        "",
    ]
    for section in memo.sections:
        out += [f"## {section.heading}", section.body, ""]
        for c in section.citations:
            out.append(f"> [{c.source_id}] {c.title} — {c.locator}: “{c.quote}”")
        out.append("")

    if memo.open_questions:
        out += ["## Open Questions", *[f"- {q}" for q in memo.open_questions], ""]

    if resp.sources:
        out += ["## Sources", *[f"- `{s}`" for s in resp.sources], ""]

    status = "PASS" if v.ok else "FLAGGED"
    out += [
        "---",
        f"### Verification: {status}",
        f"- citations: {v.citations_resolved}/{v.citations_checked} resolved "
        f"(hallucination rate {v.hallucinated_citation_rate:.0%})",
        f"- rubric coverage: {v.coverage_score:.0%}",
        *[f"- ⚠️ {f}" for f in v.flags],
    ]
    return "\n".join(out)
