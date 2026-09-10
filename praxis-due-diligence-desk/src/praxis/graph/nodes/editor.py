"""Editor — synthesize the memo, one section per rubric item.

The prose is the model's; the overall recommendation + confidence are computed
deterministically from the evidence (``graph/recommend.py``). Runs on the fast
model tier.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from praxis.graph.context import (
    evidence_indices_by_section,
    evidence_source_ids,
    llm_from,
    render_case,
    render_evidence_by_section,
)
from praxis.graph.prompts import EDITOR
from praxis.graph.recommend import derive_recommendation
from praxis.graph.state import DossierState
from praxis.schemas import PLACEHOLDER_SOURCE_IDS, RUBRIC_SECTIONS, DossierMemo, MemoSection


def _refs(finding) -> str:
    return str(finding.evidence_refs) if finding else "[]"


def _context(state: DossierState) -> str:
    plan = state["plan"]
    thesis = plan.thesis if plan else "(no plan)"
    report = state["redteam"]
    flagged = (
        "; ".join(report.unsupported_claims) if report and report.unsupported_claims else "none"
    )
    real_sources = [s for s in evidence_source_ids(state) if s not in PLACEHOLDER_SOURCE_IDS]
    return (
        f"Subject: {state['subject']}\n"
        f"Thesis: {thesis}\n\n"
        f"Evidence by rubric section:\n{render_evidence_by_section(state)}\n\n"
        f"EVIDENCE SOURCES: {', '.join(real_sources) or 'none'}\n\n"
        f"{render_case('BULL', state['bull'])}\nbull evidence_refs: {_refs(state['bull'])}\n\n"
        f"{render_case('BEAR', state['bear'])}\nbear evidence_refs: {_refs(state['bear'])}\n\n"
        f"Red Team flagged as unsupported (exclude these): {flagged}\n"
        f"Write exactly one section per rubric item, headed verbatim, and cite the "
        f"evidence you use."
    )


def _reconcile_sections(sections: list[MemoSection], state: DossierState) -> list[MemoSection]:
    """Force the memo onto the rubric: one section per rubric item in order, each
    carrying at least the citations of the evidence gathered for it. Any extra
    sections the model wrote are kept after the rubric ones."""
    evidence = state["evidence"]
    idx_by_section = evidence_indices_by_section(state)
    by_heading = {s.heading.strip().lower(): s for s in sections}

    out: list[MemoSection] = []
    for rubric in RUBRIC_SECTIONS:
        ev_items = [evidence[i] for i in idx_by_section.get(rubric, [])]
        ev_cites = [ev.citation for ev in ev_items]
        sec = by_heading.pop(rubric.lower(), None)
        if sec is None:
            body = " ".join(ev.claim for ev in ev_items) or "No evidence was gathered here."
            out.append(MemoSection(heading=rubric, body=body, citations=ev_cites))
        else:
            sec.heading = rubric
            if not sec.citations and ev_cites:
                sec.citations = ev_cites
            out.append(sec)
    out.extend(by_heading.values())
    return out


def editor(state: DossierState, config: RunnableConfig) -> dict:
    llm = llm_from(config)
    memo = llm.generate(
        system=EDITOR,
        user=_context(state),
        schema=DossierMemo,
        role="editor",
    )
    memo.subject = state["subject"]
    memo.sections = _reconcile_sections(memo.sections, state)
    memo.recommendation, memo.confidence = derive_recommendation(state)
    return {"memo": memo}
