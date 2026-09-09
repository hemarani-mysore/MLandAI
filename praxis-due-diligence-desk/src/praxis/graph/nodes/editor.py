"""Editor — synthesize the final memo (runs on the fast model tier)."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from praxis.graph.context import (
    evidence_source_ids,
    llm_from,
    render_case,
    render_evidence,
)
from praxis.graph.prompts import EDITOR
from praxis.graph.state import DossierState
from praxis.schemas import DossierMemo


def _context(state: DossierState) -> str:
    plan = state["plan"]
    thesis = plan.thesis if plan else "(no plan)"
    report = state["redteam"]
    flagged = "; ".join(report.unsupported_claims) if report else ""
    sources = ", ".join(evidence_source_ids(state)) or "none"
    return (
        f"Subject: {state['subject']}\n"
        f"Thesis: {thesis}\n\n"
        f"Evidence:\n{render_evidence(state)}\n\n"
        f"EVIDENCE SOURCES: {sources}\n\n"
        f"{render_case('BULL', state['bull'])}\n\n"
        f"{render_case('BEAR', state['bear'])}\n\n"
        f"Red Team flagged as unsupported (exclude these): {flagged or 'none'}\n"
        f"Every citation you write must use one of the EVIDENCE SOURCES ids above."
    )


def editor(state: DossierState, config: RunnableConfig) -> dict:
    llm = llm_from(config)
    memo = llm.generate(
        system=EDITOR,
        user=_context(state),
        schema=DossierMemo,
        role="editor",
    )
    memo.subject = state["subject"]
    return {"memo": memo}
