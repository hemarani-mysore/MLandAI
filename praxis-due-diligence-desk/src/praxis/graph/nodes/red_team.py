"""Red Team — audit the cases, then decide: gap-fill loop or hand off to editor."""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from praxis.graph.context import llm_from, max_gap_loops, render_case, render_evidence
from praxis.graph.nodes.researcher import dispatch_gap_fill
from praxis.graph.prompts import RED_TEAM
from praxis.graph.state import DossierState
from praxis.schemas import RedTeamReport


def red_team(state: DossierState, config: RunnableConfig) -> dict:
    llm = llm_from(config)
    report = llm.generate(
        system=RED_TEAM,
        user=(
            f"Subject: {state['subject']}\n\n"
            f"Evidence:\n{render_evidence(state)}\n\n"
            f"{render_case('BULL', state['bull'])}\n\n"
            f"{render_case('BEAR', state['bear'])}"
        ),
        schema=RedTeamReport,
        role="red_team",
    )
    return {"redteam": report, "iteration": state["iteration"] + 1}


def route_after_red_team(
    state: DossierState, config: RunnableConfig
) -> list[Send] | Literal["editor"]:
    report = state["redteam"]
    has_gaps = bool(report and report.gap_questions)
    if has_gaps and state["iteration"] <= max_gap_loops(config):
        return dispatch_gap_fill(state)
    return "editor"
