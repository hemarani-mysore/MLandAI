"""Planner — decompose the request into a rubric-covering research plan."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from praxis.graph.context import llm_from
from praxis.graph.prompts import PLANNER
from praxis.graph.state import DossierState
from praxis.schemas import ResearchPlan


async def planner(state: DossierState, config: RunnableConfig) -> dict:
    llm = llm_from(config)
    plan = await llm.agenerate(
        system=PLANNER,
        user=f"Subject: {state['subject']}\nDepth: {state['depth']}",
        schema=ResearchPlan,
        role="planner",
    )
    plan.subject = state["subject"]
    return {"plan": plan}
