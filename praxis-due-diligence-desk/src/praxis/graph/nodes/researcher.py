"""Researcher — answer each sub-question with a cited piece of evidence.

Phase 0: a sequential loop over the plan's sub-questions (or, on a gap-fill
pass, over the Red Team's follow-up questions), each producing one ``Evidence``.
Phase 1 gives each question real hybrid retrieval; Phase 2 fans the questions
out across parallel branches with the ``Send`` API.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from praxis.graph.context import llm_from
from praxis.graph.prompts import RESEARCHER
from praxis.graph.state import DossierState
from praxis.schemas import Evidence


def _questions(state: DossierState) -> list[tuple[str, str]]:
    if state["iteration"] == 0 or not state["redteam"]:
        plan = state["plan"]
        assert plan is not None, "planner must run before researcher"
        return [(sq.section, sq.question) for sq in plan.sub_questions]
    return [("Gap-fill", q) for q in state["redteam"].gap_questions]


def researcher(state: DossierState, config: RunnableConfig) -> dict:
    llm = llm_from(config)
    gathered: list[Evidence] = []
    for section, question in _questions(state):
        evidence = llm.generate(
            system=RESEARCHER,
            user=(f"Subject: {state['subject']}\nSection: {section}\nQuestion: {question}"),
            schema=Evidence,
            role="researcher",
        )
        evidence.question = question
        evidence.section = section
        gathered.append(evidence)
    return {"evidence": gathered}
