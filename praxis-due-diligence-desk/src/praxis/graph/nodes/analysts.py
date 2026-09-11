"""Bull and Bear analysts — run concurrently, each builds one side of the case."""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig

from praxis.graph.context import llm_from, render_evidence
from praxis.graph.prompts import ANALYST_BEAR, ANALYST_BULL
from praxis.graph.state import DossierState
from praxis.schemas import Finding

_SYSTEM = {"bull": ANALYST_BULL, "bear": ANALYST_BEAR}


def _clip_refs(refs: list[int], n: int) -> list[int]:
    """Keep only in-range evidence indices, deduped, in first-seen order."""
    seen: dict[int, None] = {}
    for r in refs:
        if 0 <= r < n:
            seen.setdefault(r, None)
    return list(seen)


async def _analyst(
    state: DossierState, config: RunnableConfig, stance: Literal["bull", "bear"]
) -> dict:
    llm = llm_from(config)
    finding = await llm.agenerate(
        system=_SYSTEM[stance],
        user=f"Subject: {state['subject']}\n\nEvidence:\n{render_evidence(state)}",
        schema=Finding,
        role=f"{stance}_analyst",
    )
    finding.stance = stance
    finding.evidence_refs = _clip_refs(finding.evidence_refs, len(state["evidence"]))
    return {stance: finding}


async def bull(state: DossierState, config: RunnableConfig) -> dict:
    return await _analyst(state, config, "bull")


async def bear(state: DossierState, config: RunnableConfig) -> dict:
    return await _analyst(state, config, "bear")
