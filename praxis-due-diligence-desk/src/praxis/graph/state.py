"""The shared state threaded through the graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from praxis.schemas import (
    DossierMemo,
    Evidence,
    Finding,
    RedTeamReport,
    ResearchPlan,
    VerificationReport,
)


class DossierState(TypedDict):
    subject: str
    depth: str

    plan: ResearchPlan | None
    # Researcher branches append; ``operator.add`` merges concurrent writes.
    evidence: Annotated[list[Evidence], operator.add]

    bull: Finding | None
    bear: Finding | None
    redteam: RedTeamReport | None
    iteration: int

    memo: DossierMemo | None
    verification: VerificationReport | None


def initial_state(subject: str, depth: str) -> DossierState:
    return DossierState(
        subject=subject,
        depth=depth,
        plan=None,
        evidence=[],
        bull=None,
        bear=None,
        redteam=None,
        iteration=0,
        memo=None,
        verification=None,
    )
