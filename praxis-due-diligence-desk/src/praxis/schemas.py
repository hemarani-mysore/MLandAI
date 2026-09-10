"""Typed contracts for the whole pipeline.

Every agent boundary in Praxis is a Pydantic model: the planner emits a
``ResearchPlan``, researchers emit ``Evidence``, analysts emit ``Finding``, the
critic emits a ``RedTeamReport``, the editor emits a ``DossierMemo``, and the
verifier emits a ``VerificationReport``. Keeping these strict makes the graph
testable and the eval framework (Phase 4) straightforward.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Stance = Literal["supports", "refutes", "neutral"]
Recommendation = Literal["proceed", "proceed_with_conditions", "pass", "insufficient_evidence"]
Depth = Literal["quick", "standard", "deep"]

# The fixed rubric every plan must cover, so dossiers are comparable across
# subjects and the eval harness can score section coverage.
RUBRIC_SECTIONS: tuple[str, ...] = (
    "Overview",
    "Market & Competition",
    "Team & Funding",
    "Technology & IP",
    "Traction & Financials",
    "Risks & Red Flags",
)

# Source ids that stand in for "nothing real was retrieved". A citation pointing
# at one of these is not grounded — the verifier and the recommendation heuristic
# both treat it as unresolved.
PLACEHOLDER_SOURCE_IDS: frozenset[str] = frozenset({"no-source", "stub-001"})


class Citation(BaseModel):
    """A pointer back to a specific piece of source material."""

    source_id: str = Field(description="Stable id of the ingested source or URL")
    title: str
    locator: str = Field(description="Page ref ('p.12'), section, or full URL")
    quote: str = Field(description="The exact span the claim rests on")


class Evidence(BaseModel):
    """One researched fact with its provenance."""

    question: str
    section: str = "Overview"
    claim: str
    stance: Stance = "neutral"
    citation: Citation
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class SubQuestion(BaseModel):
    section: str
    question: str
    source_hints: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    subject: str
    thesis: str = Field(description="What decision this dossier informs")
    sub_questions: list[SubQuestion] = Field(min_length=1)


class Finding(BaseModel):
    """A bull or bear case, built only from supplied evidence."""

    stance: Literal["bull", "bear"]
    summary: str
    points: list[str] = Field(min_length=1)
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Step-by-step chain of thought, surfaced in the UI",
    )
    evidence_refs: list[int] = Field(
        default_factory=list, description="Indices into the shared evidence list"
    )


class RedTeamReport(BaseModel):
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    gap_questions: list[str] = Field(
        default_factory=list, description="Follow-up questions to close the gaps"
    )


class MemoSection(BaseModel):
    heading: str
    body: str
    citations: list[Citation] = Field(default_factory=list)


class DossierMemo(BaseModel):
    subject: str
    summary: str
    recommendation: Recommendation
    confidence: float = Field(ge=0.0, le=1.0)
    sections: list[MemoSection] = Field(min_length=1)
    open_questions: list[str] = Field(default_factory=list)


class VerificationReport(BaseModel):
    citations_checked: int
    citations_resolved: int
    hallucinated_citation_rate: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    flags: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.flags


class DossierRequest(BaseModel):
    subject: str = Field(min_length=2)
    depth: Depth = "standard"


class DossierResponse(BaseModel):
    request: DossierRequest
    plan: ResearchPlan
    memo: DossierMemo
    verification: VerificationReport
    evidence_count: int
    iterations: int
    sources: list[str] = Field(
        default_factory=list, description="Distinct corpus source ids the evidence drew on"
    )
