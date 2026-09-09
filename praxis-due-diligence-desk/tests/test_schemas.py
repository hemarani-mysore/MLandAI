import pytest
from pydantic import ValidationError

from praxis.schemas import Citation, DossierRequest, Evidence, ResearchPlan


def test_dossier_request_rejects_tiny_subject():
    with pytest.raises(ValidationError):
        DossierRequest(subject="A")


def test_evidence_confidence_bounds():
    cite = Citation(source_id="s1", title="t", locator="p.1", quote="q")
    with pytest.raises(ValidationError):
        Evidence(question="q", claim="c", citation=cite, confidence=1.5)


def test_research_plan_requires_a_question():
    with pytest.raises(ValidationError):
        ResearchPlan(subject="X", thesis="t", sub_questions=[])
