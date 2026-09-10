from praxis.graph import run_dossier
from praxis.graph.nodes.analysts import _clip_refs
from praxis.graph.nodes.editor import _reconcile_sections
from praxis.graph.state import initial_state
from praxis.llm import FakeStructuredLLM
from praxis.schemas import (
    RUBRIC_SECTIONS,
    Citation,
    DossierMemo,
    DossierRequest,
    Evidence,
    MemoSection,
)


def _ev(section: str, source: str = "src-a") -> Evidence:
    return Evidence(
        question="q",
        section=section,
        claim=f"claim about {section}",
        stance="neutral",
        citation=Citation(source_id=source, title="T", locator="p.1", quote="q"),
    )


def _state_with_full_evidence():
    state = initial_state("Acme", "standard")
    state["evidence"] = [_ev(s) for s in RUBRIC_SECTIONS]
    return state


def test_clip_refs_drops_out_of_range_and_dedupes():
    assert _clip_refs([0, 2, 5, 2, -1], 3) == [0, 2]
    assert _clip_refs([], 5) == []


def test_reconcile_fills_missing_rubric_sections_and_backfills_citations():
    state = _state_with_full_evidence()
    model_sections = [MemoSection(heading="Overview", body="written by model", citations=[])]

    out = _reconcile_sections(model_sections, state)

    assert [s.heading for s in out] == list(RUBRIC_SECTIONS)
    overview = next(s for s in out if s.heading == "Overview")
    assert overview.body == "written by model"  # model prose kept
    assert overview.citations[0].source_id == "src-a"  # citations backfilled
    assert all(s.citations for s in out)  # every rubric section ends up cited


def test_reconcile_keeps_extra_sections_after_the_rubric():
    state = _state_with_full_evidence()
    out = _reconcile_sections([MemoSection(heading="Appendix", body="x")], state)

    assert [s.heading for s in out][: len(RUBRIC_SECTIONS)] == list(RUBRIC_SECTIONS)
    assert out[-1].heading == "Appendix"


def test_editor_recommendation_is_derived_not_taken_from_the_model():
    # The model insists "proceed" at 0.99; with no corpus the verdict must be
    # recomputed from the (empty) evidence.
    over = DossierMemo(
        subject="Acme",
        summary="s",
        recommendation="proceed",
        confidence=0.99,
        sections=[MemoSection(heading=s, body="b") for s in RUBRIC_SECTIONS],
    )
    llm = FakeStructuredLLM(overrides={"editor": [over]})
    resp = run_dossier(DossierRequest(subject="Acme Robotics"), llm=llm)

    assert resp.memo.recommendation == "insufficient_evidence"
    assert resp.memo.confidence <= 0.35
