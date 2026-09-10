from praxis.graph.nodes.verifier import verifier
from praxis.graph.state import initial_state
from praxis.schemas import (
    RUBRIC_SECTIONS,
    Citation,
    Evidence,
    MemoSection,
    ResearchPlan,
    SubQuestion,
)
from praxis.schemas import DossierMemo as Memo


def _plan() -> ResearchPlan:
    return ResearchPlan(
        subject="Acme",
        thesis="t",
        sub_questions=[SubQuestion(section=s, question="q") for s in RUBRIC_SECTIONS],
    )


def _ev(section: str, source: str) -> Evidence:
    return Evidence(
        question="q",
        section=section,
        claim="c",
        stance="neutral",
        citation=Citation(source_id=source, title="T", locator="p.1", quote="q"),
    )


def _run(memo: Memo, evidence: list[Evidence]):
    state = initial_state("Acme", "standard")
    state["plan"] = _plan()
    state["evidence"] = evidence
    state["memo"] = memo
    return verifier(state, {})["verification"]


def _memo(sections: list[MemoSection]) -> Memo:
    return Memo(
        subject="Acme", summary="s", recommendation="pass", confidence=0.3, sections=sections
    )


def test_placeholder_citations_do_not_count_as_grounded():
    memo = _memo(
        [
            MemoSection(
                heading=s,
                body="b",
                citations=[Citation(source_id="no-source", title="T", locator="-", quote="-")],
            )
            for s in RUBRIC_SECTIONS
        ]
    )
    report = _run(memo, [_ev(s, "no-source") for s in RUBRIC_SECTIONS])

    assert report.citations_resolved == 0
    assert report.hallucinated_citation_rate == 0.0  # placeholder != hallucinated
    assert any("no grounded sources" in f for f in report.flags)


def test_missing_and_uncited_sections_are_flagged():
    memo = _memo([MemoSection(heading="Overview", body="b", citations=[])])
    report = _run(memo, [_ev("Overview", "real-1")])

    assert any("rubric section not covered: Market & Competition" in f for f in report.flags)
    assert any("section has no citation: Overview" in f for f in report.flags)
    assert report.coverage_score < 1.0


def test_clean_grounded_memo_passes():
    memo = _memo(
        [
            MemoSection(
                heading=s,
                body="b",
                citations=[Citation(source_id="real-1", title="T", locator="p.1", quote="q")],
            )
            for s in RUBRIC_SECTIONS
        ]
    )
    report = _run(memo, [_ev(s, "real-1") for s in RUBRIC_SECTIONS])

    assert report.flags == []
    assert report.coverage_score == 1.0
    assert report.hallucinated_citation_rate == 0.0
