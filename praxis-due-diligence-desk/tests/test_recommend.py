from praxis.graph.recommend import derive_recommendation
from praxis.graph.state import initial_state
from praxis.schemas import RUBRIC_SECTIONS, Citation, Evidence, Finding, RedTeamReport


def _ev(section: str, stance: str = "neutral", source: str = "real-src") -> Evidence:
    return Evidence(
        question="q",
        section=section,
        claim="c",
        stance=stance,  # type: ignore[arg-type]
        citation=Citation(source_id=source, title="t", locator="p.1", quote="x"),
    )


def _state(evidence, *, bull=None, bear=None, redteam=None):
    s = initial_state("Acme", "standard")
    s["evidence"] = evidence
    s["bull"], s["bear"], s["redteam"] = bull, bear, redteam
    return s


def _finding(stance: str, n_points: int):
    return Finding(stance=stance, summary="s", points=[f"p{i}" for i in range(n_points)])  # type: ignore[arg-type]


def test_thin_evidence_is_insufficient():
    rec, conf = derive_recommendation(_state([_ev("Overview")]))
    assert rec == "insufficient_evidence"
    assert conf <= 0.35


def test_all_placeholder_is_insufficient():
    ev = [_ev(s, source="no-source") for s in RUBRIC_SECTIONS]
    rec, _ = derive_recommendation(_state(ev))
    assert rec == "insufficient_evidence"


def test_contradiction_forces_pass():
    ev = [_ev(s) for s in RUBRIC_SECTIONS]
    rec, _ = derive_recommendation(
        _state(ev, redteam=RedTeamReport(contradictions=["bull and bear disagree on revenue"]))
    )
    assert rec == "pass"


def test_clean_support_heavy_proceeds():
    ev = [_ev(s, stance="supports") for s in RUBRIC_SECTIONS]
    rec, conf = derive_recommendation(
        _state(ev, bull=_finding("bull", 3), bear=_finding("bear", 1), redteam=RedTeamReport())
    )
    assert rec == "proceed"
    assert 0.2 <= conf <= 0.9


def test_some_refutation_adds_conditions():
    ev = [_ev(s, stance="supports") for s in RUBRIC_SECTIONS[:4]]
    ev += [_ev(s, stance="refutes") for s in RUBRIC_SECTIONS[4:]]
    rec, _ = derive_recommendation(
        _state(ev, bull=_finding("bull", 2), bear=_finding("bear", 2), redteam=RedTeamReport())
    )
    assert rec == "proceed_with_conditions"


def test_confidence_always_in_bounds():
    for ev in ([], [_ev("Overview")], [_ev(s) for s in RUBRIC_SECTIONS]):
        _, conf = derive_recommendation(_state(ev))
        assert 0.0 <= conf <= 0.9
