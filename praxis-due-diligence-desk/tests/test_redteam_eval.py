"""`evals/redteam_eval.py` — the offline (`--gate`) path replays real,
recorded `red_team` verdicts and correctly detects every seeded error."""

import importlib.util
import sys
from pathlib import Path

_EVAL = Path(__file__).parents[1] / "evals" / "redteam_eval.py"
_spec = importlib.util.spec_from_file_location("redteam_eval", _EVAL)
assert _spec and _spec.loader
redteam_eval = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = redteam_eval  # FIXTURES' Evidence/Finding are pydantic models
_spec.loader.exec_module(redteam_eval)


def test_gate_passes_on_the_recorded_fixtures():
    assert redteam_eval.main(["--gate"]) == 0


def test_detection_rate_is_at_least_the_floor_on_the_recorded_fixtures():
    import asyncio

    report = asyncio.run(redteam_eval.evaluate(gate=True))
    assert report["seeded_error_detection_rate"] >= redteam_eval.REDTEAM_FLOOR
    assert report["n"] == len(redteam_eval.FIXTURES)


def test_every_fixture_plants_its_keyword_in_the_bull_or_bear_case():
    for fixture in redteam_eval.FIXTURES:
        case_text = " ".join(fixture["bull"].points + fixture["bear"].points).lower()
        assert fixture["planted_keyword"].lower() in case_text


def test_detected_is_false_when_neither_list_mentions_the_keyword():
    from praxis.schemas import RedTeamReport

    report = RedTeamReport(unsupported_claims=["something unrelated"], contradictions=[])
    assert redteam_eval._detected(report, "$50M Series C") is False


def test_detected_is_true_when_the_keyword_appears_in_either_list():
    from praxis.schemas import RedTeamReport

    report = RedTeamReport(contradictions=["claims a $50M Series C with no supporting evidence"])
    assert redteam_eval._detected(report, "$50M Series C") is True
