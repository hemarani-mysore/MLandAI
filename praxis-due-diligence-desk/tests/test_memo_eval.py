"""`evals/memo_eval.py`'s precision/recall/F1 math — pure, no LLM, no I/O."""

import importlib.util
from pathlib import Path

_EVAL = Path(__file__).parents[1] / "evals" / "memo_eval.py"
_spec = importlib.util.spec_from_file_location("memo_eval", _EVAL)
assert _spec and _spec.loader
memo_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(memo_eval)


def _rows(*pairs: tuple[str, str]) -> list[dict]:
    """Each pair is (judge_label, expert_label)."""
    return [{"judge_label": j, "expert_label": e} for j, e in pairs]


def test_perfect_agreement_scores_one_on_everything():
    rows = _rows(("pass", "pass"), ("fail", "fail"), ("pass", "pass"), ("fail", "fail"))
    metrics = memo_eval.score(rows)
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "agreement": 1.0}


def test_a_false_positive_fail_hurts_precision_not_recall():
    # judge says fail, expert says pass -> one false positive on the "fail" class
    rows = _rows(("fail", "fail"), ("fail", "pass"), ("pass", "pass"))
    metrics = memo_eval.score(rows)
    assert metrics["precision"] == 0.5  # 1 true positive / 2 predicted-fail
    assert metrics["recall"] == 1.0  # the one real fail was still caught
    assert metrics["agreement"] == 2 / 3


def test_a_missed_fail_hurts_recall_not_precision():
    # judge says pass, expert says fail -> a missed detection (false negative)
    rows = _rows(("pass", "fail"), ("pass", "pass"), ("fail", "fail"))
    metrics = memo_eval.score(rows)
    assert metrics["precision"] == 1.0  # the one predicted-fail was correct
    assert metrics["recall"] == 0.5  # 1 of 2 real fails caught
    assert metrics["agreement"] == 2 / 3


def test_no_predicted_fails_gives_zero_precision_and_recall_not_a_crash():
    rows = _rows(("pass", "fail"), ("pass", "pass"))
    metrics = memo_eval.score(rows)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_empty_rows_score_as_zero_not_a_crash():
    metrics = memo_eval.score([])
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "agreement": 0.0}


def test_display_name_titlecases_the_kebab_case_id():
    assert memo_eval._display_name("orbitfleet-logistics") == "Orbitfleet Logistics"
