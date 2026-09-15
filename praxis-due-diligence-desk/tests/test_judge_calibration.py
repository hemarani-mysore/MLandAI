"""Eval-of-eval: does the judge prompt itself still work? Runs entirely
offline (the `test` split's recorded verdicts, see
`evals/cassettes/judge_test.json`) so this is a safe, fast CI guard against a
silently-broken JUDGE_PROMPT — not a live judge quality check."""

import importlib.util
from pathlib import Path

_EVAL = Path(__file__).parents[1] / "evals" / "memo_eval.py"
_spec = importlib.util.spec_from_file_location("memo_eval", _EVAL)
assert _spec and _spec.loader
memo_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(memo_eval)


def test_gate_passes_on_the_recorded_test_split():
    assert memo_eval.main(["--split", "test", "--gate"]) == 0


def test_judge_f1_on_the_recorded_test_split_is_at_least_070():
    report = memo_eval.evaluate("test")
    assert report["metrics"]["f1"] >= 0.70, report
