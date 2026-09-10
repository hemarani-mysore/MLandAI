"""The memo-structure eval is a CI gate — keep it runnable from pytest too."""

import importlib.util
from pathlib import Path

_EVAL = Path(__file__).parents[1] / "evals" / "memo_structure_eval.py"
_spec = importlib.util.spec_from_file_location("memo_structure_eval", _EVAL)
assert _spec and _spec.loader
mse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mse)


def test_gate_passes_on_a_real_run():
    assert mse.main(["--gate"]) == 0


def test_self_check_catches_a_broken_memo():
    assert mse.main(["--self-check"]) == 0
    memo, report = mse._broken_run()
    assert any(not ok for _, ok in mse._checks(memo, report))
