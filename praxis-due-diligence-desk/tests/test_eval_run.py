"""`evals/run.py --suite all --split test --gate` — fully offline, writes a
report JSON with every suite's metrics, exits 0."""

import importlib.util
import json
import sys
from pathlib import Path

_EVAL = Path(__file__).parents[1] / "evals" / "run.py"
_spec = importlib.util.spec_from_file_location("eval_run", _EVAL)
assert _spec and _spec.loader
eval_run = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eval_run
_spec.loader.exec_module(eval_run)


def test_all_suites_gate_pass_on_the_test_split(tmp_path):
    report_path = tmp_path / "report.json"

    rc = eval_run.main(
        ["--suite", "all", "--split", "test", "--gate", "--report", str(report_path)]
    )

    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["all_gates_passed"] is True
    assert set(report["results"]) == set(eval_run.SUITES)
    for name, result in report["results"].items():
        assert result["gate_passed"] is True, (name, result)


def test_report_carries_every_suites_metric_keys(tmp_path):
    report_path = tmp_path / "report.json"
    eval_run.main(["--suite", "all", "--split", "test", "--report", str(report_path)])
    report = json.loads(report_path.read_text())

    assert "metrics" in report["results"]["rag"]
    assert "checks" in report["results"]["structure"]
    assert "metrics" in report["results"]["memo"]
    assert "caught" in report["results"]["citation"]
    assert "seeded_error_detection_rate" in report["results"]["redteam"]


def test_a_single_suite_runs_alone():
    report = eval_run.run("rag", split="test", gate=False)
    assert set(report["results"]) == {"rag"}


def test_gate_flag_off_never_exits_nonzero_even_if_something_failed(monkeypatch):
    # Force a failure by patching one runner to report a failed gate.
    monkeypatch.setitem(eval_run._RUNNERS, "rag", lambda gate, split: (False, {}))
    rc_no_gate = eval_run.main(["--suite", "rag", "--split", "test"])
    assert rc_no_gate == 0

    rc_gate = eval_run.main(["--suite", "rag", "--split", "test", "--gate"])
    assert rc_gate == 1
