"""Orchestrator — run one or every eval suite, aggregate a single JSON report.

    uv run python evals/run.py --suite all --split test --gate    # fully offline (CI)
    uv run python evals/run.py --suite all --split dev            # live judge, calibration
    uv run python evals/run.py --suite memo --split online        # nightly drift check
    uv run python evals/run.py --suite rag --report evals/reports/latest.json

`--split` only applies to the `memo` suite (`dev`/`test`/`online` — see
`memo_eval.py`); the other suites ignore it. Every suite's gate status
(pass/fail against its own floor) is always computed and included in the
report; `--gate` only decides whether an overall failure makes this process
exit non-zero — matching every individual `evals/*.py` script's own `--gate`
convention. Each suite is loaded dynamically from its own file (the same
`importlib.util.spec_from_file_location` pattern the test suite already uses
for `evals/*.py`) and its report-producing function is called exactly once —
deliberately *not* by shelling out to `python evals/<name>.py` as a
subprocess, which would either double the (possibly live-model, real-money)
work or require reimplementing each script's CLI printing here. This trades
away each suite's own richly-formatted stdout report (still available by
running that `evals/<name>.py` directly) for a single aggregated JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).parent
SUITES = ("rag", "structure", "memo", "citation", "redteam")


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # some of these define pydantic models / dataclasses
    spec.loader.exec_module(module)
    return module


def _run_rag(*, gate: bool) -> tuple[bool, dict]:
    mod = _load("rag_eval", "rag_eval.py")
    report = mod.evaluate()
    ok = all(v >= mod.THRESHOLDS[k] for k, v in report["metrics"].items())
    return ok, report


def _run_structure(*, gate: bool) -> tuple[bool, dict]:
    mod = _load("memo_structure_eval", "memo_structure_eval.py")
    memo, report = mod._real_run()
    checks = mod._checks(memo, report)
    ok = all(v for _, v in checks)
    return ok, {"checks": [{"name": n, "ok": v} for n, v in checks]}


def _run_memo(*, split: str, gate: bool) -> tuple[bool, dict]:
    mod = _load("memo_eval", "memo_eval.py")
    report = mod.evaluate(split)
    ok = report["metrics"]["f1"] >= mod.MEMO_F1_FLOOR
    return ok, report


def _run_citation(*, gate: bool) -> tuple[bool, dict]:
    mod = _load("citation_eval", "citation_eval.py")
    report = mod.self_check()
    return report["caught"], report


def _run_redteam(*, gate: bool) -> tuple[bool, dict]:
    mod = _load("redteam_eval", "redteam_eval.py")
    # redteam_eval's own `gate` flag is what picks recorded (offline) vs live
    # — the same flag run.py's own --gate carries, so it stays offline exactly
    # when this process's own --gate would (matching every other suite here).
    report = asyncio.run(mod.evaluate(gate=gate))
    ok = report["seeded_error_detection_rate"] >= mod.REDTEAM_FLOOR
    return ok, report


_RUNNERS = {
    "rag": lambda gate, split: _run_rag(gate=gate),
    "structure": lambda gate, split: _run_structure(gate=gate),
    "memo": lambda gate, split: _run_memo(split=split, gate=gate),
    "citation": lambda gate, split: _run_citation(gate=gate),
    "redteam": lambda gate, split: _run_redteam(gate=gate),
}


def run(suite: str, *, split: str, gate: bool) -> dict:
    names = SUITES if suite == "all" else (suite,)
    results = {}
    all_ok = True
    for name in names:
        print(f"\n=== {name} ===")
        ok, report = _RUNNERS[name](gate, split)
        all_ok = all_ok and ok
        results[name] = {"gate_passed": ok, **report}
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "suite": suite,
        "split": split,
        "gate": gate,
        "all_gates_passed": all_ok,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    parser.add_argument("--split", choices=("dev", "test", "online"), default="test")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--report", metavar="PATH", help="write the aggregated JSON report here")
    args = parser.parse_args(argv)

    report = run(args.suite, split=args.split, gate=args.gate)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        print(f"\nwrote {out}")

    print(f"\n{'=' * 40}")
    for name, result in report["results"].items():
        mark = "ok " if result["gate_passed"] else "FAIL"
        print(f"  {mark} {name}")

    if args.gate and not report["all_gates_passed"]:
        print("\nGATE FAILED", file=sys.stderr)
        return 1
    if args.gate:
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
