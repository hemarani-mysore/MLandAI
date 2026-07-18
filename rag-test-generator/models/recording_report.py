"""
models/recording_report.py — Data models for the Recording Pipeline run report
================================================================================
These dataclasses represent the structured output of the record → generate →
execute → self-heal → report pipeline (agents/recording_graph.py).
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class AttemptRecord:
    attempt_number : int
    passed         : bool
    error_summary  : str = ""


@dataclass
class RecordingRunReport:
    url             : str
    scenario_name   : str
    generated_at    : str
    attempts        : list[AttemptRecord] = field(default_factory=list)
    final_status    : str = ""   # "passed" | "gave_up" | "max_attempts_reached"
    give_up_reason  : str = ""
    final_spec_path : str = ""

    # ── serialisation ────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def save_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render_markdown())

    def _render_markdown(self) -> str:
        status_label = {
            "passed"              : "✅ Passed",
            "gave_up"             : "🛑 Gave up",
            "max_attempts_reached": "⚠️ Max attempts reached",
        }.get(self.final_status, self.final_status)

        lines = [
            "# Recording Run Report",
            f"\n**URL:** {self.url}",
            f"**Scenario:** {self.scenario_name}",
            f"**Generated:** {self.generated_at}",
            f"**Final status:** {status_label}",
        ]

        if self.give_up_reason:
            lines.append(f"**Give-up reason:** {self.give_up_reason}")

        lines += [
            f"**Final spec:** `{self.final_spec_path}`",
            "\n---\n",
            "## Attempts",
            "",
            "| Attempt | Result | Error |",
            "|---|---|---|",
        ]
        for a in self.attempts:
            result = "✅ Passed" if a.passed else "❌ Failed"
            lines.append(f"| {a.attempt_number} | {result} | {a.error_summary or '—'} |")

        return "\n".join(lines)


if __name__ == "__main__":
    # Quick smoke test: python models/recording_report.py
    report = RecordingRunReport(
        url           = "https://example.com",
        scenario_name = "example-checkout",
        generated_at  = "2026-07-16 12:00:00",
        attempts      = [
            AttemptRecord(1, False, "Timeout waiting for selector #submit"),
            AttemptRecord(2, True),
        ],
        final_status    = "passed",
        final_spec_path = "outputs/recorded_tests/example-checkout.spec.js",
    )
    print(report._render_markdown())
