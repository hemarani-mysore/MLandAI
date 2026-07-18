"""
agents/report_agent.py — Recording Run Report Agent
=====================================================
Responsible for:
  - Building a RecordingRunReport from the final graph state
  - Saving it as JSON + Markdown
  - Copying the final spec file into the recorded-tests output directory

This is the FIFTH and final agent in the recording pipeline.
Input  : Final RecordingState from agents/recording_graph.py
Output : Path to the saved Markdown report
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from config import RECORDED_TESTS_DIR
from models.recording_report import AttemptRecord, RecordingRunReport

if TYPE_CHECKING:
    from agents.recording_graph import RecordingState

console = Console()


def run(state: "RecordingState") -> Path:
    """
    Main entry point for the report agent.

    Args:
        state : Final RecordingState after the execute/fix loop has ended

    Returns:
        Path to the saved Markdown report
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST REPORT AGENT[/bold cyan]")
    console.print("=" * 60)

    scenario_name = state["scenario_name"]

    if state["execution_passed"]:
        final_status = "passed"
    elif state["give_up"]:
        final_status = "gave_up"
    else:
        final_status = "max_attempts_reached"

    # Copy the final spec into the persistent artifacts directory
    final_spec_path = RECORDED_TESTS_DIR / f"{scenario_name}.spec.js"
    spec_path = Path(state["spec_path"])
    if spec_path.exists():
        shutil.copy(spec_path, final_spec_path)

    report = RecordingRunReport(
        url             = state["url"],
        scenario_name   = scenario_name,
        generated_at    = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        attempts        = [AttemptRecord(**a) for a in state["attempts_history"]],
        final_status    = final_status,
        give_up_reason  = state.get("give_up_reason", ""),
        final_spec_path = str(final_spec_path),
    )

    json_path = RECORDED_TESTS_DIR / f"{scenario_name}_report.json"
    md_path   = RECORDED_TESTS_DIR / f"{scenario_name}_report.md"
    report.save_json(json_path)
    report.save_markdown(md_path)

    style = "green" if final_status == "passed" else "yellow"
    console.print(Panel(
        f"[bold]Recording run complete[/bold]\n\n"
        f"  Status     : {final_status}\n"
        f"  Attempts   : {len(report.attempts)}\n"
        f"  Final spec : {final_spec_path}\n"
        f"  Report     : {md_path}",
        title="[bold]Recording Run Report[/bold]",
        border_style=style,
    ))

    return md_path


# ─────────────────────────────────────────────────────────
# Quick test — run directly
# python agents/report_agent.py
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    fake_state = {
        "url"              : "https://example.com",
        "scenario_name"    : "example-scenario",
        "spec_path"        : str(RECORDED_TESTS_DIR / "example-scenario.spec.js"),
        "execution_passed" : True,
        "give_up"          : False,
        "give_up_reason"   : "",
        "attempts_history" : [{"attempt_number": 1, "passed": True, "error_summary": ""}],
    }
    Path(fake_state["spec_path"]).write_text("// placeholder spec")
    run(fake_state)
