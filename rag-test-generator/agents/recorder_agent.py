"""
agents/recorder_agent.py — Browser Recording Agent
====================================================
Responsible for:
  - Opening the target URL in a real browser via Playwright's `codegen`
  - Letting the user interact with the page while actions are recorded
  - Returning the raw recorded Playwright test script

This is the FIRST agent in the recording pipeline.
Input  : URL to record + scenario name
Output : Raw recorded Playwright test script (string)

NOTE: This step is interactive and blocking by design — codegen opens a real
browser + Inspector window, the user performs actions, and closing the browser
ends the recording. It cannot be run unattended or in CI.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from config import PLAYWRIGHT_RUNNER_DIR

console = Console()


def run(url: str, scenario_name: str) -> str:
    """
    Main entry point for the recorder agent.

    Pipeline:
        1. Launch `npx playwright codegen <url>` targeting the playwright-test format
        2. Wait for the user to record actions and close the browser
        3. Read back the raw recorded script

    Args:
        url           : Live URL to open and record a scenario from
        scenario_name : Name used for the temporary raw-recording file

    Returns:
        Raw recorded Playwright test script (string)
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]RECORDER AGENT[/bold cyan]")
    console.print("=" * 60)

    raw_path = PLAYWRIGHT_RUNNER_DIR / "tests" / f"_raw_{scenario_name}.spec.js"

    console.print(Panel(
        f"[bold]Opening browser for:[/bold] {url}\n\n"
        f"Interact with the page to record your scenario.\n"
        f"[dim]Close the browser window when you're done — recording stops automatically.[/dim]",
        title="[bold cyan]🎥 Recording[/bold cyan]",
        border_style="cyan",
    ))

    result = subprocess.run(
        [
            "npx", "playwright", "codegen", url,
            "--target", "playwright-test",
            "-o", str(raw_path),
        ],
        cwd=str(PLAYWRIGHT_RUNNER_DIR),
        check=False,
    )

    if result.returncode != 0 or not raw_path.exists():
        console.print("[red]❌ Recording failed — no script was produced.[/red]")
        raise RuntimeError(
            f"`playwright codegen` exited with code {result.returncode} "
            f"and no output file was found at {raw_path}"
        )

    raw_recording = raw_path.read_text(encoding="utf-8")
    raw_path.unlink(missing_ok=True)  # temporary — the cleaned spec is what gets kept

    console.print(f"[green]✅ Recorded {len(raw_recording.splitlines())} lines of actions[/green]")

    return raw_recording


# ─────────────────────────────────────────────────────────
# Quick test — run directly
# python agents/recorder_agent.py <url> [scenario-name]
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_url  = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    test_name = sys.argv[2] if len(sys.argv) > 2 else "manual-test"

    recording = run(test_url, test_name)
    console.print(f"\n[bold]Raw recording:[/bold]\n{recording}")
