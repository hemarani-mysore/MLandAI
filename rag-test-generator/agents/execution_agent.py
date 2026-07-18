"""
agents/execution_agent.py — Test Execution Agent
==================================================
Responsible for:
  - Running a generated .spec.js file with `npx playwright test`
  - Parsing the JSON reporter output into a typed pass/fail result

This is the THIRD agent in the recording pipeline (and also re-entered on
every fix/retry loop iteration).
Input  : Path to a .spec.js file inside playwright_runner/tests/
Output : ExecutionResult (passed, output, error, dom_snapshot)

NOTE: This is the first subprocess-execution code in this repo — the existing
5-agent pipeline only ever generates test text, it never runs it.
"""

import base64
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel

from config import PLAYWRIGHT_EXEC_TIMEOUT_S, PLAYWRIGHT_RUNNER_DIR

console = Console()


_MAX_DOM_SNAPSHOT_CHARS = 20_000


@dataclass
class ExecutionResult:
    passed       : bool
    output       : str   # human-readable summary (stdout or stats-derived)
    error        : str    # concatenated error messages from failing tests, "" if passed
    dom_snapshot : str = ""   # page HTML captured on failure, "" if none/passed


def _iter_tests(suites: list[dict]):
    """Recursively walk the JSON reporter's (possibly nested) suites for test results."""
    for suite in suites:
        for spec in suite.get("specs", []):
            for test in spec.get("tests", []):
                yield spec, test
        # suites can nest (e.g. describe blocks)
        yield from _iter_tests(suite.get("suites", []))


def _strip_scripts_and_styles(html: str) -> str:
    """
    Drop <script>/<style> content before truncating. Real-world pages front-load
    huge amounts of inline JS/CSS, which would otherwise eat the entire character
    budget before the actual markup (the part with selectors) is even reached.
    """
    html = re.sub(r"<script\b[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style\b[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    return html


def _read_dom_snapshot(attachments: list[dict]) -> str:
    """
    Read the 'dom-snapshot' attachment (page.content() captured by the
    afterEach hook injected by agents/recording_graph.py). Playwright's JSON
    reporter inlines small attachments as base64 in `body`; larger ones get
    written to disk and referenced by `path` instead — handle both.
    """
    for attachment in attachments:
        if attachment.get("name") != "dom-snapshot":
            continue
        body = attachment.get("body")
        if body:
            html = base64.b64decode(body).decode("utf-8", errors="ignore")
            return _strip_scripts_and_styles(html)[:_MAX_DOM_SNAPSHOT_CHARS]
        path = attachment.get("path")
        if path and Path(path).exists():
            html = Path(path).read_text(encoding="utf-8", errors="ignore")
            return _strip_scripts_and_styles(html)[:_MAX_DOM_SNAPSHOT_CHARS]
    return ""


def _parse_reporter_json(raw: str) -> ExecutionResult:
    """
    Parse `npx playwright test --reporter=json` stdout into an ExecutionResult.
    Pure function — no subprocess involved — so it's directly unit-testable.
    """
    data  = json.loads(raw)
    stats = data.get("stats", {})
    passed = stats.get("unexpected", 0) == 0 and stats.get("expected", 0) > 0

    if passed:
        return ExecutionResult(passed=True, output=f"{stats.get('expected', 0)} test(s) passed", error="")

    errors = []
    dom_snapshot = ""
    for spec, test in _iter_tests(data.get("suites", [])):
        for result in test.get("results", []):
            if result.get("status") not in ("passed", "skipped"):
                err = result.get("error", {}) or {}
                message = err.get("message", "").strip()
                if message:
                    errors.append(f"{spec.get('title', 'test')}: {message}")
                if not dom_snapshot:
                    dom_snapshot = _read_dom_snapshot(result.get("attachments", []))

    # Top-level fatal errors (e.g. syntax errors) that never made it into a suite/spec
    for err in data.get("errors", []):
        message = err.get("message", "").strip()
        if message:
            errors.append(message)

    error_text = "\n".join(errors) or "Test failed with no captured error message."
    return ExecutionResult(passed=False, output=error_text, error=error_text, dom_snapshot=dom_snapshot)


def run(spec_path: Path) -> ExecutionResult:
    """
    Main entry point for the execution agent.

    Args:
        spec_path : Path to the .spec.js file to run (inside playwright_runner/tests/)

    Returns:
        ExecutionResult describing pass/fail + error details
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST EXECUTION AGENT[/bold cyan]")
    console.print("=" * 60)

    relative_path = spec_path.relative_to(PLAYWRIGHT_RUNNER_DIR)
    console.print(f"\n[cyan]▶ Running {relative_path}...[/cyan]")

    try:
        result = subprocess.run(
            ["npx", "playwright", "test", str(relative_path), "--reporter=json"],
            cwd=str(PLAYWRIGHT_RUNNER_DIR),
            capture_output=True,
            text=True,
            timeout=PLAYWRIGHT_EXEC_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        console.print(f"[red]❌ Timed out after {PLAYWRIGHT_EXEC_TIMEOUT_S}s[/red]")
        return ExecutionResult(passed=False, output="", error=f"Test run timed out after {PLAYWRIGHT_EXEC_TIMEOUT_S}s")

    try:
        execution_result = _parse_reporter_json(result.stdout)
    except json.JSONDecodeError:
        # `npx` itself failed before Playwright could emit JSON (e.g. missing deps)
        error_text = result.stderr.strip() or result.stdout.strip() or "Unknown execution failure"
        execution_result = ExecutionResult(passed=False, output=error_text, error=error_text)

    if execution_result.passed:
        console.print(Panel(f"[bold green]✅ {execution_result.output}[/bold green]", border_style="green"))
    else:
        console.print(Panel(f"[bold red]❌ Test failed[/bold red]\n\n{execution_result.error}", border_style="red"))

    return execution_result


# ─────────────────────────────────────────────────────────
# Quick test — run directly
# python agents/execution_agent.py <spec-file-relative-to-playwright_runner>
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    spec_arg = sys.argv[1] if len(sys.argv) > 1 else "tests/example.spec.js"
    outcome = run(PLAYWRIGHT_RUNNER_DIR / spec_arg)
    console.print(f"\n[bold]Passed:[/bold] {outcome.passed}")
