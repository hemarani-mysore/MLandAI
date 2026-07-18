"""
agents/fix_agent.py — Test Fixing Agent
=========================================
Responsible for:
  - Taking a failing .spec.js file + its Playwright error output
  - Calling OpenAI to produce a fixed version, or decide to give up

This is the FOURTH agent in the recording pipeline — re-entered on every
failed execution until the test passes or the LLM decides to stop.
Input  : Failing spec code + error output + attempt counters + DOM snapshot on failure
Output : FixResult (fixed_code, give_up, reason)
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from openai import OpenAI
from rich.console import Console

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE

console = Console()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


@dataclass
class FixResult:
    fixed_code : str
    give_up    : bool
    reason     : str


def _call_openai_json(prompt: str, max_retries: int = 4) -> dict:
    """Call OpenAI in JSON mode and return the parsed result, with retry on transient errors."""
    for attempt in range(max_retries):
        try:
            response = _get_client().chat.completions.create(
                model           = OPENAI_MODEL,
                temperature     = OPENAI_TEMPERATURE,
                response_format = {"type": "json_object"},
                messages        = [{"role": "user", "content": prompt}],
            )
            return json.loads(response.choices[0].message.content)
        except (openai.APIConnectionError, openai.APIStatusError, openai.RateLimitError) as exc:
            is_rate_limit = isinstance(exc, openai.RateLimitError)
            wait = 30 if is_rate_limit else 15
            if attempt == max_retries - 1:
                raise
            console.print(f"\n     [yellow]⏳ OpenAI error ({exc.__class__.__name__}). Retrying in {wait}s...[/yellow]")
            time.sleep(wait)
    return {}


_PROMPT_TEMPLATE = """\
You are a senior QA engineer debugging a failing Playwright test.

FAILING SPEC CODE:
{spec_code}

PLAYWRIGHT ERROR OUTPUT:
{error_output}
{dom_snapshot_section}
ATTEMPT {attempt} of {max_attempts} allowed.

If a PAGE HTML AT TIME OF FAILURE section is present above, do this check FIRST,
before considering timing/waits: for every CSS selector or locator string used in
FAILING SPEC CODE (e.g. `input[name="q"]`, `.some-class`, `#some-id`), search for
that exact tag+attribute combination in the PAGE HTML. If it is not there, the
selector itself is wrong — even if an element with a similar attribute exists under
a DIFFERENT tag name. For example: if the code uses `input[name="q"]` but the HTML
only contains `<textarea name="q">`, the fix is to change the selector's tag to
`textarea` — an element that doesn't exist will never become visible no matter how
long you wait, so increasing timeouts or adding waitForSelector calls will NOT fix
a wrong-tag selector.

Then respond with a JSON object with exactly these fields:
{{
  "fixed_code": "the corrected, complete .spec.js file content (empty string if giving up)",
  "give_up": false,
  "reason": "one sentence explaining your fix, or why you're giving up"
}}

Rules:
- Only set "give_up" to true if the failure is not fixable by changing the test itself
  (e.g. the page requires credentials/state you don't have, or the target site is unreachable)
- If the DOM check above found a wrong-tag or wrong-attribute selector, fix that selector —
  do not just add waits/timeouts around a selector that can never match
- Otherwise, fix the timing or assertion that caused the failure
- "fixed_code" must be complete, valid JavaScript — the whole file, not a diff
- Keep any existing `test.afterEach` hook in the file unchanged — it's infrastructure, not part of the test logic
- Return ONLY the JSON object, no explanation, no markdown"""


def run(
    spec_code    : str,
    error_output : str,
    attempt      : int,
    max_attempts : int,
    dom_snapshot : str = "",
) -> FixResult:
    """
    Main entry point for the fix agent.

    Args:
        spec_code    : Current (failing) .spec.js source
        error_output : Playwright's error/stack trace from the failed run
        attempt      : Current attempt number (1-indexed)
        max_attempts : Hard cap on attempts (safety net alongside the LLM's own give-up decision)
        dom_snapshot : Page HTML captured at the moment of failure (from the afterEach hook
                       injected by agents/recording_graph.py), "" if unavailable

    Returns:
        FixResult with the fixed code and the LLM's give-up decision
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST FIXING AGENT[/bold cyan]")
    console.print("=" * 60)
    console.print(f"[cyan]🔧 Attempt {attempt}/{max_attempts} — asking OpenAI for a fix...[/cyan]")

    dom_snapshot_section = (
        f"\nPAGE HTML AT TIME OF FAILURE:\n{dom_snapshot}\n" if dom_snapshot else "\n"
    )

    prompt = _PROMPT_TEMPLATE.format(
        spec_code             = spec_code,
        error_output          = error_output,
        dom_snapshot_section  = dom_snapshot_section,
        attempt               = attempt,
        max_attempts          = max_attempts,
    )
    data = _call_openai_json(prompt)

    result = FixResult(
        fixed_code = data.get("fixed_code", "") or spec_code,
        give_up    = bool(data.get("give_up", False)),
        reason     = data.get("reason", ""),
    )

    if result.give_up:
        console.print(f"[yellow]🛑 LLM decided to give up: {result.reason}[/yellow]")
    else:
        console.print(f"[green]✅ Fix produced: {result.reason}[/green]")

    return result


# ─────────────────────────────────────────────────────────
# Quick test — run directly
# python agents/fix_agent.py
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_spec = """\
import { test, expect } from '@playwright/test';

test('example', async ({ page }) => {
  await page.goto('https://example.com/');
  await page.getByRole('link', { name: 'Wrong Link Text' }).click();
});"""
    sample_error = "TimeoutError: locator.click: Timeout 30000ms exceeded waiting for getByRole('link', { name: 'Wrong Link Text' })"

    fix = run(sample_spec, sample_error, attempt=1, max_attempts=5)
    console.print(f"\n[bold]Fixed code:[/bold]\n{fix.fixed_code}")
