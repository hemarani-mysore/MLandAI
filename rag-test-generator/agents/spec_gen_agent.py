"""
agents/spec_gen_agent.py — Spec Generation Agent
==================================================
Responsible for:
  - Taking the raw Playwright codegen recording
  - Calling OpenAI to turn it into a clean, well-asserted .spec.js file

This is the SECOND agent in the recording pipeline.
Input  : Raw recorded Playwright script + scenario name
Output : Cleaned Playwright .spec.js source code (string)
"""

import sys
import time
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


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    return text


def _call_openai_code(prompt: str, max_retries: int = 4) -> str:
    """Call OpenAI and return raw code, with retry on transient errors."""
    for attempt in range(max_retries):
        try:
            response = _get_client().chat.completions.create(
                model       = OPENAI_MODEL,
                temperature = OPENAI_TEMPERATURE,
                messages    = [{"role": "user", "content": prompt}],
            )
            return _strip_code_fences(response.choices[0].message.content)
        except (openai.APIConnectionError, openai.APIStatusError, openai.RateLimitError) as exc:
            is_rate_limit = isinstance(exc, openai.RateLimitError)
            wait = 30 if is_rate_limit else 15
            if attempt == max_retries - 1:
                raise
            console.print(f"\n     [yellow]⏳ OpenAI error ({exc.__class__.__name__}). Retrying in {wait}s...[/yellow]")
            time.sleep(wait)
    return ""


_PROMPT_TEMPLATE = """\
You are a senior QA engineer cleaning up a Playwright `codegen` recording into a
polished, maintainable end-to-end test.

RAW RECORDED SCRIPT (from `npx playwright codegen --target playwright-test`):
{raw_recording}

Rewrite this into a complete, valid Playwright test file for scenario "{scenario_name}".
Requirements:
- Import: import {{ test, expect }} from "@playwright/test";
- Give the test() call a clear, descriptive name based on what the recording does
- Add meaningful expect() assertions for the key outcomes of the scenario
  (e.g. page title, visible text, URL after navigation) — don't just replay clicks
- Keep all the original recorded actions and selectors intact
- Output ONLY valid JavaScript — no explanation, no markdown fences"""


def run(raw_recording: str, scenario_name: str) -> str:
    """
    Main entry point for the spec generation agent.

    Args:
        raw_recording : Raw Playwright codegen output from recorder_agent.run()
        scenario_name : Human-readable scenario name (used in the prompt)

    Returns:
        Cleaned Playwright .spec.js source code (string)
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]SPEC GENERATION AGENT[/bold cyan]")
    console.print("=" * 60)

    prompt = _PROMPT_TEMPLATE.format(raw_recording=raw_recording, scenario_name=scenario_name)
    spec_code = _call_openai_code(prompt)

    console.print(f"[green]✅ Generated {len(spec_code.splitlines())} lines of spec code[/green]")

    return spec_code


# ─────────────────────────────────────────────────────────
# Quick test — run directly
# python agents/spec_gen_agent.py
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_recording = """\
import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('https://example.com/');
  await page.getByRole('link', { name: 'More information...' }).click();
});"""

    spec = run(sample_recording, "example-more-info")
    console.print(f"\n[bold]Generated spec:[/bold]\n{spec}")
