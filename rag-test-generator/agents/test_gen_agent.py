"""
agents/test_gen_agent.py — Playwright Test Generation Agent
============================================================
Responsible for:
  - Reading the TestGapReport from the analysis agent
  - For each test gap: retrieving relevant code from ChromaDB via RAG
  - Calling OpenAI to generate a valid Playwright .spec.ts or pytest file
  - Saving generated tests to outputs/generated_tests/

This is the FOURTH agent in the pipeline.
Input  : TestGapReport + ChromaDB collection
Output : .spec.ts / .py test files in outputs/generated_tests/
"""

import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from config import OPENAI_MODEL, OPENAI_TEMPERATURE, OPENAI_API_KEY, OUTPUT_DIR
from agents.embedding_agent import retrieve
from models.report import TestGap, TestGapReport

console = Console()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ─────────────────────────────────────────────────────────
# Shared test boilerplate (injected into every prompt)
# ─────────────────────────────────────────────────────────

_PLAYWRIGHT_CONVENTIONS = textwrap.dedent("""
PLAYWRIGHT TEST CONVENTIONS FOR THIS REPO:
- Import: import { expect, test } from "@playwright/test"
- Import helpers from: "./utils/random", "./utils/user", "./utils/privateApi", "./config.ts"
- Available helpers:
    randomEmail()       → unique test email
    randomPassword()    → secure random password
    logInUser(page, email, password)   → logs in and waits for /
    logOutUser(page)                   → logs out via user menu
    createUser({ email, password })    → creates user via private API
    firstSuperuser, firstSuperuserPassword  → superuser credentials from config
- Use page.getByTestId("...") for form inputs
- Use page.getByRole("button", { name: "..." }) for buttons
- Use page.getByRole("heading", { name: "..." }) for headings
- Use page.getByRole("link", { name: "..." }) for links
- Add test.use({ storageState: { cookies: [], origins: [] } }) for unauthenticated tests
- Group related tests with test.describe()
- Use beforeAll/beforeEach for shared setup
""").strip()

_PYTEST_CONVENTIONS = textwrap.dedent("""
PYTEST CONVENTIONS FOR THIS REPO:
- Import: import pytest, from fastapi.testclient import TestClient
- The test client is available as a fixture called `client`
- Authentication tokens come from the `superuser_token_headers` fixture
- Use `normal_user_token_headers` for regular user auth
- Helper: `get_user_authentication_headers(client, email, password)`
- Test functions are named test_<action>_<scenario>
- Use pytest.mark.parametrize for multiple input cases
- Assert response.status_code before accessing response.json()
""").strip()


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _build_context(chunks: list[dict], max_chars: int = 10_000) -> str:
    parts, total = [], 0
    for c in chunks:
        snippet = f"// File: {c['file_path']}\n{c['content']}\n"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n---\n".join(parts)


def _call_openai_code(prompt: str, max_retries: int = 4) -> str:
    """Call OpenAI and return raw code, with retry on transient errors."""
    for attempt in range(max_retries):
        try:
            response = _get_client().chat.completions.create(
                model       = OPENAI_MODEL,
                temperature = OPENAI_TEMPERATURE,
                messages    = [{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            return text
        except (openai.APIConnectionError, openai.APIStatusError, openai.RateLimitError) as exc:
            is_rate_limit = isinstance(exc, openai.RateLimitError)
            wait = 30 if is_rate_limit else 15
            if attempt == max_retries - 1:
                raise
            console.print(f"\n     [yellow]⏳ OpenAI error ({exc.__class__.__name__}). Retrying in {wait}s...[/yellow]")
            time.sleep(wait)
    return ""


def _gap_output_file(gap: TestGap) -> str:
    """Determine the output filename for a given gap."""
    name = gap.name.lower()

    # UI pages → .spec.ts  (gap name is "Page Name /route" — use only the page name)
    if gap.gap_type == "ui_page":
        page_name = name.split("/")[0].strip()   # "login ", "recover password ", …
        slug = page_name.replace(" ", "-").strip("-")
        return f"{slug}.spec.ts"

    # API endpoints → group into utils-api.spec.ts
    if gap.gap_type == "api_endpoint":
        return "utils-api.spec.ts"

    # Python functions → test_<slug>.py
    if "security" in name or "password" in name or "hash" in name:
        return "test_security.py"
    if "error" in name or "utils" in name or "initials" in name:
        return "test_frontend_utils.spec.ts"
    if "database" in name or "retry" in name or "pre_start" in name:
        return "test_startup.py"

    # Fallback
    slug = name.split()[0].replace("/", "").replace("(", "").replace(")", "")
    return f"test_{slug}.py" if gap.file_path.endswith(".py") else f"{slug}.spec.ts"


# ─────────────────────────────────────────────────────────
# Generators per gap type
# ─────────────────────────────────────────────────────────

def _generate_ui_test(gap: TestGap, collection) -> str:
    source_chunks = retrieve(collection, f"{gap.name} {gap.route if hasattr(gap, 'route') else ''} React component page", top_k=6)
    test_chunks   = retrieve(collection, "Playwright spec test page goto click fill expect", top_k=4)
    context       = _build_context(source_chunks + test_chunks)

    prompt = f"""\
You are a senior QA engineer writing Playwright end-to-end tests for a React + FastAPI application.

{_PLAYWRIGHT_CONVENTIONS}

EXISTING SOURCE CODE CONTEXT:
{context}

GAP TO COVER:
- Page/Component : {gap.name}
- Source file    : {gap.file_path}
- Gap reason     : {gap.reason}
- Priority       : {gap.priority}

Write a complete Playwright .spec.ts test file for this page.
Cover:
  1. Page loads and key UI elements are visible
  2. Happy-path user interaction (main flow)
  3. Validation / error state (invalid input, auth failure, etc.)
  4. Access control (if applicable — unauthenticated user is redirected)

Rules:
- Output ONLY valid TypeScript — no explanation, no markdown fences
- Follow the exact conventions listed above
- Use realistic test data (randomEmail(), randomPassword())
- Each test must be independent (no shared mutable state between tests)
- Do not duplicate tests that already exist for sign-up, items, or user-settings pages"""

    return _call_openai_code(prompt)


def _generate_api_test(gaps: list[TestGap], collection) -> str:
    """Generate a single .spec.ts covering multiple API endpoint gaps."""
    queries = " ".join(g.name for g in gaps)
    source_chunks = retrieve(collection, f"FastAPI endpoint route {queries}", top_k=8)
    test_chunks   = retrieve(collection, "Playwright request API test fetch", top_k=4)
    context       = _build_context(source_chunks + test_chunks)

    gaps_desc = "\n".join(
        f"  - {g.method if hasattr(g, 'method') else ''} {g.name} ({g.file_path}): {g.reason}"
        for g in gaps
    )

    prompt = f"""\
You are a senior QA engineer writing Playwright API tests for a FastAPI backend.

{_PLAYWRIGHT_CONVENTIONS}

EXISTING SOURCE CODE CONTEXT:
{context}

GAPS TO COVER:
{gaps_desc}

Write a complete Playwright .spec.ts file that tests these API endpoints using
Playwright's built-in `request` fixture (page.request or the APIRequestContext).

For each endpoint cover:
  1. Happy path (valid request → expected 200/201 response)
  2. Auth failure (no token or invalid token → 401/403)
  3. Invalid input (missing required fields → 422)

Rules:
- Output ONLY valid TypeScript — no explanation, no markdown fences
- Use the `request` fixture from "@playwright/test"
- Base URL is process.env.VITE_API_URL or "http://localhost:8000"
- Follow the exact conventions listed above"""

    return _call_openai_code(prompt)


def _generate_python_test(gap: TestGap, collection) -> str:
    source_chunks = retrieve(collection, f"{gap.name} {gap.file_path}", top_k=6)
    test_chunks   = retrieve(collection, "pytest test function assert import", top_k=4)
    context       = _build_context(source_chunks + test_chunks)

    prompt = f"""\
You are a senior Python engineer writing pytest unit tests for a FastAPI application.

{_PYTEST_CONVENTIONS}

EXISTING SOURCE CODE CONTEXT:
{context}

GAP TO COVER:
- Function/module : {gap.name}
- Source file     : {gap.file_path}
- Gap reason      : {gap.reason}
- Priority        : {gap.priority}

Write a complete pytest test file covering this gap.
Cover:
  1. Happy-path behaviour (valid inputs → expected output)
  2. Edge cases (empty string, None, boundary values)
  3. Error / exception cases (invalid input raises expected error)

Rules:
- Output ONLY valid Python — no explanation, no markdown fences
- Follow the pytest conventions listed above
- Import the function under test with the correct module path
- Each test function must have a clear docstring explaining what it verifies"""

    return _call_openai_code(prompt)


# ─────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────

def run(report: TestGapReport, collection) -> dict[str, str]:
    """
    Main entry point for the test generation agent.

    Pipeline:
        1. Group test gaps by output file
        2. For each group, retrieve RAG context + call OpenAI
        3. Save generated test files to outputs/generated_tests/

    Args:
        report     : TestGapReport from analysis_agent.run()
        collection : ChromaDB collection from embedding_agent.run()

    Returns:
        dict mapping output filename → generated test code
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]TEST GENERATION AGENT[/bold cyan]")
    console.print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group API gaps together into one file
    api_gaps = [g for g in report.test_gaps if g.gap_type == "api_endpoint"]
    other_gaps = [g for g in report.test_gaps if g.gap_type != "api_endpoint"]

    # Build a work list: (output_filename, gaps_list)
    work: list[tuple[str, list[TestGap]]] = []
    if api_gaps:
        work.append(("utils-api.spec.ts", api_gaps))
    for gap in other_gaps:
        fname = _gap_output_file(gap)
        work.append((fname, [gap]))

    # Deduplicate: if same file appears multiple times, merge the gaps
    merged: dict[str, list[TestGap]] = {}
    for fname, gaps in work:
        merged.setdefault(fname, []).extend(gaps)

    generated: dict[str, str] = {}

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating tests...", total=len(merged))

        for filename, gaps in merged.items():
            console.print(f"\n  [cyan]→ {filename}[/cyan] ({len(gaps)} gap(s))")

            try:
                if filename.endswith(".py"):
                    code = _generate_python_test(gaps[0], collection)
                elif filename == "utils-api.spec.ts":
                    code = _generate_api_test(gaps, collection)
                else:
                    code = _generate_ui_test(gaps[0], collection)

                out_path = OUTPUT_DIR / filename
                out_path.write_text(code)
                generated[filename] = code
                console.print(f"     [green]✅ Saved → {out_path}[/green]")

            except Exception as exc:
                console.print(f"     [red]❌ Failed: {exc}[/red]")

            progress.advance(task, 1)

    # Summary panel
    console.print(Panel(
        f"[bold]Test generation complete[/bold]\n\n"
        + "\n".join(
            f"  • [cyan]{fname}[/cyan] — {len(gaps)} gap(s)"
            for fname, gaps in merged.items()
            if fname in generated
        )
        + f"\n\n  [dim]Output: {OUTPUT_DIR}[/dim]",
        title="[bold green]Generated Tests[/bold green]",
        border_style="green",
    ))

    return generated


# ─────────────────────────────────────────────────────────
# Quick test — run directly
# python agents/test_gen_agent.py
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from dataclasses import fields
    from agents.embedding_agent import get_chroma_collection
    from models.report import TestGapReport, APIEndpoint, UIPage, TestGap

    # Load saved report
    report_path = Path(__file__).parent.parent / "outputs" / "test_gap_report.json"
    raw = json.loads(report_path.read_text())

    report = TestGapReport(
        repo_url      = raw["repo_url"],
        generated_at  = raw["generated_at"],
        api_endpoints = [APIEndpoint(**e) for e in raw["api_endpoints"]],
        ui_pages      = [UIPage(**p)      for p in raw["ui_pages"]],
        auth_flows    = raw["auth_flows"],
        test_gaps     = [TestGap(**g)     for g in raw["test_gaps"]],
    )

    collection = get_chroma_collection()
    generated  = run(report, collection)

    console.print(f"\n[bold green]Done! {len(generated)} test files generated.[/bold green]")
