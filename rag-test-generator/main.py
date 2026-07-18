"""
main.py — CLI entry point for RAG Test Generator
=================================================
Usage:
  python main.py --help
  python main.py verify-setup
  python main.py analyze --repo <github-url>
  python main.py generate --repo <github-url> [--push] [--dry-run] [--reset]
  python main.py record --url <live-url> [--name <scenario-name>] [--max-attempts N]
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

app     = typer.Typer(help="Agentic RAG Test Generator — AI-powered Playwright test generation")
console = Console()


def _print_banner():
    text = Text()
    text.append("RAG Test Generator\n", style="bold cyan")
    text.append("AI-powered Playwright test generation from any GitHub repo\n", style="dim")
    text.append("Stack: OpenAI · ChromaDB · Playwright · GitHub API", style="dim")
    console.print(Panel(text, border_style="cyan"))


def _step(n: int, label: str):
    console.print(Rule(f"[bold cyan]Step {n}: {label}[/bold cyan]", style="cyan"))


def _slugify_scenario_name(url: str, name: str | None) -> str:
    """Build a filesystem/URL-safe scenario name, auto-derived from the URL + timestamp if not given."""
    if name:
        base = name
    else:
        parsed = urlparse(url)
        base = f"{parsed.netloc}{parsed.path}"

    slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    if not name:
        slug = f"{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return slug or "recording"


def _load_report(report_path: Path):
    """Deserialise a TestGapReport from the JSON file saved by the analysis agent."""
    from models.report import TestGapReport, APIEndpoint, UIPage, TestGap
    raw = json.loads(report_path.read_text())
    return TestGapReport(
        repo_url      = raw["repo_url"],
        generated_at  = raw["generated_at"],
        api_endpoints = [APIEndpoint(**e) for e in raw["api_endpoints"]],
        ui_pages      = [UIPage(**p)      for p in raw["ui_pages"]],
        auth_flows    = raw["auth_flows"],
        test_gaps     = [TestGap(**g)     for g in raw["test_gaps"]],
    )


# ─────────────────────────────────────────────────────────
# verify-setup
# ─────────────────────────────────────────────────────────

@app.command()
def verify_setup():
    """Verify all dependencies and environment variables are configured correctly."""
    _print_banner()
    console.print("\n[bold]Verifying setup...[/bold]\n")

    checks_passed = True

    console.print("[cyan]1. Checking environment variables...[/cyan]")
    try:
        from config import validate_config
        validate_config()
        console.print("   [green]✅ All environment variables set[/green]")
    except EnvironmentError as e:
        console.print(f"   [red]❌ {e}[/red]")
        checks_passed = False

    console.print("[cyan]2. Testing OpenAI API connection...[/cyan]")
    try:
        from openai import OpenAI
        from config import OPENAI_API_KEY, OPENAI_MODEL
        client   = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model    = OPENAI_MODEL,
            messages = [{"role": "user", "content": "Say 'API connection successful' in exactly those words."}],
        )
        console.print(f"   [green]✅ OpenAI API — {response.choices[0].message.content.strip()}[/green]")
    except Exception as e:
        console.print(f"   [red]❌ OpenAI API error: {e}[/red]")
        checks_passed = False

    console.print("[cyan]3. Testing ChromaDB...[/cyan]")
    try:
        import chromadb
        from config import CHROMA_DIR
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection("setup-test-check")
        collection.add(documents=["ping"], ids=["ping"])
        client.delete_collection("setup-test-check")
        console.print("   [green]✅ ChromaDB working[/green]")
    except Exception as e:
        console.print(f"   [red]❌ ChromaDB error: {e}[/red]")
        checks_passed = False

    console.print("[cyan]4. Testing GitHub API...[/cyan]")
    try:
        from github import Github, Auth
        from config import GITHUB_TOKEN
        g    = Github(auth=Auth.Token(GITHUB_TOKEN))
        user = g.get_user()
        console.print(f"   [green]✅ GitHub API — logged in as {user.login}[/green]")
    except Exception as e:
        console.print(f"   [red]❌ GitHub API error: {e}[/red]")
        checks_passed = False

    console.print("[cyan]5. Testing GitPython...[/cyan]")
    try:
        import git
        console.print(f"   [green]✅ GitPython {git.__version__}[/green]")
    except Exception as e:
        console.print(f"   [red]❌ GitPython error: {e}[/red]")
        checks_passed = False

    console.print("[cyan]6. Testing Node.js / Playwright (recording pipeline)...[/cyan]")
    try:
        from config import PLAYWRIGHT_RUNNER_DIR
        result = subprocess.run(
            ["npx", "playwright", "--version"],
            cwd=str(PLAYWRIGHT_RUNNER_DIR),
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            console.print(f"   [green]✅ {result.stdout.strip()}[/green]")
        else:
            raise RuntimeError(result.stderr.strip() or "npx playwright --version failed")
    except Exception as e:
        console.print(f"   [red]❌ Node/Playwright error: {e}[/red]")
        console.print("   [dim]Run: cd playwright_runner && npm install && npx playwright install --with-deps chromium[/dim]")
        checks_passed = False

    if not checks_passed:
        console.print("\n[red]Some checks failed. Fix the errors above before continuing.[/red]")
        raise typer.Exit(1)

    console.print("\n[bold green]All checks passed — ready to generate tests.[/bold green]")
    console.print("\n  [cyan]python main.py generate --repo <github-url> --push[/cyan]\n")


# ─────────────────────────────────────────────────────────
# analyze
# ─────────────────────────────────────────────────────────

@app.command()
def analyze(
    repo  : str  = typer.Option(..., "--repo", "-r", help="GitHub repository URL to analyze"),
    reset : bool = typer.Option(False, "--reset", help="Re-clone and re-embed even if cached"),
):
    """
    Ingest a GitHub repo and produce a test gap report (no test generation).
    Runs Steps 1–3 of the pipeline: ingest → embed → analyze.
    """
    _print_banner()

    from config import validate_config, OUTPUT_DIR
    try:
        validate_config()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Step 1 — Ingest
    _step(1, "Ingestion Agent")
    from agents.ingestion_agent import run as ingest
    code_files = ingest(repo)

    # Step 2 — Embed
    _step(2, "Embedding Agent")
    from agents.embedding_agent import run as embed
    collection = embed(code_files, reset=reset)

    # Step 3 — Analyze
    _step(3, "Analysis Agent")
    from agents.analysis_agent import run as analyse
    report = analyse(collection, repo_url=repo)

    console.print(f"\n[bold green]Analysis complete.[/bold green]")
    console.print(f"  Endpoints : {len(report.api_endpoints)} found, {len(report.untested_endpoints)} untested")
    console.print(f"  UI pages  : {len(report.ui_pages)} found, {len(report.untested_pages)} untested")
    console.print(f"  Test gaps : {len(report.test_gaps)}")
    console.print(f"  Report    : {OUTPUT_DIR.parent / 'test_gap_report.md'}")
    console.print(f"\n  Run [cyan]python main.py generate --repo {repo} --push[/cyan] to generate and push tests.\n")


# ─────────────────────────────────────────────────────────
# generate  (full pipeline)
# ─────────────────────────────────────────────────────────

@app.command()
def generate(
    repo    : str  = typer.Option(...,   "--repo",    "-r", help="GitHub repository URL"),
    push    : bool = typer.Option(False, "--push",          help="Push generated tests to GitHub and open a PR"),
    dry_run : bool = typer.Option(False, "--dry-run",       help="Generate tests locally but skip the GitHub push"),
    reset   : bool = typer.Option(False, "--reset",         help="Re-clone and re-embed even if already cached"),
    reuse_report: bool = typer.Option(False, "--reuse-report", help="Skip ingest/embed/analyze and reuse the saved test_gap_report.json"),
):
    """
    Full pipeline: ingest → embed → analyze → generate tests → (optionally) push to GitHub.

    Examples:

      # Full run with GitHub push
      python main.py generate --repo https://github.com/fastapi/full-stack-fastapi-template --push

      # Dry run (no push)
      python main.py generate --repo https://github.com/fastapi/full-stack-fastapi-template --dry-run

      # Skip re-embedding (already done) and reuse saved report
      python main.py generate --repo https://github.com/... --push --reuse-report
    """
    _print_banner()

    from config import validate_config, OUTPUT_DIR
    try:
        validate_config()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    report_json = OUTPUT_DIR.parent / "test_gap_report.json"

    if reuse_report and report_json.exists():
        # ── Fast path: skip ingestion / embedding / analysis ──
        console.print("\n[yellow]⚡ --reuse-report: loading saved test_gap_report.json[/yellow]")
        report = _load_report(report_json)

        _step(4, "Test Generation Agent")
        from agents.embedding_agent import get_chroma_collection
        collection = get_chroma_collection()
        from agents.test_gen_agent import run as generate_tests
        generate_tests(report, collection)

    else:
        # ── Full pipeline ──────────────────────────────────────

        # Step 1 — Ingest
        _step(1, "Ingestion Agent")
        from agents.ingestion_agent import run as ingest
        code_files = ingest(repo)

        # Step 2 — Embed
        _step(2, "Embedding Agent")
        from agents.embedding_agent import run as embed
        collection = embed(code_files, reset=reset)

        # Step 3 — Analyze
        _step(3, "Analysis Agent")
        from agents.analysis_agent import run as analyse
        report = analyse(collection, repo_url=repo)

        # Step 4 — Generate tests
        _step(4, "Test Generation Agent")
        from agents.test_gen_agent import run as generate_tests
        generate_tests(report, collection)

    # Step 5 — Push to GitHub (optional)
    if push and not dry_run:
        _step(5, "GitHub Push Agent")
        from agents.github_agent import run as push_to_github
        pr_url = push_to_github(report, generated_dir=OUTPUT_DIR)
        console.print(Panel(
            f"[bold green]Pipeline complete![/bold green]\n\n"
            f"  Repo    : {repo}\n"
            f"  Tests   : {OUTPUT_DIR}\n"
            f"  PR      : {pr_url}",
            border_style="green",
        ))
    else:
        reason = "(--dry-run)" if dry_run else "(--push not set)"
        console.print(f"\n[yellow]Skipping GitHub push {reason}[/yellow]")
        console.print(Panel(
            f"[bold green]Pipeline complete![/bold green]\n\n"
            f"  Repo  : {repo}\n"
            f"  Tests : {OUTPUT_DIR}\n\n"
            f"  To push: [cyan]python main.py generate --repo {repo} --push --reuse-report[/cyan]",
            border_style="green",
        ))


# ─────────────────────────────────────────────────────────
# record  (recording pipeline)
# ─────────────────────────────────────────────────────────

@app.command()
def record(
    url          : str = typer.Option(..., "--url", "-u", help="Live URL to record a scenario from"),
    name         : str = typer.Option(None, "--name", "-n", help="Scenario name (default: auto-slug from URL + timestamp)"),
    max_attempts : int = typer.Option(None, "--max-attempts", help="Override MAX_FIX_ATTEMPTS from config"),
):
    """
    Record a live browser scenario, generate a Playwright test, self-heal it
    until it passes (or the LLM gives up), and produce a run report.

    Example:

      python main.py record --url https://example.com --name checkout-flow
    """
    _print_banner()

    from config import validate_recording_config, MAX_FIX_ATTEMPTS
    try:
        validate_recording_config()
    except EnvironmentError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    scenario_name = _slugify_scenario_name(url, name)
    attempts_cap  = max_attempts or MAX_FIX_ATTEMPTS

    from agents.recording_graph import build_graph, RecordingState

    initial_state: RecordingState = {
        "url"              : url,
        "scenario_name"    : scenario_name,
        "raw_recording"    : "",
        "spec_code"        : "",
        "spec_path"        : "",
        "attempt"          : 0,
        "max_attempts"     : attempts_cap,
        "execution_passed" : False,
        "execution_output" : "",
        "dom_snapshot"     : "",
        "give_up"          : False,
        "give_up_reason"   : "",
        "attempts_history" : [],
        "report_path"      : "",
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    style = "green" if final_state["execution_passed"] else "yellow"
    console.print(Panel(
        f"[bold]Recording pipeline complete![/bold]\n\n"
        f"  Scenario : {scenario_name}\n"
        f"  Passed   : {final_state['execution_passed']}\n"
        f"  Attempts : {len(final_state['attempts_history'])}\n"
        f"  Report   : {final_state['report_path']}",
        border_style=style,
    ))


if __name__ == "__main__":
    app()
