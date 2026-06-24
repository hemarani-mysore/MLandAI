"""
main.py — CLI entry point for RAG Test Generator
=================================================
Usage:
  python main.py --help
  python main.py verify-setup
  python main.py analyze --repo <github-url>
  python main.py generate --repo <github-url> [--push] [--dry-run] [--reset]
"""

import json
from pathlib import Path

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
    text.append("Stack: Gemini API · ChromaDB · Playwright · GitHub API", style="dim")
    console.print(Panel(text, border_style="cyan"))


def _step(n: int, label: str):
    console.print(Rule(f"[bold cyan]Step {n}: {label}[/bold cyan]", style="cyan"))


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
        from config import validate_config, GOOGLE_API_KEY, GITHUB_TOKEN, GEMINI_MODEL
        validate_config()
        console.print("   [green]✅ All environment variables set[/green]")
    except EnvironmentError as e:
        console.print(f"   [red]❌ {e}[/red]")
        checks_passed = False

    console.print("[cyan]2. Testing Gemini API connection...[/cyan]")
    try:
        from google import genai
        from config import GOOGLE_API_KEY, GEMINI_MODEL
        client   = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = "Say 'API connection successful' in exactly those words.",
        )
        console.print(f"   [green]✅ Gemini API — {response.text.strip()}[/green]")
    except Exception as e:
        console.print(f"   [red]❌ Gemini API error: {e}[/red]")
        checks_passed = False

    console.print("[cyan]3. Testing ChromaDB...[/cyan]")
    try:
        import chromadb
        from config import CHROMA_DIR
        client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection("_setup_test")
        collection.add(documents=["ping"], ids=["ping"])
        client.delete_collection("_setup_test")
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


if __name__ == "__main__":
    app()
