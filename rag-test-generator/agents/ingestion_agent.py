"""
agents/ingestion_agent.py — GitHub Ingestion Agent
====================================================
Responsible for:
  - Taking a GitHub URL as input
  - Cloning the repo locally
  - Delegating file parsing to file_parser
  - Returning structured CodeFile objects ready for embedding

This is the FIRST agent in the pipeline.
Input  : GitHub repo URL (string)
Output : List of CodeFile objects
"""

import shutil
import re
from pathlib import Path
from datetime import datetime

import git
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import our config and utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CLONE_DIR,
    SUPPORTED_EXTENSIONS,
    SKIP_DIRS,
    MAX_FILE_SIZE_BYTES,
)
from utils.file_parser import parse_repo, get_file_summary, CodeFile

console = Console()


def extract_repo_name(github_url: str) -> str:
    """
    Extract repo name from a GitHub URL.

    Examples:
        https://github.com/fastapi/full-stack-fastapi-template  → full-stack-fastapi-template
        https://github.com/fastapi/full-stack-fastapi-template/ → full-stack-fastapi-template
        git@github.com:fastapi/full-stack-fastapi-template.git  → full-stack-fastapi-template
    """
    # Remove trailing slash and .git
    url = github_url.rstrip("/").removesuffix(".git")
    # Last segment is the repo name
    return url.split("/")[-1]


def validate_github_url(url: str) -> bool:
    """
    Basic validation that the URL looks like a GitHub repo URL.
    We don't want to clone arbitrary URLs.
    """
    pattern = r"^https://github\.com/[\w\-\.]+/[\w\-\.]+$"
    return bool(re.match(pattern, url.rstrip("/").removesuffix(".git")))


def clone_repo(github_url: str, force_reclone: bool = False) -> Path:
    """
    Clone a GitHub repo to the local CLONE_DIR.

    Args:
        github_url    : Public GitHub repository URL
        force_reclone : If True, delete existing clone and re-clone fresh

    Returns:
        Path to the cloned repo on disk

    Why we clone locally:
        We need to walk the full file tree and read every file.
        The GitHub API has rate limits and doesn't easily support
        bulk file reading — local clone is faster and more reliable.
    """
    repo_name  = extract_repo_name(github_url)
    # Add timestamp suffix to avoid conflicts with multiple runs
    clone_path = CLONE_DIR / repo_name

    # If already cloned, reuse unless force_reclone is set
    if clone_path.exists():
        if force_reclone:
            console.print(f"[yellow]🗑  Removing existing clone: {clone_path}[/yellow]")
            shutil.rmtree(clone_path)
        else:
            console.print(f"[yellow]📁 Repo already cloned at: {clone_path}[/yellow]")
            console.print(f"   Use force_reclone=True to re-clone fresh.")
            return clone_path

    console.print(f"\n[cyan]📥 Cloning repo: {github_url}[/cyan]")
    console.print(f"   Destination: {clone_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Cloning repository...", total=None)
        git.Repo.clone_from(
            github_url,
            clone_path,
            depth=1,        # Shallow clone — we only need current state, not full history
                            # depth=1 is much faster for large repos
        )

    console.print(f"[green]✅ Cloned successfully → {clone_path}[/green]")
    return clone_path


def run(
    github_url    : str,
    force_reclone : bool = False,
    verbose       : bool = True,
) -> list[CodeFile]:
    """
    Main entry point for the ingestion agent.

    Pipeline:
        1. Validate the GitHub URL
        2. Clone the repo locally (or reuse existing clone)
        3. Parse all code files
        4. Return list of CodeFile objects

    Args:
        github_url    : Public GitHub repository URL
        force_reclone : Delete and re-clone even if already cloned
        verbose       : Print detailed progress

    Returns:
        List of CodeFile objects with content loaded

    Example:
        >>> from agents.ingestion_agent import run
        >>> files = run("https://github.com/fastapi/full-stack-fastapi-template")
        >>> print(f"Found {len(files)} code files")
    """
    console.print("\n" + "="*60)
    console.print("[bold cyan]INGESTION AGENT[/bold cyan]")
    console.print("="*60)

    # Step 1: Validate URL
    if not validate_github_url(github_url):
        raise ValueError(
            f"Invalid GitHub URL: {github_url}\n"
            f"Expected format: https://github.com/owner/repo-name"
        )
    console.print(f"[green]✅ Valid GitHub URL[/green]")

    # Step 2: Clone repo
    clone_path = clone_repo(github_url, force_reclone=force_reclone)

    # Step 3: Parse code files
    code_files = parse_repo(
        repo_path      = str(clone_path),
        supported_exts = SUPPORTED_EXTENSIONS,
        skip_dirs      = SKIP_DIRS,
        max_file_size  = MAX_FILE_SIZE_BYTES,
        verbose        = verbose,
    )

    # Step 4: Print summary
    summary = get_file_summary(code_files)
    console.print(f"\n[bold]📊 Ingestion Summary:[/bold]")
    console.print(f"   Total files : {summary['total_files']}")
    console.print(f"   Total lines : {summary['total_lines']:,}")
    console.print(f"   Total size  : {summary['total_size_kb']} KB")
    console.print(f"   Languages   : {summary['languages']}")

    if not code_files:
        console.print("\n[red]⚠️  No code files found! Check the repo URL or supported extensions.[/red]")

    return code_files


# ─────────────────────────────────────────────
# Quick test — run this file directly to test
# python agents/ingestion_agent.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TEST_REPO = "https://github.com/fastapi/full-stack-fastapi-template"

    console.print(f"\n[bold]Testing Ingestion Agent[/bold]")
    console.print(f"Target repo: {TEST_REPO}\n")

    files = run(TEST_REPO, verbose=True)

    console.print(f"\n[bold]Sample files found:[/bold]")
    for f in files[:5]:
        console.print(f"  📄 {f.path} ({f.language}, {f.size_bytes} bytes)")

    console.print(f"\n[bold green]✅ Ingestion agent working correctly![/bold green]")
