"""
utils/file_parser.py — File filtering and reading utilities
============================================================
Responsible for:
  - Walking a cloned repo directory
  - Filtering only relevant code files
  - Reading file content safely
  - Returning structured CodeFile objects
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from rich.console import Console

console = Console()


@dataclass
class CodeFile:
    """
    Represents a single parsed code file from the repo.

    Attributes:
        path        : Relative path from repo root (e.g. "backend/app/main.py")
        absolute_path: Full path on disk
        language    : Detected language ("python", "typescript", etc.)
        content     : Full file content as string
        size_bytes  : File size in bytes
        extension   : File extension (e.g. ".py")
    """
    path          : str
    absolute_path : str
    language      : str
    content       : str
    size_bytes    : int
    extension     : str


# Map file extensions to language names
EXTENSION_TO_LANGUAGE = {
    ".py"  : "python",
    ".ts"  : "typescript",
    ".tsx" : "typescript-react",
    ".js"  : "javascript",
    ".jsx" : "javascript-react",
}


def detect_language(extension: str) -> str:
    """Return human-readable language name for a file extension."""
    return EXTENSION_TO_LANGUAGE.get(extension.lower(), "unknown")


def should_skip_dir(dir_name: str, skip_dirs: set) -> bool:
    """Return True if this directory should be skipped during traversal."""
    return dir_name in skip_dirs or dir_name.startswith(".")


def parse_repo(
    repo_path      : str,
    supported_exts : set,
    skip_dirs      : set,
    max_file_size  : int,
    verbose        : bool = True
) -> list[CodeFile]:
    """
    Walk a cloned repo and return a list of CodeFile objects.

    Args:
        repo_path      : Path to the cloned repo on disk
        supported_exts : Set of file extensions to include (e.g. {".py", ".ts"})
        skip_dirs      : Set of directory names to skip
        max_file_size  : Skip files larger than this (bytes)
        verbose        : Print progress to console

    Returns:
        List of CodeFile objects with content loaded

    Why we filter:
        - node_modules, .git etc. are not user code — they'd pollute embeddings
        - Very large files (>100KB) are usually generated/minified — not useful
        - Binary files would cause encoding errors
    """
    repo_root  = Path(repo_path)
    code_files = []
    skipped    = {"too_large": 0, "encoding_error": 0, "wrong_ext": 0, "skip_dir": 0}

    if verbose:
        console.print(f"\n[cyan]📂 Scanning repo: {repo_root.name}[/cyan]")

    for root, dirs, files in os.walk(repo_root):
        # Modify dirs in-place to prevent os.walk from descending into skip dirs
        # This is more efficient than checking after the fact
        dirs[:] = [d for d in dirs if not should_skip_dir(d, skip_dirs)]

        for filename in files:
            filepath  = Path(root) / filename
            extension = filepath.suffix.lower()

            # Skip unsupported file types
            if extension not in supported_exts:
                skipped["wrong_ext"] += 1
                continue

            # Skip files that are too large
            try:
                size = filepath.stat().st_size
            except OSError:
                continue

            if size > max_file_size:
                skipped["too_large"] += 1
                if verbose:
                    console.print(f"   [yellow]⚠ Skipping large file: {filepath.name} ({size//1024}KB)[/yellow]")
                continue

            # Read file content
            try:
                content = filepath.read_text(encoding="utf-8", errors="strict")
            except UnicodeDecodeError:
                # Binary or non-UTF8 file — skip it
                skipped["encoding_error"] += 1
                continue
            except OSError as e:
                console.print(f"   [red]❌ Could not read {filepath}: {e}[/red]")
                continue

            # Skip empty files — nothing to analyze
            if not content.strip():
                continue

            # Build relative path from repo root for cleaner display
            relative_path = str(filepath.relative_to(repo_root))

            code_files.append(CodeFile(
                path          = relative_path,
                absolute_path = str(filepath),
                language      = detect_language(extension),
                content       = content,
                size_bytes    = size,
                extension     = extension,
            ))

    if verbose:
        console.print(f"\n[green]✅ Parsed {len(code_files)} code files[/green]")
        console.print(f"   Skipped: {skipped['too_large']} too large, "
                      f"{skipped['encoding_error']} encoding errors, "
                      f"{skipped['wrong_ext']} wrong extension")

        # Show language breakdown
        lang_counts: dict[str, int] = {}
        for f in code_files:
            lang_counts[f.language] = lang_counts.get(f.language, 0) + 1
        console.print("\n   [bold]Language breakdown:[/bold]")
        for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            console.print(f"     {lang:25s}: {count} files")

    return code_files


def get_file_summary(code_files: list[CodeFile]) -> dict:
    """
    Return a summary dict of the parsed files.
    Useful for logging and the test gap report.
    """
    total_lines = sum(f.content.count("\n") for f in code_files)
    total_size  = sum(f.size_bytes for f in code_files)

    lang_counts: dict[str, int] = {}
    for f in code_files:
        lang_counts[f.language] = lang_counts.get(f.language, 0) + 1

    return {
        "total_files" : len(code_files),
        "total_lines" : total_lines,
        "total_size_kb": round(total_size / 1024, 1),
        "languages"   : lang_counts,
    }
