"""
tests/test_ingestion.py — Unit tests for Chunk 2
==================================================
Tests the ingestion agent and file parser utilities.

Run with:
    pytest tests/test_ingestion.py -v
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_parser import parse_repo, get_file_summary, detect_language, CodeFile
from agents.ingestion_agent import extract_repo_name, validate_github_url


# ─────────────────────────────────────────────
# Tests for extract_repo_name
# ─────────────────────────────────────────────

def test_extract_repo_name_standard():
    url = "https://github.com/fastapi/full-stack-fastapi-template"
    assert extract_repo_name(url) == "full-stack-fastapi-template"

def test_extract_repo_name_with_trailing_slash():
    url = "https://github.com/fastapi/full-stack-fastapi-template/"
    assert extract_repo_name(url) == "full-stack-fastapi-template"

def test_extract_repo_name_with_git_suffix():
    url = "https://github.com/fastapi/full-stack-fastapi-template.git"
    assert extract_repo_name(url) == "full-stack-fastapi-template"

def test_extract_repo_name_simple():
    url = "https://github.com/owner/my-repo"
    assert extract_repo_name(url) == "my-repo"


# ─────────────────────────────────────────────
# Tests for validate_github_url
# ─────────────────────────────────────────────

def test_validate_github_url_valid():
    assert validate_github_url("https://github.com/fastapi/full-stack-fastapi-template") == True

def test_validate_github_url_with_git():
    assert validate_github_url("https://github.com/owner/repo.git") == True

def test_validate_github_url_invalid_not_github():
    assert validate_github_url("https://gitlab.com/owner/repo") == False

def test_validate_github_url_invalid_no_repo():
    assert validate_github_url("https://github.com/owner") == False

def test_validate_github_url_invalid_empty():
    assert validate_github_url("") == False


# ─────────────────────────────────────────────
# Tests for detect_language
# ─────────────────────────────────────────────

def test_detect_language_python():
    assert detect_language(".py") == "python"

def test_detect_language_typescript():
    assert detect_language(".ts") == "typescript"

def test_detect_language_tsx():
    assert detect_language(".tsx") == "typescript-react"

def test_detect_language_javascript():
    assert detect_language(".js") == "javascript"

def test_detect_language_unknown():
    assert detect_language(".xyz") == "unknown"

def test_detect_language_case_insensitive():
    assert detect_language(".PY") == "python"


# ─────────────────────────────────────────────
# Tests for parse_repo (uses a temp directory)
# ─────────────────────────────────────────────

@pytest.fixture
def temp_repo():
    """
    Creates a temporary directory that mimics a small repo.
    Used to test parse_repo without needing to clone a real repo.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create some code files
        (root / "main.py").write_text("def hello():\n    return 'hello'\n")
        (root / "app.ts").write_text("const x: number = 1;\n")
        (root / "component.tsx").write_text("export default function App() { return <div/>; }\n")

        # Create files that should be skipped
        (root / "README.md").write_text("# Project\n")           # wrong extension
        (root / "package-lock.json").write_text("{}")             # wrong extension

        # Create a directory that should be skipped
        node_modules = root / "node_modules"
        node_modules.mkdir()
        (node_modules / "lib.js").write_text("module.exports = {};")  # should be skipped

        # Create a nested code file
        src = root / "src"
        src.mkdir()
        (src / "utils.py").write_text("def add(a, b):\n    return a + b\n")

        yield root


def test_parse_repo_finds_code_files(temp_repo):
    files = parse_repo(
        repo_path      = str(temp_repo),
        supported_exts = {".py", ".ts", ".tsx", ".js"},
        skip_dirs      = {"node_modules", ".git"},
        max_file_size  = 100_000,
        verbose        = False,
    )
    # Should find main.py, app.ts, component.tsx, src/utils.py
    assert len(files) == 4


def test_parse_repo_skips_node_modules(temp_repo):
    files = parse_repo(
        repo_path      = str(temp_repo),
        supported_exts = {".py", ".ts", ".tsx", ".js"},
        skip_dirs      = {"node_modules", ".git"},
        max_file_size  = 100_000,
        verbose        = False,
    )
    paths = [f.path for f in files]
    # node_modules/lib.js should NOT be in results
    assert not any("node_modules" in p for p in paths)


def test_parse_repo_skips_large_files(temp_repo):
    files = parse_repo(
        repo_path      = str(temp_repo),
        supported_exts = {".py", ".ts", ".tsx", ".js"},
        skip_dirs      = {"node_modules"},
        max_file_size  = 10,   # Very small limit — should skip almost everything
        verbose        = False,
    )
    # All files are > 10 bytes so none should be returned
    assert len(files) == 0


def test_parse_repo_correct_languages(temp_repo):
    files = parse_repo(
        repo_path      = str(temp_repo),
        supported_exts = {".py", ".ts", ".tsx", ".js"},
        skip_dirs      = {"node_modules"},
        max_file_size  = 100_000,
        verbose        = False,
    )
    lang_map = {Path(f.path).name: f.language for f in files}
    assert lang_map["main.py"]       == "python"
    assert lang_map["app.ts"]        == "typescript"
    assert lang_map["component.tsx"] == "typescript-react"


def test_parse_repo_content_loaded(temp_repo):
    files = parse_repo(
        repo_path      = str(temp_repo),
        supported_exts = {".py"},
        skip_dirs      = {"node_modules"},
        max_file_size  = 100_000,
        verbose        = False,
    )
    py_files = {Path(f.path).name: f for f in files}
    assert "def hello" in py_files["main.py"].content


# ─────────────────────────────────────────────
# Tests for get_file_summary
# ─────────────────────────────────────────────

def test_get_file_summary():
    files = [
        CodeFile("a.py", "/a.py", "python", "x = 1\ny = 2\n", 10, ".py"),
        CodeFile("b.ts", "/b.ts", "typescript", "const x = 1;\n", 13, ".ts"),
    ]
    summary = get_file_summary(files)
    assert summary["total_files"] == 2
    assert summary["languages"]["python"] == 1
    assert summary["languages"]["typescript"] == 1
    assert summary["total_lines"] == 3   # 2 newlines in a.py + 1 in b.ts
