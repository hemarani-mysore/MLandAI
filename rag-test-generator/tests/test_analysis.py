"""
tests/test_analysis.py — Unit tests for agents/analysis_agent.py
=================================================================
Tests the pure-Python helpers (context building, output parsing).
Gemini API calls are not tested here — they are exercised manually
via `python agents/analysis_agent.py`.

Run with:
    pytest tests/test_analysis.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.analysis_agent import _build_context
from models.report import APIEndpoint, UIPage, TestGap


# ─────────────────────────────────────────────
# _build_context
# ─────────────────────────────────────────────

def _make_chunk(file_path: str, content: str, score: float = 0.9) -> dict:
    return {
        "file_path"  : file_path,
        "content"    : content,
        "language"   : "python",
        "chunk_type" : "function",
        "name"       : "func",
        "score"      : score,
    }


def test_build_context_single_chunk():
    chunks  = [_make_chunk("a.py", "def foo(): pass")]
    context = _build_context(chunks)
    assert "a.py" in context
    assert "def foo(): pass" in context


def test_build_context_multiple_chunks():
    chunks = [
        _make_chunk("a.py", "x = 1"),
        _make_chunk("b.py", "y = 2"),
    ]
    context = _build_context(chunks)
    assert "a.py" in context
    assert "b.py" in context


def test_build_context_empty():
    assert _build_context([]) == ""


def test_build_context_respects_max_chars():
    big_content = "x" * 5_000
    chunks = [_make_chunk(f"file{i}.py", big_content) for i in range(5)]
    context = _build_context(chunks, max_chars=8_000)
    # Context must not exceed max_chars by more than one chunk's overhead
    assert len(context) <= 8_000 + 200   # allow small header overhead


def test_build_context_separates_chunks():
    chunks = [
        _make_chunk("a.py", "code_a"),
        _make_chunk("b.py", "code_b"),
    ]
    context = _build_context(chunks)
    assert "---" in context   # separator between chunks


# ─────────────────────────────────────────────
# APIEndpoint model parsing (output of find_api_endpoints)
# ─────────────────────────────────────────────

def test_api_endpoint_defaults():
    ep = APIEndpoint(
        method="get", path="/api/v1/users",
        description="list users", file_path="users.py",
    )
    assert ep.has_test is False


def test_api_endpoint_method_stored_as_given():
    ep = APIEndpoint("POST", "/api/v1/items", "create item", "items.py", has_test=True)
    assert ep.method == "POST"
    assert ep.has_test is True


# ─────────────────────────────────────────────
# UIPage model
# ─────────────────────────────────────────────

def test_ui_page_defaults():
    page = UIPage(name="Login", route="/login", file_path="login.tsx")
    assert page.has_test is False


def test_ui_page_with_test():
    page = UIPage("Signup", "/signup", "signup.tsx", has_test=True)
    assert page.has_test is True


# ─────────────────────────────────────────────
# TestGap validation helpers
# ─────────────────────────────────────────────

def test_test_gap_fields():
    gap = TestGap(
        name      = "POST /api/v1/users",
        file_path = "users.py",
        gap_type  = "api_endpoint",
        priority  = "high",
        reason    = "No test for user creation endpoint",
    )
    assert gap.gap_type == "api_endpoint"
    assert gap.priority == "high"


def test_test_gap_types():
    valid_types = {"api_endpoint", "ui_page", "function"}
    for t in valid_types:
        gap = TestGap("x", "x.py", t, "low", "reason")
        assert gap.gap_type in valid_types


def test_test_gap_priorities():
    for p in ("high", "medium", "low"):
        gap = TestGap("x", "x.py", "function", p, "reason")
        assert gap.priority == p
