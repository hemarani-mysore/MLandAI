"""
tests/test_gen.py — Unit tests for agents/test_gen_agent.py
============================================================
Tests filename routing and context-building helpers.
OpenAI API calls are not tested here.

Run with:
    pytest tests/test_gen.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.test_gen_agent import _gap_output_file, _build_context
from agents.github_agent import _get_file_destination
from models.report import TestGap


# ─────────────────────────────────────────────
# _gap_output_file  (filename routing)
# ─────────────────────────────────────────────

def _ui_gap(name: str) -> TestGap:
    return TestGap(name, "frontend/src/routes/x.tsx", "ui_page", "high", "reason")

def _api_gap(name: str) -> TestGap:
    return TestGap(name, "backend/app/api/routes/utils.py", "api_endpoint", "low", "reason")

def _fn_gap(name: str, file_path: str = "backend/app/core/security.py") -> TestGap:
    return TestGap(name, file_path, "function", "high", "reason")


def test_ui_gap_produces_spec_ts():
    assert _gap_output_file(_ui_gap("Login /login")).endswith(".spec.ts")

def test_ui_gap_slug_uses_page_name_only():
    # "Login /login" → "login.spec.ts"  (not "login-login.spec.ts")
    filename = _gap_output_file(_ui_gap("Login /login"))
    assert filename == "login.spec.ts"

def test_ui_gap_multiword_slug():
    filename = _gap_output_file(_ui_gap("Recover Password /recover-password"))
    assert filename == "recover-password.spec.ts"

def test_ui_gap_strips_special_chars():
    filename = _gap_output_file(_ui_gap("Not Found /*"))
    assert "*" not in filename

def test_api_gap_produces_utils_api_spec():
    assert _gap_output_file(_api_gap("GET /api/v1/utils/health-check/")) == "utils-api.spec.ts"

def test_security_fn_gap_routes_to_test_security():
    filename = _gap_output_file(_fn_gap("get_password_hash"))
    assert filename == "test_security.py"

def test_password_fn_gap_routes_to_test_security():
    filename = _gap_output_file(_fn_gap("verify_password"))
    assert filename == "test_security.py"

def test_error_fn_gap_routes_to_spec_ts():
    gap = _fn_gap("handleError", "frontend/src/utils.ts")
    filename = _gap_output_file(gap)
    assert filename.endswith(".spec.ts") or filename.endswith(".py")

def test_database_fn_gap_routes_to_test_startup():
    gap = _fn_gap("database retry logic", "backend/app/backend_pre_start.py")
    filename = _gap_output_file(gap)
    assert filename == "test_startup.py"


# ─────────────────────────────────────────────
# _build_context
# ─────────────────────────────────────────────

def _chunk(file_path: str, content: str) -> dict:
    return {"file_path": file_path, "content": content, "score": 0.9,
            "language": "python", "chunk_type": "function", "name": "f"}


def test_build_context_includes_file_path():
    ctx = _build_context([_chunk("users.py", "def create(): pass")])
    assert "users.py" in ctx


def test_build_context_includes_content():
    ctx = _build_context([_chunk("a.py", "def foo(): return 42")])
    assert "def foo(): return 42" in ctx


def test_build_context_empty_list():
    assert _build_context([]) == ""


def test_build_context_truncates_at_max_chars():
    chunks = [_chunk(f"f{i}.py", "x" * 3000) for i in range(10)]
    ctx = _build_context(chunks, max_chars=10_000)
    assert len(ctx) <= 10_500   # allow small overhead


# ─────────────────────────────────────────────
# _get_file_destination  (github_agent helper)
# ─────────────────────────────────────────────

def test_spec_ts_goes_to_frontend_tests():
    dest = _get_file_destination("login.spec.ts")
    assert dest == "frontend/tests/login.spec.ts"


def test_py_goes_to_backend_tests():
    dest = _get_file_destination("test_security.py")
    assert dest == "backend/tests/test_security.py"


def test_parentheses_stripped_from_filename():
    dest = _get_file_destination("error-(error).spec.ts")
    assert "(" not in dest
    assert ")" not in dest
    assert dest == "frontend/tests/error-error.spec.ts"


def test_admin_spec_destination():
    dest = _get_file_destination("admin.spec.ts")
    assert dest == "frontend/tests/admin.spec.ts"
