"""
tests/test_report.py — Unit tests for models/report.py
=======================================================
Tests the TestGapReport data model: properties, serialisation,
and Markdown rendering — all without any API calls.

Run with:
    pytest tests/test_report.py -v
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.report import APIEndpoint, UIPage, TestGap, TestGapReport


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_report() -> TestGapReport:
    return TestGapReport(
        repo_url     = "https://github.com/owner/repo",
        generated_at = "2026-06-24 10:00:00",
        api_endpoints = [
            APIEndpoint("GET",  "/api/users",  "List users",  "users.py", has_test=True),
            APIEndpoint("POST", "/api/users",  "Create user", "users.py", has_test=False),
            APIEndpoint("GET",  "/api/health", "Health check","utils.py", has_test=False),
        ],
        ui_pages = [
            UIPage("Login",     "/login",     "login.tsx",   has_test=True),
            UIPage("Dashboard", "/",          "index.tsx",   has_test=False),
            UIPage("Admin",     "/admin",     "admin.tsx",   has_test=False),
        ],
        auth_flows = [
            "User authenticates with email and password to get a JWT token.",
            "Token is presented on each protected API request.",
        ],
        test_gaps = [
            TestGap("POST /api/users",  "users.py",  "api_endpoint", "high",   "No test for user creation"),
            TestGap("Dashboard /",       "index.tsx", "ui_page",      "medium", "Main page untested"),
            TestGap("Admin /admin",      "admin.tsx", "ui_page",      "high",   "Admin page untested"),
            TestGap("GET /api/health",  "utils.py",  "api_endpoint", "low",    "Health check untested"),
        ],
    )


# ─────────────────────────────────────────────
# Properties
# ─────────────────────────────────────────────

def test_untested_endpoints():
    report = _make_report()
    untested = report.untested_endpoints
    assert len(untested) == 2
    assert all(not e.has_test for e in untested)


def test_untested_pages():
    report = _make_report()
    untested = report.untested_pages
    assert len(untested) == 2
    assert all(not p.has_test for p in untested)


def test_fully_tested_report_has_no_gaps():
    report = TestGapReport(
        repo_url     = "https://github.com/x/y",
        generated_at = "2026-01-01",
        api_endpoints = [APIEndpoint("GET", "/", "root", "main.py", has_test=True)],
        ui_pages      = [UIPage("Home", "/", "index.tsx", has_test=True)],
    )
    assert report.untested_endpoints == []
    assert report.untested_pages == []


# ─────────────────────────────────────────────
# JSON serialisation
# ─────────────────────────────────────────────

def test_to_dict_contains_all_keys():
    d = _make_report().to_dict()
    assert "repo_url"      in d
    assert "generated_at"  in d
    assert "api_endpoints" in d
    assert "ui_pages"      in d
    assert "auth_flows"    in d
    assert "test_gaps"     in d


def test_to_dict_endpoint_fields():
    d = _make_report().to_dict()
    ep = d["api_endpoints"][0]
    assert ep["method"]   == "GET"
    assert ep["path"]     == "/api/users"
    assert ep["has_test"] is True


def test_save_json(tmp_path):
    report = _make_report()
    out = tmp_path / "report.json"
    report.save_json(out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["repo_url"] == report.repo_url
    assert len(loaded["api_endpoints"]) == 3
    assert len(loaded["test_gaps"]) == 4


def test_save_json_creates_parent_dirs(tmp_path):
    report = _make_report()
    out = tmp_path / "nested" / "deep" / "report.json"
    report.save_json(out)
    assert out.exists()


# ─────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────

def test_save_markdown(tmp_path):
    report = _make_report()
    out = tmp_path / "report.md"
    report.save_markdown(out)
    assert out.exists()
    content = out.read_text()
    assert "# Test Gap Report" in content


def test_markdown_contains_repo_url():
    md = _make_report()._render_markdown()
    assert "https://github.com/owner/repo" in md


def test_markdown_contains_endpoints():
    md = _make_report()._render_markdown()
    assert "/api/users" in md
    assert "GET" in md
    assert "POST" in md


def test_markdown_contains_ui_pages():
    md = _make_report()._render_markdown()
    assert "Login" in md
    assert "Dashboard" in md


def test_markdown_contains_auth_flows():
    md = _make_report()._render_markdown()
    assert "JWT" in md


def test_markdown_priority_sections():
    md = _make_report()._render_markdown()
    assert "High Priority" in md
    assert "Medium Priority" in md
    assert "Low Priority" in md


def test_markdown_tested_vs_untested_icons():
    md = _make_report()._render_markdown()
    assert "✅" in md   # tested entries
    assert "❌" in md   # untested entries


def test_markdown_summary_counts():
    md = _make_report()._render_markdown()
    # 3 endpoints, 3 pages, 4 gaps
    assert "3" in md
    assert "4" in md
