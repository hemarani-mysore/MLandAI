"""
agents/analysis_agent.py — RAG Analysis Agent
==============================================
Responsible for:
  - Querying ChromaDB with targeted RAG queries
  - Sending retrieved code to OpenAI for structured analysis
  - Identifying API endpoints, UI pages, auth flows, and test gaps
  - Producing a TestGapReport saved as JSON + Markdown

This is the THIRD agent in the pipeline.
Input  : ChromaDB collection (from embedding_agent)
Output : TestGapReport (JSON + Markdown saved to outputs/)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    OUTPUT_DIR,
    TOP_K_RESULTS,
)
from agents.embedding_agent import retrieve
from models.report import APIEndpoint, UIPage, TestGap, TestGapReport

console = Console()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _build_context(chunks: list[dict], max_chars: int = 12_000) -> str:
    """Concatenate retrieved chunks into a single context string for OpenAI."""
    parts = []
    total = 0
    for chunk in chunks:
        snippet = f"### File: {chunk['file_path']}\n{chunk['content']}\n"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n---\n".join(parts)


def _call_openai_json(prompt: str) -> dict:
    """
    Call OpenAI with JSON output mode and return the parsed result.
    Note: OpenAI's json_object mode requires the top-level response to be a
    JSON object, not a bare array — every prompt below asks for a named
    wrapper key (e.g. {"endpoints": [...]}) which callers then unwrap.
    """
    response = _get_client().chat.completions.create(
        model           = OPENAI_MODEL,
        temperature     = OPENAI_TEMPERATURE,
        response_format = {"type": "json_object"},
        messages        = [{"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────────────────────
# Query 1: API Endpoints
# ─────────────────────────────────────────────────────────

def find_api_endpoints(collection) -> list[APIEndpoint]:
    console.print("\n[cyan]🔍 Query 1: Finding API endpoints...[/cyan]")

    source_chunks = retrieve(collection, "FastAPI router APIRouter route decorator HTTP method endpoint", top_k=12)
    test_chunks   = retrieve(collection, "pytest test client API endpoint HTTP request response", top_k=8)
    context       = _build_context(source_chunks + test_chunks)

    prompt = f"""You are analyzing a FastAPI backend codebase.

CODEBASE CONTEXT (source + test files):
{context}

Extract EVERY API endpoint defined with FastAPI decorators (@router.get, @router.post, etc.).
For each endpoint also check whether a test function exists in the test files.

Return a JSON object with exactly this shape:
{{
  "endpoints": [
    {{
      "method": "GET",
      "path": "/api/v1/users",
      "description": "one sentence describing what this endpoint does",
      "file_path": "backend/app/api/routes/users.py",
      "has_test": true
    }}
  ]
}}

Rules:
- method must be uppercase: GET, POST, PUT, DELETE, PATCH
- path must start with /
- has_test is true only if you can see a corresponding test in the context
- Return ONLY the JSON object, no explanation, no markdown"""

    data = _call_openai_json(prompt)["endpoints"]
    endpoints = []
    for item in data:
        try:
            endpoints.append(APIEndpoint(
                method      = item.get("method", "GET").upper(),
                path        = item.get("path", ""),
                description = item.get("description", ""),
                file_path   = item.get("file_path", ""),
                has_test    = bool(item.get("has_test", False)),
            ))
        except Exception:
            continue

    console.print(f"   [green]Found {len(endpoints)} endpoints "
                  f"({sum(e.has_test for e in endpoints)} tested)[/green]")
    return endpoints


# ─────────────────────────────────────────────────────────
# Query 2: UI Pages
# ─────────────────────────────────────────────────────────

def find_ui_pages(collection) -> list[UIPage]:
    console.print("\n[cyan]🔍 Query 2: Finding UI pages...[/cyan]")

    source_chunks = retrieve(collection, "React page component route createFileRoute useNavigate", top_k=12)
    test_chunks   = retrieve(collection, "Playwright test page goto click fill expect", top_k=8)
    context       = _build_context(source_chunks + test_chunks)

    prompt = f"""You are analyzing a React + TypeScript frontend codebase.

CODEBASE CONTEXT (source + Playwright test files):
{context}

Identify all distinct UI pages / routes in this application.
For each page also check whether a Playwright test (.spec.ts) exists for it.

Return a JSON object with exactly this shape:
{{
  "pages": [
    {{
      "name": "Login",
      "route": "/login",
      "file_path": "frontend/src/routes/login.tsx",
      "has_test": true
    }}
  ]
}}

Rules:
- name should be a human-readable page name (e.g. Login, Dashboard, User Settings)
- route must start with /
- has_test is true only if a Playwright spec covers this page
- Return ONLY the JSON object, no explanation, no markdown"""

    data = _call_openai_json(prompt)["pages"]
    pages = []
    for item in data:
        try:
            pages.append(UIPage(
                name      = item.get("name", ""),
                route     = item.get("route", ""),
                file_path = item.get("file_path", ""),
                has_test  = bool(item.get("has_test", False)),
            ))
        except Exception:
            continue

    console.print(f"   [green]Found {len(pages)} pages "
                  f"({sum(p.has_test for p in pages)} tested)[/green]")
    return pages


# ─────────────────────────────────────────────────────────
# Query 3: Auth Flows
# ─────────────────────────────────────────────────────────

def find_auth_flows(collection) -> list[str]:
    console.print("\n[cyan]🔍 Query 3: Mapping authentication flows...[/cyan]")

    chunks  = retrieve(collection, "authentication login JWT token OAuth password middleware security", top_k=12)
    context = _build_context(chunks)

    prompt = f"""You are analyzing a full-stack application's authentication system.

CODEBASE CONTEXT:
{context}

Identify all distinct authentication and authorization flows in this codebase.

Return a JSON object with exactly this shape:
{{
  "flows": [
    "User logs in with email/password and receives a JWT access token",
    "Superuser can reset any user's password via the admin endpoint"
  ]
}}
Each string describes one auth flow in one sentence.

Return ONLY the JSON object, no explanation, no markdown."""

    data = _call_openai_json(prompt)["flows"]
    flows = [str(f) for f in data if f]

    console.print(f"   [green]Found {len(flows)} auth flows[/green]")
    return flows


# ─────────────────────────────────────────────────────────
# Query 4: Test Gaps
# ─────────────────────────────────────────────────────────

def find_test_gaps(
    collection,
    endpoints : list[APIEndpoint],
    pages     : list[UIPage],
) -> list[TestGap]:
    console.print("\n[cyan]🔍 Query 4: Identifying test gaps...[/cyan]")

    # Build a summary of what we already know is untested
    untested_ep = [f"{e.method} {e.path} ({e.file_path})" for e in endpoints if not e.has_test]
    untested_ui = [f"{p.name} {p.route} ({p.file_path})" for p in pages if not p.has_test]

    # Retrieve additional context about utility functions that might lack tests
    func_chunks = retrieve(collection, "utility function helper CRUD database operation business logic", top_k=10)
    context     = _build_context(func_chunks)

    prompt = f"""You are a senior QA engineer reviewing a full-stack application for test coverage gaps.

UNTESTED API ENDPOINTS:
{chr(10).join(f'  - {e}' for e in untested_ep) or '  (none found)'}

UNTESTED UI PAGES:
{chr(10).join(f'  - {p}' for p in untested_ui) or '  (none found)'}

ADDITIONAL CODEBASE CONTEXT (utility / business logic):
{context}

Produce a prioritized list of test gaps. Include:
1. All untested API endpoints listed above
2. All untested UI pages listed above
3. Any critical utility functions in the context that lack tests

Return a JSON object with exactly this shape:
{{
  "gaps": [
    {{
      "name": "POST /api/v1/users/signup",
      "file_path": "backend/app/api/routes/users.py",
      "gap_type": "api_endpoint",
      "priority": "high",
      "reason": "User registration endpoint has no automated test"
    }}
  ]
}}

Rules:
- gap_type must be one of: api_endpoint, ui_page, function
- priority must be one of: high, medium, low
  - high   = auth, user data, money, security
  - medium = core CRUD, main user flows
  - low    = edge cases, admin-only, minor utilities
- Return ONLY the JSON object, no explanation, no markdown"""

    data = _call_openai_json(prompt)["gaps"]
    gaps = []
    valid_types     = {"api_endpoint", "ui_page", "function"}
    valid_priorities = {"high", "medium", "low"}

    for item in data:
        try:
            gap_type = item.get("gap_type", "function")
            priority = item.get("priority", "medium")
            gaps.append(TestGap(
                name      = item.get("name", ""),
                file_path = item.get("file_path", ""),
                gap_type  = gap_type  if gap_type  in valid_types      else "function",
                priority  = priority  if priority  in valid_priorities  else "medium",
                reason    = item.get("reason", ""),
            ))
        except Exception:
            continue

    console.print(f"   [green]Found {len(gaps)} test gaps "
                  f"({sum(g.priority == 'high' for g in gaps)} high priority)[/green]")
    return gaps


# ─────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────

def run(collection, repo_url: str = "") -> TestGapReport:
    """
    Main entry point for the analysis agent.

    Pipeline:
        1. Find all API endpoints (RAG + OpenAI)
        2. Find all UI pages (RAG + OpenAI)
        3. Map authentication flows (RAG + OpenAI)
        4. Identify test gaps (RAG + OpenAI)
        5. Build TestGapReport and save to outputs/

    Args:
        collection : ChromaDB collection from embedding_agent.run()
        repo_url   : Source GitHub URL (for the report header)

    Returns:
        TestGapReport with all findings
    """
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]ANALYSIS AGENT[/bold cyan]")
    console.print("=" * 60)

    # Run the four RAG queries
    endpoints  = find_api_endpoints(collection)
    pages      = find_ui_pages(collection)
    auth_flows = find_auth_flows(collection)
    gaps       = find_test_gaps(collection, endpoints, pages)

    # Assemble report
    report = TestGapReport(
        repo_url      = repo_url,
        generated_at  = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        api_endpoints = endpoints,
        ui_pages      = pages,
        auth_flows    = auth_flows,
        test_gaps     = sorted(
            gaps,
            key=lambda g: {"high": 0, "medium": 1, "low": 2}[g.priority]
        ),
    )

    # Save outputs
    out_dir = OUTPUT_DIR.parent  # outputs/
    json_path = out_dir / "test_gap_report.json"
    md_path   = out_dir / "test_gap_report.md"

    report.save_json(json_path)
    report.save_markdown(md_path)

    # Print summary panel
    console.print(Panel(
        f"[bold]Analysis complete[/bold]\n\n"
        f"  API endpoints : {len(endpoints)} found, {len(report.untested_endpoints)} untested\n"
        f"  UI pages      : {len(pages)} found, {len(report.untested_pages)} untested\n"
        f"  Auth flows    : {len(auth_flows)}\n"
        f"  Test gaps     : {len(gaps)} "
        f"({sum(g.priority == 'high' for g in gaps)} high, "
        f"{sum(g.priority == 'medium' for g in gaps)} medium, "
        f"{sum(g.priority == 'low' for g in gaps)} low)\n\n"
        f"  [dim]Saved → {json_path}[/dim]\n"
        f"  [dim]Saved → {md_path}[/dim]",
        title="[bold green]Test Gap Report[/bold green]",
        border_style="green",
    ))

    return report


# ─────────────────────────────────────────────────────────
# Quick test — run directly
# python agents/analysis_agent.py
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.embedding_agent import get_chroma_collection

    TEST_REPO = "https://github.com/fastapi/full-stack-fastapi-template"

    collection = get_chroma_collection()
    report     = run(collection, repo_url=TEST_REPO)

    console.print("\n[bold]Sample gaps:[/bold]")
    for gap in report.test_gaps[:5]:
        console.print(f"  [{gap.priority.upper()}] {gap.name} — {gap.reason}")
