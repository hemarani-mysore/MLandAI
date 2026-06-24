"""
models/report.py — Data models for the Test Gap Report
=======================================================
These dataclasses represent the structured output of the analysis agent.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class APIEndpoint:
    method      : str          # GET | POST | PUT | DELETE | PATCH
    path        : str          # /api/v1/users/{id}
    description : str          # What the endpoint does
    file_path   : str          # Source file where it's defined
    has_test    : bool = False  # Whether a test exists for it


@dataclass
class UIPage:
    name      : str            # Login, Dashboard, UserSettings …
    route     : str            # /login, /dashboard …
    file_path : str            # Frontend component file
    has_test  : bool = False


@dataclass
class TestGap:
    name      : str   # Endpoint path, page name, or function name
    file_path : str
    gap_type  : str   # "api_endpoint" | "ui_page" | "function"
    priority  : str   # "high" | "medium" | "low"
    reason    : str   # Why this needs a test


@dataclass
class TestGapReport:
    repo_url       : str
    generated_at   : str
    api_endpoints  : list[APIEndpoint] = field(default_factory=list)
    ui_pages       : list[UIPage]      = field(default_factory=list)
    auth_flows     : list[str]         = field(default_factory=list)
    test_gaps      : list[TestGap]     = field(default_factory=list)

    # ── derived counts ──────────────────────────────────────
    @property
    def untested_endpoints(self) -> list[APIEndpoint]:
        return [e for e in self.api_endpoints if not e.has_test]

    @property
    def untested_pages(self) -> list[UIPage]:
        return [p for p in self.ui_pages if not p.has_test]

    # ── serialisation ────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    def save_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self._render_markdown())

    def _render_markdown(self) -> str:
        tested_ep  = sum(1 for e in self.api_endpoints if e.has_test)
        tested_ui  = sum(1 for p in self.ui_pages      if p.has_test)
        high_gaps  = [g for g in self.test_gaps if g.priority == "high"]
        med_gaps   = [g for g in self.test_gaps if g.priority == "medium"]
        low_gaps   = [g for g in self.test_gaps if g.priority == "low"]

        lines = [
            "# Test Gap Report",
            f"\n**Repo:** {self.repo_url}",
            f"**Generated:** {self.generated_at}",
            "\n---\n",
            "## Summary",
            f"| | Found | Tested | Untested |",
            f"|---|---|---|---|",
            f"| API Endpoints | {len(self.api_endpoints)} | {tested_ep} | {len(self.untested_endpoints)} |",
            f"| UI Pages      | {len(self.ui_pages)}      | {tested_ui} | {len(self.untested_pages)}      |",
            f"| Test Gaps     | {len(self.test_gaps)}     | —          | {len(self.test_gaps)}           |",
            "\n---\n",
        ]

        # Auth flows
        if self.auth_flows:
            lines += ["## Authentication Flows", ""]
            for flow in self.auth_flows:
                lines.append(f"- {flow}")
            lines.append("\n---\n")

        # API Endpoints
        lines += [
            "## API Endpoints",
            "",
            "| Method | Path | File | Tested |",
            "|--------|------|------|--------|",
        ]
        for e in self.api_endpoints:
            status = "✅" if e.has_test else "❌"
            lines.append(f"| `{e.method}` | `{e.path}` | {e.file_path} | {status} |")

        # UI Pages
        lines += [
            "\n---\n",
            "## UI Pages",
            "",
            "| Page | Route | File | Tested |",
            "|------|-------|------|--------|",
        ]
        for p in self.ui_pages:
            status = "✅" if p.has_test else "❌"
            lines.append(f"| {p.name} | `{p.route}` | {p.file_path} | {status} |")

        # Test gaps
        lines += ["\n---\n", "## Test Gaps\n"]
        for label, gaps in [("🔴 High Priority", high_gaps), ("🟡 Medium Priority", med_gaps), ("🟢 Low Priority", low_gaps)]:
            if gaps:
                lines.append(f"### {label}\n")
                for g in gaps:
                    lines.append(f"- **{g.name}** (`{g.file_path}`)")
                    lines.append(f"  - Type: {g.gap_type}")
                    lines.append(f"  - Reason: {g.reason}\n")

        return "\n".join(lines)
