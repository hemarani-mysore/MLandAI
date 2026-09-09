"""Helpers for reading per-run config and rendering state into prompts."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from praxis.config import get_settings
from praxis.graph.state import DossierState
from praxis.llm import StructuredLLM, get_llm


def llm_from(config: RunnableConfig) -> StructuredLLM:
    configurable = (config or {}).get("configurable", {})
    return configurable.get("llm") or get_llm()


def max_gap_loops(config: RunnableConfig) -> int:
    configurable = (config or {}).get("configurable", {})
    return int(configurable.get("max_gap_loops", get_settings().max_gap_loops))


def render_evidence(state: DossierState) -> str:
    lines = []
    for i, ev in enumerate(state["evidence"]):
        lines.append(
            f"[{i}] ({ev.stance}; conf={ev.confidence:.2f}) {ev.claim} "
            f"— {ev.citation.title} {ev.citation.locator}"
        )
    return "\n".join(lines) if lines else "(no evidence gathered yet)"


def render_case(label: str, finding) -> str:
    if finding is None:
        return f"{label}: (none)"
    points = "\n".join(f"  - {p}" for p in finding.points)
    return f"{label}: {finding.summary}\n{points}"
