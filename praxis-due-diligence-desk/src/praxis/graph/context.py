"""Helpers for reading per-run config and rendering state into prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

from praxis.config import get_settings
from praxis.graph.state import DossierState
from praxis.llm import StructuredLLM, get_llm
from praxis.schemas import RUBRIC_SECTIONS

if TYPE_CHECKING:
    from praxis.rag import Corpus


def _configurable(config: RunnableConfig) -> dict:
    return (config or {}).get("configurable", {})


def llm_from(config: RunnableConfig) -> StructuredLLM:
    return _configurable(config).get("llm") or get_llm()


def corpus_from(config: RunnableConfig) -> Corpus | None:
    """The per-run corpus, if one was supplied; otherwise ``None`` and
    ``search_corpus`` falls back to the process singleton."""
    return _configurable(config).get("corpus")


def max_gap_loops(config: RunnableConfig) -> int:
    return int(_configurable(config).get("max_gap_loops", get_settings().max_gap_loops))


def _evidence_line(i: int, ev) -> str:
    return (
        f"[{i}] ({ev.citation.source_id}; {ev.stance}; conf={ev.confidence:.2f}) "
        f"{ev.claim} — {ev.citation.title} {ev.citation.locator}"
    )


def render_evidence(state: DossierState) -> str:
    lines = [_evidence_line(i, ev) for i, ev in enumerate(state["evidence"])]
    return "\n".join(lines) if lines else "(no evidence gathered yet)"


def render_evidence_by_section(state: DossierState) -> str:
    """Evidence grouped under rubric-ordered headings, keeping the global ``[i]``
    index on each item so analyst ``evidence_refs`` still line up. Used by the
    editor so it can write one section per rubric item."""
    by_section: dict[str, list[str]] = {}
    for i, ev in enumerate(state["evidence"]):
        by_section.setdefault(ev.section, []).append(_evidence_line(i, ev))

    ordered = list(RUBRIC_SECTIONS) + [s for s in by_section if s not in RUBRIC_SECTIONS]
    blocks = [
        f"## {section}\n" + "\n".join(by_section[section])
        for section in ordered
        if by_section.get(section)
    ]
    return "\n\n".join(blocks) if blocks else "(no evidence gathered yet)"


def evidence_indices_by_section(state: DossierState) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, ev in enumerate(state["evidence"]):
        out.setdefault(ev.section, []).append(i)
    return out


def evidence_source_ids(state: DossierState) -> list[str]:
    seen: dict[str, None] = {}
    for ev in state["evidence"]:
        seen.setdefault(ev.citation.source_id, None)
    return list(seen)


def render_case(label: str, finding) -> str:
    if finding is None:
        return f"{label}: (none)"
    points = "\n".join(f"  - {p}" for p in finding.points)
    return f"{label}: {finding.summary}\n{points}"
