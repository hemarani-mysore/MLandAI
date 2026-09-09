"""The structured-LLM boundary.

Nodes never talk to a model directly; they call ``StructuredLLM.generate(...)``
with a Pydantic schema and a role, or ``.complete(...)`` for a plain string.
Two implementations:

* ``LangChainStructuredLLM`` — real models (OpenAI / Anthropic / Nebius). Role
  decides the tier (strong vs fast).
* ``FakeStructuredLLM`` — deterministic canned objects, so the whole graph, the
  API, and CI run with no API keys and no network.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, SecretStr

from praxis.config import Settings, get_settings
from praxis.schemas import (
    Citation,
    DossierMemo,
    Evidence,
    Finding,
    MemoSection,
    RedTeamReport,
    ResearchPlan,
    SubQuestion,
)

T = TypeVar("T", bound=BaseModel)


class StructuredLLM:
    """Interface: given a prompt + schema + role, return a schema instance."""

    def generate(self, *, system: str, user: str, schema: type[T], role: str) -> T:
        raise NotImplementedError

    def complete(self, *, system: str, user: str, role: str = "util") -> str:
        raise NotImplementedError


def _build_models(settings: Settings) -> dict[str, Any]:
    if settings.llm_provider == "nebius":
        from langchain_openai import ChatOpenAI

        key = SecretStr(os.environ.get("NEBIUS_API_KEY", ""))
        base = settings.nebius_base_url
        return {
            "strong": ChatOpenAI(model=settings.strong_model, base_url=base, api_key=key),
            "fast": ChatOpenAI(model=settings.fast_model, base_url=base, api_key=key),
        }

    from langchain.chat_models import init_chat_model

    return {
        "strong": init_chat_model(settings.strong_model),
        "fast": init_chat_model(settings.fast_model),
    }


class LangChainStructuredLLM(StructuredLLM):
    def __init__(self, settings: Settings) -> None:
        self._fast_roles = set(settings.fast_roles)
        self._models = _build_models(settings)

    def _model(self, role: str) -> Any:
        return self._models["fast" if role in self._fast_roles else "strong"]

    def generate(self, *, system: str, user: str, schema: type[T], role: str) -> T:
        model = self._model(role).with_structured_output(schema)
        result = model.invoke([("system", system), ("human", user)])
        assert isinstance(result, schema)
        return result

    def complete(self, *, system: str, user: str, role: str = "util") -> str:
        result = self._models["fast"].invoke([("system", system), ("human", user)])
        return str(getattr(result, "content", result))


class FakeStructuredLLM(StructuredLLM):
    """Deterministic responses. Pass ``overrides={role: [obj, ...]}`` to script
    specific turns (used to exercise the gap-fill loop in tests)."""

    def __init__(self, overrides: dict[str, list[BaseModel]] | None = None) -> None:
        self._overrides = {k: list(v) for k, v in (overrides or {}).items()}
        self.calls: dict[str, int] = defaultdict(int)

    def generate(self, *, system: str, user: str, schema: type[T], role: str) -> T:
        self.calls[role] += 1
        queued = self._overrides.get(role)
        if queued:
            obj = queued.pop(0)
            assert isinstance(obj, schema)
            return obj
        return _canned(schema, system=system, user=user)  # type: ignore[return-value]

    def complete(self, *, system: str, user: str, role: str = "util") -> str:
        self.calls[role] += 1
        snippet = _field(user, "Chunk:")
        if not snippet and user.strip():
            snippet = user.strip().splitlines()[-1]
        return f"Context: {' '.join(snippet.split()[:12])}".strip()


def _field(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _first_source_id(user: str) -> str | None:
    """Pull a real source id from a rendered evidence/context block, if present."""
    m = re.search(r"EVIDENCE SOURCES:\s*([^\n]+)", user)
    if m:
        first = m.group(1).split(",")[0].strip()
        if first:
            return first
    m = re.search(r"\[\d+\]\s*\(([A-Za-z0-9._:-]+);", user)  # "[0] (src-id; supports) ..."
    return m.group(1) if m else None


def _canned(schema: type[BaseModel], *, system: str, user: str) -> BaseModel:
    subject = _field(user, "Subject:") or "the subject"
    grounded = _first_source_id(user)
    cite = Citation(
        source_id=grounded or "stub-001",
        title="Cited source" if grounded else "Placeholder source",
        locator="p.1",
        quote="(deterministic stub citation)",
    )

    if schema is ResearchPlan:
        from praxis.schemas import RUBRIC_SECTIONS

        return ResearchPlan(
            subject=subject,
            thesis=f"Assess {subject} as a partner / investment / acquisition target.",
            sub_questions=[
                SubQuestion(section=s, question=f"What is {subject}'s position on {s.lower()}?")
                for s in RUBRIC_SECTIONS
            ],
        )
    if schema is Evidence:
        return Evidence(
            question=_field(user, "Question:") or "n/a",
            section=_field(user, "Section:") or "Overview",
            claim=f"{subject} presents a typical profile for its stage and sector.",
            stance="neutral",
            citation=cite,
            confidence=0.5,
        )
    if schema is Finding:
        stance: Literal["bull", "bear"] = "bear" if "bear" in system.lower() else "bull"
        return Finding(
            stance=stance,
            summary=f"{stance.title()} case for {subject} (stub synthesis).",
            points=[f"{stance.title()} consideration 1", f"{stance.title()} consideration 2"],
            reasoning_steps=["Reviewed the shared evidence.", "Weighed it for this stance."],
        )
    if schema is RedTeamReport:
        return RedTeamReport()
    if schema is DossierMemo:
        return DossierMemo(
            subject=subject,
            summary=f"Deterministic stub due-diligence summary for {subject}.",
            recommendation="insufficient_evidence",
            confidence=0.4,
            sections=[
                MemoSection(
                    heading="Overview",
                    body=f"{subject} — full multi-section synthesis lands in Phase 2.",
                    citations=[cite],
                )
            ],
            open_questions=["Everything: this is an early-phase skeleton."],
        )
    raise ValueError(f"FakeStructuredLLM has no canned response for {schema.__name__}")


def get_llm(settings: Settings | None = None) -> StructuredLLM:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return FakeStructuredLLM()
    return LangChainStructuredLLM(settings)
