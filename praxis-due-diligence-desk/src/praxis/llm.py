"""The structured-LLM boundary.

Nodes never talk to a model directly; they call ``StructuredLLM.generate(...)``
with a Pydantic schema and a role. Two implementations:

* ``LangChainStructuredLLM`` — real models via ``init_chat_model`` +
  ``with_structured_output``. Role decides the tier (strong vs fast).
* ``FakeStructuredLLM`` — deterministic canned objects, so the whole graph,
  the API, and CI run with no API keys and no network.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Literal, TypeVar

from pydantic import BaseModel

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


class LangChainStructuredLLM(StructuredLLM):
    def __init__(self, settings: Settings) -> None:
        from langchain.chat_models import init_chat_model

        self._fast_roles = set(settings.fast_roles)
        self._models = {
            "strong": init_chat_model(settings.strong_model),
            "fast": init_chat_model(settings.fast_model),
        }

    def generate(self, *, system: str, user: str, schema: type[T], role: str) -> T:
        tier = "fast" if role in self._fast_roles else "strong"
        model = self._models[tier].with_structured_output(schema)
        result = model.invoke([("system", system), ("human", user)])
        assert isinstance(result, schema)
        return result


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


def _field(text: str, label: str) -> str:
    match = re.search(rf"{re.escape(label)}\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _canned(schema: type[BaseModel], *, system: str, user: str) -> BaseModel:
    subject = _field(user, "Subject:") or "the subject"
    stub = Citation(
        source_id="stub-001",
        title="Placeholder source",
        locator="p.1",
        quote="(Phase 1 wires real retrieval; this is a deterministic stub.)",
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
            citation=stub,
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
                    body=f"{subject} — populated by real agents once Phase 1/2 land.",
                    citations=[stub],
                )
            ],
            open_questions=["Everything: this is the Phase 0 skeleton."],
        )
    raise ValueError(f"FakeStructuredLLM has no canned response for {schema.__name__}")


def get_llm(settings: Settings | None = None) -> StructuredLLM:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return FakeStructuredLLM()
    return LangChainStructuredLLM(settings)
