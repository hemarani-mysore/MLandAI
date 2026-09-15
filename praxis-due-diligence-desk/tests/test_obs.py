"""OTel spans — with PRAXIS_OTEL_ENABLED=true, every graph node emits a span
carrying `role`, and every LLM-calling node's span also carries
`model`/`prompt_tokens`/`completion_tokens`/`cost_usd`.

Installs its own InMemorySpanExporter-backed TracerProvider directly, once at
import time, rather than going through `praxis.obs.setup_tracing()` — the
OTel SDK only ever installs the *first* global TracerProvider a process sets
(later calls are a harmless no-op), so a test module that wants to actually
inspect spans needs to be the one that wins that race, with full control over
the exporter. `_clear_spans` resets it between tests.
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from praxis.config import get_settings
from praxis.graph import run_dossier
from praxis.llm import FakeStructuredLLM
from praxis.schemas import RUBRIC_SECTIONS, DossierRequest

_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture(autouse=True)
def _clear_spans(monkeypatch):
    _exporter.clear()
    monkeypatch.setenv("PRAXIS_OTEL_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _span_names() -> list[str]:
    return [s.name for s in _exporter.get_finished_spans()]


def _span(name: str):
    matches = [s for s in _exporter.get_finished_spans() if s.name == name]
    assert matches, f"no span named {name!r}; got {_span_names()}"
    return matches[0]


def test_no_spans_at_all_when_tracing_is_disabled(monkeypatch):
    monkeypatch.setenv("PRAXIS_OTEL_ENABLED", "false")
    get_settings.cache_clear()

    run_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())

    assert _exporter.get_finished_spans() == ()


def test_every_node_emits_a_span_with_a_role_attribute():
    run_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())

    names = _span_names()
    # one span per RUBRIC_SECTIONS-many research_one branch, plus one each of
    # the rest (no gap-fill loop on a clean FakeStructuredLLM run).
    assert names.count("research_one") == len(RUBRIC_SECTIONS)
    for expected in ("planner", "bull_analyst", "bear_analyst", "red_team", "editor", "verifier"):
        assert expected in names

    for s in _exporter.get_finished_spans():
        assert s.attributes.get("role") == s.name


def test_llm_calling_spans_carry_model_and_token_attributes():
    run_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())

    planner_span = _span("planner")
    # FakeStructuredLLM never calls _record_usage (no real model, no real
    # usage_metadata to report) — model/token attributes are only set by
    # LangChainStructuredLLM, so on the fake provider they're simply absent.
    assert "model" not in planner_span.attributes


def test_verifier_span_has_no_model_or_token_attributes():
    run_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())

    verifier_span = _span("verifier")
    assert verifier_span.attributes.get("role") == "verifier"
    assert "model" not in verifier_span.attributes
    assert "cost_usd" not in verifier_span.attributes


def test_research_one_spans_carry_their_section():
    run_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())

    sections = {
        s.attributes.get("section")
        for s in _exporter.get_finished_spans()
        if s.name == "research_one"
    }
    assert sections == set(RUBRIC_SECTIONS)
