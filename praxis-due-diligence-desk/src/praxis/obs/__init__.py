"""Observability: OpenTelemetry spans around every graph node.

``span(name, **attributes)`` wraps a node's body — the six LLM-calling nodes
(``planner``, ``research_one``, the bull/bear analysts, ``red_team``,
``editor``) also get ``model``/``prompt_tokens``/``completion_tokens``/
``cost_usd`` set on that same span from inside ``LangChainStructuredLLM``
(see ``llm.py``), via ``trace.get_current_span().set_attribute(...)`` — a
no-op on the default non-recording span, so it's always safe to call even
when tracing is disabled. ``verifier`` gets a span with just ``role`` — it
makes no LLM call.

Gated on ``PRAXIS_OTEL_ENABLED`` (default off, so a fresh checkout / CI never
pays tracing overhead or needs a collector). ``PRAXIS_OTEL_ENDPOINT`` unset ->
spans print to stdout via ``ConsoleSpanExporter``; set it to any real OTLP/HTTP
collector (a self-hosted LangSmith or Phoenix OTLP ingester, Jaeger, ...) to
export there instead — there is no direct LangSmith SDK integration here, only
the standard OTLP wire protocol, since no LangSmith key exists on this
machine to build and verify one against.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from praxis.config import get_settings

_TRACER_NAME = "praxis"


def setup_tracing() -> None:
    """Configure the global ``TracerProvider`` from settings. Call once at
    process start (API/CLI/MCP entrypoints) — idempotent: a no-op if
    ``PRAXIS_OTEL_ENABLED`` is unset, and the OTel SDK itself only ever
    installs the *first* global provider a process sets (later calls are a
    harmless no-op with a logged warning), so calling this more than once is
    safe. Tests that need to inspect spans install their own
    ``InMemorySpanExporter``-backed provider directly instead of going
    through this function (see ``tests/test_obs.py``)."""
    settings = get_settings()
    if not settings.otel_enabled:
        return
    provider = TracerProvider()
    exporter = (
        OTLPSpanExporter(endpoint=settings.otel_endpoint)
        if settings.otel_endpoint
        else ConsoleSpanExporter()
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    if not get_settings().otel_enabled:
        yield
        return
    tracer = trace.get_tracer(_TRACER_NAME)
    clean = {k: v for k, v in attributes.items() if v is not None}
    with tracer.start_as_current_span(name, attributes=clean):  # type: ignore[arg-type]
        yield
