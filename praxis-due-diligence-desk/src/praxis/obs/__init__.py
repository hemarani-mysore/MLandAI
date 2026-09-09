"""Observability hooks.

Phase 4 wires OpenTelemetry spans around every node (role, model, tokens, cost,
judge score) and exports to LangSmith / Phoenix. For now this is a no-op so the
call sites can already exist.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from praxis.config import get_settings


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    if not get_settings().otel_enabled:
        yield
        return
    # Phase 4: real span.
    yield
