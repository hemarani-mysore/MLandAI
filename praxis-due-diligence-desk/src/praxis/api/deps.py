"""FastAPI dependencies (overridable in tests via ``app.dependency_overrides``)."""

from __future__ import annotations

from praxis.db import db_session_dep
from praxis.rag import Corpus, get_corpus


def corpus_dep() -> Corpus:
    return get_corpus()


__all__ = ["corpus_dep", "db_session_dep"]
