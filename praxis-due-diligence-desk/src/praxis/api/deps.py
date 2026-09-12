"""FastAPI dependencies (overridable in tests via ``app.dependency_overrides``)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from arq.connections import ArqRedis, create_pool

from praxis.db import db_session_dep
from praxis.rag import Corpus, get_corpus
from praxis.worker import WorkerSettings


def corpus_dep() -> Corpus:
    return get_corpus()


async def redis_dep() -> AsyncGenerator[ArqRedis, None]:
    """A fresh arq connection per request — `create_pool` is itself async, so
    (unlike the SQLAlchemy engine) this can't be a bare `@lru_cache` singleton."""
    redis = await create_pool(WorkerSettings.redis_settings)
    try:
        yield redis
    finally:
        await redis.aclose()


__all__ = ["corpus_dep", "db_session_dep", "redis_dep"]
