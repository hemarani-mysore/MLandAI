"""Engine / session plumbing for the dossier-run store.

Mirrors ``rag/corpus.py``'s ``get_corpus()``: a plain ``@lru_cache`` singleton,
not a FastAPI ``lifespan``-managed resource. ``create_async_engine`` is a sync,
lazily-connecting call, so this works fine — and it keeps the dependency
trivially overridable in tests via ``app.dependency_overrides``, the same way
``corpus_dep`` already is. (A bare ``TestClient(app)`` — the existing test
fixture's pattern, no ``with`` — does not run FastAPI's ``lifespan`` startup,
so anything that lived only on ``app.state`` would be unset in every test.)
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from praxis.config import get_settings
from praxis.db.models import Base

_init_lock = asyncio.Lock()
_initialized = False


def _database_url() -> str:
    return get_settings().database_url or "sqlite+aiosqlite:///./praxis.db"


@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(_database_url())


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def ensure_schema() -> None:
    """Create tables if they don't exist yet. Idempotent; safe to call more
    than once (guarded so the actual DDL only runs on the first call)."""
    global _initialized
    if _initialized:
        return
    async with _init_lock:
        if _initialized:
            return
        async with get_engine().begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        _initialized = True


async def db_session_dep() -> AsyncGenerator[AsyncSession, None]:
    await ensure_schema()
    async with get_sessionmaker()() as session:
        yield session
