"""The ingestion worker — arq jobs so corpus ingestion doesn't block a request.

Run for real with: ``uv run arq praxis.worker.WorkerSettings``
"""

from __future__ import annotations

import asyncio
from typing import Any

from arq.connections import RedisSettings

from praxis.config import get_settings
from praxis.rag import ingest_source


async def ingest_task(
    ctx: dict[str, Any],
    *,
    text: str | None = None,
    path: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Ingest one document into the process-wide corpus singleton.

    ``ingest_source`` is sync (a real embedder makes a blocking HTTP call
    inside it) — run it off the event loop, same reasoning as
    ``research_one``'s retrieval call.
    """
    result = await asyncio.to_thread(ingest_source, text=text, path=path, title=title)
    return result.model_dump(mode="json")


class WorkerSettings:
    """arq reads this class's attributes directly off `__dict__` (`arq.worker.get_kwargs`)
    — `redis_settings` must be an actual `RedisSettings` instance, not a method."""

    functions = (ingest_task,)
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
