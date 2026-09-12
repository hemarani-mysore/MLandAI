"""The LangGraph checkpointer, with our own schema types allow-listed.

Without this, checkpointing any ``DossierState`` logs (and, in a future
LangGraph version, will refuse) ``Deserializing unregistered type
praxis.schemas.ResearchPlan from checkpoint`` — ``JsonPlusSerializer`` only
allow-lists a fixed set of stdlib/langchain types for its msgpack extension
format by default. Every place that opens a checkpointer should go through
``open_checkpointer`` so this is configured exactly once.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from praxis.schemas import (
    Citation,
    DossierMemo,
    Evidence,
    Finding,
    MemoSection,
    RedTeamReport,
    ResearchPlan,
    SubQuestion,
    VerificationReport,
)

_ALLOWED_TYPES = [
    Citation,
    Evidence,
    SubQuestion,
    ResearchPlan,
    Finding,
    RedTeamReport,
    MemoSection,
    DossierMemo,
    VerificationReport,
]


@asynccontextmanager
async def open_checkpointer(path: str) -> AsyncIterator[AsyncSqliteSaver]:
    """Open a checkpointer backed by its own SQLite file at ``path``
    (``:memory:`` works too). Ready to use — tables are created if missing."""
    async with aiosqlite.connect(path) as conn:
        saver = AsyncSqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_TYPES)
        )
        await saver.setup()
        yield saver
