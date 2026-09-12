"""SQLAlchemy models — one table today: a durable record of each dossier run.

``RunEvent`` (per-node streaming events) lands in Phase 2 slice 5 alongside the
SSE endpoint that would actually populate it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class DossierRun(Base):
    """One `POST /dossiers` call, from creation through its final outcome."""

    __tablename__ = "dossier_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    subject: Mapped[str] = mapped_column(String)
    depth: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="running")  # running|succeeded|failed
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    memo: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    verification: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sources: Mapped[list | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
