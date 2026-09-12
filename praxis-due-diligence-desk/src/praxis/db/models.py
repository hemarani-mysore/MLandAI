"""SQLAlchemy models: a durable record of each dossier run, plus the per-node
events streamed while it ran."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
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


class RunEvent(Base):
    """One event streamed while a `DossierRun` executed (`GET /dossiers/stream`).
    Only the streaming path writes these — a plain `POST /dossiers` run has
    nothing to stream, so it has no events."""

    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("dossier_runs.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)  # order within the run, 0-based
    kind: Mapped[str] = mapped_column(String)  # node_start|node_end|complete|error
    node: Mapped[str | None] = mapped_column(String, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
