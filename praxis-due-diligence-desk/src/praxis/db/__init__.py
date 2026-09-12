"""Durable storage for dossier runs. SQLite by default (``PRAXIS_DATABASE_URL``
empty); Postgres + Alembic migrations land in Phase 5 alongside the rest of the
deploy story."""

from praxis.db.models import Base, DossierRun
from praxis.db.session import db_session_dep, ensure_schema, get_engine, get_sessionmaker

__all__ = [
    "Base",
    "DossierRun",
    "db_session_dep",
    "ensure_schema",
    "get_engine",
    "get_sessionmaker",
]
