"""`praxis.db.session._database_url()` — the same URL resolution both the app
and `alembic/env.py` build their engine from."""

from praxis.config import get_settings
from praxis.db.session import _database_url


def test_defaults_to_sqlite_when_unset(monkeypatch):
    monkeypatch.setenv("PRAXIS_DATABASE_URL", "")
    get_settings.cache_clear()

    assert _database_url() == "sqlite+aiosqlite:///./praxis.db"

    get_settings.cache_clear()


def test_a_configured_postgres_url_passes_through_unchanged(monkeypatch):
    pg_url = "postgresql+asyncpg://praxis:secret@localhost:5432/praxis"
    monkeypatch.setenv("PRAXIS_DATABASE_URL", pg_url)
    get_settings.cache_clear()

    assert _database_url() == pg_url

    get_settings.cache_clear()
