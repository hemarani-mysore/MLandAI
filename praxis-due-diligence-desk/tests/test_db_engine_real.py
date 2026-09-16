"""Round-trips a `DossierRun` through the app's *own* engine construction
(`get_engine()`/`ensure_schema()`/`get_sessionmaker()`, driven by whatever
`PRAXIS_DATABASE_URL` resolves to) — unlike `test_db_models.py`, which uses
`conftest.py`'s isolated in-memory-SQLite `db_session` fixture and so never
actually exercises `_database_url()`'s resolution or a real Postgres
connection. Passes against the SQLite default locally; the `postgres-contract`
CI job points `PRAXIS_DATABASE_URL` at a real `postgres:16` service container,
so this same test is what proves the app really works against Postgres."""

import praxis.db.session as db_session_mod
from praxis.db import DossierRun


async def test_dossier_run_round_trips_through_the_apps_own_engine():
    db_session_mod.get_engine.cache_clear()
    db_session_mod.get_sessionmaker.cache_clear()
    db_session_mod._initialized = False
    try:
        await db_session_mod.ensure_schema()
        async with db_session_mod.get_sessionmaker()() as session:
            run = DossierRun(subject="Acme Robotics", depth="standard", status="succeeded")
            session.add(run)
            await session.commit()

            fetched = await session.get(DossierRun, run.id)
            assert fetched is not None
            assert fetched.subject == "Acme Robotics"
    finally:
        await db_session_mod.get_engine().dispose()
        db_session_mod.get_engine.cache_clear()
        db_session_mod.get_sessionmaker.cache_clear()
        db_session_mod._initialized = False
