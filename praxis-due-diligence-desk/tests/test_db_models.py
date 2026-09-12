"""DossierRun round-trips through the async engine, including its JSON columns."""

import datetime as dt

from sqlalchemy import select

from praxis.db import DossierRun


async def test_dossier_run_round_trips_through_the_session(db_session):
    run = DossierRun(
        subject="Acme Robotics",
        depth="standard",
        status="succeeded",
        memo={"summary": "s", "recommendation": "pass"},
        verification={"coverage_score": 1.0},
        sources=["acme-10-k-abc123"],
    )
    db_session.add(run)
    await db_session.commit()

    fetched = await db_session.get(DossierRun, run.id)
    assert fetched is not None
    assert fetched.subject == "Acme Robotics"
    assert fetched.status == "succeeded"
    assert fetched.memo == {"summary": "s", "recommendation": "pass"}
    assert fetched.verification == {"coverage_score": 1.0}
    assert fetched.sources == ["acme-10-k-abc123"]
    assert fetched.error is None
    assert fetched.finished_at is None


async def test_dossier_run_defaults_id_status_and_created_at(db_session):
    run = DossierRun(subject="Acme", depth="quick")
    db_session.add(run)
    await db_session.commit()

    assert run.id  # a uuid4 string, auto-generated
    assert run.status == "running"
    assert run.created_at is not None


async def test_listing_orders_newest_first(db_session):
    now = dt.datetime.now(dt.UTC)
    a = DossierRun(subject="First", depth="quick", created_at=now - dt.timedelta(minutes=1))
    b = DossierRun(subject="Second", depth="quick", created_at=now)
    db_session.add_all([a, b])
    await db_session.commit()

    rows = (
        (await db_session.execute(select(DossierRun).order_by(DossierRun.created_at.desc())))
        .scalars()
        .all()
    )
    assert [r.subject for r in rows] == ["Second", "First"]
