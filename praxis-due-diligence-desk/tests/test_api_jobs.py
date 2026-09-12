"""POST /corpus/jobs enqueues ingestion; GET /corpus/jobs/{id} reports status.

Runs as one async test (httpx.AsyncClient over the ASGI app directly) rather
than through the shared sync `client` fixture: draining the fakeredis queue
with an inline burst Worker needs to share the *same* event loop as the HTTP
calls — TestClient's requests run on its own background-thread loop, so a
separate `asyncio.run()` to drain the worker binds fakeredis's internal
asyncio primitives to a different loop than a later `client.get(...)` expects,
raising "bound to a different event loop".
"""

from unittest.mock import AsyncMock, patch

import pytest
from arq.worker import Worker
from httpx import ASGITransport, AsyncClient

from praxis.api.deps import db_session_dep, redis_dep
from praxis.api.main import app
from praxis.worker import ingest_task


async def _drain(redis_pool) -> None:
    with patch("arq.worker.log_redis_info", AsyncMock()):
        worker = Worker(functions=[ingest_task], redis_pool=redis_pool, burst=True, poll_delay=0)
        await worker.async_run()
        await worker.close()


@pytest.fixture
def async_client(db_session, redis_pool):
    # corpus_dep is deliberately left un-overridden: ingest_task (running as a
    # worker job, outside FastAPI's DI) ingests into the real process-wide
    # get_corpus() singleton, same as it would against a real worker process —
    # so /corpus/stats must read that same singleton, not a per-test fixture.
    # _isolate_corpus_state (autouse) still gives each test a fresh one.
    async def _db_override():
        yield db_session

    async def _redis_override():
        yield redis_pool

    app.dependency_overrides[db_session_dep] = _db_override
    app.dependency_overrides[redis_dep] = _redis_override
    yield AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    app.dependency_overrides.clear()


async def test_enqueue_then_poll_to_completion(async_client, redis_pool):
    enqueued = await async_client.post(
        "/corpus/jobs", json={"text": "Acme Robotics revenue was $42M.", "title": "Acme brief"}
    )
    assert enqueued.status_code == 200
    job_id = enqueued.json()["job_id"]

    pending = (await async_client.get(f"/corpus/jobs/{job_id}")).json()
    assert pending["status"] in ("deferred", "queued")

    await _drain(redis_pool)

    done = (await async_client.get(f"/corpus/jobs/{job_id}")).json()
    assert done["status"] == "complete"
    assert done["success"] is True
    assert done["result"]["title"] == "Acme brief"

    stats = (await async_client.get("/corpus/stats")).json()
    assert stats["documents"] == 1


async def test_enqueue_requires_input(async_client):
    assert (await async_client.post("/corpus/jobs", json={})).status_code == 422


async def test_poll_unknown_job_is_404(async_client):
    assert (await async_client.get("/corpus/jobs/no-such-job")).status_code == 404
