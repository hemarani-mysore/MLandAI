"""The ingestion worker, fully in-process against fakeredis — no real Redis.

Two fakeredis gotchas, both worked around here (see also conftest.redis_pool):
`Worker.main()` unconditionally calls `arq.worker.log_redis_info()`, which runs
`INFO` — a command this fakeredis version doesn't implement, so a real worker
run crashes at startup unless it's patched out. A real deployment (real Redis)
never hits this.
"""

from unittest.mock import AsyncMock, patch

from arq.jobs import Job, JobStatus
from arq.worker import Worker

from praxis.rag import get_corpus
from praxis.worker import ingest_task


async def _run_burst(redis_pool) -> None:
    with patch("arq.worker.log_redis_info", AsyncMock()):
        worker = Worker(functions=[ingest_task], redis_pool=redis_pool, burst=True, poll_delay=0)
        await worker.async_run()
        await worker.close()


async def test_ingest_task_grows_the_corpus(redis_pool):
    job = await redis_pool.enqueue_job(
        "ingest_task", text="Acme Robotics holds 14 US patents.", title="Acme brief"
    )
    assert job is not None

    await _run_burst(redis_pool)

    status = await job.status()
    assert status == JobStatus.complete

    result = await job.result(timeout=5)
    assert result["title"] == "Acme brief"
    assert result["chunks"] >= 1

    assert get_corpus().stats().documents == 1


async def test_failed_ingest_task_reports_the_error(redis_pool):
    job = await redis_pool.enqueue_job("ingest_task")  # neither text nor path -> raises
    assert job is not None

    await _run_burst(redis_pool)

    info = await Job(job.job_id, redis_pool).result_info()
    assert info is not None
    assert info.success is False
