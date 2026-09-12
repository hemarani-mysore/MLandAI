"""A checkpointed run actually writes checkpoints, retrievable by thread_id."""

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from praxis.graph import arun_dossier
from praxis.graph.build import build_graph
from praxis.llm import FakeStructuredLLM
from praxis.schemas import RUBRIC_SECTIONS, DossierRequest


async def test_checkpointer_persists_state_under_the_given_thread_id():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()

        resp = await arun_dossier(
            DossierRequest(subject="Acme Robotics"),
            llm=FakeStructuredLLM(),
            checkpointer=saver,
            thread_id="t-checkpoint-1",
        )
        assert resp.evidence_count == len(RUBRIC_SECTIONS)

        graph = build_graph(saver)
        snapshot = await graph.aget_state({"configurable": {"thread_id": "t-checkpoint-1"}})

        assert snapshot.values["subject"] == "Acme Robotics"
        assert snapshot.values["memo"] is not None
        assert snapshot.next == ()  # the run reached END

        history = [
            c
            async for c in graph.aget_state_history(
                {"configurable": {"thread_id": "t-checkpoint-1"}}
            )
        ]
        assert len(history) > 1  # one checkpoint per superstep, not just the final one


async def test_a_run_without_a_checkpointer_needs_no_thread_id():
    # unchanged default behaviour — checkpointing is opt-in
    resp = await arun_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())
    assert resp.evidence_count == len(RUBRIC_SECTIONS)


async def test_different_threads_do_not_share_state():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        await saver.setup()

        await arun_dossier(
            DossierRequest(subject="Acme Robotics"),
            llm=FakeStructuredLLM(),
            checkpointer=saver,
            thread_id="thread-a",
        )
        await arun_dossier(
            DossierRequest(subject="Globex Corp"),
            llm=FakeStructuredLLM(),
            checkpointer=saver,
            thread_id="thread-b",
        )

        graph = build_graph(saver)
        snap_a = await graph.aget_state({"configurable": {"thread_id": "thread-a"}})
        snap_b = await graph.aget_state({"configurable": {"thread_id": "thread-b"}})

        assert snap_a.values["subject"] == "Acme Robotics"
        assert snap_b.values["subject"] == "Globex Corp"
