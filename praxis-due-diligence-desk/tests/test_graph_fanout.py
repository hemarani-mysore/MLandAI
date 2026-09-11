"""Researcher fan-out via `Send` — dispatch shape and observed concurrency."""

import asyncio

from praxis.graph import run_dossier
from praxis.graph.nodes.researcher import ResearchTask, dispatch_gap_fill, dispatch_research
from praxis.graph.state import initial_state
from praxis.llm import FakeStructuredLLM
from praxis.rag import ingest_source
from praxis.schemas import RUBRIC_SECTIONS, DossierRequest, RedTeamReport, ResearchPlan, SubQuestion


def _plan() -> ResearchPlan:
    return ResearchPlan(
        subject="Acme",
        thesis="t",
        sub_questions=[SubQuestion(section=s, question=f"Q about {s}") for s in RUBRIC_SECTIONS],
    )


def test_dispatch_research_emits_one_send_per_sub_question():
    state = initial_state("Acme", "standard")
    state["plan"] = _plan()

    sends = dispatch_research(state)

    assert len(sends) == len(RUBRIC_SECTIONS)
    assert all(s.node == "research_one" for s in sends)
    tasks: list[ResearchTask] = [s.arg for s in sends]
    assert {t["section"] for t in tasks} == set(RUBRIC_SECTIONS)
    assert all(t["subject"] == "Acme" for t in tasks)


def test_dispatch_gap_fill_emits_one_send_per_follow_up():
    state = initial_state("Acme", "standard")
    state["redteam"] = RedTeamReport(gap_questions=["What is ARR?", "Who are the customers?"])

    sends = dispatch_gap_fill(state)

    assert len(sends) == 2
    assert all(s.node == "research_one" for s in sends)
    assert all(s.arg["section"] == "Gap-fill" for s in sends)


class _ConcurrencyTrackingLLM(FakeStructuredLLM):
    """Records how many `agenerate` calls for role="researcher" are in flight
    at once. `asyncio.sleep` yields to the event loop, so overlap here proves
    the branches genuinely interleave on the loop — a blocking `time.sleep`
    would serialize everything and always read back 1."""

    def __init__(self, *, delay: float = 0.05) -> None:
        super().__init__()
        self._delay = delay
        self._lock = asyncio.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    async def agenerate(self, *, system, user, schema, role):  # type: ignore[override]
        if role != "researcher":
            return await super().agenerate(system=system, user=user, schema=schema, role=role)
        async with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(self._delay)
            return self.generate(system=system, user=user, schema=schema, role=role)
        finally:
            async with self._lock:
                self._in_flight -= 1


def test_research_branches_run_concurrently_bounded_by_the_cap(corpus):
    ingest_source(
        text=(
            "Overview Market Competition Team Funding Technology IP Traction "
            "Financials Risks Red Flags — a document covering every section."
        ),
        title="Everything doc",
        corpus=corpus,
    )
    llm = _ConcurrencyTrackingLLM()

    resp = run_dossier(
        DossierRequest(subject="Acme Robotics"),
        llm=llm,
        corpus=corpus,
        research_concurrency=3,
    )

    assert resp.evidence_count == len(RUBRIC_SECTIONS)  # unchanged behaviour
    assert 1 < llm.max_in_flight <= 3


def test_gap_fill_loop_still_bounded_with_fanout():
    llm = FakeStructuredLLM(
        overrides={
            "red_team": [
                RedTeamReport(coverage_gaps=["financials thin"], gap_questions=["What is ARR?"]),
                RedTeamReport(),  # second pass: clean -> editor
            ]
        }
    )
    resp = run_dossier(DossierRequest(subject="Acme Robotics"), llm=llm, max_gap_loops=1)

    assert resp.iterations == 2
    assert llm.calls["red_team"] == 2
    assert resp.evidence_count == len(RUBRIC_SECTIONS) + 1
