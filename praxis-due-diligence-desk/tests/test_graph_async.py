"""The graph is async end to end (`arun_dossier`); `run_dossier` is a thin sync
wrapper over it via `asyncio.run`. Both paths must agree."""

from praxis.graph import arun_dossier, run_dossier
from praxis.llm import FakeStructuredLLM
from praxis.schemas import RUBRIC_SECTIONS, DossierRequest


async def test_arun_dossier_produces_a_full_response():
    resp = await arun_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())

    assert resp.plan.subject == "Acme Robotics"
    assert resp.evidence_count == len(RUBRIC_SECTIONS)
    assert [s.heading for s in resp.memo.sections] == list(RUBRIC_SECTIONS)
    assert resp.verification.hallucinated_citation_rate == 0.0


def test_sync_wrapper_matches_the_async_entrypoint():
    request = DossierRequest(subject="Acme Robotics")
    sync_resp = run_dossier(request, llm=FakeStructuredLLM())

    assert sync_resp.plan.subject == "Acme Robotics"
    assert sync_resp.evidence_count == len(RUBRIC_SECTIONS)
    assert sync_resp.memo.recommendation == "insufficient_evidence"  # no corpus, either path
