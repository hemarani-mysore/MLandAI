from praxis.graph import run_dossier
from praxis.llm import FakeStructuredLLM
from praxis.schemas import (
    RUBRIC_SECTIONS,
    DossierMemo,
    DossierRequest,
    MemoSection,
    RedTeamReport,
)


def test_happy_path_produces_a_memo():
    resp = run_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())

    assert resp.plan.subject == "Acme Robotics"
    assert len(resp.plan.sub_questions) == len(RUBRIC_SECTIONS)
    assert isinstance(resp.memo, DossierMemo)
    # one section per rubric item, in rubric order
    assert [s.heading for s in resp.memo.sections] == list(RUBRIC_SECTIONS)
    assert resp.evidence_count == len(RUBRIC_SECTIONS)
    assert resp.iterations == 1  # one red-team pass, no gap-fill
    # No corpus -> nothing is grounded -> honest verdict + flag, but no hallucination.
    assert resp.memo.recommendation == "insufficient_evidence"
    assert resp.verification.hallucinated_citation_rate == 0.0
    assert any("no grounded sources" in f for f in resp.verification.flags)


def test_gap_fill_loop_runs_then_terminates():
    llm = FakeStructuredLLM(
        overrides={
            "red_team": [
                RedTeamReport(coverage_gaps=["financials thin"], gap_questions=["What is ARR?"]),
                RedTeamReport(),  # second pass: clean -> go to editor
            ]
        }
    )
    resp = run_dossier(DossierRequest(subject="Acme Robotics"), llm=llm, max_gap_loops=1)

    assert resp.iterations == 2
    assert llm.calls["red_team"] == 2
    # first pass: 6 rubric questions; gap-fill pass: 1 follow-up question
    assert resp.evidence_count == len(RUBRIC_SECTIONS) + 1


def test_researcher_grounds_evidence_in_the_corpus(corpus):
    from praxis.rag import ingest_source

    ingest_source(
        text=(
            "Acme Robotics 2025 revenue was 42 million dollars. The main risk is "
            "supplier concentration. Competitors: Globex and Nimbus. 14 US patents."
        ),
        title="Acme 10-K",
        corpus=corpus,
    )
    resp = run_dossier(
        DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM(), corpus=corpus
    )

    assert len(resp.sources) == 1
    assert resp.sources[0].startswith("acme-10-k-")
    assert resp.evidence_count == len(RUBRIC_SECTIONS)
    # a fully grounded, rubric-complete memo -> verifier passes clean
    assert [s.heading for s in resp.memo.sections] == list(RUBRIC_SECTIONS)
    assert all(s.citations for s in resp.memo.sections)
    assert resp.verification.hallucinated_citation_rate == 0.0
    assert resp.verification.coverage_score == 1.0
    assert resp.verification.flags == []
    assert resp.memo.recommendation != "insufficient_evidence"


def test_no_corpus_yields_no_source_evidence():
    resp = run_dossier(DossierRequest(subject="Acme Robotics"), llm=FakeStructuredLLM())
    assert resp.sources == []
    assert resp.evidence_count == len(RUBRIC_SECTIONS)
    assert resp.memo.confidence <= 0.35  # thin evidence -> low confidence


def test_verifier_flags_unresolvable_citations():
    from praxis.schemas import Citation

    bad_memo = DossierMemo(
        subject="Acme Robotics",
        summary="s",
        recommendation="pass",
        confidence=0.3,
        sections=[
            MemoSection(
                heading="Overview",
                body="b",
                citations=[Citation(source_id="ghost", title="?", locator="p.9", quote="?")],
            )
        ],
    )
    llm = FakeStructuredLLM(overrides={"editor": [bad_memo]})
    resp = run_dossier(DossierRequest(subject="Acme Robotics"), llm=llm)

    # the fabricated "ghost" citation is caught (other sections get reconciled in)
    assert resp.verification.hallucinated_citation_rate > 0
    assert any("do not resolve" in f for f in resp.verification.flags)
