"""`evals/citation_eval.py` — the fabricated-citation self-check actually
catches the injection (offline, deterministic — no LLM call needed for this
part; `verifier` does no I/O)."""

import importlib.util
from pathlib import Path

_EVAL = Path(__file__).parents[1] / "evals" / "citation_eval.py"
_spec = importlib.util.spec_from_file_location("citation_eval", _EVAL)
assert _spec and _spec.loader
citation_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(citation_eval)


def test_self_check_catches_the_fabricated_citation():
    report = citation_eval.self_check()
    assert report["caught"] is True
    assert report["hallucinated_citation_rate"] > 0
    assert any("do not resolve to gathered evidence" in f for f in report["flags"])


def test_gate_passes():
    assert citation_eval.main(["--gate"]) == 0


def test_a_memo_with_no_fabrication_would_not_be_falsely_caught():
    # Sanity check on the fixture itself: real_evidence()'s source_id is what
    # every *non*-fabricated citation in the injected memo also uses, so the
    # only hallucinated one is the deliberately planted "ghost-citation-007".
    evidence = citation_eval._real_evidence()
    memo = citation_eval._memo_with_a_fabricated_citation()
    cited_ids = {c.source_id for section in memo.sections for c in section.citations}
    real_ids = {ev.citation.source_id for ev in evidence}
    fabricated = cited_ids - real_ids
    assert fabricated == {"ghost-citation-007"}
