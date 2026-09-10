"""Deterministic memo-structure eval — no LLM, no network, safe as a CI gate.

Runs one dossier over the frozen fixture corpus with the fake provider and
asserts the memo has the shape a real analyst memo must have:

    * one section per rubric item, in rubric order
    * every section carries at least one citation
    * every citation resolves to gathered evidence (0 hallucination)
    * full rubric coverage
    * recommendation is a valid verdict
    * the verifier raised no flags

    uv run python evals/memo_structure_eval.py             # report
    uv run python evals/memo_structure_eval.py --gate      # exit 1 on any failure
    uv run python evals/memo_structure_eval.py --self-check  # prove it catches a broken memo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import get_args

from praxis.graph import run_dossier
from praxis.graph.nodes.verifier import verifier
from praxis.rag import ingest_source
from praxis.rag.corpus import Corpus
from praxis.rag.embed import HashEmbedder
from praxis.rag.rerank import LexicalReranker
from praxis.rag.vector_store import QdrantVectorStore
from praxis.schemas import (
    RUBRIC_SECTIONS,
    Citation,
    DossierMemo,
    DossierRequest,
    MemoSection,
    Recommendation,
    ResearchPlan,
    SubQuestion,
    VerificationReport,
)

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "datasets" / "corpus"


def _corpus() -> Corpus:
    embedder = HashEmbedder(256)
    corpus = Corpus(
        embedder=embedder,
        vector_store=QdrantVectorStore(collection="memo_structure_eval", dim=embedder.dim),
        reranker=LexicalReranker(),
    )
    for path in sorted(CORPUS_DIR.glob("*.md")):
        ingest_source(path=str(path), corpus=corpus)
    return corpus


def _checks(memo: DossierMemo, v: VerificationReport) -> list[tuple[str, bool]]:
    headings = [s.heading for s in memo.sections[: len(RUBRIC_SECTIONS)]]
    return [
        ("sections match the rubric in order", headings == list(RUBRIC_SECTIONS)),
        ("every section is cited", all(s.citations for s in memo.sections)),
        ("no hallucinated citations", v.hallucinated_citation_rate == 0.0),
        ("full rubric coverage", v.coverage_score == 1.0),
        ("recommendation is a valid verdict", memo.recommendation in get_args(Recommendation)),
        ("verifier raised no flags", not v.flags),
    ]


def _real_run() -> tuple[DossierMemo, VerificationReport]:
    resp = run_dossier(DossierRequest(subject="Acme Robotics"), corpus=_corpus())
    return resp.memo, resp.verification


def _broken_run() -> tuple[DossierMemo, VerificationReport]:
    """A malformed memo (2 sections, one uncited, one fabricated citation) — the
    checks must reject it."""
    memo = DossierMemo(
        subject="Acme Robotics",
        summary="s",
        recommendation="proceed",
        confidence=0.5,
        sections=[
            MemoSection(heading="Overview", body="b", citations=[]),
            MemoSection(
                heading="Risks & Red Flags",
                body="b",
                citations=[Citation(source_id="ghost", title="?", locator="-", quote="-")],
            ),
        ],
    )
    plan = ResearchPlan(
        subject="Acme Robotics",
        thesis="t",
        sub_questions=[SubQuestion(section=s, question="q") for s in RUBRIC_SECTIONS],
    )
    report = verifier({"memo": memo, "plan": plan, "evidence": []}, {})["verification"]
    return memo, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="exit non-zero on any failed check")
    parser.add_argument(
        "--self-check", action="store_true", help="run against a deliberately broken memo"
    )
    args = parser.parse_args(argv)

    memo, report = _broken_run() if args.self_check else _real_run()
    results = _checks(memo, report)

    print(
        f"memo-structure eval — subject 'Acme Robotics'{' [self-check]' if args.self_check else ''}\n"
    )
    for name, ok in results:
        print(f"  {'ok ' if ok else 'FAIL'} {name}")
    failed = [name for name, ok in results if not ok]

    if args.self_check:
        if failed:
            print("\nSELF-CHECK PASSED — the gate rejects a broken memo")
            return 0
        print("\nSELF-CHECK FAILED — a broken memo slipped through", file=sys.stderr)
        return 1

    if args.gate:
        if failed:
            print(f"\nGATE FAILED: {failed}", file=sys.stderr)
            return 1
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
