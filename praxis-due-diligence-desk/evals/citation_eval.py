"""Citation integrity: proves the verifier's hallucination-detection logic
actually catches a fabricated citation, and (opt-in, nightly-only, never
gated) reports how often a claim isn't really entailed by its own cited quote.

    uv run python evals/citation_eval.py             # self-check report
    uv run python evals/citation_eval.py --gate       # exit 1 if the injection is NOT caught
    uv run python evals/citation_eval.py --nli        # also runs unsupported_claim_rate (real LLM calls)

This does *not* re-check "zero hallucinated citations on a normal run" —
`memo_structure_eval.py --gate` already covers that happy path. What's new
here: (a) proof that a genuinely fabricated citation (a source_id that is
neither real evidence nor a known placeholder) gets caught in an otherwise
well-formed, fully-cited memo — no existing test plants one — and (b) the
NLI-style claim<->quote entailment check, which nothing else in the project
does at all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from pydantic import BaseModel

from praxis.graph.nodes.verifier import verifier
from praxis.graph.state import initial_state
from praxis.llm import StructuredLLM, get_llm
from praxis.schemas import (
    RUBRIC_SECTIONS,
    Citation,
    DossierMemo,
    Evidence,
    MemoSection,
    ResearchPlan,
    SubQuestion,
)


class EntailmentResult(BaseModel):
    """Is a claim actually supported by the quote it's cited against? Local
    to this file — an eval-only concern, like `evals/metric.py`'s `JudgeResult`."""

    entailed: bool


ENTAILMENT_PROMPT = """\
Decide whether the CLAIM is entailed (directly supported) by the QUOTE. \
Answer `entailed: true` only if the quote actually states or clearly implies \
the claim — a quote that is merely on the same topic without stating the \
claim's specific content does not count as entailing it.
"""


def _real_evidence(source_id: str = "real-source") -> list[Evidence]:
    return [
        Evidence(
            question=f"Question about {s}",
            section=s,
            claim=f"{s} claim",
            stance="neutral",
            citation=Citation(source_id=source_id, title="Real Source", locator="p.1", quote="q"),
        )
        for s in RUBRIC_SECTIONS
    ]


def _memo_with_a_fabricated_citation() -> DossierMemo:
    """Every section cites `real-source` (genuinely gathered evidence) except
    one, which *also* cites `ghost-citation-007` — a source_id that was never
    gathered and isn't a known placeholder (`no-source`/`stub-001`)."""
    sections = []
    for s in RUBRIC_SECTIONS:
        citations = [
            Citation(source_id="real-source", title="Real Source", locator="p.1", quote="q")
        ]
        if s == "Risks & Red Flags":
            citations.append(
                Citation(
                    source_id="ghost-citation-007", title="Fabricated", locator="p.99", quote="q"
                )
            )
        sections.append(MemoSection(heading=s, body="b", citations=citations))
    return DossierMemo(
        subject="Acme", summary="s", recommendation="pass", confidence=0.5, sections=sections
    )


def self_check() -> dict:
    state = initial_state("Acme", "standard")
    state["plan"] = ResearchPlan(
        subject="Acme",
        thesis="t",
        sub_questions=[SubQuestion(section=s, question="q") for s in RUBRIC_SECTIONS],
    )
    state["evidence"] = _real_evidence()
    state["memo"] = _memo_with_a_fabricated_citation()
    report = verifier(state, {})["verification"]
    caught = report.hallucinated_citation_rate > 0 and any(
        "do not resolve to gathered evidence" in f for f in report.flags
    )
    return {
        "caught": caught,
        "hallucinated_citation_rate": report.hallucinated_citation_rate,
        "flags": report.flags,
    }


async def _gather_real_evidence(llm: StructuredLLM) -> list[Evidence]:
    """Real evidence from the researcher node against the eval fixture
    corpus — genuine claims grounded in real retrieved text, unlike
    `_real_evidence()` above (that fixture's claim/quote text is a template
    like `"Overview claim"`/`"q"`, built only to exercise citation
    *resolution*, not meaningful to judge for entailment)."""
    from pathlib import Path

    from praxis.graph.nodes.researcher import ResearchTask, research_one
    from praxis.rag import ingest_source
    from praxis.rag.corpus import Corpus
    from praxis.rag.embed import HashEmbedder
    from praxis.rag.rerank import LexicalReranker
    from praxis.rag.vector_store import QdrantVectorStore

    embedder = HashEmbedder(256)
    corpus = Corpus(
        embedder=embedder,
        vector_store=QdrantVectorStore(collection="citation_eval_nli", dim=embedder.dim),
        reranker=LexicalReranker(),
    )
    corpus_dir = Path(__file__).parent / "datasets" / "corpus"
    for path in sorted(corpus_dir.glob("*.md")):
        ingest_source(path=str(path), corpus=corpus)

    questions = [
        ("Overview", "What does Acme Robotics do?"),
        ("Traction & Financials", "What is Acme Robotics' revenue?"),
        ("Risks & Red Flags", "What risks does Globex face?"),
    ]
    config = {"configurable": {"llm": llm, "corpus": corpus}}
    evidence: list[Evidence] = []
    for section, question in questions:
        task = ResearchTask(subject="Acme Robotics", section=section, question=question)
        result = await research_one(task, config)
        evidence.extend(result["evidence"])
    return evidence


async def _entailed(llm: StructuredLLM, evidence: Evidence) -> bool:
    result = await llm.agenerate(
        system=ENTAILMENT_PROMPT,
        user=f"CLAIM: {evidence.claim}\n\nQUOTE: {evidence.citation.quote}",
        schema=EntailmentResult,
        role="entailment",
    )
    return result.entailed


async def unsupported_claim_rate() -> float:
    """Fraction of real, freshly-researched evidence whose claim is NOT
    entailed by its own cited quote — one cheap LLM call per item.
    Nightly-only, never gated: a small model doing NLI-by-prompting is
    noisier than the deterministic resolution check above, on purpose not
    held to a hard floor here."""
    llm = get_llm()
    evidence = await _gather_real_evidence(llm)
    if not evidence:
        return 0.0
    results = await asyncio.gather(*(_entailed(llm, ev) for ev in evidence))
    return 1 - (sum(results) / len(results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate", action="store_true", help="exit non-zero if the injection is not caught"
    )
    parser.add_argument(
        "--nli", action="store_true", help="also run the nightly claim<->quote entailment check"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = self_check()
    if args.nli:
        report["unsupported_claim_rate"] = asyncio.run(unsupported_claim_rate())

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("citation-integrity self-check — fabricated citation injection\n")
        mark = "ok " if report["caught"] else "FAIL"
        print(
            f"  {mark} fabricated citation caught (rate={report['hallucinated_citation_rate']:.3f})"
        )
        for f in report["flags"]:
            print(f"       flag: {f}")
        if "unsupported_claim_rate" in report:
            print(
                f"\n  unsupported_claim_rate (nightly, not gated): {report['unsupported_claim_rate']:.3f}"
            )

    if args.gate:
        if not report["caught"]:
            print("\nGATE FAILED: a fabricated citation was not caught", file=sys.stderr)
            return 1
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
