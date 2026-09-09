"""Deterministic retrieval eval — no LLM, no network, safe as a CI gate.

Ingests the frozen fixture corpus, runs every golden question through
``search_corpus``, and scores whether the expected document is retrieved.

    uv run python evals/rag_eval.py            # report
    uv run python evals/rag_eval.py --gate     # exit 1 if below thresholds

Metrics
    recall@k   fraction of questions whose expected doc appears in the top-k
    hit@1      fraction where the expected doc is ranked first
    MRR        mean reciprocal rank of the expected doc
    answer_hit fraction where a top chunk literally contains an expected string
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from praxis.rag import ingest_source, search_corpus
from praxis.rag.corpus import Corpus
from praxis.rag.embed import HashEmbedder
from praxis.rag.rerank import LexicalReranker
from praxis.rag.vector_store import QdrantVectorStore

HERE = Path(__file__).parent
CORPUS_DIR = HERE / "datasets" / "corpus"
QUESTIONS = HERE / "datasets" / "golden_questions.jsonl"

K = 5
THRESHOLDS = {"recall@k": 0.85, "hit@1": 0.60, "mrr": 0.70, "answer_hit": 0.70}


def _load_corpus() -> tuple[Corpus, dict[str, str]]:
    embedder = HashEmbedder(256)
    corpus = Corpus(
        embedder=embedder,
        vector_store=QdrantVectorStore(collection="rag_eval", dim=embedder.dim),
        reranker=LexicalReranker(),
    )
    doc_to_source: dict[str, str] = {}
    for path in sorted(CORPUS_DIR.glob("*.md")):
        result = ingest_source(path=str(path), corpus=corpus)
        doc_to_source[path.name] = result.source_id
    return corpus, doc_to_source


def _questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def evaluate() -> dict:
    corpus, doc_to_source = _load_corpus()
    rows = []
    for q in _questions():
        expected_source = doc_to_source[q["expected_doc"]]
        hits = search_corpus(q["question"], k=K, corpus=corpus)
        ranked_sources = [h.chunk.source_id for h in hits]

        rank = next((i + 1 for i, s in enumerate(ranked_sources) if s == expected_source), 0)
        top_text = " ".join(h.chunk.raw_text.lower() for h in hits[:2])
        answer_hit = any(s.lower() in top_text for s in q.get("answer_contains", []))

        rows.append(
            {
                "id": q["id"],
                "in_topk": rank != 0,
                "hit@1": rank == 1,
                "rr": 1.0 / rank if rank else 0.0,
                "answer_hit": answer_hit,
                "rank": rank,
            }
        )

    n = len(rows)
    metrics = {
        "recall@k": sum(r["in_topk"] for r in rows) / n,
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "answer_hit": sum(r["answer_hit"] for r in rows) / n,
    }
    return {"k": K, "n": n, "metrics": metrics, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="exit non-zero if below thresholds")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"RAG retrieval eval — {report['n']} questions, k={report['k']}\n")
        for name, value in report["metrics"].items():
            floor = THRESHOLDS[name]
            flag = "" if value >= floor else f"  ✗ below {floor:.2f}"
            print(f"  {name:<12} {value:.3f}{flag}")
        misses = [r["id"] for r in report["rows"] if not r["in_topk"]]
        if misses:
            print(f"\n  not retrieved: {', '.join(misses)}")

    if args.gate:
        failed = {k: v for k, v in report["metrics"].items() if v < THRESHOLDS[k]}
        if failed:
            print(f"\nGATE FAILED: {failed}", file=sys.stderr)
            return 1
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
