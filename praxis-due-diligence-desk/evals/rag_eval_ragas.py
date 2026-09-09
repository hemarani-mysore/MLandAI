"""RAGAS eval — richer retrieval + answer quality, but needs an LLM + real
embeddings and network. NOT run in CI. Run manually with keys:

    uv sync --extra ragas
    PRAXIS_LLM_PROVIDER=openai PRAXIS_EMBEDDING_PROVIDER=openai \\
        uv run python evals/rag_eval_ragas.py

Scores faithfulness, answer relevancy, context precision and context recall over
the golden questions, using the same fixture corpus as ``rag_eval.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def _build_records() -> list[dict]:
    from praxis.graph.prompts import RESEARCHER
    from praxis.llm import get_llm
    from praxis.rag import ingest_source, search_corpus
    from praxis.rag.corpus import build_corpus

    corpus = build_corpus()
    for path in sorted((HERE / "datasets" / "corpus").glob("*.md")):
        ingest_source(path=str(path), corpus=corpus)

    llm = get_llm()
    questions = [
        json.loads(line)
        for line in (HERE / "datasets" / "golden_questions.jsonl").read_text().splitlines()
        if line.strip()
    ]

    records = []
    for q in questions:
        contexts = [h.chunk.raw_text for h in search_corpus(q["question"], k=5, corpus=corpus)]
        answer = llm.complete(
            system=RESEARCHER,
            user=f"Question: {q['question']}\n\nContext:\n" + "\n\n".join(contexts),
        )
        records.append(
            {
                "question": q["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ", ".join(q.get("answer_contains", [])),
            }
        )
    return records


def main() -> int:
    if importlib.util.find_spec("ragas") is None:
        print("ragas not installed — run: uv sync --extra ragas", file=sys.stderr)
        return 2

    from praxis.config import get_settings

    if get_settings().llm_provider == "fake" or get_settings().embedding_provider == "fake":
        print(
            "set PRAXIS_LLM_PROVIDER and PRAXIS_EMBEDDING_PROVIDER to a real provider",
            file=sys.stderr,
        )
        return 2

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    result = evaluate(
        Dataset.from_list(_build_records()),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
