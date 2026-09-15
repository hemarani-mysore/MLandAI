"""Judge-vs-expert calibration: does the LLM-as-judge agree with the expert
`label` on each golden-dossier item? Scored as precision/recall/F1 with
`"fail"` as the positive class — the signal that matters is whether the judge
actually catches a bad memo.

    uv run python evals/memo_eval.py --split dev             # live judge, report
    uv run python evals/memo_eval.py --split test             # ALWAYS offline (see below)
    uv run python evals/memo_eval.py --split test --gate      # exit 1 if F1 < MEMO_F1_FLOOR
    uv run python evals/memo_eval.py --split online           # live judge, memo generated fresh

`--split test` never makes a network call, regardless of `PRAXIS_LLM_PROVIDER`
or `--gate`: it drives `FakeStructuredLLM` with real verdicts recorded once
against an actual model (`evals/cassettes/judge_test.json`, captured by
`evals/cassettes/_record_judge_test.py`), so the CI gate is fully
deterministic. `dev`/`online` always call the live judge via `get_llm()` —
`PRAXIS_LLM_PROVIDER=fake` will raise there (see `evals/metric.py`), since a
"fake" calibration number would be worse than a clear error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
GOLDEN_DIR = HERE / "datasets" / "golden_dossiers"
CASSETTE = HERE / "cassettes" / "judge_test.json"

# Sibling-module imports without polluting sys.path with anything that could
# shadow a real installed package — `evals/datasets/` must never be resolvable
# as a bare `datasets` (the real PyPI package `rag_eval_ragas.py` needs), so
# both siblings are added by their own directory, never at a broader scope.
for _p in (HERE, GOLDEN_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import loader as golden  # noqa: E402
from metric import BinaryLLMJudgeMetric, JudgeResult  # noqa: E402

from praxis.graph import run_dossier  # noqa: E402
from praxis.llm import FakeStructuredLLM, get_llm  # noqa: E402
from praxis.rag import ingest_source  # noqa: E402
from praxis.rag.corpus import Corpus  # noqa: E402
from praxis.rag.embed import HashEmbedder  # noqa: E402
from praxis.rag.rerank import LexicalReranker  # noqa: E402
from praxis.rag.vector_store import QdrantVectorStore  # noqa: E402
from praxis.render import to_markdown  # noqa: E402
from praxis.schemas import DossierRequest  # noqa: E402

MEMO_F1_FLOOR = 0.75


def _display_name(item_id: str) -> str:
    return item_id.replace("-", " ").title()


def _fresh_memo_text(item: golden.GoldenItem) -> str:
    """`online` items have no frozen `memo.md` — ingest their `sources/` into
    a scratch corpus and generate one for real, the same shape
    `memo_structure_eval.py`'s `_corpus()` helper uses."""
    embedder = HashEmbedder(256)
    corpus = Corpus(
        embedder=embedder,
        vector_store=QdrantVectorStore(collection=f"memo_eval_{item.id}", dim=embedder.dim),
        reranker=LexicalReranker(),
    )
    for path in sorted(item.sources_dir.glob("*.md")):
        ingest_source(path=str(path), corpus=corpus)
    resp = run_dossier(DossierRequest(subject=_display_name(item.id)), corpus=corpus)
    return to_markdown(resp)


def _load_cassette() -> dict[str, dict]:
    return json.loads(CASSETTE.read_text())


def _judge_for_split(split: str) -> tuple[BinaryLLMJudgeMetric, list]:
    """Returns the judge to use and the items in the fixed order the judge
    will be called in — for `test`, that order must match the order
    `evals/cassettes/_record_judge_test.py` recorded the cassette in."""
    items = sorted(golden.load_index(split=split), key=lambda i: i.id)
    if split != "test":
        return BinaryLLMJudgeMetric(get_llm()), items

    cassette = _load_cassette()
    missing = [item.id for item in items if item.id not in cassette]
    if missing:
        raise KeyError(f"evals/cassettes/judge_test.json is missing verdicts for: {missing}")
    overrides = {"judge": [JudgeResult(**cassette[item.id]) for item in items]}
    return BinaryLLMJudgeMetric(FakeStructuredLLM(overrides=overrides)), items


def evaluate(split: str) -> dict:
    judge, items = _judge_for_split(split)
    rows = []
    for item in items:
        memo_text = _fresh_memo_text(item) if item.memo_path is None else item.read_memo()
        result = judge.run(
            subject=_display_name(item.id),
            sources_text=item.read_sources(),
            memo_text=memo_text,
        )
        rows.append(
            {
                "id": item.id,
                "expert_label": item.label,
                "judge_label": result.label,
                "agree": result.label == item.label,
                "judge_critique": result.critique,
                "expert_critique": item.critique,
            }
        )

    return {"split": split, "n": len(rows), "metrics": score(rows), "rows": rows}


def score(rows: list[dict]) -> dict:
    """Precision/recall/F1/agreement from `{"judge_label", "expert_label"}`
    rows, with `"fail"` as the positive class — pure, no LLM, no I/O."""
    tp = sum(1 for r in rows if r["judge_label"] == "fail" and r["expert_label"] == "fail")
    fp = sum(1 for r in rows if r["judge_label"] == "fail" and r["expert_label"] == "pass")
    fn = sum(1 for r in rows if r["judge_label"] == "pass" and r["expert_label"] == "fail")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    agreement = (
        sum(1 for r in rows if r["judge_label"] == r["expert_label"]) / len(rows) if rows else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "agreement": agreement}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "online"), required=True)
    parser.add_argument("--gate", action="store_true", help="exit non-zero if F1 < floor")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = evaluate(args.split)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"memo-eval (judge vs expert) — split={report['split']}, n={report['n']}\n")
        for name, value in report["metrics"].items():
            print(f"  {name:<10} {value:.3f}")
        print()
        for row in report["rows"]:
            mark = "ok " if row["agree"] else "DISAGREE"
            print(
                f"  {mark} {row['id']:<24} expert={row['expert_label']:<4} judge={row['judge_label']:<4}"
            )
            if not row["agree"]:
                print(f"       judge:  {row['judge_critique']}")
                print(f"       expert: {row['expert_critique']}")

    if args.gate:
        f1 = report["metrics"]["f1"]
        if f1 < MEMO_F1_FLOOR:
            print(f"\nGATE FAILED: f1={f1:.3f} < floor={MEMO_F1_FLOOR}", file=sys.stderr)
            return 1
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
