# evals/

The evaluation framework. Built out across phases; the deterministic retrieval
slice is live and gates CI today.

## Live now (Phase 1)

```
datasets/
  corpus/*.md                  # 4 frozen fixture documents (Acme, Globex, Nimbus)
  golden_questions.jsonl       # 14 questions, each with an expected_doc + answer_contains
rag_eval.py                    # deterministic retrieval eval  (no LLM, no network)  <-- CI gate
memo_structure_eval.py         # deterministic memo-shape eval (no LLM, no network) <-- CI gate
rag_eval_ragas.py              # RAGAS answer-quality eval     (needs keys; not in CI)
```

`memo_structure_eval.py` runs one dossier over the fixture corpus with the fake
provider and asserts the memo has analyst-memo shape: one section per rubric item
in order, every section cited, 0 hallucinated citations, full coverage, a valid
recommendation, no verifier flags. `--self-check` runs it against a deliberately
broken memo to prove the gate bites.

```bash
make eval-structure          # report
make eval-structure-gate     # exit 1 on any failed check  <-- runs in CI
```

`rag_eval.py` ingests the fixture corpus, runs every golden question through
`search_corpus`, and reports:

| metric | meaning |
|---|---|
| `recall@k` | expected doc appears in the top-k (k=5) |
| `hit@1` | expected doc ranked first |
| `mrr` | mean reciprocal rank of the expected doc |
| `answer_hit` | a top-2 chunk literally contains an expected answer string |

```bash
make eval-rag          # report
make eval-rag-gate     # exit 1 if any metric falls below its floor  <-- runs in CI
```

Thresholds are in `rag_eval.py` (`THRESHOLDS`). They sit below the current
numbers so the gate catches a real regression without flapping.

## Planned

- **Phase 4** — `datasets/golden_dossiers/` (subjects with expert pass/fail labels +
  critiques), `metric.py` (`BinaryLLMJudgeMetric`), `memo_eval.py` (judge-vs-expert
  **F1**), `citation_eval.py` (NLI claim↔span), `redteam_eval.py` (seeded-error
  detection). Added to the CI eval gate on a fixed, cheap slice.

The deterministic citation-integrity slice already runs inside every dossier via
[`src/praxis/graph/nodes/verifier.py`](../src/praxis/graph/nodes/verifier.py).
