# evals/ (Phase 4)

The evaluation framework. Lands in Phase 4; sketched here so the layout is fixed.

```
datasets/
  golden_dossiers/
    index.yaml                 # split -> [subject]; splits: dev, test, online
    <subject>/
      sources/                 # frozen source docs (RAG is reproducible)
      expected.md              # reference memo
      label                    # pass | fail  (expert judgement)
      critique                 # 1-3 sentences, why
  golden_questions.jsonl       # {question, expected_answer, expected_source}  -> RAG eval
metric.py                      # BinaryLLMJudgeMetric -> {label, critique}
memo_eval.py                   # judge vs expert labels -> precision / recall / F1
rag_eval.py                    # RAGAS: faithfulness, context precision, context recall
citation_eval.py               # deterministic: every claim resolves; hallucinated-cite rate
redteam_eval.py                # seed a false claim -> did the critic catch it?
run.py                         # --split {dev,test,online} [--gate]
```

**CI gate:** `run.py --split test --gate` runs a small fixed slice with a cheap
model (or recorded responses) and fails the build if `F1 < 0.75`,
`faithfulness < 0.85`, or `hallucinated_citation_rate > 0`.

The deterministic slice of this already exists today in
[`src/praxis/graph/nodes/verifier.py`](../src/praxis/graph/nodes/verifier.py).
