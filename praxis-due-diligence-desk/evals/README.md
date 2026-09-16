# evals/

The evaluation framework. Built out across phases; every gate below runs in
CI today via `evals/run.py` / the individual `make eval-*-gate` targets.

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

## Live now (Phase 4)

```
datasets/golden_dossiers/       # 8 subjects (dev=4, test=3, online=1), 2 deliberate fails
metric.py                       # BinaryLLMJudgeMetric — reads (subject, sources, memo)
memo_eval.py                    # --split {dev,test,online}: judge-vs-expert F1  <-- CI gate (test)
citation_eval.py                # fabricated-citation injection self-check + --nli  <-- CI gate
redteam_eval.py                 # seeded false-claim detection, 4 fixtures  <-- CI gate
run.py                          # --suite {rag,structure,memo,citation,redteam,all} --gate --report
cassettes/                      # real verdicts recorded once, replayed offline for CI
```

The judge (`metric.py`) is deliberately **not** reference-free with respect
to the sources — it reads the raw source material alongside the memo. Both
of the dataset's deliberate fails (`driftwood-materials`: a disclosed risk
reduced to vague language and partly omitted; `ferrovia-rail-tech`: a sparse
source inflated with invented specifics) are only catchable by comparing the
memo against what its sources actually said — see
`datasets/golden_dossiers/README.md` for the full writeup, including how the
prompt was calibrated against a real model to reach 4/4 dev + 3/3 test
agreement with the expert labels.

`--split test` and `redteam_eval.py --gate` never make a network call —
both replay real verdicts recorded once against an actual model
(`_record_judge_test.py`, `_record_redteam.py`), through the same
`FakeStructuredLLM(overrides=...)` scripting the rest of the test suite
already uses, not a new cassette dependency.

```bash
make eval-dev          # judge-vs-expert F1, live model, dev split (report)
make eval-test-gate    # judge-vs-expert F1 on the recorded test split  <-- CI gate
make eval-citation-gate
make eval-redteam-gate
make eval-gate          # every gate above via evals/run.py, one command  <-- what CI runs
make eval               # every suite, live judge, writes a timestamped report
```

The deterministic citation-*resolution* check (every citation must trace back
to real gathered evidence) already runs inside every dossier via
[`src/praxis/graph/nodes/verifier.py`](../src/praxis/graph/nodes/verifier.py)
and is asserted by `memo_structure_eval.py`; `citation_eval.py` adds the
fabrication-injection proof and the NLI check on top, rather than duplicating it.

OpenTelemetry spans (`src/praxis/obs/`) wrap every graph node
(`role`, and for the six LLM-calling ones `model`/`prompt_tokens`/
`completion_tokens`/`cost_usd`) — gated on `PRAXIS_OTEL_ENABLED`, console
by default or a real OTLP/HTTP collector via `PRAXIS_OTEL_ENDPOINT`.
`DossierResponse.cost_usd`/`.tokens` report the same run-level totals.

Nightly (`.github/workflows/nightly.yml`) runs the full suite — `dev` (judge
calibration) + `test` (still run live-adjacent here to catch behavioural
drift, even though it doesn't block PRs) + `online` (fresh generation, the
actual drift check) — against a real cheap model, uploading a report
artifact. Skips cleanly if `OPENAI_API_KEY` isn't set as a repo secret yet.
