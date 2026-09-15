# golden_dossiers/

8 synthetic due-diligence subjects for the Phase 4 LLM-as-judge harness
(`evals/metric.py`, `evals/memo_eval.py`). None are real companies.

```
index.yaml              # id, split, expert label (pass/fail), critique
loader.py               # load_index() -> list[GoldenItem]
<id>/
  sources/*.md           # 1-2 short source documents (the raw evidence)
  memo.md                 # the frozen memo the judge grades (dev/test only)
```

## Splits

| split | items | purpose | judge mode |
|---|---|---|---|
| `dev` | 4 | judge calibration, run by hand | live LLM |
| `test` | 3 | the CI gate | recorded verdicts (`evals/cassettes/`), zero network calls |
| `online` | 1 | nightly drift check | live LLM, memo generated fresh from `sources/` each run (no frozen `memo.md`) |

## How an item is judged

`BinaryLLMJudgeMetric` (`evals/metric.py`) is **not** reference-free in the
sense of ignoring the sources — it reads `(subject, sources text, memo text)`
and decides pass/fail against the rubric. This matters: a memo can be
internally coherent and *still* fail, if it misrepresents or omits something
the sources plainly state (see `driftwood-materials`) or invents specifics
the sources never gave it (see `ferrovia-rail-tech`). Feeding the judge only
the memo, without the sources, cannot catch either failure mode — that's an
explicit design point, not an oversight.

There is no separate hand-authored "gold" reference memo (`--reference` /
`expected.md` from the original design sketch) — the expert `label` is the
ground truth the judge is scored against; a second gold memo per item isn't
needed for F1 and would double the authoring cost. Documented as a
not-built extension point, not a silent gap.

## The two deliberate "fail" items

- **`driftwood-materials`** — the sources disclose a Q2 customer loss (38% of
  revenue, quality-control driven) and a preliminary FTC pricing inquiry. The
  frozen memo mentions the loss only as vague "revenue volatility" and omits
  the FTC inquiry from Risks & Red Flags entirely, then recommends `proceed`
  at 78% confidence — a judge with the sources in view should catch the gap
  between what's disclosed and what the memo represents.
- **`ferrovia-rail-tech`** — the source is one sparse paragraph with no
  financials, market sizing, or IP status. The frozen memo nonetheless states
  a specific ($2.1B) market size and describes "promising traction," neither
  of which the source supports, and recommends `proceed` — a well-behaved
  system should have concluded `insufficient_evidence` instead of inventing
  detail to fill the gap.

## Authoring notes

Each subject is an independent synthetic scenario (a different industry,
company stage, and specific risk shape) so the set isn't dominated by one
failure pattern. `pass` items still name a real risk and reflect it honestly
in the recommendation (`proceed_with_conditions` where warranted) — a pass
label means the memo is *faithful to its sources*, not that the company is
risk-free.
