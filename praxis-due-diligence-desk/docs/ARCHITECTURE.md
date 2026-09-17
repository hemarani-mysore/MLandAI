# Architecture

```
                                   ┌─────────────┐
                    stdio/HTTP     │  MCP host   │
                 ┌─────────────────┤ (Claude Code│
                 │                 │  Cursor...) │
                 │                 └─────────────┘
                 ▼
   ┌──────────────────────┐        ┌──────────────────┐
   │   praxis-mcp (mcp)   │        │   FastAPI (api)   │◀── curl / browser
   │  4 tools, 3 resources│        │ /dossiers /health  │
   │  2 prompts            │        │ /ready /corpus/*   │
   └──────────┬────────────┘        └─────────┬─────────┘
              │                                │
              └───────────────┬────────────────┘
                               ▼
                    ┌────────────────────┐
                    │  LangGraph runtime  │   arun_dossier / run_dossier
                    │  (src/praxis/graph) │   — one shared entry point,
                    └──────────┬──────────┘     no logic fork per surface
                               │
     planner ──Send──▶▶ research_one × N ──(fan-in)──┬─▶ bull ──┐
               (≤ PRAXIS_RESEARCH_CONCURRENCY)        └─▶ bear ──┴─▶ red_team
                                                                       │
                        ┌──────────────────────────────────(gaps? loop<cap)
                        ▼
                     editor ─▶ verifier ─▶ END
                        │
          ┌─────────────┼──────────────────┬─────────────────┐
          ▼             ▼                  ▼                 ▼
   ┌────────────┐ ┌───────────┐    ┌──────────────┐   ┌─────────────┐
   │   Qdrant    │ │  Redis    │    │  Postgres /   │   │  praxis-    │
   │ (dense +    │ │ (arq job  │    │  SQLite       │   │  worker     │
   │  BM25→RRF)  │ │  queue)   │    │ (DossierRun,  │   │ (ingestion, │
   │             │ │           │    │  RunEvent)    │   │  arq)       │
   └────────────┘ └───────────┘    └──────────────┘   └─────────────┘
```

One image, three roles (`api` / `worker` / `mcp`), selected by command
override — not three separate Dockerfiles; they share nearly all their
dependencies and a separate image per role would be maintenance overhead
with no real benefit at this scale (`Dockerfile`, `docker-compose.yml`).

## Request paths

- **`POST /dossiers`** — synchronous: `arun_dossier` runs to completion,
  persisted as a `DossierRun` row from creation through outcome, checkpointed
  under `thread_id = run.id` (resumable, though nothing exposes "resume" yet).
- **`GET /dossiers/stream`** — the same graph, streamed node-by-node over SSE
  (`graph.astream_events`), each frame persisted as a `RunEvent` so a stream
  watched live is still replayable afterwards via `GET /dossiers/{id}/events`.
- **`POST /corpus/jobs`** — ingestion on the `arq` worker instead of blocking
  the request; `GET /corpus/jobs/{id}` polls status.
- **MCP tools** (`create_dossier`, `search_corpus`, `ingest_document`,
  `get_evidence`) call the exact same library functions the API does — no
  logic fork, verified by `tests/test_mcp_parity.py`.

## Data & state

| Store | Holds | Dev default | Prod |
|---|---|---|---|
| Qdrant | dense vectors + BM25 index | in-memory (`:memory:`) or self-hosted in k8s | Qdrant Cloud |
| Redis | the `arq` ingestion job queue | self-hosted in k8s / `fakeredis` in tests | Upstash |
| Postgres/SQLite | `DossierRun`, `RunEvent`, Alembic-migrated | SQLite (`./praxis.db` or an in-cluster `/data` file) | Neon Postgres |
| LangGraph checkpoints | per-run graph state, resumable | always its own SQLite file (`PRAXIS_CHECKPOINT_DB_PATH`), regardless of the main DB backend | same |

## Observability

Every graph node emits an OpenTelemetry span (`role`; the six LLM-calling
nodes also get `model`/`prompt_tokens`/`completion_tokens`/`cost_usd`, set
from inside the `StructuredLLM` boundary itself — one instrumentation point,
not six). Gated on `PRAXIS_OTEL_ENABLED`; exports to stdout by default or a
real OTLP/HTTP collector via `PRAXIS_OTEL_ENDPOINT` (no direct LangSmith SDK
integration — no key on the machine this was built on to verify one against;
any OTLP-speaking collector, including a self-hosted LangSmith/Phoenix
ingester, works). `DossierResponse.cost_usd`/`.tokens` report the same
per-run totals.

## Eval framework

`evals/run.py` orchestrates 5 suites (retrieval, memo-structure, LLM-as-judge
F1 vs expert labels, citation-integrity, seeded-error detection) into one
report; all 5 gate CI fully offline (recorded real verdicts, not a mocked
judge — see `evals/README.md`), with a nightly workflow doing a live-model
run for drift detection.

## CI/CD

GitHub Actions workflows live at the **monorepo root**
(`../.github/workflows/`, one level above this project directory) —
GitHub Actions never discovers workflows inside a subdirectory, a real
mistake made and fixed in Phase 5 (see `docs/IMPLEMENTATION_PLAN.md`).
`quality` (lint/type/test × 2 Python versions) + `eval-gate` (5 gates) +
`mcp-contract` + `postgres-contract` (a real `postgres:16` service
container) + `build` (pushes `ghcr.io/hemarani-mysore/praxis`) +
`k8s-validate` (`kustomize build | kubeconform -strict` on both overlays) +
`kind-smoke` (a real ephemeral Kubernetes cluster: full rollout, a live
`/health`/`/ready`/`POST /dossiers` smoke test, and a NetworkPolicy check
proving an unrelated pod is refused while an `api`-labelled one succeeds) all
run on every push; `deploy` (Fly) is gated on `main` + a repo secret that
doesn't exist yet — deploy-ready, not deployed (`PRODUCTION.md`).
