# Praxis — Due-Diligence Research Desk

A production multi-agent system that turns a **subject** — a company, a technology,
a vendor, an acquisition target — into a **cited due-diligence memo**: summary →
overview → bull case → bear case → risks → open questions → recommendation with a
calibrated confidence. Every claim links back to the evidence it rests on.

Built with **LangGraph** + **FastAPI**. Designed to exercise the full production
stack — RAG, an MCP server, multi-agent orchestration, an eval framework wired into
CI, Docker/Kubernetes — in one coherent project.

- [`PLAN.md`](PLAN.md) — design, requirement mapping, architecture, scope
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — per-phase steps,
  acceptance criteria, and the test matrix (the doc to work from)

## Status

| Phase | Scope | State |
|---|---|---|
| **0 — Scaffold** | repo layout, LangGraph skeleton (7 nodes, gap-fill loop), FastAPI + CLI, deterministic `fake` LLM, Docker, CI (lint + type + test) | ✅ **done** |
| **1 — RAG** | ingestion → hybrid retrieval (Qdrant dense + BM25 → RRF → rerank) → grounded citations; corpus API + CLI; deterministic retrieval eval gating CI | ✅ **done** |
| **2 — Multi-agent** | real prompts + multi-section editor + deterministic recommendation; researcher fan-out via `Send`; async graph end to end; run persistence + checkpointing; SSE streaming + ingestion worker | ✅ **done** |
| **3 — MCP** | `praxis-mcp` FastMCP server (4 tools, 3 resources, 2 prompts); researcher falls back to an external `web_search` MCP tool when the corpus has nothing; CI (`quality` × 2 Python versions, `eval-gate`, `mcp-contract`) | ✅ **done** |
| **4 — Eval framework** | golden-dossier dataset (8 subjects, 2 deliberate fails) + LLM-as-judge (F1 vs expert labels, judge sees the sources, not just the memo); citation-integrity injection + NLI checks; seeded-error detection; OTel tracing + cost/token accounting; `evals/run.py` orchestrator; CI `eval-gate` (5 sub-gates) + a nightly full-suite workflow | ✅ **done** |
| **5 — Deploy** | Postgres + Alembic migrations; one shared Docker image pushed to GHCR in CI; Kubernetes manifests validated on a real ephemeral `kind` cluster in CI (full rollout, live smoke test, NetworkPolicy proven both ways); Fly deploy job gated on a secret that doesn't exist yet — deploy-ready, not deployed (`PRODUCTION.md`) | ✅ **done** |

## Quickstart

```bash
cd praxis-due-diligence-desk
uv sync --extra dev          # or: pip install -e ".[dev]"

# End to end with NO API keys and NO network:
#   fake LLM + hashing embedder + in-memory Qdrant
uv run praxis run "Acme Robotics" --ingest evals/datasets/corpus/*.md

# Corpus + search on their own
uv run praxis ingest path/to/*.pdf path/to/*.md
uv run praxis search "what was the revenue"

# API
uv run uvicorn praxis.api.main:app --port 8000
curl -s -XPOST localhost:8000/corpus/documents -H 'content-type: application/json' \
  -d '{"text":"Acme 2025 revenue was $42M.","title":"Acme 10-K"}'
curl -s -XPOST localhost:8000/dossiers.md -H 'content-type: application/json' \
  -d '{"subject":"Acme Robotics"}'

# Every dossier run is persisted — list past runs, fetch one by id
curl -s localhost:8000/dossiers
curl -s localhost:8000/dossiers/<id>

# Watch a run node by node instead of waiting for one response
curl -N "localhost:8000/dossiers/stream?subject=Acme%20Robotics"
curl -s localhost:8000/dossiers/<id>/events   # the same events, replayed

# Ingestion can run on a worker instead of blocking the request (needs Redis —
# docker compose provides one; `uv run arq praxis.worker.WorkerSettings` for a bare run)
curl -s -XPOST localhost:8000/corpus/jobs -d '{"text":"...","title":"..."}'
curl -s localhost:8000/corpus/jobs/<job_id>
```

Real models: set `PRAXIS_LLM_PROVIDER=openai|anthropic|nebius` and
`PRAXIS_EMBEDDING_PROVIDER=openai|nebius` with the matching key in `.env`
(see [`.env.example`](.env.example)). `docker compose up --build` brings up the
API + worker + Redis + a real Qdrant.

```bash
make check           # lint + type + test + eval gates (what CI runs)
make demo            # ingest the fixture corpus, run a grounded dossier
make eval-rag        # retrieval eval report
make eval-structure  # memo-structure eval report
make eval-gate       # every eval suite's gate, fully offline (evals/run.py)
make eval            # every eval suite, live judge, writes a timestamped report
```

## The graph

```
planner ──Send──▶▶ research_one × N ──(fan-in)──┬─▶ bull ──┐
          (≤ PRAXIS_RESEARCH_CONCURRENCY)       └─▶ bear ──┴─▶ red_team ──(gaps? & loop<cap, Send)──▶▶ research_one × M
                                                                        └──(clean / cap hit)──▶ editor ─▶ verifier ─▶ END
```

Every LLM-calling node is `async def` (concurrent branches overlap on the event
loop, not just a thread pool); `arun_dossier` is the entry point the API calls
directly, `run_dossier` a sync wrapper for the CLI. `verifier` does no I/O and
stays sync.

- **planner** — decomposes the request into a plan covering a fixed 6-section rubric
- **research_one** — hybrid-retrieves the corpus for one sub-question, emits one
  `Evidence` whose citation is *grounded* (snapped to a retrieved chunk);
  `dispatch_research` fans one branch out per sub-question via LangGraph `Send`,
  running concurrently up to `PRAXIS_RESEARCH_CONCURRENCY` (default 4); a
  gap-fill pass fans out the same way via `dispatch_gap_fill`
- **bull ∥ bear** — run concurrently; each builds one side of the case, only from evidence, with visible reasoning steps and `evidence_refs` back to the items each point rests on
- **red_team** — audits both cases for unsupported claims / contradictions / coverage gaps; can loop back for gap-fill (bounded by `PRAXIS_MAX_GAP_LOOPS`)
- **editor** — writes one section per rubric item on the cheap model tier, drops anything the red team flagged; the overall recommendation + confidence are then computed deterministically from the evidence balance (`graph/recommend.py`), not left to the model
- **verifier** — deterministic: every citation must resolve to real gathered evidence (placeholders don't count); scores rubric coverage; flags missing / uncited sections and ungrounded memos

## Persistence

Every `POST /dossiers` is recorded as a `DossierRun` (async SQLAlchemy, SQLite
by default — `PRAXIS_DATABASE_URL` empty means `./praxis.db`) from creation
through its outcome, queryable via `GET /dossiers` (newest first) and
`GET /dossiers/{id}`. Each run is also checkpointed — a per-call
`AsyncSqliteSaver` (its own SQLite file, `PRAXIS_CHECKPOINT_DB_PATH`) saves
every superstep under `thread_id = run.id`, so the run's graph state is
technically resumable even though nothing exposes "resume" yet. Real
migrations (Alembic) and Postgres land in Phase 5 with the rest of the deploy
story; for now the schema is created with `Base.metadata.create_all()`.

## Streaming & the ingestion worker

`GET /dossiers/stream?subject=&depth=` runs the same graph but streams
`node_start` / `node_end` (with that node's output) / `complete` /
`error` frames over SSE as they happen (`graph.astream_events`), persisting
each as a `RunEvent` — `GET /dossiers/{id}/events` replays them afterwards.
Ingestion can also run on an **arq** worker instead of the request path:
`POST /corpus/jobs` enqueues, `GET /corpus/jobs/{id}` polls. `docker compose
up` provides Redis + the worker service; a bare local run needs its own Redis
(or `PRAXIS_REDIS_URL` pointed at one) — nothing else in the project needs it.

## Retrieval

```
ingest:   parse (pdf/md/txt/html, page-aware) ─▶ chunk (overlap + metadata)
          ─▶ [optional contextual prefix] ─▶ embed (dense) + index (BM25)

search:   dense (Qdrant)  ─┐
          sparse (BM25)   ─┴▶ RRF fuse ─▶ rerank (lexical | cross-encoder) ─▶ top-k + citations
```

Everything runs offline by default: `HashEmbedder` (deterministic), a pure-Python
`BM25Index`, and Qdrant in `:memory:` mode. Swap in real models via env; the code
paths don't change.

## MCP

`praxis-mcp` exposes the desk to any MCP host (Claude Code, Cursor, ...) — it
calls the same library code as the API and CLI, no logic fork:

- **tools** — `create_dossier(subject, depth)`, `search_corpus(query, k)`,
  `ingest_document(text|path, title)`, `get_evidence(claim, k)` (which
  retrieved chunks support/refute a claim)
- **resources** — `dossier://{id}`, `dossier://{id}/citations`, `corpus://stats`
- **prompts** — `due-diligence-brief`, `investment-memo`

```bash
uv run praxis mcp              # stdio, for a host to spawn as a subprocess
```

`.mcp.json` at the project root points Claude Code / Cursor at
`uv run python -m praxis.mcp` — open this directory in either and the tools
above are available immediately. `PRAXIS_MCP_TRANSPORT=streamable-http` (with
`PRAXIS_MCP_HOST`/`PRAXIS_MCP_PORT`) runs it as a standalone networked service
instead — what the `mcp` Docker Compose service uses.

The researcher can also reach *outward*: `PRAXIS_MCP_SERVERS` (JSON, empty by
default) configures external MCP servers via `MultiServerMCPClient`; when a
question's corpus retrieval comes back empty and a `web_search`-shaped tool is
configured, `research_one` calls it and cites the result by URL instead of the
`no-source` placeholder. Unset (the default) — corpus-only, unchanged from
Phase 2.

## Evals

```
evals/
  datasets/golden_dossiers/  8 subjects (dev=4, test=3, online=1), 2 deliberate fails
  metric.py                  BinaryLLMJudgeMetric — reads (subject, sources, memo)
  memo_eval.py                --split {dev,test,online}: judge-vs-expert F1
  citation_eval.py            fabricated-citation injection self-check + --nli
  redteam_eval.py             seeded false-claim detection, 4 fixtures
  run.py                      --suite {rag,structure,memo,citation,redteam,all} --gate --report
  cassettes/                  real verdicts recorded once, replayed offline for CI
```

The judge is deliberately **not** reference-free with respect to the sources
— it reads the raw source material alongside the memo, because catching
either of the dataset's two deliberate failure modes (a disclosed risk
reduced to vague language and partly omitted; a sparse source inflated with
invented specifics) requires comparing the memo against what its sources
actually said, not just checking the memo's own internal consistency. The
judge prompt was calibrated against a real model until dev/test agreement
with the expert labels reached 4/4 and 3/3 — see
`evals/datasets/golden_dossiers/README.md` for the worked examples.

`--split test` (the CI-gated one) and `redteam_eval.py --gate` never make a
network call: both replay real verdicts recorded once against an actual
model, through the same `FakeStructuredLLM(overrides=...)` scripting the rest
of the test suite already uses — not a new cassette dependency.

Every LLM call also carries OpenTelemetry spans (`role`, and for the six
LLM-calling nodes `model`/`prompt_tokens`/`completion_tokens`/`cost_usd`) —
gated on `PRAXIS_OTEL_ENABLED`, exporting to stdout by default or a real
OTLP/HTTP collector via `PRAXIS_OTEL_ENDPOINT`. `DossierResponse.cost_usd`/
`.tokens` report the same totals for the whole run.

```bash
make eval-gate   # every gate, offline — what CI's eval-gate job runs
make eval-dev    # judge-vs-expert F1, live model, dev split
make eval-online # nightly drift check: fresh generation + live judge
```

## Layout

```
src/praxis/
  config.py schemas.py llm.py render.py cli.py
  graph/  state.py build.py prompts.py context.py recommend.py checkpoint.py  nodes/{planner,researcher,analysts,red_team,editor,verifier}.py
  rag/    parse.py chunk.py contextual.py embed.py bm25.py vector_store.py fuse.py rerank.py corpus.py retrieve.py ingest.py
  api/    main.py deps.py sse.py
  db/     models.py session.py         # DossierRun, RunEvent, async SQLAlchemy engine/session
  worker/ __init__.py                  # arq ingest_task + WorkerSettings
  mcp/    server.py __main__.py         # praxis-mcp — tools/resources/prompts over the same library code
  tools/  mcp_tools.py                  # load_external_tools() — the researcher's web-search fallback
  obs/    __init__.py                 # real OTel setup_tracing() + span()
tests/                                 # 142 tests: graph, fan-out, async, checkpointing, db (incl. a real-
                                        #   Postgres round trip), api runs, sse, jobs, worker, retrieval, cli,
                                        #   schemas, mcp, evals (judge/citation/redteam/obs/cost), golden-dataset
                                        #   loader, /ready's deep-check paths
evals/  rag_eval.py, memo_structure_eval.py, metric.py, memo_eval.py,       # 5 gates (CI) + run.py orchestrator
        citation_eval.py, redteam_eval.py, run.py, datasets/, cassettes/
alembic/                               # migrations — praxis.db.models.Base is the source of truth
deploy/ README.md fly/{api,mcp}.toml   # k8s/{base,overlays/{dev,prod}} — see "## Deploy" below
docs/   IMPLEMENTATION_PLAN.md ARCHITECTURE.md RUNBOOK.md   # + PRODUCTION.md at the project root
.mcp.json                              # Claude Code / Cursor config for praxis-mcp
```

CI lives at the **monorepo root**, one level up from this directory —
`../.github/workflows/ci.yml` (`quality` × {3.11, 3.12}, a 5-gate `eval-gate`,
`mcp-contract`, `postgres-contract`, `build`, `k8s-validate`, `kind-smoke`,
`deploy`) and `../.github/workflows/nightly.yml` — not inside
`praxis-due-diligence-desk/` itself, since GitHub Actions only discovers
workflows at the true repository root (a real mistake, found and fixed in
Phase 5 — see `docs/IMPLEMENTATION_PLAN.md`). Both set
`defaults.run.working-directory: praxis-due-diligence-desk` so every job step
still runs from this project's own directory.

## Deploy

```bash
docker compose -f docker-compose.selfhost.yml up --build   # api+worker+mcp+qdrant+redis+postgres, zero keys
kubectl apply -k deploy/k8s/overlays/dev                    # or overlays/prod, against a real cluster
```

One shared image (`ghcr.io/hemarani-mysore/praxis`, pushed on every push to
this branch) runs all three roles by command override. Kubernetes manifests
(`deploy/k8s/`, kustomize) are validated with `kubeconform -strict` and
applied to a real ephemeral `kind` cluster in CI — full rollout, a live
`/health`/`/ready`/`POST /dossiers` smoke test, and a NetworkPolicy proven
both ways (an unrelated pod refused, an `api`-labelled one succeeds), not
just reviewed YAML. `/ready` does an actual deep check: Qdrant, the database
(`SELECT 1`), and that the LLM client constructs from the configured
credentials.

No live URL — no Fly or managed-service account exists on the machine this
was built on. The CI `deploy` job (gated on `main` + a `FLY_API_TOKEN`
secret) and `PRODUCTION.md`'s step-by-step signup + `fly deploy` runbook are
both ready for whenever that changes. See `docs/ARCHITECTURE.md` for the
full system diagram and `docs/RUNBOOK.md` for day-2 operations (redeploy,
rollback, reindex, scale, rotate keys).

## Provenance

Patterns adapted from `Arindam200/awesome-ai-apps`:
`agentfield_finance_research_agent` (bull/bear/editor committee, visible reasoning),
`maintainer_brief` (FastAPI production skeleton, multi-provider LLM),
`deep_research_writing_agents_nebius_okahu` (eval harness, MCP servers),
`mcp_toolbox_security_agent` (k8s manifests), `advanced_rag_with_reranking` (hybrid RAG + RAGAS).
