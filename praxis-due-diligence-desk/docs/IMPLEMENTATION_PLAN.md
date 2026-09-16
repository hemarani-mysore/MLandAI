# Praxis — Implementation Plan

The execution tracker. [`PLAN.md`](../PLAN.md) is the *why* (design, requirement
mapping, architecture); this is the *how* — per-phase steps, acceptance criteria,
and the tests that prove each phase is done.

**Living document.** Tick the boxes as you go. Every phase updates its own status
here, the status table in [`README.md`](../README.md), and the phase section in
`PLAN.md`, in the same commit that finishes it.

---

## Conventions

### Definition of Done (applies to every phase)

- [ ] `make check` passes → `ruff check` + `ruff format --check` + `mypy src` +
      `pytest` + every deterministic eval gate
- [ ] New code has tests; changed behaviour has changed tests
- [ ] The whole system still runs offline: `PRAXIS_LLM_PROVIDER=fake`,
      `PRAXIS_EMBEDDING_PROVIDER=fake`, in-memory stores — no keys, no network
- [ ] Tests green on Python 3.11 **and** 3.12 (the CI matrix)
- [ ] `PLAN.md` phase section + `README.md` status table + this file updated
- [ ] Committed on `praxis-due-diligence-desk` with the `Co-Authored-By` trailer,
      pushed; CI green on the branch

### Test taxonomy

| Kind | Scope | Rules |
|---|---|---|
| **unit** | one module / pure function | no IO, fake everything, deterministic |
| **integration** | several modules, API via `TestClient`, graph end-to-end | fake provider, in-memory Qdrant / SQLite |
| **contract** | MCP tool ≡ library function; API response schema stability | fake provider |
| **eval** | dataset + metric + threshold | deterministic ones **gate CI**; LLM-scored ones run **nightly** |
| **smoke / manual** | `make demo`, `docker compose up`, the live URL | run by hand, listed per phase |

### CI gate philosophy

- **Per-PR / per-push** — fast, deterministic, offline. Target < 3 minutes.
  Jobs: `quality` (lint + type + test, 3.11 & 3.12) and `eval-gate`
  (deterministic evals only).
- **Nightly** (Phase 4+) — real cheap model, full eval suite across all splits,
  uploads a report artifact. Needs `OPENAI_API_KEY` as a repo secret.

### Model tiers (keep costs sane)

`strong` (planner, analysts) vs `fast` (editor, red_team, judge, contextualiser,
verifier-NLI). Configured by `PRAXIS_FAST_ROLES` in `config.py`.

---

## Phase 0 — Scaffold  ·  ✅ done (`9cb98ce`)

**Goal:** a runnable LangGraph skeleton with the full node topology, deterministic
offline execution, API + CLI, and CI.

### What shipped
- `DossierState` + 7 nodes (`planner → researcher → {bull ∥ bear} → red_team →
  editor → verifier`) with the bounded gap-fill loop (`route_after_red_team`,
  `PRAXIS_MAX_GAP_LOOPS`)
- Typed contracts for every boundary (`src/praxis/schemas.py`)
- `FakeStructuredLLM` — canned structured output + `overrides={role: [...]}`
  scripting for tests
- FastAPI (`/health`, `/ready`, `POST /dossiers`, `POST /dossiers.md`) + `praxis`
  CLI (`run`)
- `verifier` — deterministic citation-resolution + rubric-coverage gate
- Multi-stage non-root `Dockerfile`, `docker-compose.yml`
- GitHub Actions `quality` job (ruff + mypy + pytest, 3.11 / 3.12)

### Acceptance criteria — all met
- [x] `praxis run "X"` prints a memo end to end with no keys
- [x] `POST /dossiers` returns `DossierResponse`
- [x] Gap-fill loop runs then terminates at the cap
- [x] `verifier` flags a memo whose citations don't resolve
- [x] `ruff` + `mypy` clean, CI green

### Tests in place
`tests/test_schemas.py` (3) · `tests/test_graph.py` (happy path, gap-fill loop,
citation flagging) · `tests/test_api.py` · `tests/test_cli.py`

---

## Phase 1 — RAG  ·  ✅ done (`8f1eb76`)

**Goal:** hybrid retrieval over an ingested corpus; the researcher grounds every
piece of evidence in a real source; a deterministic retrieval eval gates CI.

### What shipped
- `src/praxis/rag/` — `parse` (PDF page-aware via pymupdf, md/txt/html) →
  `chunk` (overlap + `{source, page, ordinal, locator}`) → optional `contextual`
  prefix → `embed` + index
- Hybrid retrieval: dense (**Qdrant**, `:memory:` or server) + sparse
  (pure-Python **BM25**) → client-side **RRF** → **rerank** (`LexicalReranker`
  default, `CrossEncoderReranker` via `--extra rerank`) → top-k + citations;
  cached by `(corpus, version, query, k)`
- `researcher` node retrieves per sub-question and **grounds** the `Evidence`
  citation (snaps to a retrieved chunk if the model cites anything else)
- Corpus API (`POST /corpus/documents`, `GET /corpus/stats`, `POST /corpus/search`)
  + CLI (`praxis ingest`, `praxis search`, `praxis run --ingest`)
- Nebius provider wired (chat + embeddings)
- `evals/` — frozen 4-doc fixture corpus, 14 golden questions,
  `rag_eval.py` (recall@k / hit@1 / MRR / answer_hit) with `--gate`; new
  `eval-gate` CI job

### Acceptance criteria — all met
- [x] `search_corpus` returns the right doc first for each fixture question
- [x] researcher evidence cites a real ingested `source_id`; `verifier`
      hallucination rate 0 on the grounded demo
- [x] offline path uses no model downloads / network
- [x] `rag_eval.py --gate` passes; wired into CI

### Tests in place
`test_rag_parse.py` (6) · `test_rag_chunk.py` (3) · `test_rag_bm25.py` (4, incl.
RRF) · `test_rag_corpus.py` (4) · `test_rag_ingest.py` (3) · graph +
`sources` assertions in `test_graph.py` · corpus endpoints in `test_api.py`
· **41 tests total**

### Real-provider path — verified 2026-09-10

The offline path was always tested; the real-model path (`PRAXIS_LLM_PROVIDER=openai`,
`PRAXIS_EMBEDDING_PROVIDER=openai`) had not actually been run before. Running it
surfaced and fixed three gaps:
- `.env` was read by `Settings` for `PRAXIS_*` fields, but never loaded into
  `os.environ` — so `OPENAI_API_KEY` (read directly by LangChain/`os.environ.get`)
  never reached the process. Fixed with `load_dotenv()` in `config.py` (before any
  `os.environ` already set — tests still force `fake`).
- `LangChainEmbedder` didn't pass `dimensions` to `OpenAIEmbeddings`, so
  `text-embedding-3-large` returned 3072-dim vectors against a Qdrant collection
  sized from `PRAXIS_EMBEDDING_DIM` (256) — a silent shape mismatch on upsert.
  Fixed: pass `dimensions=dim` for `text-embedding-3-*` (not for Nebius/ada-002,
  which are fixed-width); `.env.example` now defaults `PRAXIS_EMBEDDING_DIM=1536`.
- The `EDITOR` prompt (slice 1 rewrite) never asked for the `summary` field —
  gpt-4o sometimes left it empty. Fixed with an explicit instruction.
With those fixed, a real `gpt-4o` + `text-embedding-3-large` run on the fixture
corpus produced a coherent, fully-cited 6-section memo, triggered the gap-fill
loop once, and passed verification clean (0% hallucination, 100% coverage).

---

## Phase 2 — Multi-agent for real  ·  ✅ done — 5/5 slices  (~1 week)

**Goal:** real prompts + few-shots, parallel researcher fan-out, a multi-section
editor, run persistence + checkpointing, SSE streaming, ingestion worker.

**Shipped as 5 slices, each its own commit + green CI:**
1. ✅ **Real analytical memo** — prompts, `evidence_refs`, multi-section editor,
   deterministic recommendation, memo-structure eval gate.
2. ✅ **Researcher fan-out** — `research_one` dispatched via LangGraph `Send`,
   one branch per sub-question, concurrent up to `PRAXIS_RESEARCH_CONCURRENCY`.
3. ✅ **Async graph** — every LLM-calling node is `async def`; `arun_dossier` is
   the primary entry point, `run_dossier` a thin sync wrapper; the API calls
   `arun_dossier` directly (async end to end, no worker-thread-per-request).
4. ✅ **Run persistence + checkpointing** — every `POST /dossiers` is a
   `DossierRun` row (SQLite via async SQLAlchemy), queryable via
   `GET /dossiers[/{id}]`; each run is also checkpointed (`AsyncSqliteSaver`,
   its own SQLite file) under `thread_id = run.id`. Alembic + Postgres +
   `RunEvent` deferred — see the step-6 note below.
5. ✅ **SSE streaming + ingestion worker** — `GET /dossiers/stream` emits
   `node_start`/`node_end`/`complete`/`error` frames from `astream_events`,
   persisting each as a `RunEvent` (`GET /dossiers/{id}/events` replays them);
   `arq` + `fakeredis`-tested ingestion worker (`POST /corpus/jobs` /
   `GET /corpus/jobs/{id}`). Also retrofitted slice 4's checkpointer to
   allow-list our own schema types — see the step-8 note below.

### Steps

1. **Prompts & few-shots** — `src/praxis/graph/prompts.py`  ✅ *(slice 1)*
   - Rewrite each role's system prompt: explicit output contract, the 6-section
     rubric, "every claim cites an evidence id", confidence-calibration guidance
     (what 0.3 vs 0.8 means), "a weak case is a valid output".  ✅
   - `src/praxis/graph/fewshots/` — 1–2 worked examples per role (planner,
     analyst, red_team, editor) as `.md`/`.json`, injected into the prompt.
     **Deferred to Phase 4** — few-shots are only worth carrying once the eval
     framework can measure whether they help and guard their subjects.
   - **New in slice 1:** `graph/recommend.py` — the overall recommendation +
     confidence are computed deterministically from the evidence balance and
     red-team severity (the editor writes prose only), so the verdict is
     calibrated identically for every provider and is unit-testable.
   - **New in slice 1:** `PLACEHOLDER_SOURCE_IDS` in `schemas.py`; the verifier
     now treats placeholder-only citations as *ungrounded* (not hallucinated)
     and flags per-missing-section instead of one blanket low-coverage flag.

2. **Researcher fan-out (`Send`)** — `src/praxis/graph/`  ✅ *(slice 2)*
   - Split `researcher` into `dispatch_research` (returns
     `[Send("research_one", {subject, section, question}) for ...]`) and
     `research_one` (one question → one `Evidence`, appended via the `evidence`
     reducer). `research_one` is registered with its own `input_schema`
     (`ResearchTask`) — its input is that small dict, not the full `DossierState`.
   - Rewired `build.py`: `planner --Send-->> research_one` (via
     `add_conditional_edges("planner", dispatch_research, ["research_one"])`);
     `research_one →` fan-in → `bull` / `bear`. Gap-fill: `route_after_red_team`
     now returns `dispatch_gap_fill(state)` (a `Send` list) instead of the
     literal `"researcher"` when there are gaps, else `"editor"`.
   - `PRAXIS_RESEARCH_CONCURRENCY` (default 4) — set as the run's top-level
     `max_concurrency` (a native `RunnableConfig` key; LangGraph sizes the
     `ThreadPoolExecutor` from it, capping *all* concurrent steps in the run,
     not just this fan-out — acceptable since bull/bear is only ever 2 branches).
   - mypy note: `add_node(..., input_schema=ResearchTask)` for a `Send`-fanned
     node with a distinct per-branch schema hits a known overload-resolution gap
     in the installed langgraph/mypy combo (verified correct at runtime via a
     standalone probe against the official map-reduce pattern); silenced with a
     scoped `# type: ignore[arg-type,call-overload]` + comment at the call site.

3. **`evidence_refs`** — `src/praxis/graph/nodes/analysts.py`
   - Analysts return `Finding.evidence_refs` (indices into the evidence list);
     post-parse, drop out-of-range refs.
   - Editor maps `evidence_refs` → the citations it attaches per section.

4. **Multi-section editor** — `src/praxis/graph/nodes/editor.py`
   - Emit one `MemoSection` per rubric section, each citing the evidence for that
     section. This removes the `low rubric coverage` verifier flag on real runs.
   - Derive `recommendation` from the balance of bull/bear + red-team severity;
     `insufficient_evidence` when evidence is thin or all `no-source`.

5. **Async graph** — `src/praxis/llm.py`, nodes, `build.py`  ✅ *(slice 3)*
   - `StructuredLLM` gained `agenerate`/`acomplete`; `LangChainStructuredLLM`
     uses `.ainvoke()`; `FakeStructuredLLM`'s async methods delegate to the
     sync ones (no real I/O to await).
   - Every LLM-calling node is now `async def` (`planner`, `research_one`,
     `bull`, `bear`, `red_team`, `editor`); `verifier` does no I/O and stays
     sync — LangGraph runs sync nodes fine inside an async invocation.
     `research_one`'s `search_corpus` call is sync (a real embedder makes a
     blocking HTTP call inside it) — wrapped in `asyncio.to_thread` so it
     doesn't stall the event loop for other concurrent branches.
   - `arun_dossier` (async, the primary entry point, used by the API and by
     `evals`/CLI internals that already run inside a loop) + `run_dossier`, a
     thin sync wrapper (`asyncio.run(arun_dossier(...))`) for the CLI and other
     non-async callers — **must not** be called from inside an already-running
     event loop.
   - `api/main.py`'s `POST /dossiers` and `/dossiers.md` are `async def` and
     call `arun_dossier` directly (never the sync wrapper) — async end to end,
     so a slow dossier run no longer ties up one of FastAPI's worker threads.

6. **Run persistence + checkpointing** — `src/praxis/db/`  ✅ *(slice 4, scope
   narrowed — see below)*
   - Deps actually used: `sqlalchemy[asyncio]>=2` (bare `sqlalchemy` does **not**
     pull `greenlet`, which the async engine needs at runtime — confirmed by
     probe, not assumed), `aiosqlite>=0.20`, `langgraph-checkpoint-sqlite>=2`.
   - Model shipped: `DossierRun` (id uuid4, subject, depth, status
     `running|succeeded|failed`, created/finished, `memo` JSON, `verification`
     JSON, `sources` JSON, `error`). `cost_usd`/`tokens` wait for Phase 4's
     token accounting; `RunEvent` waits for slice 5 (nothing populates it
     until `astream_events` exists).
   - `PRAXIS_DATABASE_URL` empty → `sqlite+aiosqlite:///./praxis.db`.
     **Deferred to Phase 5** (alongside real Postgres): `PostgresSaver`, and
     Alembic (`alembic init` + migrations) — one table with no migration
     history yet doesn't earn that complexity; schema is created via
     `Base.metadata.create_all()` (`ensure_schema()`, idempotent, guarded to
     run once). Real migrations land when there's an actual deploy target.
   - Checkpointer: `AsyncSqliteSaver`, opened per dossier run from its own
     SQLite file (`PRAXIS_CHECKPOINT_DB_PATH`, separate from the app's own
     tables — two libraries writing to one SQLite file was judged not worth
     the risk for the file-locking upside of merging them); `thread_id =
     run.id`, so a run's checkpoint history and its DB row share an id.
   - Engine/sessionmaker are a plain `@lru_cache` singleton (mirrors
     `rag/corpus.py`'s `get_corpus()`), **not** FastAPI `lifespan`/`app.state`
     — confirmed a bare `TestClient(app)` (the existing test fixture's
     pattern) does not run `lifespan` startup, so anything living only on
     `app.state` would be unset in every test. `db_session_dep` is overridden
     in tests exactly like the existing `corpus_dep`, via an in-memory
     `StaticPool` engine (plain `:memory:` gives each pooled connection its
     own empty DB — confirmed by probe).
   - API: `GET /dossiers` (list, newest first, `limit` query param),
     `GET /dossiers/{id}` (one run, 404 if missing); `POST /dossiers` and
     `/dossiers.md` now persist via a shared `_create_and_persist` helper.
   - Found via a real-provider run (not just review): `evals/memo_structure_eval.py`
     didn't pin `llm=FakeStructuredLLM()` on its `run_dossier` call, so running
     it standalone (outside pytest, where conftest forces `fake`) silently used
     whatever `PRAXIS_LLM_PROVIDER` `.env` set — real API calls from a
     "deterministic, no-LLM" CI gate. Fixed: pinned explicitly.

7. **Ingestion worker** — `src/praxis/worker/`  ✅ *(slice 5)*
   - Deps: `arq` (runtime), `fakeredis` (dev only — no real Redis on this
     machine, or in CI). `ingest_task(ctx, *, text, path, title)` wraps
     `ingest_source` via `asyncio.to_thread` (a real embedder blocks on HTTP).
   - `POST /corpus/jobs` enqueues → `{job_id}`; `GET /corpus/jobs/{id}` → status
     (`arq.jobs.Job`, 404 on `not_found`). Kept sync `POST /corpus/documents`
     for small inline docs.
   - `docker-compose.yml`: added `redis:7-alpine` + a `worker` service
     (`replicas: 1` — the queue isn't sharded). **Not runnable here** (no
     Docker on this machine, same limitation as every earlier Docker-touching
     slice) — reviewed for correctness, not executed. Comment in the file
     flags the real gotcha: api and worker only share a corpus when both point
     at the same **real** `PRAXIS_QDRANT_URL` — with the in-memory default
     they're two separate processes with two separate empty corpora.
   - `arq` + `fakeredis` in-process has two real gotchas, both worked around
     (see `tests/conftest.py::redis_pool` and `tests/test_worker.py`):
     `FakeRedis(client_class=ArqRedis)` breaks fakeredis's kwarg introspection
     (fix: build a plain `FakeRedis`, hand `ArqRedis` its `connection_pool`);
     `Worker.main()` unconditionally calls `arq.worker.log_redis_info()` →
     `INFO`, which this fakeredis version doesn't implement (fix: patch it out
     for the burst-worker test runs only — a real deployment never hits this).

8. **SSE streaming** — `src/praxis/api/sse.py`  ✅ *(slice 5)*
   - `GET /dossiers/stream?subject=&depth=` → `text/event-stream` from
     `graph.astream_events(version="v2")`, filtered to the 7 real node names
     (LangGraph also wraps the conditional-edge router functions in
     `on_chain_start`/`on_chain_end`, attributed to the *calling* node —
     filtering on `event["name"]`, not `metadata.langgraph_node`, avoids
     double-counting).
   - Events: `node_start {node}`, `node_end {node, partial}` (partial = the
     node's output dict, `BaseModel`/`list[BaseModel]` values serialized via
     `model_dump(mode="json")`), `complete {DossierResponse}`, `error
     {message}`. Every frame is persisted as a `RunEvent`
     (`GET /dossiers/{id}/events` replays them, ordered by `seq` — a small
     addition beyond the acceptance criteria, since otherwise `RunEvent` would
     be write-only). The final state comes from `graph.aget_state(...)` after
     the checkpointed run reaches `END`, not from re-merging partials by hand.
   - **Retrofit, found while building this:** checkpointing any `DossierState`
     was logging `Deserializing unregistered type praxis.schemas.ResearchPlan
     from checkpoint. This will be blocked in a future version` —
     `JsonPlusSerializer` only allow-lists a fixed stdlib/langchain set by
     default. New `graph/checkpoint.py::open_checkpointer()` configures a
     `JsonPlusSerializer(allowed_msgpack_modules=[...])` with our schema
     classes and is now the single way a checkpointer gets opened — slice 4's
     `POST /dossiers` path switched to it too. Verified with a `caplog` test
     that fails without the fix (confirmed by temporarily reverting it).
   - **Known real-provider risk, not fixed here:** `astream_events` appears to
     force the underlying chat model onto its streaming code path even though
     nodes only ever call `.ainvoke()`. One real run hit langchain-openai's
     `stream_chunk_timeout` after ~16 minutes on a stalled OpenAI response
     (dead TCP-layer connection, no data) — the endpoint degraded correctly
     (an `error` frame, the run marked `failed`, no server crash or hang for
     other requests) but a single stuck node can take a long time to
     self-resolve. Tuning `stream_chunk_timeout` / a request-level timeout is
     future hardening (Phase 4/5), not attempted in this slice.

### Acceptance criteria

- [x] `praxis run` (fake provider, fixture corpus) yields a memo with **all 6**
      rubric sections, each with ≥ 1 resolvable citation  *(slice 1)*
- [x] `verifier` returns **no** `low rubric coverage` flag on `make demo`  *(slice 1)*
- [x] a no-corpus run is honestly flagged: `insufficient_evidence` +
      "no grounded sources", hallucination rate 0  *(slice 1)*
- [x] `eval-gate` still green; new `evals/memo_structure_eval.py` added to the
      gate (sections == rubric, every section cited, recommendation ∈ enum,
      `--self-check` proves it catches a broken memo)  *(slice 1)*
- [x] Researcher branches run concurrently — a test asserts observed parallelism
      ≤ `PRAXIS_RESEARCH_CONCURRENCY` and > 1  *(slice 2, `test_graph_fanout.py`)*
- [x] Gap-fill loop still bounded; `iterations` reported correctly  *(slice 2)*
- [x] `POST /dossiers` → `GET /dossiers/{id}` returns the persisted run;
      `GET /dossiers` lists it  *(slice 4, `test_api_runs.py`)*
- [x] a checkpointer actually writes checkpoints under the run's thread_id,
      not just accepted silently  *(slice 4, `test_graph_checkpointing.py`)*
- [x] CI green using SQLite  *(slice 4 — fakeredis is slice 5's worker, not this)*
- [x] `GET /dossiers/stream` emits ≥ 7 node events and ends with `complete`
      *(slice 5, `test_api_sse.py`)*
- [x] `POST /corpus/jobs` → poll → `succeeded`, `GET /corpus/stats` shows growth
      *(slice 5, `test_api_jobs.py`)*
- [x] CI green using fakeredis  *(slice 5, `test_worker.py` + `test_api_jobs.py`)*
- [ ] `docker compose up` brings up api + worker + redis + qdrant, all healthy —
      **not verified**: no Docker on this machine. `docker-compose.yml` is
      written and reviewed but has never actually been run, here or in CI.

### Tests to perform

| Kind | File | Checks |
|---|---|---|
| unit | `test_graph_fanout.py` | `dispatch_research` emits N `Send`s; `evidence` accumulates across branches; concurrency cap respected |
| unit | `test_editor_multisection.py` | 6 `MemoSection`s; citations resolve to `evidence_refs`; recommendation logic (thin evidence → `insufficient_evidence`) |
| unit | `test_db_models.py` ✅ | `DossierRun` round-trips on SQLite incl. JSON columns; defaults; ordering. `RunEvent` covered via `test_api_sse.py` instead (nothing writes one outside a stream) |
| unit | `test_graph_checkpointing.py` ✅ *(slice 4, extended in slice 5)* | a checkpointed run's state is retrievable by `thread_id`; distinct threads don't share state; opt-in (no checkpointer → no thread_id needed); `open_checkpointer` allow-lists our schema types (no "unregistered type" log warning — `caplog`) |
| unit | `test_worker.py` ✅ | `ingest_task` grows the corpus and returns the `IngestResult`; a task with no input fails cleanly (arq burst `Worker` + fakeredis) |
| integration | `test_api_runs.py` ✅ | `POST /dossiers` persists; `GET /dossiers/{id}` + list, newest-first, `limit`, unknown id → 404 |
| integration | `test_api_sse.py` ✅ | stream yields ≥ 7 `node_start`/`node_end` frames ending in `complete`; the run + its events are persisted and match; a short `subject` is rejected (`TestClient` SSE) |
| integration | `test_api_jobs.py` ✅ | enqueue → poll (`deferred`/`queued`) → drain with an inline burst worker → poll (`complete`) → `/corpus/stats` grew; missing input → 422; unknown job → 404. Runs as `httpx.AsyncClient` over the ASGI app directly, not the shared sync `client` — draining fakeredis with an inline burst worker needs to share the *same* event loop as the HTTP calls |
| regression | existing 41 | ✅ all pass unchanged (82 total as of slice 5) |
| eval | `make eval-rag-gate` + `make eval-structure-gate` | ✅ retrieval unchanged; memo structure gate passes, fails on a corrupted canned memo |
| smoke | `make demo` | ✅ full 6-section grounded memo, verifier PASS |
| smoke | `docker compose up` then `curl -N localhost:8000/dossiers/stream?subject=Acme%20Robotics` | **not run** — no Docker here. The equivalent smoke test (`uv run uvicorn` + `curl -N`) was run manually and worked: real node events streaming live, persisted run + events matching |

---

## Phase 3 — MCP  ·  ✅ done

**Goal:** ship `praxis-mcp` (the desk as MCP tools/resources/prompts) and let the
researcher consume *external* MCP tools (web search, GitHub).

### What shipped
- `src/praxis/mcp/server.py` — one `FastMCP("praxis")` instance calling existing
  library code (no logic fork): tools `create_dossier`, `search_corpus`,
  `ingest_document`, `get_evidence` (retrieves top-k chunks, asks the LLM which
  *indices* support/refute a claim, then maps indices back to real `Citation`s
  — the model never invents citation fields); resources `dossier://{id}`,
  `dossier://{id}/citations`, `corpus://stats`; prompts `due-diligence-brief`,
  `investment-memo`. Transport, host, and port are settings-driven
  (`PRAXIS_MCP_TRANSPORT`/`_HOST`/`_PORT`) — stdio by default (any host spawns
  it as a subprocess), `streamable-http` for the Docker Compose service.
- `src/praxis/tools/mcp_tools.py` — `load_external_tools()`: `PRAXIS_MCP_SERVERS`
  (JSON, empty by default) → `MultiServerMCPClient` → `dict[name, BaseTool]`.
  `research_one` falls back to a `web_search`-shaped tool (if configured) when
  the corpus has nothing for a question, before the existing `no-source`
  placeholder; unset stays byte-identical to Phase 2.
- `.mcp.json` at the project root; `praxis mcp` CLI subcommand.
- `mcp` service in `docker-compose.yml`, reusing the existing root `Dockerfile`
  (it already ships `mcp`/`langchain-mcp-adapters` as core deps) with
  `command: ["praxis", "mcp"]` and `PRAXIS_MCP_TRANSPORT=streamable-http` —
  the same one-Dockerfile-three-commands pattern `api`/`worker` already use,
  not a separate `docker/mcp.Dockerfile`.
- `.github/workflows/ci.yml` — this file didn't actually exist before Phase 3
  despite Phase 0/1 above marking a `quality`/`eval-gate` workflow as shipped;
  building it now closes that gap as well as adding this phase's own
  `mcp-contract` job (`quality` × {3.11, 3.12}, `eval-gate`, `mcp-contract`).

### Scope cuts (and why)
- **No GitHub MCP consumption.** Only the plan's concretely-specified
  `web_search` fallback is built; "GitHub" appeared once in `PLAN.md`'s prose
  with no further spec, so it's out of scope here, not silently dropped.
- **External MCP consumption tested against a stub server only**
  (`tests/fixtures/stub_web_search_server.py`) — no real Exa/GitHub key exists
  on this machine. `exa-py` was not added as a dependency for the same reason.
- **`docker compose up` for the new `mcp` service is reviewed, not run** — no
  Docker on this machine, the same limitation as every earlier deploy-touching
  slice. The `streamable-http` transport itself *was* verified directly
  (started locally on a scratch port, probed with a real MCP `initialize`
  call over HTTP) — only the Compose wiring around it is untested.
- **API drift, checked rather than assumed:** an isolated dependency probe of
  `mcp>=1.2` in a scratch environment resolved to `mcp==2.2.0`, which renames
  `FastMCP` to `MCPServer` — this looked like a breaking change to plan around.
  Installing into the *actual* project (`uv sync --extra dev`) resolves
  `mcp==1.30.0` instead (constrained by `langchain-mcp-adapters`), which still
  has `FastMCP` — confirmed directly against the project's own venv before
  writing any server code. The server uses `FastMCP`, not `MCPServer`.

### Acceptance criteria
- [x] `python -m praxis.mcp` starts; a contract test enumerates **4 tools, 3
      resources, 2 prompts** with correct schemas
- [x] `create_dossier` over MCP ≡ `run_dossier` for the same input (parity
      test, fake provider; re-verified manually with a real GPT-4o run through
      the MCP tool — correct 6-section memo, and the verifier correctly
      flagged the empty-corpus/no-external-tools run as ungrounded)
- [x] `.mcp.json` present at the project root, pointing at
      `uv run python -m praxis.mcp` — loading it in an actual Claude Code
      session and asking it to create a dossier is a manual step for the user,
      not something this environment can drive itself
- [x] `PRAXIS_MCP_SERVERS` unset → dossier output unchanged from Phase 2
      (`test_researcher_web_fallback.py::test_unconfigured_stays_on_the_no_source_path`)
- [x] With a stub MCP server configured, `research_one` calls its `web_search`
      and produces an `Evidence` with a URL citation
- [x] `mcp` service added to `docker-compose.yml`, same image as `api`/`worker`
      (build reviewed for correctness, not executed — see scope cuts)

### Tests in place
`tests/test_mcp_server.py` (7 — real stdio subprocess round trip: tool/resource/
prompt listing, `ingest_document`→`search_corpus`→`get_evidence`, empty-corpus
`get_evidence`, `create_dossier`, `corpus://stats`, unknown-resource error,
prompt rendering) · `tests/test_mcp_parity.py` (1) · `tests/test_mcp_tools_loader.py`
(2 — unset → `{}`; configured against the stub → a real, invocable `web_search`
tool) · `tests/test_researcher_web_fallback.py` (2) · `tests/test_cli.py`
(+1 — `praxis mcp` dispatches to the server entrypoint)

---

## Phase 4 — Eval framework  ·  ✅ done  ·  *the portfolio centrepiece*

**Goal:** golden dossiers with expert labels; LLM-as-judge scored by **F1** vs
those labels; deterministic citation-integrity; seeded-error detection; OTel
tracing + cost accounting; an expanded CI eval gate + a nightly full run.

### What shipped (5 slices, each committed/pushed/verified independently)

1. **`evals/datasets/golden_dossiers/`** — 8 synthetic subjects (`dev`=4,
   `test`=3, `online`=1), each with `sources/*.md` and, for `dev`/`test`, a
   frozen `memo.md` the judge grades; `index.yaml` carries the expert
   `label`/`critique`; `loader.py` (`load_index()`) validates the set.
   Two deliberate fails, each engineered around a specific, checkable flaw:
   `driftwood-materials` (a disclosed 38%-of-revenue customer loss + an FTC
   inquiry, reduced in the memo to vague "revenue volatility" with the
   inquiry omitted entirely) and `ferrovia-rail-tech` (one sparse,
   pre-revenue paragraph; the memo invents a specific market-size figure and
   "promising traction" language the source never gave it).
2. **`evals/metric.py`** (`BinaryLLMJudgeMetric`) + **`evals/memo_eval.py`**
   — judge reads `(subject, source material, memo text)` — **not**
   reference-free in the sense of ignoring the sources, since both deliberate
   fails above are only catchable by comparing the memo against what its
   sources actually say. `--split test` is always offline (real verdicts
   recorded once, replayed via `FakeStructuredLLM(overrides=...)`);
   `dev`/`online` call the live judge. F1 uses `"fail"` as the positive
   class. The first prompt draft disagreed with the expert label on several
   items — every disagreement was the judge penalizing memo *tone* rather
   than checking whether the sources' facts were present — rewritten twice
   against the real model to ground verdicts in concrete, checkable facts;
   final state 4/4 dev + 3/3 test agreement.
3. **`evals/citation_eval.py`** — a fabricated-citation injection self-check
   (a real gap: no existing test planted a genuine, non-placeholder
   hallucination) + `--nli` (opt-in, nightly, never gated): one cheap LLM
   call per real `Evidence` item decides claim↔quote entailment.
   **`evals/redteam_eval.py`** — 4 inline fixtures, each planting one
   false/contradictory claim in a bull/bear `Finding`; `--gate` replays
   real, recorded `red_team` verdicts (all 4/4 detected on the first real
   recording pass — no calibration needed, `RED_TEAM` was already
   battle-tested from Phase 2).
4. **Cost/token accounting + OTel** — `StructuredLLM` gains
   `total_prompt_tokens`/`total_completion_tokens`/`total_cost_usd`,
   accumulated per-instance (one instance per run); `LangChainStructuredLLM`
   uses `with_structured_output(schema, include_raw=True)` to reach real
   `usage_metadata`, prices it against a small static
   `PRICING_USD_PER_1M` table, and sets `model`/token/`cost_usd` on the
   *current* OTel span from inside the LLM boundary — one instrumentation
   point, not six. `src/praxis/obs/` is a real `setup_tracing()` +
   `span()` now (was a no-op stub); every node wrapped;
   `DossierResponse.cost_usd`/`.tokens` added. Verified against a real model:
   17 console spans across a gap-fill run, correct attributes throughout,
   sane totals (~$0.07, ~24K tokens).
5. **`evals/run.py`** (`--suite {rag,structure,memo,citation,redteam,all}
   --split ... --gate --report ...`) — loads each suite once (never shells
   out, so a live-model suite is never accidentally run twice), aggregates
   one JSON report. CI `eval-gate` expanded to all 5 gates.
   `.github/workflows/nightly.yml` (`schedule` + `workflow_dispatch`; skips
   cleanly with a clear log line if `OPENAI_API_KEY` isn't set as a repo
   secret yet, rather than failing nightly on missing creds). Makefile:
   `eval-dev`, `eval-test(-gate)`, `eval-citation(-gate)`,
   `eval-redteam(-gate)`, `eval-online`, `eval`, `eval-gate`, `eval-report`.

### Deviations from this doc's original wording (confirmed with the user before building)

- **8 subjects, not 8–15** — meets the documented "≥8" bar without ~2x the
  authoring cost; see `evals/datasets/golden_dossiers/README.md`.
- **No `--reference`/`expected.md` reference-memo mode** — the expert `label`
  is the ground truth the judge is scored against; a second hand-authored
  gold memo per item isn't needed for F1. Documented as a not-built
  extension point, not a silent gap.
- **Cost accounting lives on the `StructuredLLM` instance, not threaded
  through `DossierState` via a reducer** — simpler (touches `llm.py` +
  response-building only, not every node's return signature), and verified
  safe under concurrent `Send` branches (the increment happens synchronously
  right after an `await`, before any other coroutine can interleave).
- **OTel exports to console (+ optional OTLP), no direct LangSmith SDK
  integration** — no LangSmith key on this machine to build and verify one
  against; any real OTLP/HTTP collector (including a self-hosted
  LangSmith/Phoenix OTLP ingester) works via `PRAXIS_OTEL_ENDPOINT`.
- **CI "recorded responses" = real verdicts via `FakeStructuredLLM.overrides`,
  not VCR/HTTP cassettes** — no new dependency, reuses this codebase's
  existing scripting mechanism exactly.
- **No `evals/online_eval.py` as a separate file** — folded into
  `memo_eval.py --split online` (same judge, same report shape, one less
  file to keep in sync with the metric it uses).
- **`citation_eval.py` doesn't re-check "zero hallucinated citations on a
  normal run"** — `memo_structure_eval.py --gate` already covers that;
  `citation_eval.py` owns the fabrication-injection proof and the NLI check
  instead, neither of which existed anywhere before.

### Acceptance criteria

- [x] `index.yaml` has 8 subjects across `dev` / `test` / `online`, each with
      `label` + `critique`
- [x] `make eval-dev` prints judge-vs-expert **F1** on the dev split (1.0 on
      the current dataset + prompt)
- [x] `make eval-test-gate` passes on the recorded slice — `--self-check`-style
      proof it *would* fail lives in `test_citation_eval.py` /
      `test_redteam_eval.py`'s injection tests, and `memo_eval.py`'s own two
      deliberate dataset fails are what the judge is scored against
- [x] `citation_eval` catches an injected bad citation
- [x] `redteam_eval` reports a detection rate; all 4 seeded fixtures caught
- [x] With `PRAXIS_OTEL_ENABLED=true`, every node emits a span with the
      expected attributes (asserted via an in-memory span exporter)
- [x] `DossierResponse` carries `cost_usd` and `tokens`
- [x] CI `eval-gate` runs all 5 sub-gates offline (rag, structure, memo-test,
      citation, redteam)
- [x] `nightly.yml` exists (manual-dispatch-safe until the secret is added)
      and produces a report artifact

### Tests in place

`tests/test_golden_dataset_loader.py` (9) · `tests/test_memo_eval.py` (6) ·
`tests/test_judge_calibration.py` (2) · `tests/test_citation_eval.py` (3) ·
`tests/test_redteam_eval.py` (5) · `tests/test_obs.py` (5) ·
`tests/test_cost_accounting.py` (7) · `tests/test_eval_run.py` (4) — 41 new
tests this phase (95 → 136).

---

## Phase 5 — Deploy  ·  ⬜  (~1 week)

**Goal:** per-service images, Kubernetes manifests validated on an ephemeral
`kind` cluster in CI, a live URL on Fly, CI/CD to GHCR + Fly.

### Steps

1. **Images** — `docker/{api,worker,mcp}.Dockerfile`
   - Shared base stage; hardened: non-root uid, `readOnlyRootFilesystem`-friendly,
     `--no-cache`, slim or distroless.
   - `.dockerignore` per build context.

2. **Compose** — `docker-compose.yml` (dev) + `docker-compose.selfhost.yml`
   (one command: api + worker + mcp + qdrant + redis + postgres, healthchecks,
   `PRAXIS_LLM_PROVIDER=fake` default so it works with zero keys).

3. **`deploy/k8s/base/`** (kustomize)
   - `namespace`, `configmap`, `secret.example.yaml`
   - `api-{deployment,service,hpa,pdb}.yaml` — HPA CPU 70%, 2→10
   - `worker-deployment.yaml` — `replicas: 1` (comment: the queue isn't sharded yet)
   - `mcp-{deployment,service}.yaml`
   - `qdrant-statefulset.yaml` + PVC (or external Qdrant Cloud via secret)
   - `ingress.yaml` (nginx) — api + mcp hosts
   - `networkpolicy.yaml` — only `api` / `worker` → datastores
   - `serviceaccount.yaml`, `kustomization.yaml`

4. **`deploy/k8s/overlays/{dev,prod}/`**
   - `dev` — kind-friendly: NodePort, in-cluster qdrant, `fake` provider
   - `prod` — real ingress host, external secrets, real provider

5. **CI additions**
   - `build` job — `docker buildx` each image → `ghcr.io/hemarani-mysore/praxis-{api,worker,mcp}:<sha>` (+ `:latest` on `main`); needs `permissions: packages: write`
   - `k8s-validate` job — `kubeconform -strict` on `kustomize build overlays/{dev,prod}`
   - `kind-smoke` job — `helm/kind-action` → `kubectl apply -k overlays/dev` →
     `kubectl rollout status` all deployments → port-forward →
     `curl /health` + `/ready` + one `POST /dossiers` → teardown
   - `deploy` job (`main` only, `environment: production`) — `flyctl deploy` for
     `api` + `mcp` from `deploy/fly/{api,mcp}.toml`

6. **Fly / managed services** (manual, documented in `PRODUCTION.md`)
   - `fly launch` api + mcp; `fly secrets set` `OPENAI_API_KEY`,
     `PRAXIS_QDRANT_URL` (Qdrant Cloud free), `PRAXIS_DATABASE_URL` (Neon free),
     `REDIS_URL` (Upstash free), `LANGSMITH_API_KEY`
   - `/ready` deep check: Qdrant + Postgres reachable + one tiny LLM ping
     (skipped when `fake`)

7. **Docs** — expand `deploy/README.md`; add `PRODUCTION.md` (Fly + self-host
   runbook), `docs/ARCHITECTURE.md`, `docs/RUNBOOK.md` (redeploy, rollback,
   reindex corpus, scale, rotate keys).

### Acceptance criteria

- [ ] `docker compose -f docker-compose.selfhost.yml up` → every service healthy;
      `curl localhost:8000/dossiers.md` works with **no keys**
- [ ] `kubeconform -strict` passes on `kustomize build overlays/{dev,prod}`
- [ ] CI `kind-smoke`: all deployments reach `Ready`; `/health` + `/ready` 200; a
      dossier `POST` returns 200
- [ ] `ghcr.io/hemarani-mysore/praxis-api:<sha>` pushed on every `main` build
- [ ] Live: `https://praxis-api.fly.dev/health` → `{"ok": true}`; `/ready` green;
      a real dossier request succeeds
- [ ] `NetworkPolicy` verified — a debug pod outside `api`/`worker` cannot reach
      Qdrant (tested in kind)
- [ ] Rollback tested once (`flyctl releases` → redeploy previous image)
- [ ] `kubectl get hpa` shows the api HPA with targets

### Tests to perform

| Kind | What | How |
|---|---|---|
| image sanity | `docker run --rm praxis-api python -c "import praxis"` | in `build` job |
| image sanity | `docker run praxis-api` then `curl /health` | ephemeral container in CI |
| manifest | `kubeconform -strict`, `kustomize build`, (`helm lint`) | `k8s-validate` job |
| smoke | rollout + `/health` + `/ready` + `POST /dossiers` | `kind-smoke` job |
| netpol | debug pod → `nc qdrant 6333` should **fail**; from `api` should succeed | `kind-smoke` job step |
| security | `trivy image` / `docker scout` (warn, non-gating); `verify.sh` proves non-root + ro-fs | CI + manual |
| manual | hit the live URL: `make demo` against it, watch Fly logs, run a load loop and watch `kubectl get hpa` scale | by hand post-deploy |

---

## Phase 6 — Polish  ·  ⬜  (~3 days)

**Goal:** a frontend, a one-command `verify.sh`, docs, a demo, measured numbers.

### Steps

1. **`frontend/`** (Next.js + Tailwind) — subject input; live SSE run view (node
   cards + reasoning steps); memo render; **citation viewer** (click a citation →
   the source chunk / PDF page). Deploy on Vercel or as a Fly static app.
2. **`verify.sh`** — one command: `make check` + every eval gate +
   citation-integrity proof + (if `PRAXIS_API_URL` set) a live smoke. Green/red
   summary. Runs as a final CI job on `main`.
3. **`docs/ARCHITECTURE.md`** (real diagram), **`docs/RUNBOOK.md`**.
4. **Demo** — 2–3 min screen recording (`docs/DEMO.md` script); blog / dev.to
   draft (`docs/WRITEUP.md`).
5. **`docs/PERFORMANCE.md`** — $/dossier, p50/p95 latency, tokens per node, from a
   real batch, pulled from OTel.
6. Finalise `PLAN.md` §6 resume bullets with the real measured numbers.
7. README: CI + eval-gate badges, a hero GIF.

### Acceptance criteria

- [ ] Frontend deployed: enter a subject → watch nodes stream → read a cited memo
      → click a citation → see the source span
- [ ] `./verify.sh` exits 0 on a clean checkout
- [ ] Demo video recorded and linked in `README.md`
- [ ] `docs/WRITEUP.md` drafted
- [ ] `docs/PERFORMANCE.md` has real measured numbers
- [ ] README shows passing CI + eval-gate badges

### Tests to perform

| Kind | What | How |
|---|---|---|
| e2e | `frontend/` Playwright — enter subject, node cards appear, memo renders, citation click opens the viewer | against a `fake`-provider backend, in CI |
| script | `verify.sh` | final CI job on `main` |
| manual | full walkthrough on the live URL; mobile viewport; a deliberately bad subject (no corpus) still renders cleanly | by hand |

---

## Appendix A — Environment variables

| Var | Default | Phase | Notes |
|---|---|---|---|
| `PRAXIS_LLM_PROVIDER` | `fake` | 0 | `fake` / `openai` / `anthropic` / `nebius` |
| `PRAXIS_STRONG_MODEL` | `openai:gpt-4o` | 0 | planner + analysts |
| `PRAXIS_FAST_MODEL` | `openai:gpt-4o-mini` | 0 | editor, red_team, judge, contextualiser |
| `PRAXIS_MAX_GAP_LOOPS` | `1` | 0 | gap-fill loop cap |
| `PRAXIS_EMBEDDING_PROVIDER` | `fake` | 1 | `fake` / `openai` / `nebius` |
| `PRAXIS_EMBEDDING_MODEL` | `text-embedding-3-large` | 1 | |
| `PRAXIS_QDRANT_URL` | *(empty)* | 1 | empty → in-memory |
| `PRAXIS_CHUNK_SIZE` / `_OVERLAP` | `900` / `150` | 1 | |
| `PRAXIS_CONTEXTUAL_CHUNKS` | `false` | 1 | 1 LLM call per chunk |
| `PRAXIS_RETRIEVAL_K` | `6` | 1 | |
| `PRAXIS_RERANKER` | `lexical` | 1 | `lexical` / `cross-encoder` / `none` |
| `PRAXIS_RESEARCH_CONCURRENCY` | `4` | 2 | parallel researcher branches |
| `PRAXIS_DATABASE_URL` | *(empty)* | 2 | empty → `sqlite+aiosqlite:///./praxis.db` |
| `PRAXIS_CHECKPOINT_DB_PATH` | `./praxis_checkpoints.db` | 2 | LangGraph checkpointer, its own SQLite file |
| `PRAXIS_REDIS_URL` | `redis://localhost:6379` | 2 | the arq ingestion queue |
| `PRAXIS_MCP_SERVERS` | *(empty)* | 3 | JSON map of external MCP servers |
| `PRAXIS_OTEL_ENABLED` | `false` | 4 | |
| `PRAXIS_OTEL_ENDPOINT` | *(empty)* | 4 | OTLP collector |
| `LANGSMITH_API_KEY` | *(empty)* | 4 | tracing backend |
| `MEMO_F1_FLOOR` | `0.75` | 4 | CI memo-eval gate |
| `REDTEAM_FLOOR` | `0.6` | 4 | CI red-team-eval gate |

## Appendix B — Command reference

```bash
make install        # uv sync --extra dev
make check          # lint + type + test + eval gates  (CI parity)
make demo           # ingest fixture corpus, run a grounded dossier
make api            # uvicorn on :8000

make eval-rag             # retrieval eval report
make eval-rag-gate        # retrieval eval as a pass/fail gate
make eval-structure       # memo-structure eval report
make eval-structure-gate  # memo-structure eval as a pass/fail gate
# Phase 4:
make eval-dev       # judge-vs-expert F1, dev split
make eval-test      # gated memo eval, recorded slice
make eval           # full suite -> evals/reports/<ts>.json

praxis run "<subject>" [--depth quick|standard|deep] [--ingest PATH...] [--json]
praxis ingest PATH...
praxis search "<query>" [-k N]
uv run arq praxis.worker.WorkerSettings   # run the ingestion worker (needs Redis)
# Phase 3:
praxis mcp          # run the MCP stdio server
```

## Appendix C — CI job map

| Job | Trigger | Gates the PR? | Phase added |
|---|---|---|---|
| `quality` (3.11, 3.12) | push / PR | yes | 0 |
| `eval-gate` | push / PR | yes | 1 (expanded in 4) |
| `mcp-contract` | push / PR | yes | 3 |
| `build` → GHCR | push | no (main: publishes) | 5 |
| `k8s-validate` | push / PR | yes | 5 |
| `kind-smoke` | push / PR | yes | 5 |
| `deploy` (Fly) | push to `main` | — | 5 |
| `verify` | push to `main` | yes | 6 |
| `nightly` (full eval, real model) | schedule | no (reports) | 4 |
