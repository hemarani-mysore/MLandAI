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

## Phase 2 — Multi-agent for real  ·  🟡 in progress — slice 3/5 done  (~1 week)

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
4. ⬜ Run persistence + checkpointing, SQLite (step 6, `GET /dossiers[/{id}]`).
5. ⬜ SSE streaming + ingestion worker (steps 7–8).

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

6. **Run persistence + checkpointing** — `src/praxis/db/`
   - Deps: `sqlalchemy>=2`, `psycopg[binary]`, `alembic`, `langgraph-checkpoint-postgres`.
   - Models: `DossierRun` (id, subject, depth, status, created/finished, memo
     JSON, verification JSON, sources, cost_usd, tokens), `RunEvent`
     (run_id, seq, node, ts, payload JSON).
   - `PRAXIS_DATABASE_URL` empty → SQLite file (dev/CI); else Postgres.
   - Compile the graph with a checkpointer (`MemorySaver` for SQLite,
     `PostgresSaver` otherwise); `thread_id = run_id`.
   - `alembic init`, first migration; `alembic upgrade head` on API startup
     (guarded) or documented as a deploy step.
   - API: `GET /dossiers` (list), `GET /dossiers/{id}` (one run), keep
     `POST /dossiers` (now persists).

7. **Ingestion worker** — `src/praxis/worker/`
   - Dep: `arq` + `redis`. `ingest_task(payload)` wraps `ingest_source`.
   - `POST /corpus/jobs` enqueues → `{job_id}`; `GET /corpus/jobs/{id}` → status.
   - Keep sync `POST /corpus/documents` for small inline docs.
   - `docker-compose.yml`: add `redis` + a `worker` service (replicas: 1).

8. **SSE streaming** — `src/praxis/api/sse.py`
   - `GET /dossiers/stream?subject=&depth=` → `text/event-stream` from
     `graph.astream_events(version="v2")`.
   - Events: `node_start {node}`, `node_end {node, reasoning_steps, partial}`,
     `complete {DossierResponse}`, `error {message}`. Persist each as a `RunEvent`.

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
- [ ] `POST /dossiers` → `GET /dossiers/{id}` returns the persisted run;
      `GET /dossiers` lists it  *(slice 4)*
- [ ] `GET /dossiers/stream` emits ≥ 7 node events and ends with `complete`  *(slice 5)*
- [ ] `POST /corpus/jobs` → poll → `succeeded`, `GET /corpus/stats` shows growth  *(slice 5)*
- [ ] CI green using SQLite + fakeredis; `docker compose up` brings up
      api + worker + redis + qdrant, all healthy  *(slices 4–5)*

### Tests to perform

| Kind | File | Checks |
|---|---|---|
| unit | `test_graph_fanout.py` | `dispatch_research` emits N `Send`s; `evidence` accumulates across branches; concurrency cap respected |
| unit | `test_editor_multisection.py` | 6 `MemoSection`s; citations resolve to `evidence_refs`; recommendation logic (thin evidence → `insufficient_evidence`) |
| unit | `test_db_models.py` | `DossierRun` / `RunEvent` round-trip on SQLite; JSON columns |
| unit | `test_worker.py` | `ingest_task` grows the corpus (arq test harness + fakeredis) |
| integration | `test_api_runs.py` | `POST /dossiers` persists; `GET /dossiers/{id}` + list |
| integration | `test_api_sse.py` | stream yields ordered node events ending in `complete` (TestClient SSE) |
| integration | `test_api_jobs.py` | enqueue → status transitions → corpus updated |
| regression | existing 41 | updated for async where signatures change; still green |
| eval | `make eval-rag-gate` + `make eval-structure-gate` | retrieval unchanged; memo structure gate passes, fails on a corrupted canned memo |
| smoke | `make demo` | full 6-section grounded memo, verifier PASS |
| smoke | `docker compose up` then `curl -N localhost:8000/dossiers/stream?subject=Acme%20Robotics` | node events stream live |

---

## Phase 3 — MCP  ·  ⬜  (~3 days)

**Goal:** ship `praxis-mcp` (the desk as MCP tools/resources/prompts) and let the
researcher consume *external* MCP tools (web search, GitHub).

### Steps

1. **Deps** — `mcp` (FastMCP), `langchain-mcp-adapters`. `exa-py` optional.

2. **`src/praxis/mcp/server.py`** — a FastMCP server calling `praxis` library code
   (no logic fork):
   - **tools**: `create_dossier(subject, depth="standard")`,
     `search_corpus(query, k=6)`, `ingest_document(text|url, title)`,
     `get_evidence(claim)` → `{supporting: [...], refuting: [...]}`
   - **resources**: `dossier://{id}`, `dossier://{id}/citations`, `corpus://stats`
   - **prompts**: `due-diligence-brief`, `investment-memo`
   - transports: stdio (`python -m praxis.mcp`) and streamable-HTTP (for deploy)

3. **`.mcp.json`** at project root — Claude Code / Cursor config pointing at
   `uv run python -m praxis.mcp`.

4. **`praxis mcp`** CLI subcommand → runs the stdio server.

5. **External MCP consumption** — `src/praxis/tools/mcp_tools.py`
   - `PRAXIS_MCP_SERVERS` (JSON): e.g. `{"exa": {...}, "github": {...}}`.
   - Load via `MultiServerMCPClient` → LangChain tools.
   - `research_one`: after corpus retrieval, if hits are empty/weak **and**
     `web_search` is available, call it; merge results as `Evidence` with URL
     citations (`locator` = the URL).
   - No `PRAXIS_MCP_SERVERS` → unchanged corpus-only behaviour.

6. **Deploy artefact** — `docker/mcp.Dockerfile`; `mcp` service in compose.

### Acceptance criteria

- [ ] `python -m praxis.mcp` starts; a contract test enumerates **4 tools, 3
      resources, 2 prompts** with correct schemas
- [ ] `create_dossier` over MCP ≡ `run_dossier` for the same input (parity test)
- [ ] `.mcp.json` loads in Claude Code; `search_corpus` returns hits (manual)
- [ ] `PRAXIS_MCP_SERVERS` unset → dossier output byte-identical to Phase 2
- [ ] With a stub MCP server configured, `research_one` calls its `web_search`
      and produces an `Evidence` with a URL citation
- [ ] `mcp` image builds; `docker compose up` exposes it

### Tests to perform

| Kind | File | Checks |
|---|---|---|
| unit | `test_mcp_server.py` | in-process client lists + calls every tool/resource/prompt; schema assertions |
| unit | `test_mcp_tools_loader.py` | mocked `MultiServerMCPClient` → tools wired into researcher |
| contract | `test_mcp_parity.py` | `create_dossier` MCP result == `run_dossier` result |
| integration | `test_researcher_web_fallback.py` | tiny fake FastMCP with canned `web_search`; researcher uses it when corpus empty; URL citation present |
| CI | new `mcp-contract` job | runs the MCP unit + contract tests (offline, fake LLM) |
| manual | Claude Code | add `praxis-mcp`, ask "create a dossier on Acme Robotics", confirm round-trip + citations |

---

## Phase 4 — Eval framework  ·  ⬜  (~1 week)  ·  *the portfolio centrepiece*

**Goal:** golden dossiers with expert labels; LLM-as-judge scored by **F1** vs
those labels; deterministic citation-integrity; seeded-error detection; OTel
tracing + cost accounting; an expanded CI eval gate + a nightly full run.

### Steps

1. **Dataset** — `evals/datasets/golden_dossiers/`
   - `index.yaml`: splits `dev` (judge calibration), `test` (gate), `online`
     (fresh-generation drift check).
   - Per subject: `sources/*.md` (frozen), `expected.md` (reference memo),
     `label` (`pass` / `fail`), `critique` (1–3 sentences).
   - 8–15 subjects. Include deliberate fails: thin evidence that should force
     `insufficient_evidence`; a planted contradiction the memo must surface.

2. **`evals/metric.py`** — `BinaryLLMJudgeMetric`
   - `JudgeResult(label: Literal["pass","fail"], critique: str)` structured output.
   - Judge prompt: plan + evidence + generated memo + rubric → pass/fail + why.
     Reference-free by default; `--reference` also feeds `expected.md`.
   - Runs on the `fast` tier; temperature 0.

3. **`evals/memo_eval.py`** — offline judge-vs-expert
   - For each labelled item: generate (or load) memo → judge → collect
     `(pred, expert_label)`.
   - Report precision / recall / **F1** / raw agreement. `--split dev|test`,
     `--gate` (fail if F1 < `MEMO_F1_FLOOR`, default 0.75).

4. **`evals/citation_eval.py`**
   - Deterministic: every memo sentence carries a citation whose `source_id`
     resolves → `hallucinated_citation_rate` (gate: == 0).
   - NLI: is each claim entailed by its cited span? → `unsupported_claim_rate`
     (small model; **nightly**, not gated).

5. **`evals/redteam_eval.py`**
   - Inject a known-false claim into the evidence before analysts run; measure
     whether `red_team` flags it → `seeded_error_detection_rate`
     (gate: ≥ `REDTEAM_FLOOR`, default 0.6, on recorded responses).

6. **`evals/online_eval.py`** — generate fresh memos on `online` + judge. Nightly.

7. **`evals/run.py`** — `--suite {rag,memo,citation,redteam,all} --split ...
   --gate --report evals/reports/<ts>.json`.

8. **Observability** — `src/praxis/obs/`
   - Real OTel: `setup_tracing()` (OTLP → LangSmith / console), wrap every node in
     a span (`span()` already stubbed).
   - Attributes: `role`, `model`, `prompt_tokens`, `completion_tokens`,
     `cost_usd`, and for the judge `score`.
   - Token/cost accounting in `llm.py` → `DossierState.cost_usd` (add the reducer)
     → `DossierResponse.cost_usd` + `.tokens`.
   - `PRAXIS_OTEL_ENABLED`, `PRAXIS_OTEL_ENDPOINT`, `LANGSMITH_API_KEY`.

9. **CI `eval-gate` expansion** — all deterministic / recorded, offline, < 3 min:
   - `rag_eval.py --gate` (exists)
   - `memo_eval.py --split test --gate` on a **5–8 item fixed slice** with
     **recorded LLM responses** (`evals/cassettes/`, VCR-style)
   - `citation_eval.py --gate` (resolution part)
   - `redteam_eval.py --gate` (recorded)

10. **Nightly** — `.github/workflows/nightly.yml`: `run.py --suite all
    --split dev,test,online` with a real cheap model (`OPENAI_API_KEY` secret);
    uploads `evals/reports/<ts>.json` as an artifact; optional summary comment.

11. **Makefile** — `eval`, `eval-dev`, `eval-test`, `eval-online`, `eval-report`.

### Acceptance criteria

- [ ] `index.yaml` ≥ 8 subjects across `dev` / `test` / `online`, each with
      `label` + `critique`
- [ ] `make eval-dev` prints judge-vs-expert **F1** on the dev split
- [ ] `make eval-test` (`--gate`) passes on the recorded slice; **fails** when a
      canned memo is deliberately corrupted (test)
- [ ] `citation_eval` catches an injected bad citation
- [ ] `redteam_eval` reports a detection rate; a seeded false claim is caught in
      ≥ 1 fixture
- [ ] With `PRAXIS_OTEL_ENABLED=true`, every node emits a span with the expected
      attributes (asserted via an in-memory span exporter)
- [ ] `DossierResponse` carries `cost_usd` and `tokens`
- [ ] CI `eval-gate` runs all sub-gates offline in < 3 min
- [ ] `nightly.yml` exists (manual-dispatch until the secret is added) and
      produces a report artifact

### Tests to perform

| Kind | File | Checks |
|---|---|---|
| unit | `test_metric.py` | `JudgeResult` parsing; judge-prompt rendering includes rubric + evidence |
| unit | `test_memo_eval.py` | precision / recall / F1 math on synthetic `(pred, label)` pairs |
| unit | `test_citation_eval.py` | resolution pass; a fabricated citation is flagged |
| unit | `test_redteam_eval.py` | injection helper; detection on a canned red-team report |
| unit | `test_obs.py` | `InMemorySpanExporter` sees one span per node with `role` / `model` / token attrs |
| unit | `test_cost_accounting.py` | token + `cost_usd` sums propagate to `DossierResponse` |
| integration | `test_eval_run.py` | `run.py --suite all --split dev` writes a report JSON with every metric key |
| eval-of-eval | `test_judge_calibration.py` | on `dev` (recorded responses) judge F1 ≥ 0.70 — guards a broken judge prompt |
| CI | `eval-gate` expanded | 4 sub-gates; `--report` artifact uploaded |
| manual | one real run | `make eval-test` with a real key; inspect a LangSmith trace; sanity-check a critique |

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
| `PRAXIS_DATABASE_URL` | *(empty)* | 2 | empty → SQLite file |
| `REDIS_URL` | `redis://localhost:6379` | 2 | ingestion queue |
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

make eval-rag       # retrieval eval report
make eval-rag-gate  # retrieval eval as a pass/fail gate
# Phase 4:
make eval-dev       # judge-vs-expert F1, dev split
make eval-test      # gated memo eval, recorded slice
make eval           # full suite -> evals/reports/<ts>.json

praxis run "<subject>" [--depth quick|standard|deep] [--ingest PATH...] [--json]
praxis ingest PATH...
praxis search "<query>" [-k N]
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
