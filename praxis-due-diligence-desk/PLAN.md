# Praxis — Design & Phase Plan

**Goal:** one production-grade, deployed AI system for an AI-Engineer portfolio that
genuinely exercises: (1) LLM prompting + embeddings + RAG, (2) an MCP server with
tools, (3) Docker + Kubernetes + CI/CD, (4) multi-agent orchestration, (5) an
evaluation framework, (6) a team of role-specialized agents.

**Product:** a *due-diligence research desk*. Input: a subject (company / technology
/ vendor / acquisition target). Output: a cited due-diligence memo — summary,
overview, bull case, bear case, risks, open questions, and a recommendation
(`proceed` / `proceed_with_conditions` / `pass` / `insufficient_evidence`) with a
calibrated confidence. Every claim deep-links to the evidence it came from.

**Backbone:** LangGraph (explicit state machine, checkpointing, human-in-the-loop
ready — and the framework most named in AI-Engineer job descriptions).

**"In production" means:** an always-on live URL (Fly.io) + real Kubernetes
manifests validated on an ephemeral `kind` cluster in CI.

---

## 1. How each requirement is satisfied

| # | Requirement | In Praxis | Pattern adapted from `awesome-ai-apps` |
|---|---|---|---|
| 1 | LLM prompting, embedding, RAG | Ingestion → chunk + contextual augmentation → Qdrant hybrid (dense `Qwen3-Embedding-8B` / `text-embedding-3-large` + BM25) → RRF fusion → cross-encoder rerank → page-level citations. Every agent boundary is a Pydantic structured output. | `advanced_rag_with_reranking/src/boeing_rag/*`, `mcp_toolbox_security_agent/agent/embeddings.py`, `maintainer_brief/backend/app/intelligence/llm.py` |
| 2 | MCP server with tools | `praxis-mcp` (FastMCP) exposes `create_dossier`, `search_corpus`, `ingest_document`, `get_evidence` + resources (`dossier://{id}`, `corpus://stats`) + prompts. The **researcher node also consumes** an external web-search MCP (Exa) and a GitHub MCP. | `deep_research_writing_agents_nebius_okahu/src/research/server.py` + `routers/*`, `deep_researcher_agent/server.py` |
| 3 | Docker, k8s, CI/CD | Multi-stage non-root Dockerfiles per service; `docker-compose` (dev + self-host); `deploy/k8s` base + kustomize overlays (Deployments, HPA, PDB, NetworkPolicy, Ingress, secrets); GitHub Actions: lint → type → test → **eval-gate** → build→GHCR → `kind` smoke → deploy. | `mcp_toolbox_security_agent/deploy/{k8s,compose}/*`, `maintainer_brief/backend/{Dockerfile,fly.toml}` + `PRODUCTION.md` |
| 4 | Multi-agent | LangGraph graph with parallel branches (bull ∥ bear), a bounded evaluator-optimizer gap-fill loop (red_team → researcher), and a deterministic final gate (verifier). | `agentfield_finance_research_agent/src/reasoners.py`, `deep_researcher_agent/agents.py`, `due_diligence_agent/agents.py` |
| 5 | Eval framework | Golden-dossier dataset with splits; LLM-as-judge (`BinaryLLMJudgeMetric` → pass/fail + critique → **F1 vs expert labels**); RAGAS for retrieval; deterministic citation-integrity; seeded-error detection for the red team; offline + online eval; `make eval-*`; CI regression gate. | `deep_research_writing_agents_nebius_okahu/src/writing/evals/*` + `scripts/run_*evaluation.py`, `advanced_rag_with_reranking/evals/*` |
| 6 | Team of agents | 6 roles, each its own system prompt + tool set + model tier (strong: planner/analysts; fast: editor/red_team — the calibration trick from Argus). | `content_team_agent/`, `due_diligence_agent/prompts.py` |

No single agent in the source repo has all six — Praxis is the deliberate fusion.

---

## 2. Architecture

```
                 Next.js frontend (live URL) — subject input · streaming run · memo + citation viewer
                                     │ SSE / REST
┌────────────────────────────────────▼───────────────────────────────────────────────┐
│ praxis-api (FastAPI)   POST /dossiers ─▶ LangGraph run                               │
│                                                                                     │
│   planner ─▶ researcher ─┬─▶ bull ─┐                                                 │
│                          └─▶ bear ─┴─▶ red_team ─(gaps)─▶ researcher                 │
│                                              └─(clean)──▶ editor ─▶ verifier ─▶ END  │
│                                                                                     │
│   researcher uses: hybrid RAG (Qdrant) + web-search MCP (Exa) + GitHub MCP           │
│   every node emits an OpenTelemetry span (role, model, tokens, cost, judge score)    │
└───────┬───────────────────────┬────────────────────┬───────────────────┬────────────┘
        │                       │                    │                   │
     Qdrant                praxis-worker          Postgres           praxis-mcp
   (vectors)            (ingestion jobs)     (dossiers, runs, evals)  (FastMCP: 4 tools)
```

`praxis-mcp` calls the same graph code the API does — a thin wrapper, not a fork.

---

## 3. Typed contracts (already in `src/praxis/schemas.py`)

`Citation` · `Evidence(question, section, claim, stance, citation, confidence)` ·
`SubQuestion` · `ResearchPlan(subject, thesis, sub_questions)` ·
`Finding(stance, summary, points, reasoning_steps, evidence_refs)` ·
`RedTeamReport(unsupported_claims, contradictions, coverage_gaps, gap_questions)` ·
`DossierMemo(subject, summary, recommendation, confidence, sections[], open_questions)` ·
`VerificationReport(citations_checked, citations_resolved, hallucinated_citation_rate, coverage_score, flags)`.

Fixed rubric (`RUBRIC_SECTIONS`): Overview · Market & Competition · Team & Funding ·
Technology & IP · Traction & Financials · Risks & Red Flags. The planner must cover
all six, so dossiers are comparable and coverage is measurable.

---

## 4. Phase plan

Each phase ends with something runnable and a green CI.

### Phase 0 — Scaffold ✅ (done)
Repo layout · `DossierState` + 7 nodes wired in LangGraph with the bounded gap-fill
loop · deterministic `FakeStructuredLLM` (whole system runs with no keys) · FastAPI
(`/health`, `/ready`, `/dossiers`, `/dossiers.md`) · `praxis` CLI · Dockerfile +
compose · ruff + mypy + pytest · GitHub Actions (lint + type + test, py3.11/3.12) ·
tests cover the happy path, the gap-fill loop, and citation-integrity flagging.

### Phase 1 — RAG (~1 week)
- `praxis-worker`: ingestion queue (arq/RQ). Fetch → parse (`pymupdf` for text +
  page coords; a VLM for tables/figures) → chunk with metadata → **contextual
  augmentation** (1-sentence LLM context per chunk, cached by content hash).
- Embed dense + sparse BM25 → one **Qdrant hybrid collection**.
- Retrieval: hybrid → RRF → **cross-encoder rerank** (`bge-reranker-v2` / Cohere) →
  top-k with `{source, page, span}` citations. Cache by `(query, corpus_version)`.
- `search_corpus` tool; researcher node uses it instead of the stub.
- `evals/golden_questions.jsonl` (~50) + `rag_eval.py` (RAGAS: faithfulness,
  context precision, context recall). `make eval-rag`.
- Nebius provider wired (`ChatOpenAI` base-url shim).
- compose gains `qdrant` + `postgres`.

### Phase 2 — Multi-agent for real (~1 week)
- Real, tightened prompts with eval-covered few-shots.
- Researcher fan-out via LangGraph `Send` — one branch per sub-question, concurrent.
- `evidence_refs` populated so the memo and UI can trace each point.
- `MemorySaver`/`PostgresSaver` checkpointer → resumable runs.
- SSE endpoint streaming `graph.astream_events` → frontend shows each node glowing
  with its reasoning steps (Argus UX).
- Dossier + run history persisted in Postgres.

### Phase 3 — MCP (~3 days)
- `src/praxis/mcp/server.py` (FastMCP): tools `create_dossier`, `search_corpus`,
  `ingest_document`, `get_evidence`; resources `dossier://{id}`,
  `dossier://{id}/citations`, `corpus://stats`; prompt `due-diligence-brief`.
- Researcher consumes an external **web-search MCP** (Exa) + **GitHub MCP**
  (`langchain-mcp-adapters`).
- `.mcp.json` so it drops into Claude Code / Cursor. MCP contract test in CI.

### Phase 4 — Eval framework (~1 week) — the portfolio centrepiece
- `evals/datasets/golden_dossiers/` — 8–15 subjects, each with frozen `sources/`,
  `expected.md`, an expert `label` (pass/fail) + `critique`; `index.yaml` splits
  (`dev`, `test`, `online`).
- `metric.py` — `BinaryLLMJudgeMetric` → `{label, critique}` structured output.
- `memo_eval.py` — judge vs expert labels → precision / recall / **F1**
  (okahu pattern, near-verbatim).
- `citation_eval.py` — deterministic: every claim resolves; NLI claim↔span;
  hallucinated-citation rate.
- `redteam_eval.py` — inject a known-false claim into evidence → detection rate.
- `run.py --split {dev,test,online} [--gate]`; online eval generates fresh + judges.
- OpenTelemetry spans on every node → LangSmith (or Phoenix). `PRAXIS_OTEL_ENABLED`.
- **CI eval-gate**: fixed small slice, cheap model / recorded responses; fail PR if
  `F1 < 0.75` or `faithfulness < 0.85` or `hallucinated_citation_rate > 0`.
  Full eval runs nightly.

### Phase 5 — Deploy (~1 week)
- Dockerfiles for `api`, `mcp`, `worker`, `frontend` (hardened: non-root,
  `readOnlyRootFilesystem`, dropped caps).
- `deploy/k8s/base` + `overlays/{dev,prod}` (kustomize) or a Helm chart:
  Deployments, Services, Ingress, HPA on `api`, PDB, NetworkPolicy (only
  `api`/`worker` reach the datastores), ConfigMap, `secrets.example.yaml`.
- CI: `kubeconform` + `kustomize build` + `helm lint`; then `kind` — apply
  `overlays/dev`, wait for rollout, `curl /health` on every service.
- CI deploy on `main`: `flyctl deploy` for `api` + `mcp` + `frontend`;
  Qdrant Cloud + Neon Postgres free tiers. `/ready` checks both + one LLM ping.
- `ARCHITECTURE.md`, `RUNBOOK.md`, `PRODUCTION.md`.

### Phase 6 — Polish (~3 days)
Demo video · dev.to writeup · cost & latency notes from OTel · `verify.sh` that
proves the eval gate + citation integrity in one command.

**Total: ~5–7 weeks part-time.** Phases 0–2 already demo well; 4 is the
differentiator; 5 is the "deployed in production" claim.

---

## 5. Scope guardrails

- **No custom scraper** — web access is Exa / an MCP, always.
- **Freeze the eval corpus** — golden dossiers ship their own `sources/`; no live
  fetching in evals or the numbers aren't reproducible.
- **Cap the gap-fill loop** (`PRAXIS_MAX_GAP_LOOPS`, default 1). Enforced in
  `route_after_red_team`.
- **CI eval-gate must be cheap & fast** — recorded responses or a small model on a
  5–8 item slice. Full eval is nightly, not per-PR.
- **One `worker` replica** for ingestion, or externalize the queue (the
  `maintainer_brief` "cron fires N times" lesson).
- **Cache aggressively** — embeddings by content hash, retrieval by
  `(query, corpus_version)`, dev LLM calls on disk.
- **Cheap model for judge / editor / verifier**; strong model only for planner +
  analysts.

---

## 6. Resume / interview points this unlocks

- Designed & deployed a production multi-agent due-diligence system (LangGraph,
  FastAPI, Next.js, Kubernetes) — a 6-agent graph with an adversarial red-team loop
  producing analyst memos where every claim is citation-linked to source evidence.
- Hybrid RAG: Qdrant dense + BM25 with RRF fusion, contextual chunking,
  cross-encoder reranking, page-level citations; RAGAS faithfulness 0.9+.
- Built an eval framework — golden dataset with expert labels, LLM-as-judge scored
  by F1 against those labels, plus deterministic citation-integrity and
  seeded-error-detection checks — wired into GitHub Actions as a regression gate
  that blocks PRs on quality drops.
- Packaged the system as an MCP server (FastMCP), callable from any MCP host; the
  agents themselves also consume external MCP tools.
- CI/CD on GitHub Actions → GHCR; Kubernetes manifests (HPA, NetworkPolicy, PDB)
  validated on an ephemeral `kind` cluster in CI; live deployment on Fly.io;
  OpenTelemetry tracing into LangSmith.
