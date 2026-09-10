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
| **2 — Multi-agent** | real prompts, multi-section editor, `evidence_refs`, deterministic recommendation, memo-structure eval gate *(slice 1 ✅)* · `Send` fan-out, async graph, run persistence, SSE, ingestion worker *(slices 2–5 ⬜)* | 🟡 **in progress** |
| 3 — MCP | `praxis-mcp` FastMCP server (tools/resources/prompts); agents consume external MCP tools | ⬜ |
| 4 — Eval framework | golden dossiers, LLM-as-judge (F1 vs expert labels), citation-integrity + seeded-error checks, OTel tracing | ⬜ |
| 5 — Deploy | per-service Dockerfiles, k8s manifests validated on `kind` in CI, live URL on Fly | ⬜ |

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
```

Real models: set `PRAXIS_LLM_PROVIDER=openai|anthropic|nebius` and
`PRAXIS_EMBEDDING_PROVIDER=openai|nebius` with the matching key in `.env`
(see [`.env.example`](.env.example)). `docker compose up --build` brings up the
API + a real Qdrant.

```bash
make check           # lint + type + test + eval gates (what CI runs)
make demo            # ingest the fixture corpus, run a grounded dossier
make eval-rag        # retrieval eval report
make eval-structure  # memo-structure eval report
```

## The graph

```
planner ─▶ researcher ─┬─▶ bull ──┐
                       └─▶ bear ──┴─▶ red_team ──(gaps? & loop<cap)──▶ researcher
                                              └──(clean / cap hit)──▶ editor ─▶ verifier ─▶ END
```

- **planner** — decomposes the request into a plan covering a fixed 6-section rubric
- **researcher** — hybrid-retrieves the corpus per sub-question, emits one `Evidence`
  whose citation is *grounded* (snapped to a retrieved chunk); Phase 2 fans this out
- **bull ∥ bear** — run concurrently; each builds one side of the case, only from evidence, with visible reasoning steps and `evidence_refs` back to the items each point rests on
- **red_team** — audits both cases for unsupported claims / contradictions / coverage gaps; can loop back for gap-fill (bounded by `PRAXIS_MAX_GAP_LOOPS`)
- **editor** — writes one section per rubric item on the cheap model tier, drops anything the red team flagged; the overall recommendation + confidence are then computed deterministically from the evidence balance (`graph/recommend.py`), not left to the model
- **verifier** — deterministic: every citation must resolve to real gathered evidence (placeholders don't count); scores rubric coverage; flags missing / uncited sections and ungrounded memos

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

## Layout

```
src/praxis/
  config.py schemas.py llm.py render.py cli.py
  graph/  state.py build.py prompts.py context.py recommend.py  nodes/{planner,researcher,analysts,red_team,editor,verifier}.py
  rag/    parse.py chunk.py contextual.py embed.py bm25.py vector_store.py fuse.py rerank.py corpus.py retrieve.py ingest.py
  api/    main.py deps.py
  obs/    __init__.py                 # OTel hooks (Phase 4)
tests/                                 # 56 tests: graph, retrieval, api, cli, schemas, recommend, verifier
evals/  rag_eval.py + memo_structure_eval.py + fixture corpus   # two deterministic gates (CI)
deploy/ README.md                      # Phase 5
docs/   IMPLEMENTATION_PLAN.md         # per-phase steps + acceptance criteria + tests
```

## Provenance

Patterns adapted from `Arindam200/awesome-ai-apps`:
`agentfield_finance_research_agent` (bull/bear/editor committee, visible reasoning),
`maintainer_brief` (FastAPI production skeleton, multi-provider LLM),
`deep_research_writing_agents_nebius_okahu` (eval harness, MCP servers),
`mcp_toolbox_security_agent` (k8s manifests), `advanced_rag_with_reranking` (hybrid RAG + RAGAS).
