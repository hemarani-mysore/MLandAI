# Praxis — Due-Diligence Research Desk

A production multi-agent system that turns a **subject** — a company, a technology,
a vendor, an acquisition target — into a **cited due-diligence memo**: summary →
overview → bull case → bear case → risks → open questions → recommendation with a
calibrated confidence. Every claim links back to the evidence it rests on.

Built with **LangGraph** + **FastAPI**. Designed to exercise the full production
stack — RAG, an MCP server, multi-agent orchestration, an eval framework wired into
CI, Docker/Kubernetes — in one coherent project. Full design and phase plan in
[`PLAN.md`](PLAN.md).

## Status

| Phase | Scope | State |
|---|---|---|
| **0 — Scaffold** | repo layout, LangGraph skeleton (7 nodes, gap-fill loop), FastAPI + CLI, deterministic `fake` LLM, Docker, CI (lint + type + test) | ✅ **done** |
| 1 — RAG | ingestion → Qdrant hybrid (dense + BM25) → rerank → page-level citations; RAGAS eval | ⬜ next |
| 2 — Multi-agent | real prompts, `Send` fan-out for researchers, SSE streaming of node events | ⬜ |
| 3 — MCP | `praxis-mcp` FastMCP server (tools/resources/prompts); agents consume external MCP tools | ⬜ |
| 4 — Eval framework | golden dataset, LLM-as-judge (F1 vs expert labels), citation-integrity + seeded-error checks, OTel tracing, CI eval-gate | ⬜ |
| 5 — Deploy | per-service Dockerfiles, k8s manifests validated on `kind` in CI, live URL on Fly | ⬜ |

## Quickstart

```bash
cd praxis-due-diligence-desk
uv sync --extra dev          # or: pip install -e ".[dev]"

# CLI — runs end to end with NO API keys (deterministic `fake` provider)
uv run praxis run "Acme Robotics" --depth standard

# API
uv run uvicorn praxis.api.main:app --port 8000
curl -s -XPOST localhost:8000/dossiers.md \
  -H 'content-type: application/json' -d '{"subject":"Acme Robotics"}'
```

To use real models: set `PRAXIS_LLM_PROVIDER=openai` (or `anthropic`) and the
matching API key in `.env` (see [`.env.example`](.env.example)).

```bash
make check      # lint + type + test (what CI runs)
make demo       # a sample dossier
docker compose up --build
```

## The graph

```
planner ─▶ researcher ─┬─▶ bull ──┐
                       └─▶ bear ──┴─▶ red_team ──(gaps? & loop<cap)──▶ researcher
                                              └──(clean / cap hit)──▶ editor ─▶ verifier ─▶ END
```

- **planner** — decomposes the request into a plan covering a fixed 6-section rubric
- **researcher** — one cited `Evidence` per sub-question (Phase 1: real hybrid retrieval; Phase 2: parallel fan-out)
- **bull ∥ bear** — run concurrently; each builds one side of the case, only from evidence, with visible reasoning steps
- **red_team** — audits both cases for unsupported claims / contradictions / coverage gaps; can loop back for gap-fill (bounded by `PRAXIS_MAX_GAP_LOOPS`)
- **editor** — writes the memo on the cheap model tier; drops anything the red team flagged
- **verifier** — deterministic: every citation must resolve to gathered evidence; scores rubric coverage; raises flags

## Layout

```
src/praxis/
  config.py schemas.py llm.py render.py cli.py
  graph/    state.py build.py prompts.py context.py  nodes/{planner,researcher,analysts,red_team,editor,verifier}.py
  api/      main.py
  obs/      __init__.py          # OTel hooks (Phase 4)
tests/                            # graph (incl. gap-fill loop), API, CLI, schemas
evals/    README.md               # Phase 4
deploy/   README.md               # Phase 5
```

## Provenance

Patterns are adapted from `Arindam200/awesome-ai-apps`:
`agentfield_finance_research_agent` (bull/bear/editor committee, visible reasoning),
`maintainer_brief` (FastAPI production skeleton, multi-provider LLM),
`deep_research_writing_agents_nebius_okahu` (eval harness, MCP servers),
`mcp_toolbox_security_agent` (k8s manifests), `advanced_rag_with_reranking` (RAG + RAGAS).
