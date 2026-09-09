# MLandAI — branch: `praxis-due-diligence-desk`

> Part of [MLandAI](https://github.com/hemarani-mysore/MLandAI). Each project lives on
> its own branch; `main` holds the index. This branch holds **Praxis**.

## Praxis — a production due-diligence research desk

A multi-agent (LangGraph) system that turns a subject — a company, a technology, a
vendor, an acquisition target — into a **cited due-diligence memo**: summary →
overview → bull case → bear case → risks → open questions → recommendation with a
confidence score. Every claim links back to the evidence it came from.

One project, exercising the full production stack:

| Capability | In Praxis |
|---|---|
| LLM prompting · embeddings · RAG | Hybrid retrieval (dense + BM25) over an ingested corpus, cross-encoder reranking, page-level citations, structured LLM outputs everywhere |
| MCP server with tools | `praxis-mcp` (FastMCP) exposes the desk as tools/resources/prompts; the research agents also *consume* external MCP tools |
| Docker · Kubernetes · CI/CD | Per-service Dockerfiles, k8s manifests (HPA, NetworkPolicy, PDB) validated on `kind` in CI, GitHub Actions → GHCR → live URL |
| Multi-agent | LangGraph graph: Planner → Researcher (fan-out) → {Bull ∥ Bear} → Red-team → Editor → Verifier, with an evaluator-optimizer gap-fill loop |
| Eval framework | Golden dataset + LLM-as-judge (F1 vs expert labels), RAGAS retrieval metrics, deterministic citation-integrity + seeded-error checks — run as a CI regression gate |
| Team of agents | Six role-specialized agents with their own prompts, tools, and model tiers |

**The project is in [`praxis-due-diligence-desk/`](praxis-due-diligence-desk/).**
See its [`README.md`](praxis-due-diligence-desk/README.md) for setup and
[`PLAN.md`](praxis-due-diligence-desk/PLAN.md) for the full design and phase plan.
