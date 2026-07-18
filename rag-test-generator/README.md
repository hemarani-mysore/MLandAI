# RAG Test Generator

An agentic AI system that ingests any public GitHub repository, identifies untested code paths using RAG (Retrieval-Augmented Generation), auto-generates Playwright and pytest test stubs, and opens a Pull Request — with zero human intervention.

A second, independent pipeline can also record a live browser scenario, generate an executable test for it, and self-heal it in a loop with an LLM until it passes.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-purple)
![Playwright](https://img.shields.io/badge/Playwright-TypeScript-green?logo=playwright)
![LangGraph](https://img.shields.io/badge/LangGraph-recording%20pipeline-1C3C3C)
![pytest](https://img.shields.io/badge/pytest-94%20passing-brightgreen?logo=pytest)

---

## What it does

```
GitHub URL
    │
    ▼
┌─────────────────────────┐
│  1. Ingestion Agent     │  Clone repo → parse Python/TS/JS files
└──────────┬──────────────┘
           │
    ▼
┌─────────────────────────┐
│  2. Embedding Agent     │  Chunk code → embed with OpenAI → store in ChromaDB
└──────────┬──────────────┘
           │
    ▼
┌─────────────────────────┐
│  3. Analysis Agent      │  RAG queries → find endpoints, pages, auth flows, gaps
└──────────┬──────────────┘
           │  test_gap_report.json / .md
    ▼
┌─────────────────────────┐
│  4. Test Gen Agent      │  OpenAI generates Playwright .spec.ts + pytest files
└──────────┬──────────────┘
           │
    ▼
┌─────────────────────────┐
│  5. GitHub Push Agent   │  Fork → branch → single commit → Pull Request
└─────────────────────────┘
           │
    ▼
   PR URL
```

**Demonstrated on:** [fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)
**Generated PR:** [hemarani-mysore/full-stack-fastapi-template/pull/1](https://github.com/hemarani-mysore/full-stack-fastapi-template/pull/1)

There's also a second pipeline, `record`, described in its own section below.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM & Embeddings | OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`) |
| Vector Store | ChromaDB (local, persistent) |
| Code Parsing | Python AST + character-level chunking for TypeScript |
| Test Framework | Playwright (TypeScript + JavaScript) + pytest (Python) |
| Recording Pipeline Orchestration | LangGraph (`StateGraph` with a self-heal loop) |
| GitHub Integration | PyGitHub — fork, branch, commit, PR via API |
| CLI | Typer + Rich |
| Config | python-dotenv |

---

## Prerequisites

Install these **before** Quick Start below.

| Requirement | Why | Check | Install |
|---|---|---|---|
| **Python 3.11+** | Runs the CLI and all agents | `python --version` | [python.org](https://www.python.org/downloads/) |
| **Git** | Cloning target repos (GitPython) | `git --version` | [git-scm.com](https://git-scm.com/downloads) |
| **Node.js 18+ & npm** | Runs `npx playwright` for the `record` pipeline | `node --version && npm --version` | See below |
| **OpenAI API key** | All LLM calls + embeddings, both pipelines | — | [platform.openai.com](https://platform.openai.com/api-keys) |
| **GitHub Personal Access Token** | Only needed for `analyze`/`generate --push` (forking + opening PRs) | — | [github.com/settings/tokens](https://github.com/settings/tokens) — `repo`, `workflow` scopes |

### Installing Node.js (if you don't have it)

The `record` pipeline shells out to `npx playwright`, which needs a real Node.js/npm install — this is **not** a Python package. If `node --version` fails, install via [nvm](https://github.com/nvm-sh/nvm) (no sudo required):

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
# Open a new terminal (or `source ~/.zshrc` / `source ~/.bashrc`), then:
nvm install --lts
node --version && npm --version
```

If you skip Node.js entirely, the `analyze`/`generate` pipeline still works fine — only `record` needs it.

---

## Quick Start

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/hemarani-mysore/rag-test-generator
cd rag-test-generator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up the recording pipeline's Node project (skip if you only need `analyze`/`generate`)

```bash
cd playwright_runner
npm install
npx playwright install --with-deps chromium
cd ..
```

### 3. Configure environment

```bash
cp env.example .env
```

Fill in `.env`:

| Variable | Required for | Where to get it |
|---|---|---|
| `OPENAI_API_KEY` | Everything | [platform.openai.com](https://platform.openai.com/api-keys) |
| `OPENAI_MODEL` | Everything (optional — defaults to `gpt-4o-mini`) | — |
| `OPENAI_EMBEDDING_MODEL` | `analyze`/`generate` (optional — defaults to `text-embedding-3-small`) | — |
| `GITHUB_TOKEN` | `generate --push` only | [github.com/settings/tokens](https://github.com/settings/tokens) — `repo`, `workflow` scopes |
| `GITHUB_USERNAME` | `generate --push` only | your GitHub username |

### 4. Verify setup

```bash
python main.py verify-setup
```

This checks: env vars, OpenAI API connection, ChromaDB, GitHub API, GitPython, and Node.js/Playwright — all six should show ✅.

### 5. Run a pipeline

```bash
# GitHub-analysis pipeline
python main.py generate \
  --repo https://github.com/fastapi/full-stack-fastapi-template \
  --push

# Recording pipeline
python main.py record --url https://example.com --name checkout-flow
```

---

## CLI Reference

```bash
# Verify all dependencies are configured
python main.py verify-setup

# Just analyze (no test generation)
python main.py analyze --repo <github-url>

# Full pipeline — generate locally only
python main.py generate --repo <github-url> --dry-run

# Full pipeline — generate + push PR
python main.py generate --repo <github-url> --push

# Reuse saved analysis (skip re-embedding, saves API quota)
python main.py generate --repo <github-url> --push --reuse-report

# Force fresh re-embed from scratch
python main.py generate --repo <github-url> --push --reset

# Record a live browser scenario, self-heal it, and produce a report
python main.py record --url <live-url> [--name <scenario-name>] [--max-attempts N]
```

---

## Recording Pipeline (`record`)

A second, independent pipeline: point it at a **live URL** instead of a GitHub repo. It records a real browser scenario with Playwright, generates an executable `.spec.js` test with OpenAI, runs it, and self-heals it in a loop until it passes or the LLM decides to give up. See [Prerequisites](#prerequisites) for the one-time Node.js/`playwright_runner` setup this needs.

```
Live URL
    │
    ▼
┌─────────────────────────┐
│  1. Recorder Agent      │  npx playwright codegen — record actions in a real browser
└──────────┬──────────────┘
           │
    ▼
┌─────────────────────────┐
│  2. Spec Gen Agent      │  OpenAI turns the recording into a clean .spec.js test
└──────────┬──────────────┘
           │
    ▼
┌─────────────────────────┐
│  3. Execution Agent     │  npx playwright test — run it, parse pass/fail,
└──────────┬──────────────┘  capture page HTML on failure
           │
     ┌─────┴─────┐
     ▼           ▼
  passed      failed ──▶ 4. Fix Agent (OpenAI) ──▶ back to Execution Agent
     │           (sees the failure DOM, not just the error message —
     │            loops until it passes, the LLM gives up, or
     │            MAX_FIX_ATTEMPTS is hit)
     ▼
┌─────────────────────────┐
│  5. Report Agent        │  Saves outputs/recorded_tests/<scenario>.spec.js + report
└─────────────────────────┘
```

Built with **LangGraph** (`agents/recording_graph.py`) — the execute/fix loop is a real graph cycle with a conditional exit, not a fixed linear chain like the GitHub-analysis pipeline above.

### Usage

```bash
python main.py record --url https://example.com --name checkout-flow
```

This opens a browser — interact with the page, then close the browser window to finish recording. The tool takes it from there: generating the spec, running it, and self-healing on failure (using the actual page HTML at the moment of failure, not just the error text, so it can catch wrong-selector bugs rather than just retrying with longer timeouts).

---

## Project Structure

```
rag-test-generator/
├── main.py                        # CLI entry point (both pipelines)
├── config.py                      # All settings (loaded from .env)
├── requirements.txt
├── pytest.ini
├── env.example
│
├── agents/
│   ├── ingestion_agent.py         # Clone + parse GitHub repo
│   ├── embedding_agent.py         # Chunk + batch-embed + ChromaDB (with rate-limit retry)
│   ├── analysis_agent.py          # RAG queries → TestGapReport
│   ├── test_gen_agent.py          # OpenAI → Playwright + pytest files
│   ├── github_agent.py            # Fork + branch + commit + PR
│   │
│   ├── recorder_agent.py          # npx playwright codegen — record a browser scenario
│   ├── spec_gen_agent.py          # OpenAI → clean .spec.js from the raw recording
│   ├── execution_agent.py         # npx playwright test — run + parse pass/fail + DOM on failure
│   ├── fix_agent.py               # OpenAI → fixed .spec.js, or decide to give up
│   ├── report_agent.py            # Build + save the recording run report
│   └── recording_graph.py         # LangGraph StateGraph wiring the 5 agents above
│
├── models/
│   ├── report.py                  # TestGapReport dataclasses + JSON/MD serialisation
│   └── recording_report.py        # RecordingRunReport dataclasses + JSON/MD serialisation
│
├── utils/
│   ├── file_parser.py             # File filtering, language detection, CodeFile
│   └── code_chunker.py            # AST-based Python chunking + text splitting for TS
│
├── playwright_runner/              # Node.js project used by the `record` pipeline
│   ├── package.json                #   @playwright/test devDependency
│   ├── playwright.config.js        #   testDir: "./tests"
│   └── tests/                      #   generated .spec.js working copies (overwritten per attempt)
│
├── tests/                          # pytest unit tests (94 passing, no API/browser calls needed)
│   ├── test_ingestion.py
│   ├── test_embedding.py
│   ├── test_analysis.py
│   ├── test_report.py
│   ├── test_gen.py
│   ├── test_execution_agent.py     # JSON-reporter-parsing logic
│   └── test_recording_graph.py     # Conditional routing + scenario-name slugging
│
└── outputs/                        # gitignored — created at runtime
    ├── test_gap_report.json
    ├── test_gap_report.md
    ├── generated_tests/            # Generated .spec.ts and .py files (analyze/generate)
    ├── recorded_tests/             # Final .spec.js + run reports (record)
    └── chroma_db/                  # ChromaDB vector store
```

---

## Example Output

After running on `fastapi/full-stack-fastapi-template`:

### Test Gap Report

| | Found | Tested | Untested |
|---|---|---|---|
| API Endpoints | 19 | 17 | 2 |
| UI Pages | 10 | 1 | 9 |
| Test Gaps | 15 | — | 15 |

### Generated Files

| File | Covers |
|---|---|
| `login.spec.ts` | Login page — inputs, happy path, invalid creds, auth redirect |
| `recover-password.spec.ts` | Password recovery form — submit, invalid email |
| `reset-password.spec.ts` | Reset password — token validation, weak password, mismatch |
| `admin.spec.ts` | Admin user management — create, edit, delete, access control |
| `dashboard.spec.ts` | Dashboard — loads after login, shows welcome message |
| `items.spec.ts` | Items page — list, create, update, delete flows |
| `not-found.spec.ts` | 404 page — invalid routes render correctly |
| `utils-api.spec.ts` | API: health-check + test-email (auth, validation, happy path) |
| `test_security.py` | Python: `get_password_hash`, `verify_password` unit tests |
| `test_startup.py` | Python: database connection retry logic |

---

## Cost & Rate Limits

This project runs on the **OpenAI API** (pay-as-you-go — check current pricing at [openai.com/pricing](https://openai.com/api/pricing/)):

| Model | Used for |
|---|---|
| `text-embedding-3-small` | Embedding code chunks (batched — one API call per 50 chunks) |
| `gpt-4o-mini` | Analysis, test generation, spec generation, and self-healing fixes |

The embedding agent includes automatic **rate-limit detection and retry** on 429 responses. A partial embedding run is resumed from where it left off.

---

## Running Tests

```bash
pytest tests/ -v
# 94 passed — no API/browser calls required
```

---

## Resume / LinkedIn Description

> **Agentic RAG Test Generator** | Python · OpenAI API · ChromaDB · Playwright · GitHub API · LangGraph
>
> Built an end-to-end agentic AI system that ingests any GitHub repository using RAG (Retrieval-Augmented Generation), analyzes code coverage gaps using OpenAI, and autonomously generates Playwright test suites. The system clones the target repo, embeds 586 code chunks into ChromaDB using `text-embedding-3-small`, identifies 15 test gaps across 19 API endpoints and 10 UI pages, generates valid Playwright `.spec.ts` and pytest files, and opens a Pull Request — with zero human intervention. Demonstrated on the FastAPI full-stack template, generating 12 test files covering authentication, CRUD operations, UI flows, and security functions. A second LangGraph-based pipeline records live browser scenarios and self-heals generated Playwright tests in a loop — using the failing page's actual DOM, not just the error message — until they pass.
