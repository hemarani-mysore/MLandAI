# RAG Test Generator

An agentic AI system that ingests any public GitHub repository, identifies untested code paths using RAG (Retrieval-Augmented Generation), auto-generates Playwright and pytest test stubs, and opens a Pull Request — with zero human intervention.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![Gemini](https://img.shields.io/badge/Gemini-1.5--flash-orange?logo=google)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-purple)
![Playwright](https://img.shields.io/badge/Playwright-TypeScript-green?logo=playwright)
![pytest](https://img.shields.io/badge/pytest-79%20passing-brightgreen?logo=pytest)

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
│  2. Embedding Agent     │  Chunk code → embed with Gemini → store in ChromaDB
└──────────┬──────────────┘
           │
    ▼
┌─────────────────────────┐
│  3. Analysis Agent      │  RAG queries → find endpoints, pages, auth flows, gaps
└──────────┬──────────────┘
           │  test_gap_report.json / .md
    ▼
┌─────────────────────────┐
│  4. Test Gen Agent      │  Gemini generates Playwright .spec.ts + pytest files
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

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM & Embeddings | Google Gemini API (`gemini-1.5-flash`, `gemini-embedding-2`) |
| Vector Store | ChromaDB (local, persistent) |
| Code Parsing | Python AST + character-level chunking for TypeScript |
| Test Framework | Playwright (TypeScript) + pytest (Python) |
| GitHub Integration | PyGitHub — fork, branch, commit, PR via API |
| CLI | Typer + Rich |
| Config | python-dotenv |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/hemarani-mysore/rag-test-generator
cd rag-test-generator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp env.example .env
# Fill in your keys:
#   GOOGLE_API_KEY   — from aistudio.google.com
#   GITHUB_TOKEN     — from github.com/settings/tokens (repo scope)
#   GITHUB_USERNAME  — your GitHub username
```

### 3. Verify setup

```bash
python main.py verify-setup
```

### 4. Run the full pipeline

```bash
python main.py generate \
  --repo https://github.com/fastapi/full-stack-fastapi-template \
  --push
```

That's it. A PR with generated tests opens automatically.

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
```

---

## Project Structure

```
rag-test-generator/
├── main.py                    # CLI entry point
├── config.py                  # All settings (loaded from .env)
├── requirements.txt
├── pytest.ini
├── env.example
│
├── agents/
│   ├── ingestion_agent.py     # Clone + parse GitHub repo
│   ├── embedding_agent.py     # Chunk + embed + ChromaDB (with rate-limit retry)
│   ├── analysis_agent.py      # RAG queries → TestGapReport
│   ├── test_gen_agent.py      # Gemini → Playwright + pytest files
│   └── github_agent.py        # Fork + branch + commit + PR
│
├── models/
│   └── report.py              # TestGapReport dataclasses + JSON/MD serialisation
│
├── utils/
│   ├── file_parser.py         # File filtering, language detection, CodeFile
│   └── code_chunker.py        # AST-based Python chunking + text splitting for TS
│
├── tests/                     # pytest unit tests (79 passing, no API calls needed)
│   ├── test_ingestion.py
│   ├── test_embedding.py
│   ├── test_analysis.py
│   ├── test_report.py
│   └── test_gen.py
│
└── outputs/                   # gitignored — created at runtime
    ├── test_gap_report.json
    ├── test_gap_report.md
    ├── generated_tests/       # Generated .spec.ts and .py files
    └── chroma_db/             # ChromaDB vector store
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

## Free Tier Limits

This project runs entirely on the **Gemini API free tier**:

| Model | Limit | Used for |
|---|---|---|
| `gemini-embedding-2` | 100 req/min | Embedding 586 code chunks |
| `gemini-1.5-flash` | 1,500 req/day | Analysis + test generation |

The embedding agent includes automatic **rate-limit detection and retry** (reads the `retry in Xs` from the 429 response). A partial embedding run is resumed from where it left off.

---

## Running Tests

```bash
pytest tests/ -v
# 79 passed in 0.85s — no API calls required
```

---

## Resume / LinkedIn Description

> **Agentic RAG Test Generator** | Python · Gemini API · ChromaDB · Playwright · GitHub API
>
> Built an end-to-end agentic AI system that ingests any GitHub repository using RAG (Retrieval-Augmented Generation), analyzes code coverage gaps using Google Gemini, and autonomously generates Playwright test suites. The system clones the target repo, embeds 586 code chunks into ChromaDB using `gemini-embedding-2`, identifies 15 test gaps across 19 API endpoints and 10 UI pages, generates valid Playwright `.spec.ts` and pytest files, and opens a Pull Request — with zero human intervention. Demonstrated on the FastAPI full-stack template, generating 12 test files covering authentication, CRUD operations, UI flows, and security functions.
