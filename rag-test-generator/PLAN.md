# 🧪 Agentic RAG Test Generator — Implementation Plan

## Project Overview

An agentic AI system that:
1. Takes any public GitHub repo URL as input
2. Ingests and understands the codebase using RAG
3. Identifies untested or undertested code paths
4. Auto-generates Playwright test stubs
5. Pushes the generated tests back to GitHub

**Target repo:** [fastapi/full-stack-fastapi-template](https://github.com/fastapi/full-stack-fastapi-template)  
**LLM:** Google Gemini API (free tier)  
**Stack:** Python, LangChain, ChromaDB, Gemini, Playwright, GitHub API

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
│              GitHub Repo URL                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               PHASE 1: INGESTION AGENT                  │
│  • Clone repo                                           │
│  • Parse code files (Python, TS, JS)                   │
│  • Chunk code into meaningful segments                  │
│  • Embed chunks → ChromaDB vector store                 │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│               PHASE 2: ANALYSIS AGENT                   │
│  • RAG pipeline: query vector store                     │
│  • Identify all API endpoints                           │
│  • Identify all UI components/pages                     │
│  • Find functions with no test coverage                 │
│  • Generate test gap report                             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│             PHASE 3: TEST GENERATION AGENT              │
│  • Generate Playwright test stubs per endpoint/page     │
│  • Cover: happy path, auth, error cases                 │
│  • Structure tests in proper Playwright format          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              PHASE 4: GITHUB PUSH AGENT                 │
│  • Create new branch in target repo                     │
│  • Commit generated test files                          │
│  • Create Pull Request with test summary                │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| LLM | Google Gemini 1.5 Flash | Code analysis + test generation (free) |
| Embeddings | Google text-embedding-004 | Convert code to vectors (free) |
| Vector Store | ChromaDB | Store + retrieve code chunks |
| RAG Framework | LangChain | Pipeline orchestration |
| Code Parsing | AST (Python) + Tree-sitter | Extract functions, classes, endpoints |
| Test Framework | Playwright (Python) | Generated test format |
| GitHub Integration | PyGitHub + GitPython | Clone repos, push tests, create PRs |
| CLI | Rich + Typer | Beautiful terminal interface |
| Config | python-dotenv | Environment management |

---

## Implementation Chunks

---

### ✅ CHUNK 1: Project Setup & Structure (Day 1)
**Goal:** Scaffold the project, set up dependencies, verify Gemini API works

**Tasks:**
- [ ] Create folder structure
- [ ] Set up `requirements.txt`
- [ ] Create `.env` template
- [ ] Test Gemini API connection
- [ ] Test ChromaDB setup
- [ ] Write `config.py` for all settings

**Deliverable:** Running `python main.py --help` shows CLI options

**Files:**
```
rag-test-generator/
├── main.py              # CLI entry point
├── config.py            # All settings and constants
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
└── README.md            # Project overview
```

---

### ✅ CHUNK 2: GitHub Ingestion Agent (Day 2-3)
**Goal:** Clone any public GitHub repo and extract all code files

**Tasks:**
- [ ] Clone repo using GitPython
- [ ] Walk directory tree, filter relevant files (`.py`, `.ts`, `.tsx`, `.js`)
- [ ] Skip non-code files (images, lock files, migrations)
- [ ] Extract file content + metadata (path, language, size)
- [ ] Handle large repos gracefully (skip files > 100KB)
- [ ] Write unit tests for the ingestion agent

**Deliverable:** Given a GitHub URL → returns list of parsed code files with content

**Files:**
```
agents/
└── ingestion_agent.py   # Clone + parse repo
utils/
└── file_parser.py       # File filtering + reading
```

---

### ✅ CHUNK 3: Code Chunking & Embedding (Day 3-4)
**Goal:** Convert code files into searchable vector embeddings in ChromaDB

**Tasks:**
- [ ] Split code files into meaningful chunks (by function/class, not arbitrary tokens)
- [ ] Use AST parser to extract Python functions and classes as natural chunk boundaries
- [ ] Embed chunks using Google text-embedding-004
- [ ] Store in ChromaDB with metadata (file path, language, chunk type)
- [ ] Build retrieval function: given a query → return top-k relevant chunks
- [ ] Test retrieval quality with sample queries

**Deliverable:** Query "authentication endpoints" → returns relevant FastAPI auth code

**Files:**
```
agents/
└── embedding_agent.py   # Chunk + embed + store
utils/
└── code_chunker.py      # AST-based smart chunking
data/
└── chroma_db/           # Vector store (gitignored)
```

---

### ✅ CHUNK 4: RAG Analysis Agent (Day 5-6)
**Goal:** Use RAG to analyze the codebase and identify test gaps

**Tasks:**
- [ ] Build RAG chain using LangChain + Gemini + ChromaDB
- [ ] Query 1: "List all API endpoints and their HTTP methods"
- [ ] Query 2: "List all UI pages and their user interactions"
- [ ] Query 3: "Which functions have no corresponding test files?"
- [ ] Query 4: "What are the authentication and authorization flows?"
- [ ] Aggregate results into a structured `TestGapReport`
- [ ] Save report as JSON + human-readable markdown

**Deliverable:** `test_gap_report.md` listing all endpoints, pages, and gaps found

**Files:**
```
agents/
└── analysis_agent.py    # RAG queries + gap detection
models/
└── report.py            # TestGapReport data model
outputs/
└── test_gap_report.md   # Generated report
```

---

### ✅ CHUNK 5: Playwright Test Generation Agent (Day 7-9)
**Goal:** Auto-generate Playwright test stubs for every identified gap

**Tasks:**
- [ ] Design prompt template for Playwright test generation
- [ ] Generate tests for API endpoints (using Playwright's request API)
- [ ] Generate tests for UI pages (using Playwright's browser automation)
- [ ] Cover 3 scenarios per endpoint: happy path, auth failure, invalid input
- [ ] Structure output as valid `.spec.ts` Playwright test files
- [ ] Validate generated test syntax
- [ ] Group tests by feature (auth, users, items, etc.)

**Deliverable:** Valid Playwright `.spec.ts` files for every identified gap

**Files:**
```
agents/
└── test_gen_agent.py    # Test generation using Gemini
templates/
├── api_test.jinja2      # Template for API tests
└── ui_test.jinja2       # Template for UI tests
outputs/
└── generated_tests/     # Generated .spec.ts files
    ├── auth.spec.ts
    ├── users.spec.ts
    └── items.spec.ts
```

---

### ✅ CHUNK 6: GitHub Push Agent (Day 10-11)
**Goal:** Push generated tests to GitHub and create a Pull Request

**Tasks:**
- [ ] Fork or branch the target repo using GitHub API
- [ ] Create new branch: `ai-generated-tests/<timestamp>`
- [ ] Commit all generated test files
- [ ] Create Pull Request with:
  - Summary of test gaps found
  - List of generated test files
  - Coverage improvement estimate
- [ ] Handle GitHub API rate limits gracefully

**Deliverable:** Open PR in target repo with all generated tests

**Files:**
```
agents/
└── github_agent.py      # GitHub API: branch, commit, PR
```

---

### ✅ CHUNK 7: Orchestration & CLI (Day 12-13)
**Goal:** Wire all agents together into a single end-to-end pipeline

**Tasks:**
- [ ] Build `main.py` CLI with Typer
- [ ] Command: `python main.py analyze --repo <url>`
- [ ] Command: `python main.py generate --repo <url> --push`
- [ ] Add `--dry-run` flag (analyze + generate but don't push)
- [ ] Add progress bars and rich terminal output
- [ ] Add error handling and retry logic
- [ ] Log all steps to file for debugging

**Deliverable:** Single command runs the full pipeline end to end

```bash
python main.py generate \
  --repo https://github.com/fastapi/full-stack-fastapi-template \
  --push \
  --output ./my-tests
```

---

### ✅ CHUNK 8: Testing, README & GitHub Polish (Day 14-15)
**Goal:** Make the project portfolio-ready

**Tasks:**
- [ ] Write unit tests for each agent (pytest)
- [ ] Write integration test for full pipeline
- [ ] Create detailed README with:
  - Architecture diagram
  - Setup instructions
  - Demo GIF / screenshots
  - Example output
  - Tech stack badges
- [ ] Add `CONTRIBUTING.md`
- [ ] Tag release `v1.0.0`
- [ ] Record a 2-minute demo video (optional but powerful)

**Deliverable:** Professional GitHub repo ready to show interviewers

---

## Folder Structure (Final)

```
rag-test-generator/
├── main.py                    # CLI entry point
├── config.py                  # Settings
├── requirements.txt           # Dependencies
├── .env.example               # Env template
├── README.md                  # Project docs
├── PLAN.md                    # This file
│
├── agents/
│   ├── __init__.py
│   ├── ingestion_agent.py     # Clone + parse repo
│   ├── embedding_agent.py     # Chunk + embed + ChromaDB
│   ├── analysis_agent.py      # RAG + gap detection
│   ├── test_gen_agent.py      # Playwright test generation
│   └── github_agent.py        # Push to GitHub + PR
│
├── utils/
│   ├── __init__.py
│   ├── file_parser.py         # File filtering
│   └── code_chunker.py        # AST chunking
│
├── models/
│   ├── __init__.py
│   └── report.py              # Data models
│
├── templates/
│   ├── api_test.jinja2        # API test template
│   └── ui_test.jinja2         # UI test template
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_embedding.py
│   ├── test_analysis.py
│   └── test_gen.py
│
└── outputs/                   # gitignored
    ├── test_gap_report.md
    └── generated_tests/
```

---

## Resume / LinkedIn Description

> **Agentic RAG Test Generator** | Python, LangChain, Gemini API, ChromaDB, Playwright, GitHub API
>
> Built an end-to-end agentic AI system that ingests any GitHub repository using RAG (Retrieval Augmented Generation), analyzes code coverage gaps using Google Gemini, and autonomously generates Playwright test suites. The system clones the target repo, embeds code using vector embeddings, identifies untested API endpoints and UI flows, generates valid Playwright test files, and opens a Pull Request — with zero human intervention. Demonstrated on the FastAPI full-stack template, generating 15+ test stubs covering authentication, CRUD operations, and UI interactions.

---

## Timeline Summary

| Week | Chunks | Focus |
|---|---|---|
| Week 1 | 1, 2, 3 | Setup + Ingestion + Embeddings |
| Week 2 | 4, 5 | RAG Analysis + Test Generation |
| Week 3 | 6, 7, 8 | GitHub Push + CLI + Polish |

---

## Success Criteria

- [ ] Given any public GitHub URL → generates valid Playwright tests in < 5 minutes
- [ ] Generated tests cover all API endpoints found in the repo
- [ ] PR is created automatically with a meaningful description
- [ ] Project has a clean README with setup instructions
- [ ] Code is modular — each agent is independently testable
