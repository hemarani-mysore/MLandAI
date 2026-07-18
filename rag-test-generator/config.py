"""
config.py — Central configuration for RAG Test Generator
=========================================================
All settings are loaded from environment variables (.env file).
Import this module anywhere in the project to access settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file automatically when config is imported
load_dotenv()

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs/generated_tests")
CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DB_PATH", "outputs/chroma_db")
CLONE_DIR  = BASE_DIR / "outputs" / "cloned_repos"

# Recording pipeline (record → generate → execute → self-heal → report)
PLAYWRIGHT_RUNNER_DIR = BASE_DIR / "playwright_runner"
RECORDED_TESTS_DIR    = BASE_DIR / "outputs" / "recorded_tests"

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CLONE_DIR.mkdir(parents=True, exist_ok=True)
RECORDED_TESTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

# OpenAI — the project's single LLM provider (generation + embeddings), used by every agent.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────
# OPENAI MODEL SETTINGS
# ─────────────────────────────────────────────
OPENAI_MODEL           = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_TEMPERATURE     = 0.2   # Low temp = more consistent, less creative (good for code)

# ─────────────────────────────────────────────
# RECORDING PIPELINE SETTINGS
# ─────────────────────────────────────────────
MAX_FIX_ATTEMPTS          = 5    # hard safety cap, independent of the LLM's own give-up decision
PLAYWRIGHT_EXEC_TIMEOUT_S = 60   # per `npx playwright test` subprocess run

# ─────────────────────────────────────────────
# RAG / CHUNKING SETTINGS
# ─────────────────────────────────────────────
CHUNK_SIZE      = 1000   # Characters per chunk
CHUNK_OVERLAP   = 200    # Overlap between chunks (preserves context at boundaries)
TOP_K_RESULTS   = 5      # Number of chunks to retrieve per query
CHROMA_COLLECTION_NAME = "codebase"

# ─────────────────────────────────────────────
# FILE INGESTION SETTINGS
# ─────────────────────────────────────────────
# File extensions to include in analysis
SUPPORTED_EXTENSIONS = {
    ".py",   # Python
    ".ts",   # TypeScript
    ".tsx",  # TypeScript React
    ".js",   # JavaScript
    ".jsx",  # JavaScript React
}

# Directories to skip (not useful for test generation)
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".mypy_cache",
    "dist", "build", ".venv", "venv", "env", "migrations",
    ".pytest_cache", "coverage", ".next", "static"
}

# Skip files larger than this (bytes) — avoids huge generated files
MAX_FILE_SIZE_BYTES = 100_000  # 100KB

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def validate_config():
    """
    Check that required environment variables are set.
    Call this at startup to catch missing config early.
    """
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set. Get one at platform.openai.com")

    if not GITHUB_TOKEN:
        errors.append("GITHUB_TOKEN is not set. Generate at github.com/settings/tokens")

    if not GITHUB_USERNAME:
        errors.append("GITHUB_USERNAME is not set.")

    if errors:
        raise EnvironmentError(
            "\n\nMissing required environment variables:\n" +
            "\n".join(f"  ❌ {e}" for e in errors) +
            "\n\nCopy .env.example to .env and fill in your values.\n"
        )

    return True


def validate_recording_config():
    """
    Check that required environment variables for the `record` pipeline are set.
    Separate from validate_config() because the two pipelines have disjoint requirements:
    `analyze`/`generate` don't need OPENAI_API_KEY, and `record` doesn't need
    GITHUB_TOKEN/GITHUB_USERNAME.
    """
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "\n\nMissing required environment variable:\n"
            "  ❌ OPENAI_API_KEY is not set. Get one at platform.openai.com\n"
            "\nCopy env.example to .env and fill in your values.\n"
        )
    return True


if __name__ == "__main__":
    # Quick test: python config.py
    try:
        validate_config()
        print("✅ Config loaded successfully!")
        print(f"   OpenAI model     : {OPENAI_MODEL}")
        print(f"   Embedding model  : {OPENAI_EMBEDDING_MODEL}")
        print(f"   ChromaDB path    : {CHROMA_DIR}")
        print(f"   Output directory : {OUTPUT_DIR}")
        print(f"   Clone directory  : {CLONE_DIR}")
    except EnvironmentError as e:
        print(f"❌ Config error: {e}")
