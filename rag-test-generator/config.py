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

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CLONE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")

# ─────────────────────────────────────────────
# GEMINI MODEL SETTINGS
# ─────────────────────────────────────────────
# gemini-1.5-flash = fast + free tier friendly
# gemini-1.5-pro   = more powerful but uses more quota
GEMINI_MODEL           = "gemini-1.5-flash"   # 1500 req/day on free tier (vs 20 for 2.5-flash)
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-2"
GEMINI_TEMPERATURE     = 0.2    # Low temp = more consistent, less creative (good for code)
GEMINI_MAX_TOKENS      = 8192   # Max output tokens per request

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

    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY is not set. Get one at aistudio.google.com")

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


if __name__ == "__main__":
    # Quick test: python config.py
    try:
        validate_config()
        print("✅ Config loaded successfully!")
        print(f"   Gemini model     : {GEMINI_MODEL}")
        print(f"   Embedding model  : {GEMINI_EMBEDDING_MODEL}")
        print(f"   ChromaDB path    : {CHROMA_DIR}")
        print(f"   Output directory : {OUTPUT_DIR}")
        print(f"   Clone directory  : {CLONE_DIR}")
    except EnvironmentError as e:
        print(f"❌ Config error: {e}")
