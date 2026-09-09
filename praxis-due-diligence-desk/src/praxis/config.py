"""Runtime configuration, loaded from the environment (prefix ``PRAXIS_``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["fake", "openai", "anthropic", "nebius"]
EmbeddingProvider = Literal["fake", "openai", "nebius"]
Reranker = Literal["lexical", "cross-encoder", "none"]

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1"


class Settings(BaseSettings):
    """Process configuration.

    Defaults are chosen so a fresh checkout runs end to end with no API keys and
    no network: ``llm_provider="fake"`` returns deterministic canned structured
    output, ``embedding_provider="fake"`` uses a hashing embedder, and the vector
    store runs in-memory.
    """

    model_config = SettingsConfigDict(env_prefix="PRAXIS_", env_file=".env", extra="ignore")

    # --- LLM ---
    llm_provider: Provider = "fake"
    strong_model: str = "openai:gpt-4o"
    fast_model: str = "openai:gpt-4o-mini"
    # Roles that run on the cheaper/faster tier (calibration + synthesis work).
    fast_roles: tuple[str, ...] = ("editor", "red_team")
    nebius_base_url: str = NEBIUS_BASE_URL

    # --- Retrieval / RAG ---
    embedding_provider: EmbeddingProvider = "fake"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 256  # hashing embedder dim; real models override at build
    qdrant_url: str = ""  # empty -> in-memory
    qdrant_collection: str = "praxis_corpus"
    chunk_size: int = 900
    chunk_overlap: int = 150
    contextual_chunks: bool = False  # 1-sentence LLM context per chunk (costs an LLM call each)
    retrieval_k: int = 6
    retrieval_prefetch: int = 24
    reranker: Reranker = "lexical"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Graph behaviour ---
    max_gap_loops: int = 1

    # --- Observability ---
    otel_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
