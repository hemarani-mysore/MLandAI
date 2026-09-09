"""Embedders. ``HashEmbedder`` is deterministic and dependency-free (default /
CI); ``LangChainEmbedder`` wraps OpenAI-compatible embedding APIs."""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np

from praxis.config import Settings, get_settings
from praxis.rag.text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Signed hashing-trick embedding. Captures lexical overlap well enough for
    tests and the deterministic eval; not a substitute for a real model."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = float(np.linalg.norm(v))
        return (v / norm).tolist() if norm else v.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


class LangChainEmbedder:
    def __init__(self, *, model: str, dim: int, base_url: str | None, api_key: str | None) -> None:
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict = {"model": model}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._e = OpenAIEmbeddings(**kwargs)
        self.name = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._e.embed_documents(texts)


def get_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    if s.embedding_provider == "fake":
        return HashEmbedder(s.embedding_dim)
    import os

    if s.embedding_provider == "nebius":
        return LangChainEmbedder(
            model=s.embedding_model,
            dim=s.embedding_dim,
            base_url=s.nebius_base_url,
            api_key=os.environ.get("NEBIUS_API_KEY"),
        )
    return LangChainEmbedder(
        model=s.embedding_model,
        dim=s.embedding_dim,
        base_url=None,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
