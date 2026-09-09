"""Rerankers. ``LexicalReranker`` is deterministic and dependency-free (default);
``CrossEncoderReranker`` uses a sentence-transformers cross-encoder if installed
(``pip install 'praxis[rerank]'``)."""

from __future__ import annotations

from typing import Protocol

from praxis.config import Settings, get_settings
from praxis.rag.models import Chunk
from praxis.rag.text import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]: ...


class LexicalReranker:
    name = "lexical"

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        q = set(tokenize(query))
        scored: list[tuple[Chunk, float]] = []
        for chunk in chunks:
            toks = tokenize(chunk.raw_text)
            if not toks or not q:
                scored.append((chunk, 0.0))
                continue
            coverage = len(q & set(toks)) / len(q)
            density = sum(1 for t in toks if t in q) / len(toks)
            scored.append((chunk, round(0.7 * coverage + 0.3 * density, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        scores = self._model.predict([(query, c.raw_text) for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:top_k]]


def get_reranker(settings: Settings | None = None) -> Reranker | None:
    s = settings or get_settings()
    if s.reranker == "none":
        return None
    if s.reranker == "cross-encoder":
        try:
            return CrossEncoderReranker(s.reranker_model)
        except Exception:  # noqa: BLE001 - fall back rather than break retrieval
            return LexicalReranker()
    return LexicalReranker()
