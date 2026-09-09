"""A small in-memory BM25 index — the sparse/lexical retrieval channel.

Kept separate from the vector store on purpose: it is transparent, has no model
downloads, and runs identically in tests and in production. A linear scan is
fine at the corpus sizes this desk works with; swap in a real inverted index or
Qdrant sparse vectors if a corpus ever outgrows it.
"""

from __future__ import annotations

import math
from collections import Counter

from praxis.rag.text import tokenize


class BM25Index:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: dict[str, list[str]] = {}
        self._df: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def _avgdl(self) -> float:
        return sum(len(t) for t in self._docs.values()) / len(self._docs) if self._docs else 0.0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._docs:
            self.remove(doc_id)
        tokens = tokenize(text)
        self._docs[doc_id] = tokens
        for term in set(tokens):
            self._df[term] += 1

    def remove(self, doc_id: str) -> None:
        tokens = self._docs.pop(doc_id, None)
        if tokens is None:
            return
        for term in set(tokens):
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        return math.log(1 + (len(self._docs) - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = [t for t in tokenize(query) if t in self._df]
        if not terms:
            return []
        avgdl = self._avgdl or 1.0
        scored: list[tuple[str, float]] = []
        for doc_id, tokens in self._docs.items():
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score += self._idf(term) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
