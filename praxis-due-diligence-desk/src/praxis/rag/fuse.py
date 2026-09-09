"""Reciprocal Rank Fusion — merge the dense and sparse rankings."""

from __future__ import annotations

from collections.abc import Sequence


def rrf(rankings: Sequence[Sequence[str]], *, k: int = 60) -> list[tuple[str, float]]:
    """Combine ranked id lists. ``score(id) = sum 1 / (k + rank)`` across lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
