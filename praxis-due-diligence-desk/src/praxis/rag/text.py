"""Shared text utilities for retrieval."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")
# fmt: off
_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
    "with", "as", "from", "into", "per", "vs",
})
# fmt: on


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, length > 1, minus a tiny stop list."""
    return [t for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in _STOP]


def slugify(text: str, *, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "doc"
