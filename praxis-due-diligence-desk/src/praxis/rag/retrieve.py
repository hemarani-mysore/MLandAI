"""``search_corpus`` — the retrieval entry point used by the researcher node and
the API. Results are cached per ``(corpus, version, query, k)``."""

from __future__ import annotations

from collections import OrderedDict

from praxis.config import get_settings
from praxis.rag.corpus import Corpus, get_corpus
from praxis.rag.models import RetrievedChunk

_CACHE: OrderedDict[tuple, list[RetrievedChunk]] = OrderedDict()
_CACHE_MAX = 512


def search_corpus(
    query: str, *, k: int | None = None, corpus: Corpus | None = None
) -> list[RetrievedChunk]:
    settings = get_settings()
    c = corpus or get_corpus()
    top_k = k or settings.retrieval_k
    key = (id(c), c.version, query.strip().lower(), top_k)

    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return cached

    results = c.search(query, k=top_k, prefetch=settings.retrieval_prefetch)
    _CACHE[key] = results
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return results


def clear_cache() -> None:
    _CACHE.clear()
