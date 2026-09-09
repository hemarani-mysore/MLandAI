"""Retrieval layer: ingest documents, search a hybrid (dense + BM25) corpus."""

from praxis.rag.corpus import Corpus, build_corpus, get_corpus
from praxis.rag.ingest import ingest_source
from praxis.rag.models import (
    Chunk,
    CorpusStats,
    IngestResult,
    RetrievedChunk,
    SourceDoc,
)
from praxis.rag.retrieve import clear_cache, search_corpus

__all__ = [
    "Corpus",
    "build_corpus",
    "get_corpus",
    "ingest_source",
    "search_corpus",
    "clear_cache",
    "Chunk",
    "CorpusStats",
    "IngestResult",
    "RetrievedChunk",
    "SourceDoc",
]
