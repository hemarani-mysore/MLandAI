"""Ingestion pipeline: parse -> chunk -> (optional) contextualize -> index.

Synchronous today. It is written as a pure function over a ``Corpus`` so Phase 2
can wrap it in a background worker without changing the logic.
"""

from __future__ import annotations

from praxis.config import get_settings
from praxis.rag.chunk import chunk_document
from praxis.rag.contextual import contextualize
from praxis.rag.corpus import Corpus, get_corpus
from praxis.rag.models import IngestResult
from praxis.rag.parse import parse_source


def ingest_source(
    *,
    path: str | None = None,
    text: str | None = None,
    title: str | None = None,
    uri: str | None = None,
    corpus: Corpus | None = None,
    contextual: bool | None = None,
) -> IngestResult:
    settings = get_settings()
    c = corpus or get_corpus()

    doc = parse_source(path=path, text=text, title=title, uri=uri)
    chunks = chunk_document(doc, size=settings.chunk_size, overlap=settings.chunk_overlap)

    use_context = settings.contextual_chunks if contextual is None else contextual
    if use_context and chunks:
        chunks = contextualize(chunks, doc)

    c.add(chunks)
    return IngestResult(
        source_id=doc.source_id,
        title=doc.title,
        pages=len(doc.pages),
        chunks=len(chunks),
        corpus_version=c.version,
    )
