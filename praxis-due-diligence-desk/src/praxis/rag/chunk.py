"""Split a ``SourceDoc`` into overlapping, metadata-rich ``Chunk``s."""

from __future__ import annotations

from praxis.rag.models import Chunk, SourceDoc

_BREAKS = ("\n\n", "\n", ". ", " ")


def _split(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(b) for b in _BREAKS)
            if cut > size * 0.4:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(doc: SourceDoc, *, size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for page in doc.pages:
        for piece in _split(page.text, size, overlap):
            locator = f"p.{page.number}" if doc.kind == "pdf" else f"chunk {ordinal + 1}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_id}::p{page.number}::{ordinal}",
                    source_id=doc.source_id,
                    title=doc.title,
                    uri=doc.uri,
                    page=page.number,
                    ordinal=ordinal,
                    locator=locator,
                    text=piece,
                    raw_text=piece,
                )
            )
            ordinal += 1
    return chunks
