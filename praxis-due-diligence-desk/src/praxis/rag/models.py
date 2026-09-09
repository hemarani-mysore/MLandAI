"""Typed contracts for the retrieval layer."""

from __future__ import annotations

from pydantic import BaseModel, Field

from praxis.schemas import Citation


class Page(BaseModel):
    number: int
    text: str


class SourceDoc(BaseModel):
    source_id: str
    title: str
    uri: str
    kind: str  # pdf | markdown | text | html
    pages: list[Page]

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class Chunk(BaseModel):
    chunk_id: str  # "<source_id>::p<page>::<ordinal>"
    source_id: str
    title: str
    uri: str
    page: int
    ordinal: int
    locator: str  # "p.12" for PDFs, "chunk 3" otherwise
    text: str  # what gets embedded (may carry a contextual prefix)
    raw_text: str  # the original chunk span

    def citation(self, quote: str | None = None) -> Citation:
        snippet = (quote or self.raw_text).strip().replace("\n", " ")
        return Citation(
            source_id=self.source_id,
            title=self.title,
            locator=self.locator,
            quote=snippet[:280],
        )


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = 0.0  # fused (RRF) score
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    @property
    def citation(self) -> Citation:
        return self.chunk.citation()


class IngestResult(BaseModel):
    source_id: str
    title: str
    pages: int
    chunks: int
    corpus_version: int


class CorpusStats(BaseModel):
    documents: int
    chunks: int
    vectors: int
    version: int
    embedder: str = Field(description="Name of the active embedding model")
