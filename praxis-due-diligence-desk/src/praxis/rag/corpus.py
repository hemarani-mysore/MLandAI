"""The ``Corpus`` — dense vector store + BM25 index + reranker, behind one API.

dense (Qdrant) ─┐
                ├─▶ RRF fuse ─▶ rerank ─▶ top-k RetrievedChunk (with citations)
sparse (BM25) ──┘
"""

from __future__ import annotations

from functools import lru_cache

from praxis.config import Settings, get_settings
from praxis.rag.bm25 import BM25Index
from praxis.rag.embed import Embedder, get_embedder
from praxis.rag.fuse import rrf
from praxis.rag.models import Chunk, CorpusStats, RetrievedChunk
from praxis.rag.rerank import Reranker, get_reranker
from praxis.rag.vector_store import QdrantVectorStore

_DEFAULT_RERANKER: object = object()


class Corpus:
    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: QdrantVectorStore,
        bm25: BM25Index | None = None,
        reranker: Reranker | None | object = _DEFAULT_RERANKER,
    ) -> None:
        self.embedder = embedder
        self.vs = vector_store
        self.bm25 = bm25 if bm25 is not None else BM25Index()
        # not passed -> settings default; explicit None -> no reranking
        self.reranker: Reranker | None = (
            get_reranker() if reranker is _DEFAULT_RERANKER else reranker  # type: ignore[assignment]
        )
        self._chunks: dict[str, Chunk] = {}
        self.version = 0

    # --- write ---
    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        items = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.chunk_id] = chunk
            self.bm25.add(chunk.chunk_id, chunk.raw_text)
            items.append(
                (chunk.chunk_id, vector, {"source_id": chunk.source_id, "page": chunk.page})
            )
        self.vs.upsert(items)
        self.version += 1
        return len(chunks)

    # --- read ---
    def search(self, query: str, *, k: int = 6, prefetch: int = 24) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        query_vec = self.embedder.embed([query])[0]
        dense = self.vs.search(query_vec, prefetch)
        sparse = self.bm25.search(query, prefetch)
        dense_ids = [i for i, _ in dense]
        sparse_ids = [i for i, _ in sparse]

        fused = rrf([dense_ids, sparse_ids])
        fused_score = dict(fused)
        dense_rank = {i: r for r, i in enumerate(dense_ids)}
        sparse_rank = {i: r for r, i in enumerate(sparse_ids)}

        candidate_ids = [i for i, _ in fused if i in self._chunks][: max(prefetch, k)]
        candidates = [self._chunks[i] for i in candidate_ids]

        ranked: list[tuple[Chunk, float | None]]
        if self.reranker and candidates:
            ranked = list(self.reranker.rerank(query, candidates, k))
        else:
            ranked = [(c, None) for c in candidates[:k]]

        return [
            RetrievedChunk(
                chunk=chunk,
                score=round(fused_score.get(chunk.chunk_id, 0.0), 6),
                dense_rank=dense_rank.get(chunk.chunk_id),
                sparse_rank=sparse_rank.get(chunk.chunk_id),
                rerank_score=rerank_score,
            )
            for chunk, rerank_score in ranked
        ]

    def stats(self) -> CorpusStats:
        return CorpusStats(
            documents=len({c.source_id for c in self._chunks.values()}),
            chunks=len(self._chunks),
            vectors=self.vs.count(),
            version=self.version,
            embedder=self.embedder.name,
        )


def build_corpus(settings: Settings | None = None) -> Corpus:
    s = settings or get_settings()
    embedder = get_embedder(s)
    return Corpus(
        embedder=embedder,
        vector_store=QdrantVectorStore(
            collection=s.qdrant_collection, dim=embedder.dim, url=s.qdrant_url or None
        ),
        reranker=get_reranker(s),
    )


@lru_cache
def get_corpus() -> Corpus:
    """Process-wide corpus singleton (used by the API and CLI)."""
    return build_corpus()
