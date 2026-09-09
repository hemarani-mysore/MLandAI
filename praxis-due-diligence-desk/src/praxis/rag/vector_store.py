"""Dense vector store — Qdrant, in-memory by default, a server when ``PRAXIS_QDRANT_URL`` is set."""

from __future__ import annotations

import uuid

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantVectorStore:
    def __init__(self, *, collection: str, dim: int, url: str | None = None) -> None:
        from qdrant_client import QdrantClient, models

        self._models = models
        self._client = QdrantClient(url=url) if url else QdrantClient(location=":memory:")
        self.collection = collection
        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    def upsert(self, items: list[tuple[str, list[float], dict]]) -> None:
        if not items:
            return
        points = [
            self._models.PointStruct(
                id=_point_id(chunk_id),
                vector=vector,
                payload={**payload, "chunk_id": chunk_id},
            )
            for chunk_id, vector, payload in items
        ]
        self._client.upsert(self.collection, points=points)

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        response = self._client.query_points(
            self.collection, query=vector, limit=k, with_payload=True
        )
        return [((p.payload or {})["chunk_id"], p.score) for p in response.points]

    def count(self) -> int:
        return self._client.count(self.collection).count
