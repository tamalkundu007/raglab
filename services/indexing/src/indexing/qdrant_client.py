"""
Qdrant client wrapper for the indexing-service.
QdrantClient imported at module level for patchability in tests.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, VectorParams

from raglab_common.exceptions import IndexingError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel, EmbeddingModel

log = get_logger(__name__)

_DISTANCE_MAP = {
    "Cosine": Distance.COSINE,
    "Dot": Distance.DOT,
    "Euclid": Distance.EUCLID,
}


class QdrantIndexer:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        vector_size: int = 1536,
        distance: str = "Cosine",
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
        on_disk_payload: bool = True,
    ) -> None:
        self._client = QdrantClient(host=host, port=port)
        self._vector_size = vector_size
        self._distance = distance
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construct = hnsw_ef_construct
        self._on_disk_payload = on_disk_payload
        log.info("qdrant.client_created", host=host, port=port)

    def ensure_collection(self, collection_name: str) -> bool:
        try:
            existing = [c.name for c in self._client.get_collections().collections]
            if collection_name in existing:
                log.info("qdrant.collection_exists", collection=collection_name)
                return False
            distance = _DISTANCE_MAP.get(self._distance, Distance.COSINE)
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self._vector_size, distance=distance, on_disk=self._on_disk_payload),
                hnsw_config=HnswConfigDiff(m=self._hnsw_m, ef_construct=self._hnsw_ef_construct, on_disk=self._on_disk_payload),
                on_disk_payload=self._on_disk_payload,
            )
            log.info("qdrant.collection_created", collection=collection_name, vector_size=self._vector_size)
            return True
        except Exception as exc:
            raise IndexingError(f"Failed to ensure collection '{collection_name}': {exc}") from exc

    def upsert_chunks(self, collection_name: str, chunks: list[ChunkModel], embeddings: list[EmbeddingModel]) -> int:
        if len(chunks) != len(embeddings):
            raise IndexingError(f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch.")
        points = []
        for chunk, emb in zip(chunks, embeddings):
            payload: dict[str, Any] = {
                "chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id,
                "text": chunk.text, "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count, **chunk.metadata,
            }
            points.append(PointStruct(id=chunk.chunk_id, vector=emb.vector, payload=payload))
        try:
            self._client.upsert(collection_name=collection_name, points=points, wait=True)
            log.info("qdrant.upserted", collection=collection_name, count=len(points))
            return len(points)
        except Exception as exc:
            raise IndexingError(f"Qdrant upsert failed: {exc}") from exc

    def collection_info(self, collection_name: str) -> dict[str, Any]:
        try:
            info = self._client.get_collection(collection_name)
            return {"name": collection_name, "vectors_count": info.vectors_count, "indexed_vectors_count": info.indexed_vectors_count, "status": str(info.status)}
        except Exception as exc:
            raise IndexingError(f"Failed to get collection info for '{collection_name}': {exc}") from exc

    def delete_collection(self, collection_name: str) -> None:
        try:
            self._client.delete_collection(collection_name)
            log.info("qdrant.collection_deleted", collection=collection_name)
        except Exception as exc:
            raise IndexingError(f"Failed to delete collection '{collection_name}': {exc}") from exc
