"""
Tests for the indexing-service.

Covers:
- IndexingSettings defaults
- QdrantIndexer: ensure_collection, upsert_chunks, collection_info (all mocked)
- ORM model instantiation
- HTTP endpoints: /index, /collections/{name}
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from raglab_common.exceptions import IndexingError
from raglab_common.models import ChunkModel, EmbeddingModel, IngestionStatus
from indexing.settings import IndexingSettings
from indexing.qdrant_client import QdrantIndexer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(index: int = 0, doc_id: str = "doc-001") -> ChunkModel:
    return ChunkModel(
        chunk_id=str(uuid.uuid4()),
        doc_id=doc_id,
        text=f"Chunk text number {index}. It contains some information.",
        chunk_index=index,
        token_count=10,
        metadata={"chunker": "text"},
    )


def make_embedding(chunk_id: str) -> EmbeddingModel:
    return EmbeddingModel(
        chunk_id=chunk_id,
        doc_id="doc-001",
        vector=[0.1] * 1536,
        model="text-embedding-3-small",
        dimensions=1536,
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestIndexingSettings:
    def test_defaults(self):
        s = IndexingSettings()
        assert s.service_name == "indexing"
        assert s.port == 8003
        assert s.qdrant_vector_size == 1536
        assert s.qdrant_distance == "Cosine"
        assert s.qdrant_hnsw_m == 16


# ---------------------------------------------------------------------------
# QdrantIndexer
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_qdrant_client():
    """Return a QdrantIndexer with a fully mocked internal _client."""
    with patch("indexing.qdrant_client.QdrantClient") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        indexer = QdrantIndexer(host="localhost", port=6333, vector_size=1536)
        return indexer, mock_client


class TestQdrantIndexerEnsureCollection:
    def test_creates_collection_when_not_exists(self, mock_qdrant_client):
        indexer, client = mock_qdrant_client
        client.get_collections.return_value = MagicMock(collections=[])
        created = indexer.ensure_collection("test-col")
        assert created is True
        client.create_collection.assert_called_once()

    def test_skips_creation_when_exists(self, mock_qdrant_client):
        indexer, client = mock_qdrant_client
        existing = MagicMock()
        existing.name = "test-col"
        client.get_collections.return_value = MagicMock(collections=[existing])
        created = indexer.ensure_collection("test-col")
        assert created is False
        client.create_collection.assert_not_called()

    def test_raises_indexing_error_on_failure(self, mock_qdrant_client):
        indexer, client = mock_qdrant_client
        client.get_collections.side_effect = Exception("network error")
        with pytest.raises(IndexingError):
            indexer.ensure_collection("bad-col")


class TestQdrantIndexerUpsert:
    def test_upsert_calls_qdrant(self, mock_qdrant_client):
        indexer, client = mock_qdrant_client
        chunks = [make_chunk(i) for i in range(3)]
        embeddings = [make_embedding(c.chunk_id) for c in chunks]
        count = indexer.upsert_chunks("test-col", chunks, embeddings)
        assert count == 3
        client.upsert.assert_called_once()

    def test_mismatch_raises_indexing_error(self, mock_qdrant_client):
        indexer, _ = mock_qdrant_client
        chunks = [make_chunk(0)]
        embeddings = []
        with pytest.raises(IndexingError, match="mismatch"):
            indexer.upsert_chunks("test-col", chunks, embeddings)

    def test_qdrant_error_raises_indexing_error(self, mock_qdrant_client):
        indexer, client = mock_qdrant_client
        client.upsert.side_effect = Exception("qdrant down")
        chunks = [make_chunk(0)]
        embeddings = [make_embedding(chunks[0].chunk_id)]
        with pytest.raises(IndexingError, match="upsert failed"):
            indexer.upsert_chunks("test-col", chunks, embeddings)

    def test_payload_contains_chunk_fields(self, mock_qdrant_client):
        from qdrant_client.models import PointStruct
        indexer, client = mock_qdrant_client
        chunk = make_chunk(0)
        emb = make_embedding(chunk.chunk_id)
        indexer.upsert_chunks("test-col", [chunk], [emb])
        call_kwargs = client.upsert.call_args[1]
        point = call_kwargs["points"][0]
        assert point.payload["chunk_id"] == chunk.chunk_id
        assert point.payload["doc_id"] == chunk.doc_id
        assert point.payload["text"] == chunk.text


class TestQdrantIndexerCollectionInfo:
    def test_returns_info_dict(self, mock_qdrant_client):
        indexer, client = mock_qdrant_client
        mock_info = MagicMock()
        mock_info.vectors_count = 42
        mock_info.indexed_vectors_count = 42
        mock_info.status = "green"
        client.get_collection.return_value = mock_info
        info = indexer.collection_info("test-col")
        assert info["vectors_count"] == 42
        assert info["name"] == "test-col"

    def test_raises_on_missing_collection(self, mock_qdrant_client):
        indexer, client = mock_qdrant_client
        client.get_collection.side_effect = Exception("not found")
        with pytest.raises(IndexingError):
            indexer.collection_info("nonexistent")


# ---------------------------------------------------------------------------
# ORM models (instantiation only — no DB needed)
# ---------------------------------------------------------------------------


class TestORMModels:
    def test_document_record_instantiation(self):
        from indexing.db.models import DocumentRecord
        doc = DocumentRecord(
            doc_id="doc-001",
            filename="test.txt",
            content_type="text/plain",
            storage_path="/tmp/test.txt",
            collection="raglab",
            chunker_type="text",
        )
        assert doc.doc_id == "doc-001"
        assert doc.status == IngestionStatus.PENDING.value or doc.status is None  # default triggers on flush

    def test_chunk_record_instantiation(self):
        from indexing.db.models import ChunkRecord
        cr = ChunkRecord(
            chunk_id=str(uuid.uuid4()),
            doc_id="doc-001",
            collection="raglab",
            chunk_index=0,
            token_count=15,
            text_preview="This is the chunk text preview.",
        )
        assert cr.chunk_index == 0
        assert cr.collection == "raglab"


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def indexing_client():
    """TestClient with mocked Qdrant in app.state."""
    from indexing.main import app

    mock_qdrant = MagicMock()
    mock_qdrant.ensure_collection.return_value = False
    mock_qdrant.upsert_chunks.return_value = 2
    mock_qdrant.collection_info.return_value = {
        "name": "raglab",
        "vectors_count": 10,
        "indexed_vectors_count": 10,
        "status": "green",
    }

    app.state.qdrant = mock_qdrant
    app.state.session_factory = None  # skip Postgres in tests
    return TestClient(app)


class TestIndexingEndpoints:
    def test_health_returns_ok_with_qdrant(self, indexing_client):
        r = indexing_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["dependencies"]["qdrant"] == "ok"

    def test_root_returns_service_info(self, indexing_client):
        r = indexing_client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "indexing"

    def test_index_chunks_success(self, indexing_client):
        chunk = make_chunk(0)
        emb = make_embedding(chunk.chunk_id)
        payload = {
            "collection": "raglab",
            "doc_id": "doc-001",
            "filename": "test.txt",
            "chunks": [chunk.model_dump(mode="json")],
            "embeddings": [emb.model_dump(mode="json")],
        }
        r = indexing_client.post("/index", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["chunks_indexed"] == 2  # mock returns 2
        assert body["status"] == IngestionStatus.COMPLETED.value

    def test_index_mismatch_returns_422(self, indexing_client):
        chunk = make_chunk(0)
        payload = {
            "collection": "raglab",
            "doc_id": "doc-001",
            "chunks": [chunk.model_dump(mode="json")],
            "embeddings": [],  # mismatch
        }
        r = indexing_client.post("/index", json=payload)
        assert r.status_code == 422

    def test_collection_info_endpoint(self, indexing_client):
        r = indexing_client.get("/collections/raglab")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "raglab"
        assert body["vectors_count"] == 10

    def test_ensure_collection_endpoint(self, indexing_client):
        r = indexing_client.post("/collections/new-col/ensure")
        assert r.status_code == 200
        assert "created" in r.json()
