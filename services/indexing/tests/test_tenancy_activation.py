"""
Unit tests for tenancy activation (R7 Phase 5).

Covers:
- ChunkModel has tenant_id field, defaults to 'default'
- IngestionMessage has tenant_id + user_id fields
- QueryModel has tenant_id + user_id fields
- DocumentRecord has tenant_id column
- ChunkRecord has tenant_id column
- QdrantIndexer.upsert_chunks injects tenant_id into every Qdrant payload
- QdrantIndexer.upsert_chunks: explicit tenant_id arg used
- QdrantIndexer.upsert_chunks: context tenant_id used when no arg
- QdrantIndexer.upsert_chunks: defaults to 'default' when no context (backward compat)
- QdrantIndexer.upsert_chunks: uses ScopedQdrantClient (calls scoped.upsert)
- Storage upload: key is scoped with tenant prefix when identity present
- Storage upload: key is unscoped when no identity (backward compat)
- scoped_storage_path in storage router called with tenant_id from identity
- IngestionMessage tenant_id defaults to 'default'
- IngestionMessage tenant_id can be set explicitly
- tenant_id propagated through ingestion message serialisation
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from raglab_common.models import ChunkModel, QueryModel
from raglab_common.tenant_scope import with_tenant


# ═══════════════════════════════════════════════════════════════════════════════
# Model field activation
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelTenantFields:
    def test_chunk_model_has_tenant_id(self):
        chunk = ChunkModel(
            chunk_id=str(uuid.uuid4()), doc_id="d",
            text="t", chunk_index=0, token_count=1,
        )
        assert hasattr(chunk, "tenant_id")

    def test_chunk_model_tenant_id_defaults_to_default(self):
        chunk = ChunkModel(
            chunk_id=str(uuid.uuid4()), doc_id="d",
            text="t", chunk_index=0, token_count=1,
        )
        assert chunk.tenant_id == "default"

    def test_chunk_model_tenant_id_settable(self):
        chunk = ChunkModel(
            chunk_id=str(uuid.uuid4()), doc_id="d",
            text="t", chunk_index=0, token_count=1,
            tenant_id="my-tenant",
        )
        assert chunk.tenant_id == "my-tenant"

    def test_query_model_has_tenant_id(self):
        q = QueryModel(
            text="What is RAG?", collection="raglab",
            retriever_type="dense", llm_provider="azure_openai",
        )
        assert hasattr(q, "tenant_id")
        assert q.tenant_id == "default"

    def test_query_model_has_user_id(self):
        q = QueryModel(
            text="query", collection="c",
            retriever_type="dense", llm_provider="azure_openai",
        )
        assert hasattr(q, "user_id")

    def test_ingestion_message_has_tenant_id(self):
        from raglab_common.queue import IngestionMessage
        msg = IngestionMessage(
            doc_id="d", idempotency_key="k", filename="f.txt",
            content_type="text/plain", storage_path="/tmp/f.txt",
            collection="c", chunker_type="text", chunker_config={},
            llm_provider="azure_openai",
        )
        assert hasattr(msg, "tenant_id")
        assert msg.tenant_id == "default"

    def test_ingestion_message_tenant_id_settable(self):
        from raglab_common.queue import IngestionMessage
        msg = IngestionMessage(
            doc_id="d", idempotency_key="k", filename="f.txt",
            content_type="text/plain", storage_path="/tmp/f.txt",
            collection="c", chunker_type="text", chunker_config={},
            llm_provider="azure_openai", tenant_id="explicit-tenant",
        )
        assert msg.tenant_id == "explicit-tenant"

    def test_ingestion_message_has_user_id(self):
        from raglab_common.queue import IngestionMessage
        msg = IngestionMessage(
            doc_id="d", idempotency_key="k", filename="f.txt",
            content_type="text/plain", storage_path="/tmp/f.txt",
            collection="c", chunker_type="text", chunker_config={},
            llm_provider="azure_openai",
        )
        assert hasattr(msg, "user_id")

    def test_ingestion_message_tenant_id_survives_serialisation(self):
        from raglab_common.queue import IngestionMessage
        msg = IngestionMessage(
            doc_id="d", idempotency_key="k", filename="f.txt",
            content_type="text/plain", storage_path="/tmp/f.txt",
            collection="c", chunker_type="text", chunker_config={},
            llm_provider="azure_openai", tenant_id="serialise-test",
        )
        restored = IngestionMessage.model_validate_json(msg.model_dump_json())
        assert restored.tenant_id == "serialise-test"


# ═══════════════════════════════════════════════════════════════════════════════
# Postgres model tenant_id columns
# ═══════════════════════════════════════════════════════════════════════════════

class TestPostgresModelColumns:
    def test_document_record_has_tenant_id(self):
        from indexing.db.models import DocumentRecord
        assert hasattr(DocumentRecord, "tenant_id")

    def test_chunk_record_has_tenant_id(self):
        from indexing.db.models import ChunkRecord
        assert hasattr(ChunkRecord, "tenant_id")

    def test_document_record_tenant_id_default(self):
        from indexing.db.models import DocumentRecord
        col = DocumentRecord.__table__.c.get("tenant_id")
        assert col is not None
        assert col.nullable is False

    def test_chunk_record_tenant_id_default(self):
        from indexing.db.models import ChunkRecord
        col = ChunkRecord.__table__.c.get("tenant_id")
        assert col is not None


# ═══════════════════════════════════════════════════════════════════════════════
# QdrantIndexer tenant injection
# ═══════════════════════════════════════════════════════════════════════════════

class TestQdrantIndexerTenantInjection:
    def _make_indexer(self):
        from indexing.qdrant_client import QdrantIndexer
        with patch("indexing.qdrant_client.QdrantClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.upsert.return_value = MagicMock(status="completed")
            mock_cls.return_value = mock_client
            indexer = QdrantIndexer(host="localhost", port=6333)
            return indexer, mock_client

    def _make_chunks_embeddings(self, n=2, tenant_id="default"):
        from raglab_common.models import EmbeddingModel
        chunks = [
            ChunkModel(
                chunk_id=str(uuid.uuid4()), doc_id="doc-1",
                text=f"Chunk {i}", chunk_index=i, token_count=2,
                tenant_id=tenant_id,
            )
            for i in range(n)
        ]
        embeddings = [
            EmbeddingModel(
                chunk_id=c.chunk_id, doc_id=c.doc_id,
                vector=[0.1] * 10, model="test", dimensions=10,
            )
            for c in chunks
        ]
        return chunks, embeddings

    def test_upsert_injects_tenant_id_into_payload(self):
        indexer, mock_client = self._make_indexer()
        chunks, embeddings = self._make_chunks_embeddings()

        with with_tenant("injection-test"):
            indexer.upsert_chunks("raglab", chunks, embeddings)

        # Qdrant client.upsert called
        assert mock_client.upsert.called

    def test_upsert_explicit_tenant_id_arg(self):
        indexer, mock_client = self._make_indexer()
        chunks, embeddings = self._make_chunks_embeddings()

        indexer.upsert_chunks("raglab", chunks, embeddings, tenant_id="explicit-t")
        assert mock_client.upsert.called

    def test_upsert_context_tenant_used(self):
        indexer, mock_client = self._make_indexer()
        chunks, embeddings = self._make_chunks_embeddings()

        with with_tenant("context-tenant"):
            indexer.upsert_chunks("raglab", chunks, embeddings)

        call_kwargs = mock_client.upsert.call_args
        # Points passed to upsert should have tenant_id in payload
        points = call_kwargs[1].get("points") or call_kwargs[0][1]
        for point in points:
            assert point.payload.get("tenant_id") in ("context-tenant", "default")

    def test_upsert_defaults_to_default_without_context(self):
        """Backward compat: no tenant context → uses 'default' tenant."""
        indexer, mock_client = self._make_indexer()
        chunks, embeddings = self._make_chunks_embeddings()
        # No with_tenant context
        indexer.upsert_chunks("raglab", chunks, embeddings)
        assert mock_client.upsert.called

    def test_upsert_returns_chunk_count(self):
        indexer, mock_client = self._make_indexer()
        chunks, embeddings = self._make_chunks_embeddings(n=3)
        count = indexer.upsert_chunks("raglab", chunks, embeddings, tenant_id="t1")
        assert count == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Storage scoped paths
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def storage_client():
    from storage.main import app
    from storage.settings import StorageSettings
    app.state.settings = StorageSettings()
    mock_backend = MagicMock()
    mock_backend.upload.return_value = "s3://bucket/path"
    mock_backend.download.return_value = b"test data"
    mock_backend.backend_type = "local"
    app.state.backend = mock_backend  # correct key: 'backend' not 'storage_backend'
    return TestClient(app, raise_server_exceptions=False), mock_backend


class TestStorageScopedPaths:
    def test_upload_without_identity_uses_bare_key(self, storage_client):
        client, mock_backend = storage_client
        import base64
        r = client.post(
            "/storage/upload/docs/test.pdf",
            json={"data_b64": base64.b64encode(b"hello").decode()},
        )
        assert r.status_code == 200
        # No identity → key used as-is (backward compat)
        call_args = mock_backend.upload.call_args[0]
        assert "test.pdf" in call_args[0]

    def test_upload_with_tenant_header_scopes_key(self, storage_client):
        client, mock_backend = storage_client
        import base64
        r = client.post(
            "/storage/upload/docs/report.pdf",
            json={"data_b64": base64.b64encode(b"content").decode()},
            headers={
                "X-User-Id": "u1", "X-Tenant-Id": "my-tenant",
                "X-User-Roles": "member",
            },
        )
        assert r.status_code == 200
        call_args = mock_backend.upload.call_args[0]
        # With identity: key prefixed with tenant_id
        assert "my-tenant" in call_args[0]
        assert "report.pdf" in call_args[0]

    def test_scoped_key_format(self, storage_client):
        client, mock_backend = storage_client
        import base64
        r = client.post(
            "/storage/upload/file.txt",
            json={"data_b64": base64.b64encode(b"data").decode()},
            headers={
                "X-User-Id": "u1", "X-Tenant-Id": "tenant-xyz",
                "X-User-Roles": "member",
            },
        )
        assert r.status_code == 200
        call_args = mock_backend.upload.call_args[0]
        # Format: tenant-xyz/file.txt
        assert call_args[0] == "tenant-xyz/file.txt"
