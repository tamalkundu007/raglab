"""Tests for shared Pydantic models — round-trip, defaults, and validation."""
import pytest
from pydantic import ValidationError

from raglab_common.models import (
    ChunkModel,
    DocumentModel,
    EmbeddingModel,
    HealthModel,
    LLMProvider,
    QueryModel,
    ResponseModel,
    RetrieverType,
    VectorStoreType,
)


class TestChunkModel:
    def test_defaults_assigned(self, sample_chunk_data):
        chunk = ChunkModel(**sample_chunk_data)
        assert chunk.chunk_id is not None
        assert chunk.created_at is not None

    def test_round_trip(self, sample_chunk_data):
        chunk = ChunkModel(**sample_chunk_data)
        restored = ChunkModel.model_validate(chunk.model_dump())
        assert restored.chunk_id == chunk.chunk_id
        assert restored.text == chunk.text

    def test_unique_ids(self, sample_chunk_data):
        c1 = ChunkModel(**sample_chunk_data)
        c2 = ChunkModel(**sample_chunk_data)
        assert c1.chunk_id != c2.chunk_id


class TestDocumentModel:
    def test_defaults(self, sample_document_data):
        doc = DocumentModel(**sample_document_data)
        assert doc.doc_id is not None
        assert doc.metadata == {}

    def test_round_trip(self, sample_document_data):
        doc = DocumentModel(**sample_document_data)
        restored = DocumentModel.model_validate(doc.model_dump())
        assert restored.doc_id == doc.doc_id


class TestQueryModel:
    def test_defaults(self):
        q = QueryModel(text="What is RAG?", collection="test")
        assert q.retriever_type == RetrieverType.DENSE
        assert q.llm_provider == LLMProvider.AZURE_OPENAI
        assert q.top_k == 5

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            QueryModel(text="q", collection="c", top_k=0)
        with pytest.raises(ValidationError):
            QueryModel(text="q", collection="c", top_k=51)

    def test_valid_top_k_boundary(self):
        q1 = QueryModel(text="q", collection="c", top_k=1)
        q2 = QueryModel(text="q", collection="c", top_k=50)
        assert q1.top_k == 1
        assert q2.top_k == 50


class TestHealthModel:
    def test_defaults(self):
        h = HealthModel(service="test-service")
        assert h.status == "ok"
        assert h.release == "R1"
        assert h.version == "0.1.0"
        assert h.dependencies == {}


class TestEnums:
    def test_retriever_type_values(self):
        assert RetrieverType.DENSE == "dense"
        assert RetrieverType.HYBRID == "hybrid"

    def test_vector_store_values(self):
        assert VectorStoreType.QDRANT == "qdrant"
        assert VectorStoreType.FAISS == "faiss"

    def test_llm_provider_values(self):
        assert LLMProvider.AZURE_OPENAI == "azure_openai"
        assert LLMProvider.ANTHROPIC == "anthropic"
