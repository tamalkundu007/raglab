"""
Tests for DenseRetriever.

Uses a mock vector store and mock embedder — no Qdrant instance needed.
Covers: config validation, embedding, filter building, hit conversion,
        error handling, and ChunkModel output structure.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, QueryModel, RetrieverType, LLMProvider
from raglab_retrievers.dense_retriever import DenseRetriever


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def make_query(
    text: str = "What is RAG?",
    collection: str = "test_collection",
    top_k: int = 5,
    metadata_filter: dict | None = None,
) -> QueryModel:
    return QueryModel(
        text=text,
        collection=collection,
        top_k=top_k,
        retriever_type=RetrieverType.DENSE,
        llm_provider=LLMProvider.AZURE_OPENAI,
        metadata_filter=metadata_filter or {},
    )


def make_hit(chunk_id: str, doc_id: str, text: str, score: float = 0.85) -> dict:
    """Dict-style hit (mock vector store returns these)."""
    return {
        "payload": {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "chunk_index": 0,
            "token_count": len(text.split()),
            "source": "unit_test",
        },
        "score": score,
    }


def mock_embedder(text: str) -> list[float]:
    """Fixed-size embedding for tests."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


def make_vector_store(hits: list[dict] | None = None) -> MagicMock:
    """Mock vector store that returns pre-set hits."""
    vs = MagicMock()
    vs.search.return_value = hits or []
    return vs


@pytest.fixture
def default_retriever() -> DenseRetriever:
    return DenseRetriever()


@pytest.fixture
def sample_hits() -> list[dict]:
    return [
        make_hit("chunk-1", "doc-001", "First result about retrieval augmented generation.", 0.92),
        make_hit("chunk-2", "doc-001", "Second result about vector databases.", 0.85),
        make_hit("chunk-3", "doc-002", "Third result about embeddings.", 0.78),
    ]


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestDenseRetrieverConfig:
    def test_defaults(self, default_retriever):
        assert default_retriever.score_threshold == 0.0
        assert default_retriever.ef == 128
        assert default_retriever.with_payload is True
        assert default_retriever.with_vectors is False

    def test_custom_config(self):
        r = DenseRetriever(config={"score_threshold": 0.7, "ef": 64, "with_vectors": True})
        assert r.score_threshold == 0.7
        assert r.ef == 64
        assert r.with_vectors is True

    def test_invalid_score_threshold_above_one(self):
        with pytest.raises(ValueError, match="score_threshold"):
            DenseRetriever(config={"score_threshold": 1.5})

    def test_invalid_score_threshold_negative(self):
        with pytest.raises(ValueError, match="score_threshold"):
            DenseRetriever(config={"score_threshold": -0.1})

    def test_score_threshold_boundary_values(self):
        r0 = DenseRetriever(config={"score_threshold": 0.0})
        r1 = DenseRetriever(config={"score_threshold": 1.0})
        assert r0.score_threshold == 0.0
        assert r1.score_threshold == 1.0

    def test_invalid_ef_zero(self):
        with pytest.raises(ValueError, match="ef"):
            DenseRetriever(config={"ef": 0})

    def test_invalid_ef_negative(self):
        with pytest.raises(ValueError, match="ef"):
            DenseRetriever(config={"ef": -1})


# ---------------------------------------------------------------------------
# Retriever type identity
# ---------------------------------------------------------------------------


class TestRetrieverType:
    def test_retriever_type(self, default_retriever):
        assert default_retriever.retriever_type == "dense"


# ---------------------------------------------------------------------------
# retrieve() — public method error handling
# ---------------------------------------------------------------------------


class TestRetrieveErrorHandling:
    def test_no_embedder_returns_empty(self, default_retriever):
        """retrieve() must return [] when embedder is None — never raise."""
        query = make_query()
        vs = make_vector_store()
        result = default_retriever.retrieve(query, vs, embedder=None)
        assert result == []

    def test_embedder_raises_returns_empty(self, default_retriever):
        def bad_embedder(text):
            raise RuntimeError("embedding service down")

        query = make_query()
        vs = make_vector_store()
        result = default_retriever.retrieve(query, vs, embedder=bad_embedder)
        assert result == []

    def test_vector_store_raises_returns_empty(self, default_retriever):
        query = make_query()
        vs = MagicMock()
        vs.search.side_effect = ConnectionError("Qdrant unavailable")
        result = default_retriever.retrieve(query, vs, embedder=mock_embedder)
        assert result == []

    def test_embedder_returns_empty_vector_returns_empty(self, default_retriever):
        query = make_query()
        vs = make_vector_store()
        result = default_retriever.retrieve(query, vs, embedder=lambda t: [])
        assert result == []


# ---------------------------------------------------------------------------
# _retrieve() — direct tests for internal logic
# ---------------------------------------------------------------------------


class TestRetrieveInternals:
    def test_embedder_called_with_query_text(self, default_retriever):
        query = make_query(text="specific query text")
        calls = []

        def recording_embedder(text):
            calls.append(text)
            return [0.1, 0.2, 0.3]

        vs = make_vector_store()
        default_retriever._retrieve(query, vs, recording_embedder)
        assert calls == ["specific query text"]

    def test_vector_store_called_with_correct_args(self, default_retriever):
        query = make_query(top_k=3)
        vs = make_vector_store()
        default_retriever._retrieve(query, vs, mock_embedder)
        vs.search.assert_called_once()
        call_kwargs = vs.search.call_args[1]
        assert call_kwargs["collection_name"] == "test_collection"
        assert call_kwargs["limit"] == 3
        assert call_kwargs["query_vector"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_score_threshold_passed_when_nonzero(self):
        r = DenseRetriever(config={"score_threshold": 0.75})
        query = make_query()
        vs = make_vector_store()
        r._retrieve(query, vs, mock_embedder)
        call_kwargs = vs.search.call_args[1]
        assert call_kwargs["score_threshold"] == 0.75

    def test_score_threshold_none_when_zero(self, default_retriever):
        query = make_query()
        vs = make_vector_store()
        default_retriever._retrieve(query, vs, mock_embedder)
        call_kwargs = vs.search.call_args[1]
        assert call_kwargs["score_threshold"] is None

    def test_no_filter_when_empty_metadata_filter(self, default_retriever):
        query = make_query(metadata_filter={})
        vs = make_vector_store()
        default_retriever._retrieve(query, vs, mock_embedder)
        call_kwargs = vs.search.call_args[1]
        assert call_kwargs["query_filter"] is None

    def test_filter_built_from_metadata_filter(self, default_retriever):
        query = make_query(metadata_filter={"doc_id": "doc-001"})
        vs = make_vector_store()
        default_retriever._retrieve(query, vs, mock_embedder)
        call_kwargs = vs.search.call_args[1]
        assert call_kwargs["query_filter"] is not None
        assert "must" in call_kwargs["query_filter"]


# ---------------------------------------------------------------------------
# ChunkModel output
# ---------------------------------------------------------------------------


class TestChunkModelOutput:
    def test_returns_list_of_chunk_models(self, default_retriever, sample_hits):
        query = make_query()
        vs = make_vector_store(hits=sample_hits)
        results = default_retriever._retrieve(query, vs, mock_embedder)
        assert isinstance(results, list)
        assert all(isinstance(r, ChunkModel) for r in results)

    def test_result_count_matches_hits(self, default_retriever, sample_hits):
        query = make_query()
        vs = make_vector_store(hits=sample_hits)
        results = default_retriever._retrieve(query, vs, mock_embedder)
        assert len(results) == len(sample_hits)

    def test_empty_hits_returns_empty(self, default_retriever):
        query = make_query()
        vs = make_vector_store(hits=[])
        results = default_retriever._retrieve(query, vs, mock_embedder)
        assert results == []

    def test_chunk_fields_populated(self, default_retriever, sample_hits):
        query = make_query()
        vs = make_vector_store(hits=sample_hits)
        results = default_retriever._retrieve(query, vs, mock_embedder)
        first = results[0]
        assert first.chunk_id == "chunk-1"
        assert first.doc_id == "doc-001"
        assert "retrieval augmented generation" in first.text

    def test_score_in_metadata(self, default_retriever, sample_hits):
        query = make_query()
        vs = make_vector_store(hits=sample_hits)
        results = default_retriever._retrieve(query, vs, mock_embedder)
        assert results[0].metadata["score"] == pytest.approx(0.92)
        assert results[1].metadata["score"] == pytest.approx(0.85)

    def test_retriever_tag_in_metadata(self, default_retriever, sample_hits):
        query = make_query()
        vs = make_vector_store(hits=sample_hits)
        results = default_retriever._retrieve(query, vs, mock_embedder)
        assert all(r.metadata["retriever"] == "dense" for r in results)

    def test_query_id_in_metadata(self, default_retriever, sample_hits):
        query = make_query()
        vs = make_vector_store(hits=sample_hits)
        results = default_retriever._retrieve(query, vs, mock_embedder)
        assert all(r.metadata["query_id"] == str(query.query_id) for r in results)

    def test_object_style_hit_parsed(self, default_retriever):
        """DenseRetriever must handle Qdrant ScoredPoint-style objects too."""
        hit = MagicMock()
        hit.payload = {
            "chunk_id": "c-obj", "doc_id": "d-obj",
            "text": "object style hit", "chunk_index": 0, "token_count": 3,
        }
        hit.score = 0.91
        query = make_query()
        vs = make_vector_store(hits=[hit])
        results = default_retriever._retrieve(query, vs, mock_embedder)
        assert len(results) == 1
        assert results[0].chunk_id == "c-obj"
        assert results[0].metadata["score"] == pytest.approx(0.91)


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------


class TestBuildFilter:
    def test_single_key(self):
        f = DenseRetriever._build_filter({"doc_id": "abc"})
        assert f == {"must": [{"key": "doc_id", "match": {"value": "abc"}}]}

    def test_multiple_keys(self):
        f = DenseRetriever._build_filter({"doc_id": "abc", "source": "pdf"})
        assert "must" in f
        assert len(f["must"]) == 2

    def test_empty_filter(self):
        f = DenseRetriever._build_filter({})
        assert f == {"must": []}


# ---------------------------------------------------------------------------
# config_schema
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_schema_returns_dict(self):
        assert isinstance(DenseRetriever.config_schema(), dict)

    def test_schema_has_required_keys(self):
        schema = DenseRetriever.config_schema()
        for key in ["score_threshold", "ef", "with_payload", "with_vectors"]:
            assert key in schema

    def test_schema_defaults_match_class(self):
        schema = DenseRetriever.config_schema()
        assert schema["score_threshold"]["default"] == 0.0
        assert schema["ef"]["default"] == 128
        assert schema["with_payload"]["default"] is True
        assert schema["with_vectors"]["default"] is False
