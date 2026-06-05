"""Tests for RetrieverFactory — updated for R3 active retrievers."""

import pytest

from raglab_common.models import RetrieverType
from raglab_retrievers.dense_retriever import DenseRetriever
from raglab_retrievers.bm25_retriever import BM25Retriever
from raglab_retrievers.hybrid_retriever import HybridRetriever
from raglab_retrievers.mmr_retriever import MMRRetriever
from raglab_retrievers.reranker_retriever import ReRankerRetriever
from raglab_retrievers.compression_retriever import CompressionRetriever
from raglab_retrievers.factory import RetrieverFactory


class TestRetrieverFactoryCreate:
    def test_create_dense_by_string(self):
        assert isinstance(RetrieverFactory.create("dense"), DenseRetriever)

    def test_create_dense_by_enum(self):
        assert isinstance(RetrieverFactory.create(RetrieverType.DENSE), DenseRetriever)

    def test_create_bm25(self):
        assert isinstance(RetrieverFactory.create("bm25"), BM25Retriever)

    def test_create_hybrid(self):
        assert isinstance(RetrieverFactory.create("hybrid"), HybridRetriever)

    def test_create_mmr(self):
        assert isinstance(RetrieverFactory.create("mmr"), MMRRetriever)

    def test_create_reranker(self):
        assert isinstance(RetrieverFactory.create("reranker"), ReRankerRetriever)

    def test_create_compression(self):
        assert isinstance(RetrieverFactory.create("compression"), CompressionRetriever)

    def test_create_passes_config(self):
        r = RetrieverFactory.create("dense", config={"score_threshold": 0.6, "ef": 64})
        assert r.score_threshold == 0.6

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown retriever type"):
            RetrieverFactory.create("quantum_retriever")


class TestRetrieverFactoryAvailable:
    def test_all_active(self):
        entries = {e["type"]: e for e in RetrieverFactory.available()}
        for t in ["dense", "bm25", "hybrid", "mmr", "reranker", "compression"]:
            assert entries[t]["active"] is True


class TestRetrieverFactorySchema:
    def test_dense_schema(self):
        assert "score_threshold" in RetrieverFactory.schema("dense")

    def test_bm25_schema(self):
        assert "k1" in RetrieverFactory.schema("bm25")

    def test_hybrid_schema(self):
        assert "alpha" in RetrieverFactory.schema("hybrid")

    def test_mmr_schema(self):
        assert "lambda_mult" in RetrieverFactory.schema("mmr")

    def test_reranker_schema(self):
        assert "model_name" in RetrieverFactory.schema("reranker")

    def test_compression_schema(self):
        assert "strategy" in RetrieverFactory.schema("compression")

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            RetrieverFactory.schema("nonexistent")
