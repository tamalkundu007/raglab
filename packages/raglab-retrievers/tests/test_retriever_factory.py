"""
Tests for RetrieverFactory — registry, creation, stubs, schema, available().
Mirrors test_factory.py in raglab-chunkers for consistency.
"""

import pytest

from raglab_common.exceptions import NotImplementedFeatureError
from raglab_common.models import RetrieverType
from raglab_retrievers.dense_retriever import DenseRetriever
from raglab_retrievers.factory import RetrieverFactory


class TestRetrieverFactoryCreate:
    def test_create_dense_by_string(self):
        r = RetrieverFactory.create("dense")
        assert isinstance(r, DenseRetriever)

    def test_create_dense_by_enum(self):
        r = RetrieverFactory.create(RetrieverType.DENSE)
        assert isinstance(r, DenseRetriever)

    def test_create_passes_config(self):
        r = RetrieverFactory.create("dense", config={"score_threshold": 0.6, "ef": 64})
        assert r.score_threshold == 0.6
        assert r.ef == 64

    def test_create_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown retriever type"):
            RetrieverFactory.create("quantum_retriever")

    def test_create_bm25_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError) as exc_info:
            RetrieverFactory.create("bm25")
        assert "R3" in str(exc_info.value)

    def test_create_hybrid_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            RetrieverFactory.create("hybrid")

    def test_create_mmr_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            RetrieverFactory.create("mmr")

    def test_create_reranker_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            RetrieverFactory.create("reranker")

    def test_create_compression_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            RetrieverFactory.create("compression")


class TestRetrieverFactoryAvailable:
    def test_returns_list(self):
        assert isinstance(RetrieverFactory.available(), list)

    def test_dense_is_present(self):
        types = {e["type"] for e in RetrieverFactory.available()}
        assert "dense" in types

    def test_dense_is_active(self):
        entries = {e["type"]: e for e in RetrieverFactory.available()}
        assert entries["dense"]["active"] is True

    def test_stubs_are_not_active(self):
        entries = {e["type"]: e for e in RetrieverFactory.available()}
        for stub in ["bm25", "hybrid", "mmr", "reranker", "compression"]:
            assert entries[stub]["active"] is False

    def test_stubs_have_available_in_r3(self):
        entries = {e["type"]: e for e in RetrieverFactory.available()}
        for stub in ["bm25", "hybrid", "mmr", "reranker", "compression"]:
            assert entries[stub].get("available_in") == "R3"


class TestRetrieverFactorySchema:
    def test_dense_schema_returned(self):
        schema = RetrieverFactory.schema("dense")
        assert "score_threshold" in schema

    def test_dense_schema_by_enum(self):
        schema = RetrieverFactory.schema(RetrieverType.DENSE)
        assert "ef" in schema

    def test_stub_schema_accessible(self):
        schema = RetrieverFactory.schema("bm25")
        assert "_stub" in schema

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown retriever type"):
            RetrieverFactory.schema("nonexistent")
