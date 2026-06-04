"""Tests for the RAGLab exception hierarchy."""
import pytest

from raglab_common.exceptions import (
    ChunkerError,
    ConfigError,
    EmbeddingError,
    IndexingError,
    LLMError,
    NotImplementedFeatureError,
    RAGLabError,
    RetrieverError,
    StorageError,
)


class TestExceptionHierarchy:
    def test_all_inherit_from_raglab_error(self):
        for cls in [
            ChunkerError,
            RetrieverError,
            EmbeddingError,
            IndexingError,
            LLMError,
            ConfigError,
            StorageError,
            NotImplementedFeatureError,
        ]:
            assert issubclass(cls, RAGLabError)

    def test_all_inherit_from_exception(self):
        assert issubclass(RAGLabError, Exception)

    def test_catch_all_with_base(self):
        with pytest.raises(RAGLabError):
            raise ChunkerError("failed to chunk")

    def test_specific_catch(self):
        with pytest.raises(ChunkerError):
            raise ChunkerError("failed to chunk")

    def test_does_not_catch_unrelated(self):
        with pytest.raises(RetrieverError):
            raise RetrieverError("retrieval failed")
        # ChunkerError should not be caught as RetrieverError
        with pytest.raises(ChunkerError):
            try:
                raise ChunkerError("oops")
            except RetrieverError:
                pass  # should not reach here


class TestErrorCodes:
    def test_chunker_code(self):
        e = ChunkerError("msg")
        assert e.code == "CHUNKER_ERROR"

    def test_llm_code(self):
        e = LLMError("msg")
        assert e.code == "LLM_ERROR"

    def test_base_default_code(self):
        e = RAGLabError("msg")
        assert e.code == "RAGLAB_ERROR"


class TestNotImplementedFeatureError:
    def test_message_format(self):
        e = NotImplementedFeatureError("BM25Retriever", available_in="R3")
        assert "BM25Retriever" in str(e)
        assert "R3" in str(e)

    def test_feature_attribute(self):
        e = NotImplementedFeatureError("GraphRAG", available_in="R4")
        assert e.feature == "GraphRAG"
        assert e.available_in == "R4"

    def test_code(self):
        e = NotImplementedFeatureError("SomeFeature")
        assert e.code == "NOT_IMPLEMENTED"
