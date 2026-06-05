"""
Tests for TextChunker — full coverage of config validation,
chunking behaviour, ChunkModel output, and edge cases.
"""

import pytest

from raglab_chunkers.text_chunker import TextChunker
from raglab_common.models import ChunkModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_chunker() -> TextChunker:
    return TextChunker()


@pytest.fixture
def word_count_chunker() -> TextChunker:
    return TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})


@pytest.fixture
def long_text() -> str:
    return " ".join([
        f"This is sentence number {i}. It contains some useful information about topic {i}."
        for i in range(50)
    ])


@pytest.fixture
def short_text() -> str:
    return "Hello world. This is a short document."


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestTextChunkerConfig:
    def test_defaults(self, default_chunker):
        assert default_chunker.chunk_size == 500
        assert default_chunker.chunk_overlap == 50
        assert default_chunker.boundary_enforcement is True
        assert "." in default_chunker.boundary_chars
        assert default_chunker.tokenizer == "tiktoken"
        assert default_chunker.min_chunk_size == 50

    def test_custom_config(self):
        chunker = TextChunker(config={"chunk_size": 200, "chunk_overlap": 20, "tokenizer": "word_count"})
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20
        assert chunker.tokenizer == "word_count"

    def test_invalid_chunk_size_zero(self):
        with pytest.raises(ValueError, match="chunk_size"):
            TextChunker(config={"chunk_size": 0})

    def test_invalid_chunk_size_negative(self):
        with pytest.raises(ValueError, match="chunk_size"):
            TextChunker(config={"chunk_size": -10})

    def test_invalid_overlap_negative(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(config={"chunk_overlap": -1})

    def test_invalid_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(config={"chunk_size": 100, "chunk_overlap": 100})

    def test_invalid_min_chunk_size(self):
        with pytest.raises(ValueError, match="min_chunk_size"):
            TextChunker(config={"min_chunk_size": 0})

    def test_invalid_tokenizer(self):
        with pytest.raises(ValueError, match="tokenizer"):
            TextChunker(config={"tokenizer": "gpt5"})

    def test_boundary_enforcement_false(self):
        chunker = TextChunker(config={"boundary_enforcement": False})
        assert chunker.boundary_enforcement is False

    def test_custom_boundary_chars(self):
        chunker = TextChunker(config={"boundary_chars": [";", ":"]})
        assert ";" in chunker.boundary_chars
        assert "." not in chunker.boundary_chars


# ---------------------------------------------------------------------------
# Chunker type identity
# ---------------------------------------------------------------------------


class TestChunkerType:
    def test_chunker_type(self, default_chunker):
        assert default_chunker.chunker_type == "text"


# ---------------------------------------------------------------------------
# Empty and edge-case inputs (via public chunk() method)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string_returns_empty(self, default_chunker):
        assert default_chunker.chunk("", "doc-1") == []

    def test_whitespace_only_returns_empty(self, default_chunker):
        assert default_chunker.chunk("   \n\t  ", "doc-1") == []

    def test_single_word(self, word_count_chunker):
        chunks = word_count_chunker.chunk("Hello", "doc-1")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello"

    def test_single_sentence(self, word_count_chunker):
        chunks = word_count_chunker.chunk("This is a test sentence.", "doc-1")
        assert len(chunks) == 1

    def test_exception_in_chunk_returns_empty(self):
        """chunk() must not raise — returns [] on internal error."""
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 100, "chunk_overlap": 10})
        # Monkeypatch _chunk to raise
        def bad_chunk(*args, **kwargs):
            raise RuntimeError("deliberate error")
        chunker._chunk = bad_chunk
        result = chunker.chunk("some text here", "doc-1")
        assert result == []


# ---------------------------------------------------------------------------
# ChunkModel output structure
# ---------------------------------------------------------------------------


class TestChunkModelOutput:
    def test_returns_list_of_chunk_models(self, word_count_chunker, long_text):
        chunks = word_count_chunker.chunk(long_text, "doc-001")
        assert isinstance(chunks, list)
        assert all(isinstance(c, ChunkModel) for c in chunks)

    def test_doc_id_propagated(self, word_count_chunker, long_text):
        chunks = word_count_chunker.chunk(long_text, "my-doc-42")
        assert all(c.doc_id == "my-doc-42" for c in chunks)

    def test_chunk_indices_sequential(self, word_count_chunker, long_text):
        chunks = word_count_chunker.chunk(long_text, "doc-001")
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_ids_unique(self, word_count_chunker, long_text):
        chunks = word_count_chunker.chunk(long_text, "doc-001")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_token_count_positive(self, word_count_chunker, long_text):
        chunks = word_count_chunker.chunk(long_text, "doc-001")
        assert all(c.token_count > 0 for c in chunks)

    def test_metadata_attached(self, word_count_chunker, long_text):
        chunks = word_count_chunker.chunk(long_text, "doc-001", metadata={"source": "test"})
        assert all(c.metadata.get("source") == "test" for c in chunks)

    def test_chunker_metadata_in_chunk(self, word_count_chunker, long_text):
        chunks = word_count_chunker.chunk(long_text, "doc-001")
        assert all(c.metadata.get("chunker") == "text" for c in chunks)

    def test_chunk_text_non_empty(self, word_count_chunker, long_text):
        chunks = word_count_chunker.chunk(long_text, "doc-001")
        assert all(len(c.text.strip()) > 0 for c in chunks)


# ---------------------------------------------------------------------------
# Multiple chunks behaviour
# ---------------------------------------------------------------------------


class TestMultipleChunks:
    def test_long_text_produces_multiple_chunks(self, long_text):
        chunker = TextChunker(config={
            "chunk_size": 30, "chunk_overlap": 5, "tokenizer": "word_count"
        })
        chunks = chunker.chunk(long_text, "doc-001")
        assert len(chunks) > 1

    def test_all_text_covered(self, long_text):
        """All words in the original text should appear in at least one chunk."""
        chunker = TextChunker(config={
            "chunk_size": 30, "chunk_overlap": 0,
            "tokenizer": "word_count", "boundary_enforcement": False,
        })
        chunks = chunker.chunk(long_text, "doc-001")
        all_chunk_text = " ".join(c.text for c in chunks)
        # Spot-check a few distinctive words
        for word in ["sentence", "number", "information"]:
            assert word in all_chunk_text

    def test_boundary_enforcement_chunks_end_on_boundary(self, long_text):
        chunker = TextChunker(config={
            "chunk_size": 30, "chunk_overlap": 3,
            "tokenizer": "word_count",
            "boundary_enforcement": True,
            "min_chunk_size": 5,
        })
        chunks = chunker.chunk(long_text, "doc-001")
        # At least half the non-final chunks should end on a boundary char
        non_final = chunks[:-1]
        boundary_ended = sum(
            1 for c in non_final if c.text.rstrip().endswith((".", "!", "?"))
        )
        assert boundary_ended >= len(non_final) // 2


# ---------------------------------------------------------------------------
# config_schema
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_schema_returns_dict(self):
        schema = TextChunker.config_schema()
        assert isinstance(schema, dict)

    def test_schema_has_required_keys(self):
        schema = TextChunker.config_schema()
        for key in ["chunk_size", "chunk_overlap", "boundary_enforcement",
                    "boundary_chars", "tokenizer", "min_chunk_size"]:
            assert key in schema

    def test_schema_defaults_match_class_defaults(self):
        schema = TextChunker.config_schema()
        assert schema["chunk_size"]["default"] == 500
        assert schema["chunk_overlap"]["default"] == 50
        assert schema["boundary_enforcement"]["default"] is True
        assert schema["tokenizer"]["default"] == "tiktoken"
        assert schema["min_chunk_size"]["default"] == 50

    def test_tokenizer_schema_has_options(self):
        schema = TextChunker.config_schema()
        assert "options" in schema["tokenizer"]
        assert "tiktoken" in schema["tokenizer"]["options"]
        assert "word_count" in schema["tokenizer"]["options"]
