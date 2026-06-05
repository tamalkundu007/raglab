"""
Tests for ChunkerFactory — registry, creation, stubs, schema, available().
"""

import pytest

from raglab_chunkers.factory import ChunkerFactory
from raglab_chunkers.text_chunker import TextChunker
from raglab_common.exceptions import NotImplementedFeatureError
from raglab_common.models import ChunkerType


class TestChunkerFactoryCreate:
    def test_create_text_chunker_by_string(self):
        chunker = ChunkerFactory.create("text")
        assert isinstance(chunker, TextChunker)

    def test_create_text_chunker_by_enum(self):
        chunker = ChunkerFactory.create(ChunkerType.TEXT)
        assert isinstance(chunker, TextChunker)

    def test_create_passes_config(self):
        chunker = ChunkerFactory.create("text", config={"chunk_size": 200, "tokenizer": "word_count"})
        assert isinstance(chunker, TextChunker)
        assert chunker.chunk_size == 200

    def test_create_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown chunker type"):
            ChunkerFactory.create("nonexistent_chunker")

    def test_create_stub_chunker_raises_not_implemented(self):
        """R2+ stubs must raise NotImplementedFeatureError on instantiation."""
        with pytest.raises(NotImplementedFeatureError) as exc_info:
            ChunkerFactory.create("pdf")
        assert "R2" in str(exc_info.value)

    def test_create_docx_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            ChunkerFactory.create("docx")

    def test_create_markdown_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            ChunkerFactory.create("markdown")

    def test_create_html_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            ChunkerFactory.create("html")

    def test_create_excel_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            ChunkerFactory.create("excel")


class TestChunkerFactoryAvailable:
    def test_available_returns_list(self):
        result = ChunkerFactory.available()
        assert isinstance(result, list)

    def test_available_contains_text(self):
        types = {entry["type"] for entry in ChunkerFactory.available()}
        assert "text" in types

    def test_text_is_active(self):
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        assert entries["text"]["active"] is True

    def test_stubs_are_not_active(self):
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        for stub_type in ["pdf", "docx", "markdown", "html", "excel"]:
            assert entries[stub_type]["active"] is False

    def test_stubs_have_available_in(self):
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        for stub_type in ["pdf", "docx", "markdown", "html", "excel"]:
            assert "available_in" in entries[stub_type]
            assert entries[stub_type]["available_in"] == "R2"


class TestChunkerFactorySchema:
    def test_text_schema_returned(self):
        schema = ChunkerFactory.schema("text")
        assert "chunk_size" in schema

    def test_text_schema_by_enum(self):
        schema = ChunkerFactory.schema(ChunkerType.TEXT)
        assert "tokenizer" in schema

    def test_stub_schema_returned(self):
        """Stub schema should be accessible without raising."""
        schema = ChunkerFactory.schema("pdf")
        assert "_stub" in schema

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown chunker type"):
            ChunkerFactory.schema("mystery_chunker")
