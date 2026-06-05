"""Tests for ChunkerFactory — updated for R2 active chunkers."""

import pytest

from raglab_chunkers.factory import ChunkerFactory
from raglab_chunkers.text_chunker import TextChunker
from raglab_chunkers.pdf_chunker import PDFChunker
from raglab_chunkers.docx_chunker import DOCXChunker
from raglab_chunkers.markdown_chunker import MarkdownChunker
from raglab_chunkers.html_chunker import HTMLChunker
from raglab_chunkers.excel_chunker import ExcelChunker
from raglab_common.exceptions import NotImplementedFeatureError
from raglab_common.models import ChunkerType


class TestChunkerFactoryCreate:
    # R1 active
    def test_create_text_chunker_by_string(self):
        assert isinstance(ChunkerFactory.create("text"), TextChunker)

    def test_create_text_chunker_by_enum(self):
        assert isinstance(ChunkerFactory.create(ChunkerType.TEXT), TextChunker)

    def test_create_passes_config(self):
        c = ChunkerFactory.create("text", config={"chunk_size": 200, "tokenizer": "word_count"})
        assert c.chunk_size == 200

    # R2 active
    def test_create_pdf_chunker(self):
        assert isinstance(ChunkerFactory.create("pdf", config={"tokenizer": "word_count"}), PDFChunker)

    def test_create_docx_chunker(self):
        assert isinstance(ChunkerFactory.create("docx", config={"tokenizer": "word_count"}), DOCXChunker)

    def test_create_markdown_chunker(self):
        assert isinstance(ChunkerFactory.create("markdown", config={"tokenizer": "word_count"}), MarkdownChunker)

    def test_create_html_chunker(self):
        assert isinstance(ChunkerFactory.create("html", config={"tokenizer": "word_count"}), HTMLChunker)

    def test_create_excel_chunker(self):
        assert isinstance(ChunkerFactory.create("excel"), ExcelChunker)

    # Stubs
    def test_create_pdf_images_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            ChunkerFactory.create("pdf_images")

    def test_create_table_stitch_stub_raises(self):
        with pytest.raises(NotImplementedFeatureError):
            ChunkerFactory.create("table_stitch")

    def test_create_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown chunker type"):
            ChunkerFactory.create("nonexistent_chunker")


class TestChunkerFactoryAvailable:
    def test_available_returns_list(self):
        assert isinstance(ChunkerFactory.available(), list)

    def test_all_r1_r2_active(self):
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        for t in ["text", "pdf", "docx", "markdown", "html", "excel"]:
            assert entries[t]["active"] is True, f"{t} should be active"

    def test_stubs_not_active(self):
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        for t in ["pdf_images", "table_stitch"]:
            assert entries[t]["active"] is False


class TestChunkerFactorySchema:
    def test_text_schema(self):
        assert "chunk_size" in ChunkerFactory.schema("text")

    def test_pdf_schema(self):
        assert "respect_page_boundary" in ChunkerFactory.schema("pdf")

    def test_docx_schema(self):
        assert "preserve_headings" in ChunkerFactory.schema("docx")

    def test_markdown_schema(self):
        assert "split_on_headers" in ChunkerFactory.schema("markdown")

    def test_html_schema(self):
        assert "split_tags" in ChunkerFactory.schema("html")

    def test_excel_schema(self):
        assert "sheet_strategy" in ChunkerFactory.schema("excel")

    def test_stub_schema_accessible(self):
        schema = ChunkerFactory.schema("pdf_images")
        assert "_stub" in schema

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            ChunkerFactory.schema("mystery_chunker")
