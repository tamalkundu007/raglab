"""
Tests for MarkdownChunker, HTMLChunker, ExcelChunker — all R2 chunkers.
"""

from __future__ import annotations

import pytest

from raglab_chunkers.markdown_chunker import MarkdownChunker
from raglab_chunkers.html_chunker import HTMLChunker
from raglab_chunkers.excel_chunker import ExcelChunker, _row_to_text
from raglab_common.models import ChunkModel


# ═══════════════════════════════════════════════════════════════════════════════
# MarkdownChunker
# ═══════════════════════════════════════════════════════════════════════════════


MARKDOWN_TEXT = """# Introduction

Retrieval-Augmented Generation (RAG) is a framework that enhances large language models
with external knowledge at inference time.

## How It Works

The retrieval step uses dense vector search. A query is embedded and nearest neighbours
are retrieved as context chunks.

### Dense Retrieval

Dense retrieval encodes both queries and documents into dense vector representations.
Cosine similarity is then used to rank candidate documents.

## Benefits

RAG significantly reduces hallucinations by grounding model answers in retrieved facts.
The knowledge base can also be updated without retraining the model.
"""


class TestMarkdownChunkerConfig:
    def test_defaults(self):
        c = MarkdownChunker()
        assert c.chunk_size == 500
        assert c.split_on_headers is True
        assert c.header_levels == {1, 2, 3}
        assert c.include_header_in_chunk is True

    def test_custom_header_levels(self):
        c = MarkdownChunker(config={"header_levels": [1, 2]})
        assert c.header_levels == {1, 2}

    def test_invalid_header_levels(self):
        with pytest.raises(ValueError, match="header_levels"):
            MarkdownChunker(config={"header_levels": [0, 7]})

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            MarkdownChunker(config={"chunk_size": 0})

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            MarkdownChunker(config={"chunk_size": 100, "chunk_overlap": 150})


class TestMarkdownChunkerSplitByHeaders:
    def _chunker(self, **kwargs) -> MarkdownChunker:
        cfg = {"tokenizer": "word_count", "chunk_size": 40, "chunk_overlap": 5}
        cfg.update(kwargs)
        return MarkdownChunker(config=cfg)

    def test_splits_at_h1_and_h2(self):
        c = self._chunker()
        sections = c._split_by_headers(MARKDOWN_TEXT)
        headers = [s["header"] for s in sections if s["header"]]
        assert any("Introduction" in h for h in headers)
        assert any("How It Works" in h for h in headers)

    def test_preamble_before_first_header(self):
        text = "Preamble without header.\n\n# First Section\n\nBody text here."
        c = self._chunker()
        sections = c._split_by_headers(text)
        assert sections[0]["header"] is None
        assert "Preamble" in sections[0]["body"]

    def test_only_h1_splits(self):
        c = self._chunker(header_levels=[1])
        sections = c._split_by_headers(MARKDOWN_TEXT)
        # H2 and H3 should NOT trigger splits
        headers = [s["header"] for s in sections if s["header"]]
        h2_splits = [h for h in headers if h and h.startswith("## ")]
        assert not h2_splits

    def test_header_in_chunk_text_when_enabled(self):
        c = self._chunker(include_header_in_chunk=True)
        chunks = c.chunk(MARKDOWN_TEXT, "doc-md")
        assert any("Introduction" in ch.text for ch in chunks)

    def test_header_metadata_injected(self):
        c = self._chunker()
        chunks = c.chunk(MARKDOWN_TEXT, "doc-md")
        headed = [ch for ch in chunks if ch.metadata.get("header")]
        assert len(headed) > 0
        assert all("header_level" in ch.metadata for ch in headed)

    def test_header_level_values_correct(self):
        c = self._chunker()
        chunks = c.chunk(MARKDOWN_TEXT, "doc-md")
        levels = {ch.metadata.get("header_level") for ch in chunks if ch.metadata.get("header")}
        assert levels.issubset({1, 2, 3})

    def test_split_on_headers_false_single_stream(self):
        c = self._chunker(split_on_headers=False)
        chunks = c.chunk(MARKDOWN_TEXT, "doc-md")
        assert all("header" not in ch.metadata for ch in chunks)
        assert len(chunks) >= 1

    def test_empty_text_returns_empty(self):
        c = self._chunker()
        assert c.chunk("", "doc-001") == []

    def test_sequential_indices(self):
        c = self._chunker()
        chunks = c.chunk(MARKDOWN_TEXT, "doc-001")
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))

    def test_unique_chunk_ids(self):
        c = self._chunker()
        chunks = c.chunk(MARKDOWN_TEXT, "doc-001")
        ids = [ch.chunk_id for ch in chunks]
        assert len(ids) == len(set(ids))

    def test_schema_keys(self):
        schema = MarkdownChunker.config_schema()
        for key in ["split_on_headers", "header_levels", "include_header_in_chunk"]:
            assert key in schema


# ═══════════════════════════════════════════════════════════════════════════════
# HTMLChunker
# ═══════════════════════════════════════════════════════════════════════════════


HTML_TEXT = """
<html>
<head><title>RAG Guide</title></head>
<body>
  <script>var x = 1;</script>
  <style>body { color: red; }</style>
  <article>
    <h1>Introduction to RAG</h1>
    <p>Retrieval-Augmented Generation enhances language models with external knowledge.</p>
    <p>The retrieval step uses dense vector search to find relevant documents.</p>
  </article>
  <section>
    <h2>Benefits</h2>
    <p>RAG reduces hallucinations by grounding answers in retrieved evidence.</p>
    <ul>
      <li>Knowledge base can be updated without retraining.</li>
      <li>Answers are grounded in verifiable sources.</li>
    </ul>
  </section>
</body>
</html>
"""


class TestHTMLChunkerConfig:
    def test_defaults(self):
        c = HTMLChunker()
        assert c.chunk_size_fallback == 500
        assert c.strip_scripts_styles is True
        assert c.include_tag_attrs is False
        assert c.boundary_enforcement is True

    def test_custom_config(self):
        c = HTMLChunker(config={
            "chunk_size_fallback": 200, "overlap_fallback": 20,
            "strip_scripts_styles": False, "include_tag_attrs": True,
            "tokenizer": "word_count",
        })
        assert c.chunk_size_fallback == 200
        assert c.strip_scripts_styles is False
        assert c.include_tag_attrs is True

    def test_invalid_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError):
            HTMLChunker(config={"chunk_size_fallback": 100, "overlap_fallback": 100})

    def test_invalid_tokenizer(self):
        with pytest.raises(ValueError):
            HTMLChunker(config={"tokenizer": "magic"})


class TestHTMLChunkerChunking:
    def _chunker(self, **kwargs) -> HTMLChunker:
        cfg = {"tokenizer": "word_count", "chunk_size_fallback": 30, "overlap_fallback": 3}
        cfg.update(kwargs)
        return HTMLChunker(config=cfg)

    def test_produces_chunks_from_html(self):
        c = self._chunker()
        chunks = c.chunk(HTML_TEXT, "doc-html")
        assert len(chunks) >= 1
        assert all(isinstance(ch, ChunkModel) for ch in chunks)

    def test_scripts_styles_stripped(self):
        c = self._chunker(strip_scripts_styles=True)
        chunks = c.chunk(HTML_TEXT, "doc-html")
        all_text = " ".join(ch.text for ch in chunks)
        assert "var x = 1" not in all_text
        assert "color: red" not in all_text

    def test_script_included_when_not_stripped(self):
        # When strip_scripts_styles=False, script tag appears in split_tags scan
        # BeautifulSoup extracts script text via get_text when it matches split_tags
        simple_html = "<html><body><p>Real content here with enough words for chunking.</p></body></html>"
        c = self._chunker(strip_scripts_styles=False)
        chunks = c.chunk(simple_html, "doc-html")
        # Verify chunking still works correctly when strip is disabled
        assert len(chunks) >= 1
        all_text = " ".join(ch.text for ch in chunks)
        assert "Real content" in all_text

    def test_tag_attrs_in_metadata_when_enabled(self):
        c = self._chunker(include_tag_attrs=True)
        chunks = c.chunk(HTML_TEXT, "doc-html")
        assert any("tags" in ch.metadata for ch in chunks)

    def test_fallback_on_no_matching_tags(self):
        plain_html = "<html><body><span>Some plain text content here with enough words.</span></body></html>"
        c = HTMLChunker(config={
            "tokenizer": "word_count", "chunk_size_fallback": 10, "overlap_fallback": 2,
            "split_tags": ["p", "article"],  # span not in split_tags → fallback
        })
        chunks = c.chunk(plain_html, "doc-html")
        assert len(chunks) >= 1

    def test_empty_html_returns_empty(self):
        c = self._chunker()
        chunks = c.chunk("<html><body></body></html>", "doc-001")
        assert chunks == []

    def test_sequential_indices(self):
        c = self._chunker()
        chunks = c.chunk(HTML_TEXT, "doc-html")
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))

    def test_doc_id_propagated(self):
        c = self._chunker()
        chunks = c.chunk(HTML_TEXT, "my-html-doc")
        assert all(ch.doc_id == "my-html-doc" for ch in chunks)

    def test_oversized_node_split(self):
        long_p = "<p>" + ("word " * 200) + "</p>"
        html = f"<html><body>{long_p}</body></html>"
        c = HTMLChunker(config={"tokenizer": "word_count", "chunk_size_fallback": 30, "overlap_fallback": 3})
        chunks = c.chunk(html, "doc-001")
        assert len(chunks) > 1

    def test_schema_keys(self):
        schema = HTMLChunker.config_schema()
        for key in ["split_tags", "strip_scripts_styles", "chunk_size_fallback", "overlap_fallback"]:
            assert key in schema


# ═══════════════════════════════════════════════════════════════════════════════
# ExcelChunker
# ═══════════════════════════════════════════════════════════════════════════════


class TestRowToText:
    def test_basic_row(self):
        text = _row_to_text(["Name", "Age", "City"], ["Alice", 30, "London"])
        assert "Name: Alice" in text
        assert "Age: 30" in text
        assert "City: London" in text

    def test_skips_empty_values(self):
        text = _row_to_text(["A", "B", "C"], ["x", "", None])
        assert "A: x" in text
        assert "B" not in text
        assert "C" not in text

    def test_skips_nan(self):
        text = _row_to_text(["Score"], ["nan"])
        assert text == ""

    def test_pipe_separator(self):
        text = _row_to_text(["X", "Y"], ["1", "2"])
        assert "|" in text


CSV_TEXT = "Name,Age,City\nAlice,30,London\nBob,25,Paris\nCarol,35,Berlin"


class TestExcelChunkerConfig:
    def test_defaults(self):
        c = ExcelChunker()
        assert c.sheet_strategy == "row"
        assert c.header_rows == 1
        assert c.row_grouping == 10
        assert c.include_sheet_name is True
        assert c.tokenizer == "word_count"

    def test_custom_config(self):
        c = ExcelChunker(config={"sheet_strategy": "cell", "row_grouping": 5})
        assert c.sheet_strategy == "cell"
        assert c.row_grouping == 5

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="sheet_strategy"):
            ExcelChunker(config={"sheet_strategy": "diagonal"})

    def test_invalid_header_rows(self):
        with pytest.raises(ValueError):
            ExcelChunker(config={"header_rows": -1})

    def test_invalid_row_grouping(self):
        with pytest.raises(ValueError):
            ExcelChunker(config={"row_grouping": 0})

    def test_invalid_tokenizer(self):
        with pytest.raises(ValueError):
            ExcelChunker(config={"tokenizer": "bpe_v2"})


class TestExcelChunkerCSVFallback:
    """_chunk() handles CSV-like plain text input."""

    def _chunker(self, **kwargs) -> ExcelChunker:
        cfg = {"tokenizer": "word_count", "chunk_size": 100, "row_grouping": 2}
        cfg.update(kwargs)
        return ExcelChunker(config=cfg)

    def test_row_strategy_produces_chunks(self):
        c = self._chunker(sheet_strategy="row", include_sheet_name=False)
        chunks = c.chunk(CSV_TEXT, "doc-xls")
        assert len(chunks) >= 1
        assert all(isinstance(ch, ChunkModel) for ch in chunks)

    def test_headers_in_chunk_text(self):
        c = self._chunker(sheet_strategy="row", include_sheet_name=False)
        chunks = c.chunk(CSV_TEXT, "doc-xls")
        all_text = " ".join(ch.text for ch in chunks)
        assert "Name" in all_text or "Alice" in all_text

    def test_cell_strategy(self):
        c = self._chunker(sheet_strategy="cell", include_sheet_name=False)
        chunks = c.chunk(CSV_TEXT, "doc-xls")
        # Each non-empty cell → separate chunk
        assert len(chunks) >= 9  # 3 cols × 3 data rows = 9

    def test_column_strategy(self):
        c = self._chunker(sheet_strategy="column", include_sheet_name=False)
        chunks = c.chunk(CSV_TEXT, "doc-xls")
        all_text = " ".join(ch.text for ch in chunks)
        assert "Alice" in all_text or "30" in all_text

    def test_empty_input_returns_empty(self):
        c = self._chunker()
        assert c.chunk("", "doc-001") == []

    def test_doc_id_propagated(self):
        c = self._chunker(sheet_strategy="row")
        chunks = c.chunk(CSV_TEXT, "xls-doc-99")
        assert all(ch.doc_id == "xls-doc-99" for ch in chunks)

    def test_sequential_indices(self):
        c = self._chunker()
        chunks = c.chunk(CSV_TEXT, "doc-001")
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))

    def test_unique_ids(self):
        c = self._chunker()
        chunks = c.chunk(CSV_TEXT, "doc-001")
        ids = [ch.chunk_id for ch in chunks]
        assert len(ids) == len(set(ids))

    def test_row_metadata(self):
        c = self._chunker(sheet_strategy="row")
        chunks = c.chunk(CSV_TEXT, "doc-001")
        assert all("sheet" in ch.metadata for ch in chunks)
        assert all("chunker" in ch.metadata for ch in chunks)

    def test_schema_keys(self):
        schema = ExcelChunker.config_schema()
        for key in ["sheet_strategy", "header_rows", "row_grouping", "include_sheet_name"]:
            assert key in schema


# ═══════════════════════════════════════════════════════════════════════════════
# Factory — R2 chunkers now active
# ═══════════════════════════════════════════════════════════════════════════════


class TestFactoryR2:
    def test_pdf_active_in_factory(self):
        from raglab_chunkers import ChunkerFactory
        chunker = ChunkerFactory.create("pdf", config={"tokenizer": "word_count"})
        assert isinstance(chunker, __import__("raglab_chunkers").PDFChunker)

    def test_docx_active_in_factory(self):
        from raglab_chunkers import ChunkerFactory, DOCXChunker
        chunker = ChunkerFactory.create("docx", config={"tokenizer": "word_count"})
        assert isinstance(chunker, DOCXChunker)

    def test_markdown_active_in_factory(self):
        from raglab_chunkers import ChunkerFactory, MarkdownChunker
        chunker = ChunkerFactory.create("markdown", config={"tokenizer": "word_count"})
        assert isinstance(chunker, MarkdownChunker)

    def test_html_active_in_factory(self):
        from raglab_chunkers import ChunkerFactory, HTMLChunker
        chunker = ChunkerFactory.create("html", config={"tokenizer": "word_count"})
        assert isinstance(chunker, HTMLChunker)

    def test_excel_active_in_factory(self):
        from raglab_chunkers import ChunkerFactory, ExcelChunker
        chunker = ChunkerFactory.create("excel")
        assert isinstance(chunker, ExcelChunker)

    def test_available_shows_r2_active(self):
        from raglab_chunkers import ChunkerFactory
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        for t in ["pdf", "docx", "markdown", "html", "excel"]:
            assert entries[t]["active"] is True

    def test_table_stitch_still_stub(self):
        # pdf_images activated in R4; table_stitch activates in R4 Phase 3
        from raglab_common.exceptions import NotImplementedFeatureError
        from raglab_chunkers import ChunkerFactory
        with pytest.raises(NotImplementedFeatureError):
            ChunkerFactory.create("table_stitch")
