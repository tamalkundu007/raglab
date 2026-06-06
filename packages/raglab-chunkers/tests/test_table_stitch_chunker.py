"""
Unit tests for TableStitchChunker (R4 Phase 3).

Tests are entirely infra-free — pdfplumber is mocked via dependency injection
of the extracted table data structure. No real PDFs required.

Covers:
- Config validation (emit_format, stitch_threshold, column_alignment_tolerance, etc.)
- _RawTable / _StitchedTable data structures
- _clean_table_data: None handling, empty row filtering
- _split_header_rows: header detection heuristic, auto-header generation
- _stitch_tables: same-page, consecutive-page, non-consecutive, header repeat strip,
  column count tolerance, stitch_threshold limit
- _is_continuation: all branching conditions
- _rows_match: case-insensitive, length mismatch
- Emit: markdown GFM format, JSON array, CSV with headers
- _table_to_chunk: metadata (chunk_type, stitched flag, page_range, row_count, col_count)
- min_rows filter
- chunk_pdf_bytes: pdfplumber mocked, multi-page table stitched, free text chunked
- _chunk: pipe-delimited table, plain text fallback
- Factory: create, active status, schema
- config_schema: all keys present
"""

from __future__ import annotations

import json
import csv
import io
import uuid
from unittest.mock import MagicMock, patch

import pytest

from raglab_chunkers.table_stitch_chunker import (
    TableStitchChunker,
    _RawTable,
    _StitchedTable,
    _emit_markdown,
    _emit_json,
    _emit_csv,
    _VALID_EMIT_FORMATS,
)
from raglab_common.models import ChunkModel


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunker(**kwargs) -> TableStitchChunker:
    defaults = {
        "tokenizer": "word_count",
        "chunk_size": 50,
        "chunk_overlap": 5,
        "emit_format": "markdown",
        "min_rows": 1,
    }
    defaults.update(kwargs)
    return TableStitchChunker(config=defaults)


def make_raw(page: int, headers: list[str], rows: list[list[str]]) -> _RawTable:
    return _RawTable(page_number=page, headers=headers, rows=rows)


HEADERS_3 = ["Name", "Score", "Department"]
ROWS_P1 = [["Alice", "92", "Engineering"], ["Bob", "87", "Engineering"]]
ROWS_P2 = [["Carol", "91", "Marketing"], ["Dave", "85", "Marketing"]]
ROWS_P3 = [["Eve", "95", "Research"]]


def make_pdfplumber_mock(pages_tables: list[list[list[list[str]]]], page_texts: list[str] | None = None):
    """
    pages_tables: list[page][table_idx][row][cell]
    page_texts: text to return from page.extract_text() per page
    """
    page_texts = page_texts or [""] * len(pages_tables)
    mock_pdf = MagicMock()
    mock_pdf.__enter__ = lambda s: s
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.close = MagicMock()

    pages = []
    for i, (tables_data, text) in enumerate(zip(pages_tables, page_texts)):
        page = MagicMock()
        page.extract_tables.return_value = tables_data
        page.extract_text.return_value = text
        pages.append(page)

    mock_pdf.pages = pages
    return mock_pdf


# ═══════════════════════════════════════════════════════════════════════════════
# Config validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableStitchChunkerConfig:
    def test_defaults(self):
        c = TableStitchChunker()
        assert c.emit_format == "markdown"
        assert c.stitch_threshold == 3
        assert c.header_repeat_detection is True
        assert c.column_alignment_tolerance == 1
        assert c.include_page_range is True
        assert c.min_rows == 2
        assert c.chunk_text_between_tables is True
        assert c.tokenizer == "word_count"

    def test_custom_config(self):
        c = TableStitchChunker(config={
            "emit_format": "json",
            "stitch_threshold": 5,
            "min_rows": 3,
            "column_alignment_tolerance": 2,
        })
        assert c.emit_format == "json"
        assert c.stitch_threshold == 5
        assert c.min_rows == 3

    def test_invalid_emit_format(self):
        with pytest.raises(ValueError, match="emit_format"):
            TableStitchChunker(config={"emit_format": "html"})

    def test_invalid_stitch_threshold(self):
        with pytest.raises(ValueError, match="stitch_threshold"):
            TableStitchChunker(config={"stitch_threshold": 0})

    def test_invalid_column_alignment_tolerance(self):
        with pytest.raises(ValueError, match="column_alignment_tolerance"):
            TableStitchChunker(config={"column_alignment_tolerance": -1})

    def test_invalid_min_rows(self):
        with pytest.raises(ValueError, match="min_rows"):
            TableStitchChunker(config={"min_rows": 0})

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            TableStitchChunker(config={"chunk_size": 0})

    def test_invalid_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError):
            TableStitchChunker(config={"chunk_size": 50, "chunk_overlap": 50})

    def test_invalid_tokenizer(self):
        with pytest.raises(ValueError, match="tokenizer"):
            TableStitchChunker(config={"tokenizer": "unknown"})

    def test_all_valid_emit_formats(self):
        for fmt in _VALID_EMIT_FORMATS:
            c = TableStitchChunker(config={"emit_format": fmt})
            assert c.emit_format == fmt


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataStructures:
    def test_raw_table_col_count_from_headers(self):
        t = make_raw(1, HEADERS_3, ROWS_P1)
        assert t.col_count == 3

    def test_raw_table_col_count_from_rows_when_no_headers(self):
        t = _RawTable(page_number=1, headers=[], rows=[["a", "b"]])
        assert t.col_count == 2

    def test_raw_table_is_empty(self):
        t = _RawTable(page_number=1, headers=HEADERS_3, rows=[])
        assert t.is_empty()

    def test_stitched_table_initial_state(self):
        raw = make_raw(1, HEADERS_3, ROWS_P1)
        st = _StitchedTable(raw)
        assert st.headers == HEADERS_3
        assert st.rows == ROWS_P1
        assert st.page_start == 1
        assert st.page_end == 1

    def test_stitched_table_append(self):
        raw1 = make_raw(1, HEADERS_3, ROWS_P1)
        raw2 = make_raw(2, HEADERS_3, ROWS_P2)
        st = _StitchedTable(raw1)
        st.append(raw2)
        assert len(st.rows) == 4
        assert st.page_end == 2

    def test_stitched_table_data_row_count(self):
        raw = make_raw(1, HEADERS_3, ROWS_P1)
        st = _StitchedTable(raw)
        assert st.data_row_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# clean_table_data
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanTableData:
    def setup_method(self):
        self.c = make_chunker()

    def test_strips_none_cells(self):
        result = self.c._clean_table_data([[None, "hello", None]])
        assert result[0] == ["", "hello", ""]

    def test_strips_whitespace(self):
        result = self.c._clean_table_data([["  Name  ", " Score "]])
        assert result[0] == ["Name", "Score"]

    def test_removes_fully_empty_rows(self):
        result = self.c._clean_table_data([["", ""], ["Alice", "92"]])
        assert len(result) == 1
        assert result[0] == ["Alice", "92"]

    def test_preserves_mixed_rows(self):
        result = self.c._clean_table_data([
            ["Name", "Score"],
            [None, ""],
            ["Alice", "92"],
        ])
        assert len(result) == 2  # middle row removed (all empty after clean)


# ═══════════════════════════════════════════════════════════════════════════════
# split_header_rows
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitHeaderRows:
    def setup_method(self):
        self.c = make_chunker()

    def test_first_row_is_header(self):
        headers, rows = self.c._split_header_rows([HEADERS_3] + ROWS_P1)
        assert headers == HEADERS_3
        assert rows == ROWS_P1

    def test_auto_headers_for_numeric_first_row(self):
        data = [["42", "85", "91"], ["50", "77", "88"]]
        headers, rows = self.c._split_header_rows(data)
        assert headers[0].startswith("Col")
        assert len(headers) == 3
        assert len(rows) == 2

    def test_single_row_returns_header_only(self):
        headers, rows = self.c._split_header_rows([HEADERS_3])
        assert headers == HEADERS_3
        assert rows == []

    def test_empty_returns_empty(self):
        headers, rows = self.c._split_header_rows([])
        assert headers == [] and rows == []


# ═══════════════════════════════════════════════════════════════════════════════
# rows_match
# ═══════════════════════════════════════════════════════════════════════════════

class TestRowsMatch:
    def test_exact_match(self):
        assert TableStitchChunker._rows_match(["Name", "Score"], ["Name", "Score"])

    def test_case_insensitive_match(self):
        assert TableStitchChunker._rows_match(["name", "score"], ["Name", "Score"])

    def test_strip_whitespace(self):
        assert TableStitchChunker._rows_match(["  Name  ", "Score"], ["Name", "Score"])

    def test_length_mismatch_false(self):
        assert not TableStitchChunker._rows_match(["Name", "Score", "Extra"], ["Name", "Score"])

    def test_content_mismatch_false(self):
        assert not TableStitchChunker._rows_match(["Rank", "Score"], ["Name", "Score"])


# ═══════════════════════════════════════════════════════════════════════════════
# is_continuation
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsContinuation:
    def setup_method(self):
        self.c = make_chunker()

    def _stitched(self, page_start, page_end, col_count=3):
        raw = make_raw(page_start, ["A", "B", "C"][:col_count], [["x"] * col_count])
        st = _StitchedTable(raw)
        st.page_end = page_end
        return st

    def test_consecutive_same_columns_is_continuation(self):
        current = self._stitched(1, 1, col_count=3)
        fragment = make_raw(2, HEADERS_3, ROWS_P2)
        assert self.c._is_continuation(current, fragment)

    def test_non_consecutive_page_is_not_continuation(self):
        current = self._stitched(1, 1)
        fragment = make_raw(3, HEADERS_3, ROWS_P2)
        assert not self.c._is_continuation(current, fragment)

    def test_exceeds_stitch_threshold_is_not_continuation(self):
        c = TableStitchChunker(config={"stitch_threshold": 2, "min_rows": 1, "tokenizer": "word_count"})
        current = self._stitched(1, 2)  # already spans 2 pages
        fragment = make_raw(3, HEADERS_3, ROWS_P2)
        assert not c._is_continuation(current, fragment)

    def test_within_column_tolerance_is_continuation(self):
        c = TableStitchChunker(config={"column_alignment_tolerance": 1, "min_rows": 1, "tokenizer": "word_count"})
        current = self._stitched(1, 1, col_count=3)
        fragment = make_raw(2, ["A", "B"], ROWS_P2[:2])  # 2 cols (delta=1)
        assert c._is_continuation(current, fragment)

    def test_exceeds_column_tolerance_is_not_continuation(self):
        c = TableStitchChunker(config={"column_alignment_tolerance": 0, "min_rows": 1, "tokenizer": "word_count"})
        current = self._stitched(1, 1, col_count=3)
        fragment = make_raw(2, ["A", "B"], ROWS_P2[:2])  # 2 cols (delta=1 > tolerance=0)
        assert not c._is_continuation(current, fragment)


# ═══════════════════════════════════════════════════════════════════════════════
# stitch_tables
# ═══════════════════════════════════════════════════════════════════════════════

class TestStitchTables:
    def setup_method(self):
        self.c = make_chunker()

    def test_single_table_unchanged(self):
        raw = make_raw(1, HEADERS_3, ROWS_P1)
        stitched = self.c._stitch_tables([raw])
        assert len(stitched) == 1
        assert stitched[0].data_row_count == 2

    def test_two_page_table_stitched(self):
        raw1 = make_raw(1, HEADERS_3, ROWS_P1)
        raw2 = make_raw(2, HEADERS_3, ROWS_P2)
        stitched = self.c._stitch_tables([raw1, raw2])
        assert len(stitched) == 1
        assert stitched[0].data_row_count == 4
        assert stitched[0].page_start == 1
        assert stitched[0].page_end == 2

    def test_three_page_table_stitched(self):
        raw1 = make_raw(1, HEADERS_3, ROWS_P1)
        raw2 = make_raw(2, HEADERS_3, ROWS_P2)
        raw3 = make_raw(3, HEADERS_3, ROWS_P3)
        stitched = self.c._stitch_tables([raw1, raw2, raw3])
        assert len(stitched) == 1
        assert stitched[0].data_row_count == 5

    def test_non_consecutive_creates_separate_tables(self):
        raw1 = make_raw(1, HEADERS_3, ROWS_P1)
        raw2 = make_raw(3, HEADERS_3, ROWS_P2)  # page 3 — gap
        stitched = self.c._stitch_tables([raw1, raw2])
        assert len(stitched) == 2

    def test_repeated_header_stripped_on_continuation(self):
        # Page 2 starts with repeated header row
        rows_p2_with_header = [HEADERS_3] + ROWS_P2
        raw1 = make_raw(1, HEADERS_3, ROWS_P1)
        raw2 = make_raw(2, HEADERS_3, rows_p2_with_header)
        stitched = self.c._stitch_tables([raw1, raw2])
        assert len(stitched) == 1
        # Row count: 2 from p1 + 2 from p2 (header stripped)
        assert stitched[0].data_row_count == 4

    def test_header_not_stripped_when_detection_disabled(self):
        c = TableStitchChunker(config={
            "header_repeat_detection": False, "min_rows": 1, "tokenizer": "word_count"
        })
        rows_p2_with_header = [HEADERS_3] + ROWS_P2
        raw1 = make_raw(1, HEADERS_3, ROWS_P1)
        raw2 = make_raw(2, HEADERS_3, rows_p2_with_header)
        stitched = c._stitch_tables([raw1, raw2])
        # 2 + 3 rows (header row kept)
        assert stitched[0].data_row_count == 5

    def test_empty_input_returns_empty(self):
        assert self.c._stitch_tables([]) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Emit formats
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmitFormats:
    def _make_stitched(self, headers=HEADERS_3, rows=None):
        rows = rows or ROWS_P1 + ROWS_P2
        raw = make_raw(1, headers, rows)
        return _StitchedTable(raw)

    def test_markdown_has_header_row(self):
        text = _emit_markdown(self._make_stitched())
        assert "Name" in text and "Score" in text and "Department" in text

    def test_markdown_has_separator(self):
        text = _emit_markdown(self._make_stitched())
        assert "---" in text or "----" in text

    def test_markdown_has_data_rows(self):
        text = _emit_markdown(self._make_stitched())
        assert "Alice" in text and "Carol" in text

    def test_markdown_pipe_delimited(self):
        text = _emit_markdown(self._make_stitched())
        assert "|" in text

    def test_json_parses_to_list(self):
        text = _emit_json(self._make_stitched())
        data = json.loads(text)
        assert isinstance(data, list)
        assert len(data) == 4

    def test_json_row_keyed_by_header(self):
        text = _emit_json(self._make_stitched())
        data = json.loads(text)
        assert "Name" in data[0]
        assert data[0]["Name"] == "Alice"

    def test_json_all_rows_present(self):
        text = _emit_json(self._make_stitched())
        data = json.loads(text)
        names = [row["Name"] for row in data]
        assert "Alice" in names and "Carol" in names

    def test_csv_has_header_row(self):
        text = _emit_csv(self._make_stitched())
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        assert rows[0] == HEADERS_3

    def test_csv_has_data_rows(self):
        text = _emit_csv(self._make_stitched())
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        assert len(rows) == 5  # 1 header + 4 data rows

    def test_csv_data_values_correct(self):
        text = _emit_csv(self._make_stitched())
        assert "Alice" in text and "Carol" in text

    def test_empty_table_returns_empty_string(self):
        empty = _StitchedTable(make_raw(1, HEADERS_3, []))
        assert _emit_markdown(empty).strip() == ""


# ═══════════════════════════════════════════════════════════════════════════════
# table_to_chunk metadata
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableToChunk:
    def setup_method(self):
        self.c = make_chunker()

    def _stitched(self, page_start=1, page_end=1, rows=None):
        rows = rows or ROWS_P1 + ROWS_P2
        raw = make_raw(page_start, HEADERS_3, rows)
        st = _StitchedTable(raw)
        st.page_end = page_end
        return st

    def test_returns_chunk_model(self):
        chunk = self.c._table_to_chunk(self._stitched(), "doc-001", {}, 0)
        assert isinstance(chunk, ChunkModel)

    def test_chunk_type_is_table(self):
        chunk = self.c._table_to_chunk(self._stitched(), "doc-001", {}, 0)
        assert chunk.metadata["chunk_type"] == "table"

    def test_emit_format_in_metadata(self):
        chunk = self.c._table_to_chunk(self._stitched(), "doc-001", {}, 0)
        assert chunk.metadata["emit_format"] == "markdown"

    def test_row_count_in_metadata(self):
        chunk = self.c._table_to_chunk(self._stitched(), "doc-001", {}, 0)
        assert chunk.metadata["row_count"] == 4

    def test_col_count_in_metadata(self):
        chunk = self.c._table_to_chunk(self._stitched(), "doc-001", {}, 0)
        assert chunk.metadata["col_count"] == 3

    def test_single_page_stitched_false(self):
        chunk = self.c._table_to_chunk(self._stitched(1, 1), "doc-001", {}, 0)
        assert chunk.metadata["stitched"] is False

    def test_multi_page_stitched_true(self):
        chunk = self.c._table_to_chunk(self._stitched(1, 2), "doc-001", {}, 0)
        assert chunk.metadata["stitched"] is True

    def test_page_range_in_metadata(self):
        chunk = self.c._table_to_chunk(self._stitched(1, 2), "doc-001", {}, 0)
        assert chunk.metadata["page_start"] == 1
        assert chunk.metadata["page_end"] == 2

    def test_pages_stitched_in_metadata_for_multi_page(self):
        chunk = self.c._table_to_chunk(self._stitched(1, 3), "doc-001", {}, 0)
        assert chunk.metadata["pages_stitched"] == 3

    def test_page_range_excluded_when_disabled(self):
        c = TableStitchChunker(config={"include_page_range": False, "min_rows": 1, "tokenizer": "word_count"})
        chunk = c._table_to_chunk(self._stitched(), "doc-001", {}, 0)
        assert "page_start" not in chunk.metadata

    def test_below_min_rows_returns_none(self):
        c = TableStitchChunker(config={"min_rows": 10, "tokenizer": "word_count"})
        result = c._table_to_chunk(self._stitched(rows=ROWS_P1), "doc-001", {}, 0)
        assert result is None

    def test_chunk_index_set_correctly(self):
        chunk = self.c._table_to_chunk(self._stitched(), "doc-001", {}, 5)
        assert chunk.chunk_index == 5

    def test_doc_id_propagated(self):
        chunk = self.c._table_to_chunk(self._stitched(), "my-doc", {}, 0)
        assert chunk.doc_id == "my-doc"


# ═══════════════════════════════════════════════════════════════════════════════
# chunk_pdf_bytes (pdfplumber mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkPdfBytes:
    def _run(self, pages_tables, page_texts=None, **cfg_kwargs):
        mock_pdf = make_pdfplumber_mock(pages_tables, page_texts)
        cfg = {"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5, "min_rows": 1}
        cfg.update(cfg_kwargs)
        with patch("raglab_chunkers.table_stitch_chunker.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = mock_pdf
            chunker = TableStitchChunker(config=cfg)
            return chunker.chunk_pdf_bytes(b"fake-pdf", "doc-pdf")

    def test_single_page_table_produces_chunk(self):
        chunks = self._run([[[HEADERS_3] + ROWS_P1]])
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        assert len(table_chunks) == 1

    def test_two_page_table_stitched(self):
        # Page 1: table with headers + rows; Page 2: continuation rows
        chunks = self._run([
            [[HEADERS_3] + ROWS_P1],  # page 1
            [[HEADERS_3] + ROWS_P2],  # page 2 (continuation)
        ])
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        assert len(table_chunks) == 1
        assert table_chunks[0].metadata["stitched"] is True
        assert table_chunks[0].metadata["page_start"] == 1
        assert table_chunks[0].metadata["page_end"] == 2

    def test_two_separate_tables_produce_two_chunks(self):
        # Page 1 and page 3 (non-consecutive = separate tables)
        chunks = self._run([
            [[HEADERS_3] + ROWS_P1],  # page 1
            [],                        # page 2: no table
            [[HEADERS_3] + ROWS_P2],  # page 3: new table
        ])
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        assert len(table_chunks) == 2

    def test_free_text_also_chunked(self):
        chunks = self._run(
            [[[HEADERS_3] + ROWS_P1]],
            page_texts=["This document discusses employee performance metrics across departments."],
        )
        text_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "text"]
        assert len(text_chunks) >= 1

    def test_free_text_disabled(self):
        chunks = self._run(
            [[[HEADERS_3] + ROWS_P1]],
            page_texts=["Some text here."],
            chunk_text_between_tables=False,
        )
        text_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "text"]
        assert len(text_chunks) == 0

    def test_sequential_chunk_indices(self):
        chunks = self._run(
            [[[HEADERS_3] + ROWS_P1]],
            page_texts=["Free text on this page about performance."],
        )
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_header_repeat_stripped_in_pdf_mode(self):
        p2_with_header = [HEADERS_3] + ROWS_P2  # repeated header on page 2
        chunks = self._run([
            [[HEADERS_3] + ROWS_P1],
            [p2_with_header],
        ])
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        # 4 data rows total (not 5 — header stripped)
        assert table_chunks[0].metadata["row_count"] == 4

    def test_json_emit_format(self):
        chunks = self._run([[[HEADERS_3] + ROWS_P1]], emit_format="json")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        data = json.loads(table_chunks[0].text)
        assert isinstance(data, list)
        assert data[0]["Name"] == "Alice"

    def test_csv_emit_format(self):
        chunks = self._run([[[HEADERS_3] + ROWS_P1]], emit_format="csv")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        assert "Name" in table_chunks[0].text
        assert "Alice" in table_chunks[0].text

    def test_empty_pdf_returns_empty(self):
        chunks = self._run([])
        assert chunks == []


# ═══════════════════════════════════════════════════════════════════════════════
# _chunk plain text fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkTextFallback:
    def setup_method(self):
        self.c = make_chunker()

    PIPE_TABLE = """\
| Name  | Score | Department  |
|-------|-------|-------------|
| Alice | 92    | Engineering |
| Bob   | 87    | Engineering |
| Carol | 91    | Marketing   |"""

    def test_pipe_table_parsed_as_table(self):
        chunks = self.c.chunk(self.PIPE_TABLE, "doc-001")
        table_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "table"]
        assert len(table_chunks) >= 1

    def test_pipe_table_contains_data(self):
        chunks = self.c.chunk(self.PIPE_TABLE, "doc-001")
        all_text = " ".join(c.text for c in chunks)
        assert "Alice" in all_text

    def test_plain_text_chunked_via_windows(self):
        text = "This is plain text without any table structure. " * 10
        chunks = self.c.chunk(text, "doc-001")
        assert len(chunks) >= 1
        assert all(isinstance(c, ChunkModel) for c in chunks)

    def test_empty_text_returns_empty(self):
        assert self.c.chunk("", "doc-001") == []

    def test_doc_id_propagated_in_text_fallback(self):
        chunks = self.c.chunk("Some text here.", "my-doc")
        assert all(c.doc_id == "my-doc" for c in chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

class TestTableStitchFactory:
    def test_factory_creates_table_stitch_chunker(self):
        from raglab_chunkers import ChunkerFactory
        c = ChunkerFactory.create("table_stitch", config={"tokenizer": "word_count"})
        assert isinstance(c, TableStitchChunker)

    def test_table_stitch_active_in_available(self):
        from raglab_chunkers import ChunkerFactory
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        assert entries["table_stitch"]["active"] is True

    def test_all_r4_types_active(self):
        from raglab_chunkers import ChunkerFactory
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        assert entries["pdf_images"]["active"] is True
        assert entries["table_stitch"]["active"] is True

    def test_schema_via_factory(self):
        from raglab_chunkers import ChunkerFactory
        schema = ChunkerFactory.schema("table_stitch")
        for key in ["emit_format", "stitch_threshold", "header_repeat_detection",
                    "column_alignment_tolerance", "min_rows"]:
            assert key in schema


# ═══════════════════════════════════════════════════════════════════════════════
# config_schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigSchema:
    def test_schema_returns_dict(self):
        assert isinstance(TableStitchChunker.config_schema(), dict)

    def test_schema_has_all_keys(self):
        schema = TableStitchChunker.config_schema()
        for key in [
            "emit_format", "stitch_threshold", "header_repeat_detection",
            "column_alignment_tolerance", "include_page_range", "min_rows",
            "chunk_text_between_tables", "chunk_size", "chunk_overlap", "tokenizer",
        ]:
            assert key in schema, f"Missing: {key}"

    def test_schema_emit_format_options(self):
        schema = TableStitchChunker.config_schema()
        for fmt in ["markdown", "json", "csv"]:
            assert fmt in schema["emit_format"]["options"]

    def test_schema_defaults_match_class(self):
        schema = TableStitchChunker.config_schema()
        assert schema["emit_format"]["default"] == "markdown"
        assert schema["stitch_threshold"]["default"] == 3
        assert schema["header_repeat_detection"]["default"] is True
        assert schema["min_rows"]["default"] == 2
