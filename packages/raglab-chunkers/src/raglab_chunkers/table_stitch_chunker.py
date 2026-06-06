"""
TableStitchChunker — cross-page table stitching for RAG pipelines.

The problem this solves:
    PDF tables that span page breaks lose meaning when chunked naively.
    A page-by-page chunker sees:

        Page 1:  | Name  | Score | Department |
                 | Alice |  92   | Engineering|
                 | Bob   |  87   | Engineering|

        Page 2:  | Carol |  91   | Marketing  |   ← no header, orphaned rows
                 | Dave  |  85   | Marketing  |

    A retriever that fetches page 2's fragment has no idea what the columns
    mean. The downstream LLM answers wrong or says "insufficient context."

    TableStitchChunker detects that page 2's first rows are a continuation
    of page 1's table (via header repeat detection + column alignment), then
    reconstructs the logical table before chunking it.

Strategy (R4 FRS):
    1. **Extract tables per page** using pdfplumber (handles bordered + borderless).
    2. **Detect continuations** — a fragment is a continuation candidate when:
         a. Column count matches the previous page's table (within tolerance).
         b. No header row is present (header_repeat_detection=True: headers that
            appear verbatim on the next page are stripped and used as alignment).
         c. The start position (y-coordinate) is near the top of the page.
    3. **Stitch** continuation fragments onto the preceding table.
    4. **Emit** each logical table as a ChunkModel. Emit formats:
         - "markdown"  — GFM table (default, most LLM-friendly)
         - "json"      — list of row dicts keyed by header
         - "csv"       — comma-separated values

    Free text between tables is chunked via split_into_windows() (reuse rule).

Parameters:
    emit_format              : str   = "markdown"
    stitch_threshold         : int   = 3      — max pages a table can span
    header_repeat_detection  : bool  = True   — detect and strip repeated headers
    column_alignment_tolerance: int  = 1      — allowed column count delta for continuation
    include_page_range       : bool  = True   — inject page_start/page_end in metadata
    min_rows                 : int   = 2      — min data rows to emit as a table chunk
    chunk_text_between_tables: bool  = True   — also chunk free text via split_into_windows
    chunk_size               : int   = 500    — for free text chunking
    chunk_overlap            : int   = 50
    tokenizer                : str   = "word_count"

Reuse rule: free text portions use split_into_windows() — no reimplementation.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from typing import Any

from raglab_common.exceptions import ChunkerError
from raglab_common.models import ChunkModel

from raglab_chunkers._boundary import count_tokens, split_into_windows
from raglab_chunkers.base import BaseChunker

# Module-level import for test patchability
try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

_VALID_EMIT_FORMATS = ("markdown", "json", "csv")


# ── Table data structures ──────────────────────────────────────────────────────

class _RawTable:
    """A table fragment extracted from a single PDF page."""

    __slots__ = ("page_number", "headers", "rows", "bbox")

    def __init__(
        self,
        page_number: int,
        headers: list[str],
        rows: list[list[str]],
        bbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.page_number = page_number
        self.headers = headers
        self.rows = rows
        self.bbox = bbox  # (x0, top, x1, bottom)

    @property
    def col_count(self) -> int:
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)

    def is_empty(self) -> bool:
        return not self.rows


class _StitchedTable:
    """A logical table assembled from one or more _RawTable fragments."""

    def __init__(self, initial: _RawTable) -> None:
        self.headers: list[str] = initial.headers
        self.rows: list[list[str]] = list(initial.rows)
        self.page_start: int = initial.page_number
        self.page_end: int = initial.page_number

    def append(self, fragment: _RawTable) -> None:
        """Append rows from a continuation fragment."""
        self.rows.extend(fragment.rows)
        self.page_end = fragment.page_number

    @property
    def data_row_count(self) -> int:
        return len(self.rows)


# ── Emit helpers ───────────────────────────────────────────────────────────────

def _emit_markdown(table: _StitchedTable) -> str:
    """Emit GFM-compatible Markdown table."""
    if not table.headers or not table.rows:
        return ""

    col_widths = [max(len(h), max((len(str(r[i])) for r in table.rows if i < len(r)), default=0))
                  for i, h in enumerate(table.headers)]

    def _row(cells: list[str]) -> str:
        padded = [str(cells[i]) if i < len(cells) else "" for i in range(len(table.headers))]
        return "| " + " | ".join(
            c.ljust(col_widths[i]) for i, c in enumerate(padded)
        ) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [_row(table.headers), sep]
    for row in table.rows:
        lines.append(_row(row))
    return "\n".join(lines)


def _emit_json(table: _StitchedTable) -> str:
    """Emit JSON array of row dicts keyed by header."""
    result = []
    for row in table.rows:
        obj = {}
        for i, header in enumerate(table.headers):
            obj[header] = row[i] if i < len(row) else ""
        result.append(obj)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _emit_csv(table: _StitchedTable) -> str:
    """Emit CSV with header row."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    if table.headers:
        writer.writerow(table.headers)
    for row in table.rows:
        writer.writerow(row)
    return buf.getvalue().strip()


_EMITTERS = {
    "markdown": _emit_markdown,
    "json": _emit_json,
    "csv": _emit_csv,
}


# ── Main chunker ───────────────────────────────────────────────────────────────

class TableStitchChunker(BaseChunker):
    """
    Cross-page table stitching chunker for RAG pipelines. Activates in R4.

    Detects tables spanning multiple PDF pages, reconstructs them into
    logical units, and emits each as a self-contained ChunkModel.
    """

    chunker_type: str = "table_stitch"

    _DEFAULT_EMIT_FORMAT: str = "markdown"
    _DEFAULT_STITCH_THRESHOLD: int = 3
    _DEFAULT_HEADER_REPEAT_DETECTION: bool = True
    _DEFAULT_COLUMN_ALIGNMENT_TOLERANCE: int = 1
    _DEFAULT_INCLUDE_PAGE_RANGE: bool = True
    _DEFAULT_MIN_ROWS: int = 2
    _DEFAULT_CHUNK_TEXT_BETWEEN_TABLES: bool = True
    _DEFAULT_CHUNK_SIZE: int = 500
    _DEFAULT_CHUNK_OVERLAP: int = 50
    _DEFAULT_TOKENIZER: str = "word_count"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}

        self.emit_format: str = cfg.get("emit_format", self._DEFAULT_EMIT_FORMAT)
        self.stitch_threshold: int = int(cfg.get("stitch_threshold", self._DEFAULT_STITCH_THRESHOLD))
        self.header_repeat_detection: bool = bool(
            cfg.get("header_repeat_detection", self._DEFAULT_HEADER_REPEAT_DETECTION)
        )
        self.column_alignment_tolerance: int = int(
            cfg.get("column_alignment_tolerance", self._DEFAULT_COLUMN_ALIGNMENT_TOLERANCE)
        )
        self.include_page_range: bool = bool(
            cfg.get("include_page_range", self._DEFAULT_INCLUDE_PAGE_RANGE)
        )
        self.min_rows: int = int(cfg.get("min_rows", self._DEFAULT_MIN_ROWS))
        self.chunk_text_between_tables: bool = bool(
            cfg.get("chunk_text_between_tables", self._DEFAULT_CHUNK_TEXT_BETWEEN_TABLES)
        )
        self.chunk_size: int = int(cfg.get("chunk_size", self._DEFAULT_CHUNK_SIZE))
        self.chunk_overlap: int = int(cfg.get("chunk_overlap", self._DEFAULT_CHUNK_OVERLAP))
        self.tokenizer: str = cfg.get("tokenizer", self._DEFAULT_TOKENIZER)

        # Validation
        if self.emit_format not in _VALID_EMIT_FORMATS:
            raise ValueError(
                f"emit_format must be one of {_VALID_EMIT_FORMATS}, got {self.emit_format!r}"
            )
        if self.stitch_threshold < 1:
            raise ValueError(f"stitch_threshold must be >= 1, got {self.stitch_threshold}")
        if self.column_alignment_tolerance < 0:
            raise ValueError(
                f"column_alignment_tolerance must be >= 0, got {self.column_alignment_tolerance}"
            )
        if self.min_rows < 1:
            raise ValueError(f"min_rows must be >= 1, got {self.min_rows}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.tokenizer not in ("tiktoken", "word_count"):
            raise ValueError(
                f"tokenizer must be 'tiktoken' or 'word_count', got {self.tokenizer!r}"
            )

    # ── Public entry points ────────────────────────────────────────────────────

    def chunk_pdf_bytes(
        self,
        pdf_bytes: bytes,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkModel]:
        """
        Extract, stitch, and chunk tables from a PDF byte stream.

        Args:
            pdf_bytes: Raw PDF bytes.
            doc_id:    Document identifier.
            metadata:  Optional base metadata.

        Returns:
            ChunkModel list — one per logical table + free-text chunks if enabled.
        """
        if pdfplumber is None:
            raise ChunkerError("pdfplumber not installed. Run: pip install pdfplumber")

        metadata = metadata or {}
        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_bytes))
        except Exception as exc:
            raise ChunkerError(f"Failed to open PDF: {exc}") from exc

        try:
            raw_tables: list[_RawTable] = []
            free_texts: list[tuple[int, str]] = []  # (page_number, text)

            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = self._extract_page_tables(page, page_num)
                raw_tables.extend(page_tables)

                if self.chunk_text_between_tables:
                    text = page.extract_text() or ""
                    if text.strip():
                        free_texts.append((page_num, text.strip()))
        finally:
            pdf.close()

        stitched = self._stitch_tables(raw_tables)
        chunks: list[ChunkModel] = []
        chunk_index = 0

        # Emit stitched table chunks
        for table in stitched:
            chunk = self._table_to_chunk(table, doc_id, metadata, chunk_index)
            if chunk is not None:
                chunks.append(chunk)
                chunk_index += 1

        # Emit free text chunks
        if self.chunk_text_between_tables:
            for page_num, text in free_texts:
                text_chunks = self._text_to_chunks(text, doc_id, metadata, page_num, chunk_index)
                chunks.extend(text_chunks)
                chunk_index += len(text_chunks)

        return chunks

    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        """
        Fallback: chunk pre-extracted text (plain text / CSV-like input).

        When called via the standard chunk() API with plain text, attempts
        to parse simple table structures, then falls back to split_into_windows().
        """
        # Try parsing as a simple pipe-delimited table
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines and "|" in lines[0]:
            table = self._parse_pipe_table(lines)
            if table and table.data_row_count >= self.min_rows:
                chunk = self._table_to_chunk(table, doc_id, metadata, 0)
                if chunk:
                    return [chunk]

        # Fall back to text chunking
        return self._text_to_chunks(text, doc_id, metadata, page_number=None, start_index=0)

    # ── Table extraction ───────────────────────────────────────────────────────

    def _extract_page_tables(self, page: Any, page_number: int) -> list[_RawTable]:
        """Extract all tables from a pdfplumber page object."""
        raw_tables = []
        try:
            tables = page.extract_tables()
        except Exception:
            return []

        for tbl_data in tables:
            if not tbl_data:
                continue

            cleaned = self._clean_table_data(tbl_data)
            if not cleaned:
                continue

            headers, rows = self._split_header_rows(cleaned)
            if not rows:
                continue

            raw_tables.append(_RawTable(
                page_number=page_number,
                headers=headers,
                rows=rows,
                bbox=None,
            ))

        return raw_tables

    def _clean_table_data(self, raw: list[list[Any]]) -> list[list[str]]:
        """Normalise raw pdfplumber table data — strip None, whitespace."""
        cleaned = []
        for row in raw:
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(cells):  # skip fully empty rows
                cleaned.append(cells)
        return cleaned

    def _split_header_rows(
        self, rows: list[list[str]]
    ) -> tuple[list[str], list[list[str]]]:
        """
        Split table rows into headers and data rows.

        Heuristic: first row is headers if it contains non-numeric,
        non-empty strings and differs from subsequent rows.
        """
        if not rows:
            return [], []

        first_row = rows[0]
        if len(rows) < 2:
            return first_row, []

        # Header heuristic: first row is header if it looks like labels
        # (contains at least one non-numeric, non-empty string)
        def _is_header_like(row: list[str]) -> bool:
            return any(
                cell and not cell.replace(".", "").replace(",", "").replace("-", "").isnumeric()
                for cell in row
            )

        if _is_header_like(first_row):
            return first_row, rows[1:]
        else:
            # Auto-generate column headers
            auto_headers = [f"Col{i+1}" for i in range(len(first_row))]
            return auto_headers, rows

    # ── Stitching ─────────────────────────────────────────────────────────────

    def _stitch_tables(self, raw_tables: list[_RawTable]) -> list[_StitchedTable]:
        """
        Group raw table fragments into logical stitched tables.

        A fragment is appended to the current stitched table when:
          1. It immediately follows the previous fragment (page_number == prev + 1).
          2. Column count is within column_alignment_tolerance.
          3. If header_repeat_detection=True: first row of fragment matches
             the stitched table's headers → strip it (repeated header).
          4. Total page span does not exceed stitch_threshold.
        """
        if not raw_tables:
            return []

        stitched: list[_StitchedTable] = []
        current: _StitchedTable | None = None

        for fragment in raw_tables:
            if current is None:
                current = _StitchedTable(fragment)
                continue

            can_stitch = self._is_continuation(current, fragment)

            if can_stitch:
                rows_to_add = list(fragment.rows)

                # Strip repeated header row
                if (
                    self.header_repeat_detection
                    and rows_to_add
                    and self._rows_match(rows_to_add[0], current.headers)
                ):
                    rows_to_add = rows_to_add[1:]

                if rows_to_add:
                    fragment_copy = _RawTable(
                        page_number=fragment.page_number,
                        headers=fragment.headers,
                        rows=rows_to_add,
                    )
                    current.append(fragment_copy)
            else:
                stitched.append(current)
                current = _StitchedTable(fragment)

        if current is not None:
            stitched.append(current)

        return stitched

    def _is_continuation(
        self, current: _StitchedTable, fragment: _RawTable
    ) -> bool:
        """Return True if fragment is a continuation of the current stitched table."""
        # Must be on the immediately next page
        if fragment.page_number != current.page_end + 1:
            return False

        # Must not exceed stitch_threshold pages
        span = fragment.page_number - current.page_start + 1
        if span > self.stitch_threshold:
            return False

        # Column count must be within tolerance
        col_delta = abs(fragment.col_count - len(current.headers))
        if col_delta > self.column_alignment_tolerance:
            return False

        return True

    @staticmethod
    def _rows_match(row: list[str], headers: list[str]) -> bool:
        """Return True if a row matches the header row (case-insensitive strip)."""
        if len(row) != len(headers):
            return False
        return all(
            r.strip().lower() == h.strip().lower()
            for r, h in zip(row, headers)
        )

    # ── Emit ──────────────────────────────────────────────────────────────────

    def _table_to_chunk(
        self,
        table: _StitchedTable,
        doc_id: str,
        metadata: dict[str, Any],
        chunk_index: int,
    ) -> ChunkModel | None:
        """Convert a stitched table to a ChunkModel in the configured format."""
        if table.data_row_count < self.min_rows:
            return None

        emitter = _EMITTERS[self.emit_format]
        text = emitter(table)
        if not text.strip():
            return None

        chunk_meta: dict[str, Any] = {
            **metadata,
            "chunker": self.chunker_type,
            "chunk_type": "table",
            "emit_format": self.emit_format,
            "row_count": table.data_row_count,
            "col_count": len(table.headers),
            "stitched": table.page_end > table.page_start,
        }
        if self.include_page_range:
            chunk_meta["page_start"] = table.page_start
            chunk_meta["page_end"] = table.page_end
            if table.page_end > table.page_start:
                chunk_meta["pages_stitched"] = table.page_end - table.page_start + 1

        return ChunkModel(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            text=text,
            chunk_index=chunk_index,
            token_count=count_tokens(text, mode=self.tokenizer),
            metadata=chunk_meta,
        )

    def _text_to_chunks(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any],
        page_number: int | None,
        start_index: int,
    ) -> list[ChunkModel]:
        """Chunk free text via split_into_windows() (reuse rule)."""
        if not text.strip():
            return []

        raw_chunks = split_into_windows(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=self.tokenizer,
            min_chunk_size=10,
        )

        result = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_meta: dict[str, Any] = {
                **metadata,
                "chunker": self.chunker_type,
                "chunk_type": "text",
                "tokenizer": self.tokenizer,
            }
            if page_number is not None:
                chunk_meta["page_number"] = page_number

            result.append(ChunkModel(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=start_index + i,
                token_count=count_tokens(chunk_text, mode=self.tokenizer),
                metadata=chunk_meta,
            ))
        return result

    # ── Pipe-table parser (for plain text input) ───────────────────────────────

    def _parse_pipe_table(self, lines: list[str]) -> _StitchedTable | None:
        """
        Parse a simple pipe-delimited Markdown/plain-text table.
        Used by _chunk() for text-mode input.
        """
        rows = []
        for line in lines:
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if not all(set(c).issubset("-| ") for c in cells):  # skip separator rows
                    rows.append(cells)

        if not rows:
            return None

        cleaned = self._clean_table_data(rows)
        headers, data_rows = self._split_header_rows(cleaned)
        if not data_rows:
            return None

        raw = _RawTable(page_number=1, headers=headers, rows=data_rows)
        return _StitchedTable(raw)

    # ── Schema ────────────────────────────────────────────────────────────────

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "emit_format": {
                "type": "str", "default": cls._DEFAULT_EMIT_FORMAT,
                "options": list(_VALID_EMIT_FORMATS),
                "description": "Output format for each table chunk: markdown (LLM-friendly), json, csv.",
            },
            "stitch_threshold": {
                "type": "int", "default": cls._DEFAULT_STITCH_THRESHOLD,
                "min": 1, "max": 20,
                "description": "Maximum number of pages a table is allowed to span.",
            },
            "header_repeat_detection": {
                "type": "bool", "default": cls._DEFAULT_HEADER_REPEAT_DETECTION,
                "description": "Detect and strip repeated header rows on continuation pages.",
            },
            "column_alignment_tolerance": {
                "type": "int", "default": cls._DEFAULT_COLUMN_ALIGNMENT_TOLERANCE,
                "min": 0, "max": 5,
                "description": "Allowed column count difference for continuation detection.",
            },
            "include_page_range": {
                "type": "bool", "default": cls._DEFAULT_INCLUDE_PAGE_RANGE,
                "description": "Inject page_start / page_end into chunk metadata.",
            },
            "min_rows": {
                "type": "int", "default": cls._DEFAULT_MIN_ROWS,
                "min": 1, "max": 100,
                "description": "Minimum data rows for a table chunk to be emitted.",
            },
            "chunk_text_between_tables": {
                "type": "bool", "default": cls._DEFAULT_CHUNK_TEXT_BETWEEN_TABLES,
                "description": "Also chunk free text (non-table) content via split_into_windows().",
            },
            "chunk_size": {
                "type": "int", "default": cls._DEFAULT_CHUNK_SIZE,
                "min": 50, "max": 4000,
                "description": "Token window for free text chunking.",
            },
            "chunk_overlap": {
                "type": "int", "default": cls._DEFAULT_CHUNK_OVERLAP,
                "min": 0, "max": 500,
                "description": "Overlap for free text chunking.",
            },
            "tokenizer": {
                "type": "str", "default": cls._DEFAULT_TOKENIZER,
                "options": ["tiktoken", "word_count"],
                "description": "Token counting mode.",
            },
        }
