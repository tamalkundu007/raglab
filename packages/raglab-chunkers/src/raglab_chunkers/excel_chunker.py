"""
ExcelChunker — sheet/row/column-aware structured chunking for RAG.

Strategy (R2 FRS spec):
    Spreadsheets are not prose. Throwing a blind token split at a CSV loses
    the column-to-value relationship that makes each row meaningful.

    Three sheet strategies:
        "row"    — each row becomes a text representation with column headers
                   prepended. Groups of rows are then binned to chunk_size.
        "column" — each column is serialised as a list of (header, value) pairs.
        "cell"   — every non-empty cell becomes its own chunk (smallest granularity).

    Header rows:
        `header_rows=1` means the first row contains column names, which are
        prepended to every subsequent data row in the chunk text. This ensures
        a retrieved chunk is self-contained — downstream LLM never sees a raw
        "42, London, True" without knowing what those values mean.

    Sheet name injection:
        If `include_sheet_name=True`, "Sheet: <name>" is prepended to each chunk.

Parameters:
    sheet_strategy   : str  = "row"   — "row" | "column" | "cell"
    header_rows      : int  = 1       — rows to treat as column headers
    row_grouping     : int  = 10      — rows per chunk (for "row" strategy)
    include_sheet_name: bool = True   — prepend sheet name to each chunk
    tokenizer        : str  = "word_count"  — default word_count (no tiktoken dep)
    chunk_size       : int  = 500     — max tokens per chunk (fallback split)
    chunk_overlap    : int  = 0       — overlap (0 is natural for tabular data)

Reuse rule: if a row-group text exceeds chunk_size, delegates to
`_boundary.split_into_windows()`. Primary structure is always row-based first.
"""

from __future__ import annotations

import io
import uuid
from typing import Any

from raglab_common.models import ChunkModel

from raglab_chunkers._boundary import count_tokens, split_into_windows
from raglab_chunkers.base import BaseChunker


def _row_to_text(headers: list[str], row_values: list[Any]) -> str:
    """Convert a row to 'Header: Value | Header: Value' text representation."""
    parts = []
    for h, v in zip(headers, row_values):
        val = str(v).strip() if v is not None else ""
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(f"{h}: {val}")
    return " | ".join(parts)


class ExcelChunker(BaseChunker):
    """
    Sheet/row/column-aware Excel chunker using openpyxl + pandas. Activates in R2.
    """

    chunker_type: str = "excel"

    _DEFAULT_SHEET_STRATEGY: str = "row"
    _DEFAULT_HEADER_ROWS: int = 1
    _DEFAULT_ROW_GROUPING: int = 10
    _DEFAULT_INCLUDE_SHEET_NAME: bool = True
    _DEFAULT_TOKENIZER: str = "word_count"
    _DEFAULT_CHUNK_SIZE: int = 500
    _DEFAULT_CHUNK_OVERLAP: int = 0

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.sheet_strategy: str = cfg.get("sheet_strategy", self._DEFAULT_SHEET_STRATEGY)
        self.header_rows: int = int(cfg.get("header_rows", self._DEFAULT_HEADER_ROWS))
        self.row_grouping: int = int(cfg.get("row_grouping", self._DEFAULT_ROW_GROUPING))
        self.include_sheet_name: bool = bool(cfg.get("include_sheet_name", self._DEFAULT_INCLUDE_SHEET_NAME))
        self.tokenizer: str = cfg.get("tokenizer", self._DEFAULT_TOKENIZER)
        self.chunk_size: int = int(cfg.get("chunk_size", self._DEFAULT_CHUNK_SIZE))
        self.chunk_overlap: int = int(cfg.get("chunk_overlap", self._DEFAULT_CHUNK_OVERLAP))

        if self.sheet_strategy not in ("row", "column", "cell"):
            raise ValueError(f"sheet_strategy must be 'row', 'column', or 'cell', got {self.sheet_strategy!r}")
        if self.header_rows < 0:
            raise ValueError(f"header_rows must be >= 0, got {self.header_rows}")
        if self.row_grouping < 1:
            raise ValueError(f"row_grouping must be >= 1, got {self.row_grouping}")
        if self.tokenizer not in ("tiktoken", "word_count"):
            raise ValueError(f"tokenizer must be 'tiktoken' or 'word_count', got {self.tokenizer!r}")

    def chunk_excel_bytes(
        self, excel_bytes: bytes, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> list[ChunkModel]:
        """Chunk from raw Excel bytes."""
        try:
            import pandas as pd
        except ImportError as exc:
            from raglab_common.exceptions import ChunkerError
            raise ChunkerError("pandas not installed. Run: pip install pandas openpyxl") from exc

        metadata = metadata or {}
        try:
            xl = pd.ExcelFile(io.BytesIO(excel_bytes), engine="openpyxl")
        except Exception as exc:
            from raglab_common.exceptions import ChunkerError
            raise ChunkerError(f"Failed to open Excel file: {exc}") from exc

        chunks: list[ChunkModel] = []
        chunk_index = 0

        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name, header=self.header_rows - 1 if self.header_rows > 0 else None)
            except Exception:
                continue

            sheet_chunks = self._chunk_dataframe(
                df=df,
                sheet_name=sheet_name,
                doc_id=doc_id,
                metadata=metadata,
                start_index=chunk_index,
            )
            chunks.extend(sheet_chunks)
            chunk_index += len(sheet_chunks)

        return chunks

    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        """
        Chunk from plain text (CSV-like input).

        Treats the text as TSV/CSV rows, parsing first line as headers.
        """
        lines = [l for l in text.strip().split("\n") if l.strip()]
        if not lines:
            return []

        # Auto-detect delimiter
        delimiter = "\t" if "\t" in lines[0] else ","
        headers: list[str] = []
        data_rows: list[list[str]] = []

        if self.header_rows > 0 and lines:
            headers = [h.strip() for h in lines[0].split(delimiter)]
            data_rows = [l.split(delimiter) for l in lines[1:]]
        else:
            num_cols = len(lines[0].split(delimiter))
            headers = [f"Col{i+1}" for i in range(num_cols)]
            data_rows = [l.split(delimiter) for l in lines]

        return self._rows_to_chunks(headers, data_rows, "text_input", doc_id, metadata, 0)

    def _chunk_dataframe(
        self,
        df: Any,
        sheet_name: str,
        doc_id: str,
        metadata: dict[str, Any],
        start_index: int,
    ) -> list[ChunkModel]:
        """Convert a pandas DataFrame to ChunkModel list."""
        headers = [str(c) for c in df.columns.tolist()]
        data_rows = df.values.tolist()
        return self._rows_to_chunks(headers, data_rows, sheet_name, doc_id, metadata, start_index)

    def _rows_to_chunks(
        self,
        headers: list[str],
        data_rows: list[list[Any]],
        sheet_name: str,
        doc_id: str,
        metadata: dict[str, Any],
        start_index: int,
    ) -> list[ChunkModel]:
        """Apply sheet strategy to convert rows to chunks."""
        chunks: list[ChunkModel] = []
        chunk_index = start_index
        sheet_prefix = f"Sheet: {sheet_name}\n" if self.include_sheet_name else ""

        if self.sheet_strategy == "cell":
            for row_idx, row in enumerate(data_rows):
                for col_idx, value in enumerate(row):
                    val_str = str(value).strip() if value is not None else ""
                    if not val_str or val_str.lower() in ("nan", "none", ""):
                        continue
                    header = headers[col_idx] if col_idx < len(headers) else f"Col{col_idx+1}"
                    cell_text = f"{sheet_prefix}{header}: {val_str}"
                    chunks.append(ChunkModel(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        text=cell_text,
                        chunk_index=chunk_index,
                        token_count=count_tokens(cell_text, mode=self.tokenizer),
                        metadata={**metadata, "chunker": self.chunker_type, "sheet": sheet_name, "row": row_idx, "col": header},
                    ))
                    chunk_index += 1

        elif self.sheet_strategy == "column":
            for col_idx, header in enumerate(headers):
                col_values = [row[col_idx] for row in data_rows if col_idx < len(row)]
                col_text = f"{sheet_prefix}Column: {header}\n" + "\n".join(
                    str(v) for v in col_values
                    if v is not None and str(v).strip().lower() not in ("nan", "none", "")
                )
                if not col_text.strip():
                    continue
                # Split if oversized
                raw = split_into_windows(
                    text=col_text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    boundary_enforcement=False,
                    tokenizer=self.tokenizer,
                    min_chunk_size=10,
                )
                for c in raw:
                    chunks.append(ChunkModel(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        text=c,
                        chunk_index=chunk_index,
                        token_count=count_tokens(c, mode=self.tokenizer),
                        metadata={**metadata, "chunker": self.chunker_type, "sheet": sheet_name, "column": header},
                    ))
                    chunk_index += 1

        else:  # "row" strategy (default)
            # Group rows into bins of `row_grouping`
            for group_start in range(0, len(data_rows), self.row_grouping):
                group = data_rows[group_start: group_start + self.row_grouping]
                row_texts = [_row_to_text(headers, row) for row in group]
                row_texts = [t for t in row_texts if t.strip()]
                if not row_texts:
                    continue
                group_text = sheet_prefix + "\n".join(row_texts)
                # If group is oversized, split it
                if count_tokens(group_text, mode=self.tokenizer) > self.chunk_size:
                    raw = split_into_windows(
                        text=group_text,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        boundary_enforcement=False,
                        tokenizer=self.tokenizer,
                        min_chunk_size=10,
                    )
                else:
                    raw = [group_text]

                for c in raw:
                    chunks.append(ChunkModel(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        text=c,
                        chunk_index=chunk_index,
                        token_count=count_tokens(c, mode=self.tokenizer),
                        metadata={**metadata, "chunker": self.chunker_type, "sheet": sheet_name, "row_group_start": group_start},
                    ))
                    chunk_index += 1

        return chunks

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "sheet_strategy": {"type": "str", "default": cls._DEFAULT_SHEET_STRATEGY, "options": ["row", "column", "cell"], "description": "How to decompose spreadsheet data into chunks."},
            "header_rows": {"type": "int", "default": cls._DEFAULT_HEADER_ROWS, "min": 0, "max": 5, "description": "Number of header rows containing column names."},
            "row_grouping": {"type": "int", "default": cls._DEFAULT_ROW_GROUPING, "min": 1, "max": 100, "description": "Number of rows per chunk (row strategy)."},
            "include_sheet_name": {"type": "bool", "default": cls._DEFAULT_INCLUDE_SHEET_NAME, "description": "Prepend sheet name to each chunk."},
            "tokenizer": {"type": "str", "default": cls._DEFAULT_TOKENIZER, "options": ["tiktoken", "word_count"], "description": "Token counting mode."},
            "chunk_size": {"type": "int", "default": cls._DEFAULT_CHUNK_SIZE, "min": 50, "max": 4000, "description": "Max tokens per chunk (fallback split threshold)."},
            "chunk_overlap": {"type": "int", "default": cls._DEFAULT_CHUNK_OVERLAP, "min": 0, "max": 200, "description": "Overlap between chunks (0 recommended for tabular data)."},
        }
