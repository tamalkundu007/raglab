"""
MarkdownChunker — header-hierarchy-aware chunking with boundary backtracking.

Strategy (R2 FRS spec):
    1. Parse Markdown, split on headers at configured header levels (H1/H2/H3).
    2. Each header + its following body text becomes a structural unit.
    3. Run `split_into_windows()` within each structural unit.
    4. If `include_header_in_chunk=True`, prepend the header text to every
       chunk produced from that section.

Parameters:
    chunk_size           : int   = 500
    chunk_overlap        : int   = 50
    boundary_enforcement : bool  = True
    boundary_chars       : list  = ['.','!','?']
    tokenizer            : str   = "tiktoken" | "word_count"
    min_chunk_size       : int   = 50
    split_on_headers     : bool  = True  — use header hierarchy as structural units
    header_levels        : list  = [1,2,3] — which header levels trigger splits
    include_header_in_chunk: bool = True  — prepend header to each chunk

Reuse rule: token+boundary splitting via `_boundary.split_into_windows()` only.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from raglab_common.models import ChunkModel

from raglab_chunkers._boundary import count_tokens, split_into_windows
from raglab_chunkers.base import BaseChunker

# Regex: match Markdown ATX headers — group(1)=hashes, group(2)=text
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownChunker(BaseChunker):
    """
    Header-hierarchy-aware Markdown chunker.

    Splits at configured header levels (#/##/###), then applies
    token+boundary backtracking within each section. Activates in R2.
    """

    chunker_type: str = "markdown"

    _DEFAULT_CHUNK_SIZE: int = 500
    _DEFAULT_CHUNK_OVERLAP: int = 50
    _DEFAULT_BOUNDARY_ENFORCEMENT: bool = True
    _DEFAULT_BOUNDARY_CHARS: list[str] = [".", "!", "?"]
    _DEFAULT_TOKENIZER: str = "tiktoken"
    _DEFAULT_MIN_CHUNK_SIZE: int = 50
    _DEFAULT_SPLIT_ON_HEADERS: bool = True
    _DEFAULT_HEADER_LEVELS: list[int] = [1, 2, 3]
    _DEFAULT_INCLUDE_HEADER_IN_CHUNK: bool = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.chunk_size: int = int(cfg.get("chunk_size", self._DEFAULT_CHUNK_SIZE))
        self.chunk_overlap: int = int(cfg.get("chunk_overlap", self._DEFAULT_CHUNK_OVERLAP))
        self.boundary_enforcement: bool = bool(
            cfg.get("boundary_enforcement", self._DEFAULT_BOUNDARY_ENFORCEMENT)
        )
        self.boundary_chars: frozenset[str] = frozenset(
            cfg.get("boundary_chars", self._DEFAULT_BOUNDARY_CHARS)
        )
        self.tokenizer: str = cfg.get("tokenizer", self._DEFAULT_TOKENIZER)
        self.min_chunk_size: int = int(cfg.get("min_chunk_size", self._DEFAULT_MIN_CHUNK_SIZE))
        self.split_on_headers: bool = bool(
            cfg.get("split_on_headers", self._DEFAULT_SPLIT_ON_HEADERS)
        )
        self.header_levels: set[int] = set(
            cfg.get("header_levels", self._DEFAULT_HEADER_LEVELS)
        )
        self.include_header_in_chunk: bool = bool(
            cfg.get("include_header_in_chunk", self._DEFAULT_INCLUDE_HEADER_IN_CHUNK)
        )

        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.tokenizer not in ("tiktoken", "word_count"):
            raise ValueError(f"tokenizer must be 'tiktoken' or 'word_count', got {self.tokenizer!r}")
        if not self.header_levels.issubset({1, 2, 3, 4, 5, 6}):
            raise ValueError(f"header_levels must be a subset of 1-6, got {self.header_levels}")

    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        if not self.split_on_headers:
            # No header awareness — treat entire text as one stream
            return self._windows_to_chunks(
                split_into_windows(
                    text=text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    boundary_enforcement=self.boundary_enforcement,
                    boundary_chars=self.boundary_chars,
                    tokenizer=self.tokenizer,
                    min_chunk_size=self.min_chunk_size,
                ),
                doc_id=doc_id,
                metadata=metadata,
                start_index=0,
                header=None,
            )

        sections = self._split_by_headers(text)
        chunks: list[ChunkModel] = []
        chunk_index = 0

        for section in sections:
            header = section["header"]
            body = section["body"]

            section_text = f"{header}\n{body}" if (header and self.include_header_in_chunk) else body
            if not section_text.strip():
                continue

            raw_chunks = split_into_windows(
                text=section_text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                boundary_enforcement=self.boundary_enforcement,
                boundary_chars=self.boundary_chars,
                tokenizer=self.tokenizer,
                min_chunk_size=self.min_chunk_size,
            )

            for chunk_text in raw_chunks:
                chunk_meta = {
                    **metadata,
                    "chunker": self.chunker_type,
                    "tokenizer": self.tokenizer,
                    "boundary_enforcement": self.boundary_enforcement,
                }
                if header:
                    chunk_meta["header"] = header
                    chunk_meta["header_level"] = section["level"]

                chunks.append(ChunkModel(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=count_tokens(chunk_text, mode=self.tokenizer),
                    metadata=chunk_meta,
                ))
                chunk_index += 1

        return chunks

    def _split_by_headers(self, text: str) -> list[dict[str, Any]]:
        """
        Split Markdown text into sections at configured header levels.

        Returns a list of dicts: {header, level, body}
        Preamble text before the first header is returned as a section with header=None.
        """
        sections: list[dict[str, Any]] = []
        lines = text.split("\n")
        current_header: str | None = None
        current_level: int = 0
        current_body: list[str] = []

        for line in lines:
            match = _HEADER_RE.match(line)
            if match:
                level = len(match.group(1))
                if level in self.header_levels:
                    # Flush current section
                    body = "\n".join(current_body).strip()
                    if body or current_header is not None:
                        sections.append({
                            "header": current_header,
                            "level": current_level,
                            "body": body,
                        })
                    current_header = line.strip()
                    current_level = level
                    current_body = []
                    continue

            current_body.append(line)

        # Flush final section
        body = "\n".join(current_body).strip()
        if body or current_header is not None:
            sections.append({
                "header": current_header,
                "level": current_level,
                "body": body,
            })

        return sections

    def _windows_to_chunks(
        self,
        raw_chunks: list[str],
        doc_id: str,
        metadata: dict[str, Any],
        start_index: int,
        header: str | None,
    ) -> list[ChunkModel]:
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_meta = {
                **metadata,
                "chunker": self.chunker_type,
                "tokenizer": self.tokenizer,
            }
            if header:
                chunk_meta["header"] = header
            chunks.append(ChunkModel(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=start_index + i,
                token_count=count_tokens(chunk_text, mode=self.tokenizer),
                metadata=chunk_meta,
            ))
        return chunks

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "chunk_size": {"type": "int", "default": cls._DEFAULT_CHUNK_SIZE, "min": 50, "max": 4000, "description": "Target token count per chunk."},
            "chunk_overlap": {"type": "int", "default": cls._DEFAULT_CHUNK_OVERLAP, "min": 0, "max": 500, "description": "Overlap between consecutive chunks."},
            "boundary_enforcement": {"type": "bool", "default": cls._DEFAULT_BOUNDARY_ENFORCEMENT, "description": "Backtrack to nearest sentence boundary."},
            "boundary_chars": {"type": "list", "default": cls._DEFAULT_BOUNDARY_CHARS, "description": "Characters treated as sentence boundaries."},
            "tokenizer": {"type": "str", "default": cls._DEFAULT_TOKENIZER, "options": ["tiktoken", "word_count"], "description": "Token counting mode."},
            "min_chunk_size": {"type": "int", "default": cls._DEFAULT_MIN_CHUNK_SIZE, "min": 1, "max": 200, "description": "Minimum tokens per chunk."},
            "split_on_headers": {"type": "bool", "default": cls._DEFAULT_SPLIT_ON_HEADERS, "description": "Use Markdown headers as structural split boundaries."},
            "header_levels": {"type": "list", "default": cls._DEFAULT_HEADER_LEVELS, "options": [1, 2, 3, 4, 5, 6], "description": "Header levels (#/##/###) that trigger section splits."},
            "include_header_in_chunk": {"type": "bool", "default": cls._DEFAULT_INCLUDE_HEADER_IN_CHUNK, "description": "Prepend header text to every chunk in that section."},
        }
