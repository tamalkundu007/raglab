"""
raglab-chunkers — Chunker implementations for RAGLab.

Version: 0.2.0
Active in R1: TextChunker
Active in R2: PDFChunker, DOCXChunker, MarkdownChunker, HTMLChunker, ExcelChunker

Public API:
    from raglab_chunkers import ChunkerFactory, TextChunker, BaseChunker
"""

from raglab_chunkers.base import BaseChunker
from raglab_chunkers.factory import ChunkerFactory
from raglab_chunkers.text_chunker import TextChunker
from raglab_chunkers._boundary import split_into_windows, count_tokens, backtrack_to_boundary

__version__ = "0.2.0"

__all__ = [
    "BaseChunker",
    "ChunkerFactory",
    "TextChunker",
    "split_into_windows",
    "count_tokens",
    "backtrack_to_boundary",
]
