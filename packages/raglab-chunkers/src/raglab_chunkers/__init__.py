"""
raglab-chunkers — Chunker implementations for RAGLab.

Version: 0.3.0
Active in R1: TextChunker
Active in R2: PDFChunker, DOCXChunker, MarkdownChunker, HTMLChunker, ExcelChunker, HybridChunker
"""

from raglab_chunkers.base import BaseChunker
from raglab_chunkers.factory import ChunkerFactory
from raglab_chunkers.text_chunker import TextChunker
from raglab_chunkers.pdf_chunker import PDFChunker
from raglab_chunkers.docx_chunker import DOCXChunker
from raglab_chunkers.markdown_chunker import MarkdownChunker
from raglab_chunkers.html_chunker import HTMLChunker
from raglab_chunkers.excel_chunker import ExcelChunker
from raglab_chunkers.hybrid_chunker import HybridChunker
from raglab_chunkers.pdf_image_chunker import PDFImageChunker
from raglab_chunkers.table_stitch_chunker import TableStitchChunker
from raglab_chunkers.caption_service import CaptionService
from raglab_chunkers._boundary import split_into_windows, count_tokens, backtrack_to_boundary

__version__ = "0.5.0"

__all__ = [
    "BaseChunker", "ChunkerFactory",
    "TextChunker", "PDFChunker", "DOCXChunker",
    "MarkdownChunker", "HTMLChunker", "ExcelChunker", "HybridChunker",
    "split_into_windows", "count_tokens", "backtrack_to_boundary",
]
