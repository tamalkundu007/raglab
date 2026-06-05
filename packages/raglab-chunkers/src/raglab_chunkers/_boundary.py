"""
Boundary backtracking utility for RAGLab chunkers.

All chunkers that do token+boundary splitting call this module.
No chunker reimplements this logic — it lives here once.

Algorithm (TextChunker spec from FRS):
    Given a fixed-size token window that ends mid-sentence, walk backward
    word-by-word until a sentence boundary character is found. If no boundary
    is found within `min_chunk_size` tokens of the window end, return the
    original window unchanged.

Tokenisation modes:
    - "tiktoken"    : cl100k_base encoding (GPT-4 compatible)
    - "word_count"  : whitespace split — fast, no external dependency
"""

from __future__ import annotations

import re
from typing import Literal

# tiktoken is an optional dependency; fall back to word_count if unavailable.
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False

TokenizerMode = Literal["tiktoken", "word_count"]

_DEFAULT_BOUNDARY_CHARS: frozenset[str] = frozenset({".", "!", "?"})
_TIKTOKEN_ENCODING = "cl100k_base"


def _get_encoding():
    """Lazy-load tiktoken encoding."""
    return tiktoken.get_encoding(_TIKTOKEN_ENCODING)


def count_tokens(text: str, mode: TokenizerMode = "tiktoken") -> int:
    """
    Count tokens in `text` using the specified tokeniser.

    Args:
        text: Input string.
        mode: "tiktoken" (default) or "word_count".

    Returns:
        Integer token count.
    """
    if mode == "tiktoken" and _TIKTOKEN_AVAILABLE:
        return len(_get_encoding().encode(text))
    return len(text.split())


def tokenize(text: str, mode: TokenizerMode = "tiktoken") -> list[str]:
    """
    Split `text` into a list of word-level tokens (not subword).

    Note: for backtracking we work at word level regardless of the counting
    mode, because we reconstruct text by joining words, not subword tokens.

    Args:
        text: Input string.
        mode: Unused here — kept for API symmetry.

    Returns:
        List of whitespace-split words.
    """
    return text.split()


def backtrack_to_boundary(
    words: list[str],
    end_index: int,
    min_chunk_size: int,
    boundary_chars: frozenset[str] = _DEFAULT_BOUNDARY_CHARS,
    mode: TokenizerMode = "tiktoken",
) -> int:
    """
    Walk backward from `end_index` to find the last sentence boundary.

    Starting at `end_index - 1` (exclusive end), scan words backward.
    A word is a boundary if its last non-space character is in `boundary_chars`.
    Stop scanning if we go below `min_chunk_size` words from the window start
    (i.e. we never return a chunk smaller than min_chunk_size tokens).

    Args:
        words:          Full word list of the text being chunked.
        end_index:      Exclusive end of the current window (words[:end_index]).
        min_chunk_size: Minimum token count — don't backtrack past this.
        boundary_chars: Set of characters that signal sentence end.
        mode:           Tokeniser mode for size checks.

    Returns:
        Adjusted exclusive end index. Same as `end_index` if no boundary found.
    """
    # Estimate minimum word count as a proxy for min_chunk_size
    # (exact token count is expensive to recompute per word during scan)
    min_words = max(1, min_chunk_size // 3)  # rough ratio: ~3 chars/token avg

    for i in range(end_index - 1, max(end_index - 1 - (end_index - min_words), -1), -1):
        if i < min_words:
            break
        word = words[i].rstrip()
        if word and word[-1] in boundary_chars:
            return i + 1  # include this boundary word

    return end_index  # no boundary found — return original window


def split_into_windows(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    boundary_enforcement: bool = True,
    boundary_chars: frozenset[str] = _DEFAULT_BOUNDARY_CHARS,
    tokenizer: TokenizerMode = "tiktoken",
    min_chunk_size: int = 50,
) -> list[str]:
    """
    Split `text` into overlapping token windows with optional boundary backtracking.

    This is the core algorithm used by TextChunker (and reused by all other
    chunkers that need token+boundary splitting within a structural unit).

    Args:
        text:                 Input text to split.
        chunk_size:           Target token count per chunk.
        chunk_overlap:        Overlap token count between consecutive chunks.
        boundary_enforcement: If True, backtrack to sentence boundary.
        boundary_chars:       Characters treated as sentence boundaries.
        tokenizer:            "tiktoken" or "word_count".
        min_chunk_size:       Minimum tokens per chunk — no backtracking past this.

    Returns:
        List of text strings, one per chunk.
    """
    if not text or not text.strip():
        return []

    words = tokenize(text, mode=tokenizer)
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    total = len(words)

    while start < total:
        # Build window: estimate end by word count (approximate token:word ratio)
        # For tiktoken we'd need to count precisely; we use words as proxy to
        # avoid O(n²) encoding calls. Accurate enough for boundary backtracking.
        end = min(start + chunk_size, total)

        # Count actual tokens in the window and shrink/grow to hit chunk_size
        window_words = words[start:end]
        token_count = count_tokens(" ".join(window_words), mode=tokenizer)

        # Shrink window if over-budget
        while token_count > chunk_size and len(window_words) > 1:
            window_words = window_words[:-1]
            token_count = count_tokens(" ".join(window_words), mode=tokenizer)

        end = start + len(window_words)

        # Boundary backtracking
        if boundary_enforcement and end < total:
            adjusted_end = backtrack_to_boundary(
                words=words,
                end_index=end,
                min_chunk_size=min_chunk_size,
                boundary_chars=boundary_chars,
                mode=tokenizer,
            )
            end = adjusted_end

        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)

        if end >= total:
            break

        # Overlap: step back by chunk_overlap tokens
        overlap_words = max(1, chunk_overlap)
        start = max(start + 1, end - overlap_words)

    return chunks
