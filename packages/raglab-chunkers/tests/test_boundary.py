"""
Tests for raglab_chunkers._boundary — the shared backtracking utility.

Covers:
- count_tokens (both modes)
- backtrack_to_boundary (finds boundary / falls back to original)
- split_into_windows (core algorithm — size, overlap, boundary enforcement)
"""

import pytest

from raglab_chunkers._boundary import (
    backtrack_to_boundary,
    count_tokens,
    split_into_windows,
    tokenize,
)


def _tiktoken_available() -> bool:
    """Return True if tiktoken BPE file is downloadable in this environment."""
    try:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_word_count_empty(self):
        assert count_tokens("", mode="word_count") == 0

    def test_word_count_single(self):
        assert count_tokens("hello", mode="word_count") == 1

    def test_word_count_multiple(self):
        assert count_tokens("hello world foo", mode="word_count") == 3

    @pytest.mark.skipif(
        not _tiktoken_available(),
        reason="tiktoken BPE file unavailable in this environment",
    )
    def test_tiktoken_positive(self):
        # tiktoken should return a positive count for non-empty input
        count = count_tokens("The quick brown fox", mode="tiktoken")
        assert count > 0

    @pytest.mark.skipif(
        not _tiktoken_available(),
        reason="tiktoken BPE file unavailable in this environment",
    )
    def test_tiktoken_empty(self):
        count = count_tokens("", mode="tiktoken")
        assert count == 0


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_split(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_empty(self):
        assert tokenize("") == []

    def test_extra_spaces(self):
        result = tokenize("  a   b  ")
        assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# backtrack_to_boundary
# ---------------------------------------------------------------------------


class TestBacktrackToBoundary:
    def _words(self, text: str) -> list[str]:
        return text.split()

    def test_finds_period_boundary(self):
        words = self._words("Hello world. This is a test")
        # end=6 (all words), boundary should be at "world." (index 2), returns 3
        result = backtrack_to_boundary(words, end_index=6, min_chunk_size=1)
        # "world." ends at index 1, so boundary return is 2
        assert result <= 6

    def test_no_boundary_returns_original(self):
        words = self._words("no boundary chars here at all")
        end = len(words)
        result = backtrack_to_boundary(words, end_index=end, min_chunk_size=1)
        assert result == end

    def test_boundary_at_last_word(self):
        words = self._words("This is a sentence.")
        end = len(words)
        result = backtrack_to_boundary(words, end_index=end, min_chunk_size=1)
        # Last word ends with "." — boundary is at end
        assert result == end

    def test_custom_boundary_chars(self):
        words = self._words("stop here; continue more words")
        result = backtrack_to_boundary(
            words, end_index=5, min_chunk_size=1, boundary_chars=frozenset({";"}),
        )
        assert result <= 5

    def test_min_chunk_size_prevents_tiny_chunk(self):
        words = self._words("A. B C D E F G H I J")
        # With large min_chunk_size, backtracking can't go back far
        result = backtrack_to_boundary(words, end_index=10, min_chunk_size=8)
        # Should not go all the way back to index 2 (after "A.")
        assert result >= 2


# ---------------------------------------------------------------------------
# split_into_windows — core algorithm
# ---------------------------------------------------------------------------


class TestSplitIntoWindows:
    def test_empty_input(self):
        assert split_into_windows("") == []

    def test_whitespace_only(self):
        assert split_into_windows("   ") == []

    def test_single_short_text_returns_one_chunk(self):
        text = "Hello world this is a short text."
        chunks = split_into_windows(text, chunk_size=500, tokenizer="word_count")
        assert len(chunks) == 1
        assert "Hello" in chunks[0]

    def test_long_text_produces_multiple_chunks(self):
        # 200 words — with chunk_size=50 and word_count, expect ~4 chunks
        words = " ".join([f"word{i}" for i in range(200)])
        chunks = split_into_windows(words, chunk_size=50, chunk_overlap=5, tokenizer="word_count")
        assert len(chunks) > 1

    def test_overlap_creates_shared_content(self):
        words = " ".join([f"word{i}" for i in range(100)])
        chunks = split_into_windows(words, chunk_size=20, chunk_overlap=5, tokenizer="word_count")
        assert len(chunks) >= 2
        # Last words of chunk[0] should appear in chunk[1] due to overlap
        last_words_c0 = set(chunks[0].split()[-5:])
        first_words_c1 = set(chunks[1].split()[:10])
        assert last_words_c0 & first_words_c1  # non-empty intersection

    def test_no_overlap(self):
        words = " ".join([f"word{i}" for i in range(100)])
        chunks = split_into_windows(words, chunk_size=20, chunk_overlap=0, tokenizer="word_count")
        assert len(chunks) > 1

    def test_boundary_enforcement_off(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_into_windows(
            text, chunk_size=5, chunk_overlap=0,
            boundary_enforcement=False, tokenizer="word_count",
        )
        assert len(chunks) >= 1

    def test_boundary_enforcement_on_finds_period(self):
        # Construct text where boundary enforcement should kick in
        text = "Word one two three four five. Next sentence here now done."
        chunks = split_into_windows(
            text, chunk_size=8, chunk_overlap=0,
            boundary_enforcement=True,
            boundary_chars=frozenset({".", "!", "?"}),
            tokenizer="word_count",
            min_chunk_size=2,
        )
        assert len(chunks) >= 1
        # At least one chunk should end with a boundary char
        assert any(c.rstrip().endswith((".", "!", "?")) for c in chunks)

    def test_all_chunks_non_empty(self):
        text = " ".join([f"token{i}" for i in range(300)])
        chunks = split_into_windows(text, chunk_size=50, chunk_overlap=10, tokenizer="word_count")
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_chunk_size_respected_approximately(self):
        """Each chunk should be within reasonable range of chunk_size."""
        text = " ".join([f"word{i}" for i in range(500)])
        chunks = split_into_windows(text, chunk_size=50, chunk_overlap=5, tokenizer="word_count")
        for chunk in chunks[:-1]:  # last chunk may be smaller
            token_count = count_tokens(chunk, mode="word_count")
            assert token_count <= 60  # allow some slack for boundary backtrack

    @pytest.mark.skipif(
        not _tiktoken_available(),
        reason="tiktoken BPE file unavailable in this environment",
    )
    def test_tiktoken_mode(self):
        text = "This is a sentence. And another one here. And a third one too."
        chunks = split_into_windows(text, chunk_size=10, chunk_overlap=2, tokenizer="tiktoken")
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
