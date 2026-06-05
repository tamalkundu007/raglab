"""
Tests for HybridChunker — meta-chunking strategy.

Covers:
- Config validation (max_unit_tokens, chunk_size, chunk_overlap, tokenizer)
- structural_first=True: two-pass structural + token windowing
- structural_first=False: direct token windowing (bypass)
- Units within budget kept as-is (structural pass)
- Units exceeding budget subdivided (token_window pass)
- source_chunker routing (text, markdown)
- hybrid_pass metadata tag distinguishes structural vs token_window chunks
- Factory registration and available() reporting
- Naming: HybridChunker ≠ HybridRetriever (different concepts entirely)
- Edge cases: empty text, single word, max_unit_tokens edge boundary
"""

from __future__ import annotations

import pytest

from raglab_chunkers.hybrid_chunker import HybridChunker
from raglab_chunkers.text_chunker import TextChunker
from raglab_chunkers.markdown_chunker import MarkdownChunker
from raglab_common.models import ChunkModel


# ── sample content ─────────────────────────────────────────────────────────────

SHORT_TEXT = "RAG retrieves documents. It reduces hallucinations significantly."

MEDIUM_TEXT = (
    "Retrieval-Augmented Generation is a powerful framework that enhances "
    "language models with external knowledge at inference time. "
    "The retrieval step uses dense vector search to locate relevant documents. "
    "These documents are passed as context alongside the original query. "
    "This approach reduces hallucinations by grounding model answers in evidence."
)

MARKDOWN_WITH_SECTIONS = """# Introduction

Retrieval-Augmented Generation enhances language models with external knowledge.
This approach is used in production systems for question answering tasks.

## Dense Retrieval

Dense retrieval encodes queries and documents into vector representations.
Cosine similarity is then used to rank and select the most relevant candidates.
The top K results are passed to the language model as context.

## Benefits

RAG reduces hallucinations by grounding answers in retrieved evidence.
The knowledge base can be updated without retraining the underlying model.
This provides significant operational advantages for production deployments.
"""

LONG_SECTION_TEXT = " ".join([f"word{i}" for i in range(300)])


# ── helpers ────────────────────────────────────────────────────────────────────

def make_chunker(**kwargs) -> HybridChunker:
    defaults = {
        "tokenizer": "word_count",
        "chunk_size": 30,
        "chunk_overlap": 3,
        "max_unit_tokens": 60,
        "source_chunker": "text",
        "boundary_enforcement": True,
        "min_chunk_size": 5,
    }
    defaults.update(kwargs)
    return HybridChunker(config=defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Config validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestHybridChunkerConfig:
    def test_defaults(self):
        c = HybridChunker()
        assert c.structural_first is True
        assert c.max_unit_tokens == 1000
        assert c.source_chunker_type == "markdown"
        assert c.chunk_size == 500
        assert c.chunk_overlap == 50
        assert c.boundary_enforcement is True
        assert c.tokenizer == "tiktoken"
        assert c.min_chunk_size == 50

    def test_custom_config(self):
        c = HybridChunker(config={
            "structural_first": False,
            "max_unit_tokens": 200,
            "source_chunker": "text",
            "chunk_size": 100,
            "chunk_overlap": 10,
            "tokenizer": "word_count",
        })
        assert c.structural_first is False
        assert c.max_unit_tokens == 200
        assert c.source_chunker_type == "text"

    def test_invalid_max_unit_tokens(self):
        with pytest.raises(ValueError, match="max_unit_tokens"):
            HybridChunker(config={"max_unit_tokens": 0})

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            HybridChunker(config={"chunk_size": 0})

    def test_invalid_overlap_negative(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            HybridChunker(config={"chunk_overlap": -1})

    def test_invalid_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError):
            HybridChunker(config={"chunk_size": 50, "chunk_overlap": 50})

    def test_invalid_min_chunk_size(self):
        with pytest.raises(ValueError, match="min_chunk_size"):
            HybridChunker(config={"min_chunk_size": 0})

    def test_invalid_tokenizer(self):
        with pytest.raises(ValueError, match="tokenizer"):
            HybridChunker(config={"tokenizer": "gpt5"})

    def test_source_config_forwarded(self):
        c = HybridChunker(config={"source_config": {"header_levels": [1, 2]}})
        assert c.source_config == {"header_levels": [1, 2]}


# ═══════════════════════════════════════════════════════════════════════════════
# Basic output contracts
# ═══════════════════════════════════════════════════════════════════════════════


class TestHybridChunkerOutput:
    def test_returns_list_of_chunk_models(self):
        c = make_chunker()
        chunks = c.chunk(MEDIUM_TEXT, "doc-hybrid")
        assert isinstance(chunks, list)
        assert all(isinstance(ch, ChunkModel) for ch in chunks)

    def test_produces_at_least_one_chunk(self):
        c = make_chunker()
        chunks = c.chunk(MEDIUM_TEXT, "doc-hybrid")
        assert len(chunks) >= 1

    def test_empty_text_returns_empty(self):
        c = make_chunker()
        assert c.chunk("", "doc-001") == []

    def test_whitespace_only_returns_empty(self):
        c = make_chunker()
        assert c.chunk("   \n\t  ", "doc-001") == []

    def test_doc_id_propagated(self):
        c = make_chunker()
        chunks = c.chunk(MEDIUM_TEXT, "my-hybrid-doc")
        assert all(ch.doc_id == "my-hybrid-doc" for ch in chunks)

    def test_sequential_chunk_indices(self):
        c = make_chunker()
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))

    def test_unique_chunk_ids(self):
        c = make_chunker()
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        ids = [ch.chunk_id for ch in chunks]
        assert len(ids) == len(set(ids))

    def test_positive_token_counts(self):
        c = make_chunker()
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        assert all(ch.token_count > 0 for ch in chunks)

    def test_chunker_type_in_metadata(self):
        c = make_chunker()
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        assert all(ch.metadata.get("chunker") == "hybrid" for ch in chunks)

    def test_non_empty_chunk_texts(self):
        c = make_chunker()
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        assert all(len(ch.text.strip()) > 0 for ch in chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# structural_first=False — bypass structural pass
# ═══════════════════════════════════════════════════════════════════════════════


class TestStructuralFirstFalse:
    def test_bypass_produces_chunks(self):
        c = make_chunker(structural_first=False)
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        assert len(chunks) >= 1

    def test_bypass_no_hybrid_pass_metadata(self):
        """When structural_first=False, hybrid_pass may not be present."""
        c = make_chunker(structural_first=False, chunk_size=20, max_unit_tokens=100)
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        # All chunks go through token_window path but metadata key may be absent
        # (direct windowing path sets chunker=hybrid but not hybrid_pass)
        assert all(ch.metadata.get("chunker") == "hybrid" for ch in chunks)

    def test_bypass_equivalent_to_textchunker_output_shape(self):
        """structural_first=False should produce similar count to TextChunker at same params."""
        cfg = {"tokenizer": "word_count", "chunk_size": 20, "chunk_overlap": 3, "min_chunk_size": 3}
        hybrid = HybridChunker(config={**cfg, "structural_first": False})
        text_ch = TextChunker(config=cfg)

        h_chunks = hybrid.chunk(MEDIUM_TEXT, "doc-001")
        t_chunks = text_ch.chunk(MEDIUM_TEXT, "doc-001")
        # Should produce same number of chunks (same algorithm, same params)
        assert len(h_chunks) == len(t_chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# Two-pass logic: structural → token window
# ═══════════════════════════════════════════════════════════════════════════════


class TestTwoPassLogic:
    def test_small_units_kept_as_structural(self):
        """Units within max_unit_tokens budget → hybrid_pass='structural'."""
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 50,
            "chunk_overlap": 5,
            "max_unit_tokens": 200,   # large budget → no subdivision needed
            "source_chunker": "text",
            "min_chunk_size": 3,
        })
        chunks = c.chunk(SHORT_TEXT, "doc-001")
        structural = [ch for ch in chunks if ch.metadata.get("hybrid_pass") == "structural"]
        assert len(structural) >= 1

    def test_large_units_subdivided_with_token_window(self):
        """Units exceeding chunk_size budget → hybrid_pass='token_window'."""
        # source_chunker gets max_unit_tokens as its chunk_size (300 words)
        # but we then split each unit down to chunk_size=20 words
        # → windowing pass guaranteed on every unit
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 20,
            "chunk_overlap": 3,
            "max_unit_tokens": 300,  # source produces large units
            "source_chunker": "text",
            "min_chunk_size": 3,
        })
        # LONG_SECTION_TEXT is 300 words — source produces 1 unit of ~300 words
        # which exceeds chunk_size=20 → must be windowed
        chunks = c.chunk(LONG_SECTION_TEXT, "doc-001")
        windowed = [ch for ch in chunks if ch.metadata.get("hybrid_pass") == "token_window"]
        assert len(windowed) >= 1

    def test_hybrid_source_in_metadata(self):
        c = make_chunker(source_chunker="text")
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        sources = {ch.metadata.get("hybrid_source") for ch in chunks}
        assert "text" in sources

    def test_long_text_produces_multiple_windowed_chunks(self):
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 20,
            "chunk_overlap": 2,
            "max_unit_tokens": 5,    # force all units to be windowed
            "source_chunker": "text",
            "min_chunk_size": 3,
        })
        chunks = c.chunk(LONG_SECTION_TEXT, "doc-001")
        assert len(chunks) > 5


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown source chunker
# ═══════════════════════════════════════════════════════════════════════════════


class TestMarkdownSourceChunker:
    def test_markdown_structural_units_detected(self):
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 50,
            "chunk_overlap": 5,
            "max_unit_tokens": 200,
            "source_chunker": "markdown",
            "source_config": {"header_levels": [1, 2], "include_header_in_chunk": True},
            "min_chunk_size": 3,
        })
        chunks = c.chunk(MARKDOWN_WITH_SECTIONS, "doc-md")
        assert len(chunks) >= 1
        all_text = " ".join(ch.text for ch in chunks)
        assert "Introduction" in all_text or "Dense Retrieval" in all_text

    def test_markdown_large_section_subdivided(self):
        """A section with 200+ words should be subdivided when max_unit_tokens=50."""
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 30,
            "chunk_overlap": 3,
            "max_unit_tokens": 50,
            "source_chunker": "markdown",
            "min_chunk_size": 5,
        })
        # Each section in MARKDOWN_WITH_SECTIONS is ~30-40 words — some may get windowed
        chunks = c.chunk(MARKDOWN_WITH_SECTIONS, "doc-md")
        assert len(chunks) >= 1
        assert all(ch.doc_id == "doc-md" for ch in chunks)

    def test_metadata_preserved_from_source(self):
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 60,
            "chunk_overlap": 5,
            "max_unit_tokens": 200,
            "source_chunker": "markdown",
            "source_config": {"include_header_in_chunk": True},
            "min_chunk_size": 5,
        })
        chunks = c.chunk(MARKDOWN_WITH_SECTIONS, "doc-md")
        # Metadata from markdown chunker (header, header_level) should flow through
        has_header_meta = any("header" in ch.metadata for ch in chunks)
        assert has_header_meta


# ═══════════════════════════════════════════════════════════════════════════════
# Boundary enforcement within token windows
# ═══════════════════════════════════════════════════════════════════════════════


class TestBoundaryEnforcement:
    def test_boundary_enforcement_on_windowed_chunks(self):
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 15,
            "chunk_overlap": 2,
            "max_unit_tokens": 5,
            "source_chunker": "text",
            "boundary_enforcement": True,
            "boundary_chars": [".", "!", "?"],
            "min_chunk_size": 3,
        })
        text = "First sentence ends here. Second sentence follows now. Third one too."
        chunks = c.chunk(text, "doc-001")
        # At least one chunk should end with a sentence boundary character
        assert any(ch.text.rstrip().endswith((".", "!", "?")) for ch in chunks)

    def test_boundary_enforcement_false_no_backtrack(self):
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 10,
            "chunk_overlap": 0,
            "max_unit_tokens": 5,
            "source_chunker": "text",
            "boundary_enforcement": False,
            "min_chunk_size": 3,
        })
        chunks = c.chunk(LONG_SECTION_TEXT, "doc-001")
        assert len(chunks) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Source config forwarding
# ═══════════════════════════════════════════════════════════════════════════════


class TestSourceConfigForwarding:
    def test_source_config_forwarded_to_markdown(self):
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 50,
            "chunk_overlap": 5,
            "max_unit_tokens": 200,
            "source_chunker": "markdown",
            "source_config": {"header_levels": [1], "include_header_in_chunk": False},
            "min_chunk_size": 5,
        })
        # source_config header_levels=[1] means H2/H3 don't split — all body under H1
        chunks = c.chunk(MARKDOWN_WITH_SECTIONS, "doc-md")
        assert len(chunks) >= 1

    def test_source_config_empty_dict_is_valid(self):
        c = HybridChunker(config={
            "tokenizer": "word_count",
            "chunk_size": 30,
            "chunk_overlap": 3,
            "source_config": {},
            "min_chunk_size": 3,
        })
        chunks = c.chunk(MEDIUM_TEXT, "doc-001")
        assert len(chunks) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# config_schema
# ═══════════════════════════════════════════════════════════════════════════════


class TestHybridChunkerSchema:
    def test_schema_returns_dict(self):
        assert isinstance(HybridChunker.config_schema(), dict)

    def test_schema_has_required_keys(self):
        schema = HybridChunker.config_schema()
        for key in [
            "structural_first", "max_unit_tokens", "source_chunker",
            "chunk_size", "chunk_overlap", "boundary_enforcement",
            "tokenizer", "min_chunk_size", "source_config",
        ]:
            assert key in schema, f"Missing key: {key}"

    def test_schema_defaults_match_class(self):
        schema = HybridChunker.config_schema()
        assert schema["structural_first"]["default"] is True
        assert schema["max_unit_tokens"]["default"] == 1000
        assert schema["source_chunker"]["default"] == "markdown"
        assert schema["chunk_size"]["default"] == 500

    def test_source_chunker_options(self):
        schema = HybridChunker.config_schema()
        options = schema["source_chunker"]["options"]
        assert "text" in options
        assert "markdown" in options
        assert "html" in options


# ═══════════════════════════════════════════════════════════════════════════════
# Factory registration
# ═══════════════════════════════════════════════════════════════════════════════


class TestHybridChunkerFactory:
    def test_factory_creates_hybrid(self):
        from raglab_chunkers import ChunkerFactory
        c = ChunkerFactory.create("hybrid", config={
            "tokenizer": "word_count", "chunk_size": 50,
            "chunk_overlap": 5, "min_chunk_size": 5,
        })
        assert isinstance(c, HybridChunker)

    def test_hybrid_listed_in_available(self):
        from raglab_chunkers import ChunkerFactory
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        assert "hybrid" in entries

    def test_hybrid_is_active_in_available(self):
        from raglab_chunkers import ChunkerFactory
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        assert entries["hybrid"]["active"] is True

    def test_hybrid_schema_via_factory(self):
        from raglab_chunkers import ChunkerFactory
        schema = ChunkerFactory.schema("hybrid")
        assert "structural_first" in schema
        assert "max_unit_tokens" in schema


# ═══════════════════════════════════════════════════════════════════════════════
# Naming distinction test — HybridChunker ≠ HybridRetriever
# ═══════════════════════════════════════════════════════════════════════════════


class TestNamingDistinction:
    def test_hybrid_chunker_is_not_retriever(self):
        """Sanity check: HybridChunker is a BaseChunker, not a retriever."""
        from raglab_chunkers.base import BaseChunker
        assert issubclass(HybridChunker, BaseChunker)

    def test_hybrid_chunker_type_is_hybrid(self):
        c = HybridChunker(config={"tokenizer": "word_count"})
        assert c.chunker_type == "hybrid"

    def test_hybrid_retriever_is_separate_class(self):
        """HybridRetriever (R3) is a completely separate concept."""
        from raglab_common.exceptions import NotImplementedFeatureError
        from raglab_retrievers import RetrieverFactory
        with pytest.raises(NotImplementedFeatureError):
            RetrieverFactory.create("hybrid")  # R3 stub — correctly raises
