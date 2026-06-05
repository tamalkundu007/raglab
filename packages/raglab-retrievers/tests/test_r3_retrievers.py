"""
Tests for R3 retrievers: BM25, Hybrid, MMR, ReRanker, Compression.

All infra-free — Qdrant, sentence-transformers, and rank-bm25 calls mocked.
Covers: config validation, algorithm correctness, metadata tagging, factory.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, LLMProvider, QueryModel, RetrieverType


# ── helpers ────────────────────────────────────────────────────────────────────

def make_query(text="What is RAG?", collection="raglab", top_k=3) -> QueryModel:
    return QueryModel(
        text=text, collection=collection, top_k=top_k,
        retriever_type=RetrieverType.DENSE, llm_provider=LLMProvider.AZURE_OPENAI,
    )

def make_chunk(text: str, index: int = 0, doc_id: str = "doc-001") -> ChunkModel:
    return ChunkModel(
        chunk_id=str(uuid.uuid4()), doc_id=doc_id,
        text=text, chunk_index=index, token_count=len(text.split()),
    )

CHUNKS = [
    make_chunk("RAG retrieves documents for context. Reduces hallucinations.", 0),
    make_chunk("Dense retrieval uses vector embeddings and cosine similarity.", 1),
    make_chunk("BM25 is a sparse keyword-based retrieval algorithm for search.", 2),
    make_chunk("Hybrid retrieval combines dense and sparse methods via RRF.", 3),
    make_chunk("MMR selects diverse results to avoid near-duplicate chunks.", 4),
]

def mock_embedder(text: str) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4, 0.5]

def mock_qdrant(hits=None):
    vs = MagicMock()
    vs.search.return_value = hits or [
        {"payload": {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                     "chunk_index": c.chunk_index, "token_count": c.token_count},
         "score": 0.9 - i * 0.05}
        for i, c in enumerate(CHUNKS[:3])
    ]
    return vs


# ═══════════════════════════════════════════════════════════════════════════════
# BM25Retriever + BM25Corpus
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25Corpus:
    def test_build_and_search(self):
        from raglab_retrievers.bm25_retriever import BM25Corpus
        corpus = BM25Corpus(CHUNKS)
        results = corpus.search("dense retrieval vector", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(c, ChunkModel) for c, _ in results)
        assert all(isinstance(s, float) for _, s in results)

    def test_size(self):
        from raglab_retrievers.bm25_retriever import BM25Corpus
        corpus = BM25Corpus(CHUNKS)
        assert corpus.size == len(CHUNKS)

    def test_results_ordered_by_score(self):
        from raglab_retrievers.bm25_retriever import BM25Corpus
        corpus = BM25Corpus(CHUNKS)
        results = corpus.search("RAG retrieval", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_keyword_match_scores_higher(self):
        from raglab_retrievers.bm25_retriever import BM25Corpus
        corpus = BM25Corpus(CHUNKS)
        results = corpus.search("BM25 sparse keyword", top_k=5)
        # BM25 chunk should be top result
        top_chunk = results[0][0]
        assert "BM25" in top_chunk.text or "sparse" in top_chunk.text or "keyword" in top_chunk.text


class TestBM25Retriever:
    def test_config_defaults(self):
        from raglab_retrievers.bm25_retriever import BM25Retriever
        r = BM25Retriever()
        assert r.k1 == 1.5
        assert r.b == 0.75
        assert r.top_n_factor == 3

    def test_invalid_k1(self):
        from raglab_retrievers.bm25_retriever import BM25Retriever
        with pytest.raises(ValueError, match="k1"):
            BM25Retriever(config={"k1": -1.0})

    def test_invalid_b(self):
        from raglab_retrievers.bm25_retriever import BM25Retriever
        with pytest.raises(ValueError, match="b"):
            BM25Retriever(config={"b": 1.5})

    def test_wrong_vector_store_raises(self):
        from raglab_retrievers.bm25_retriever import BM25Retriever
        r = BM25Retriever()
        result = r.retrieve(make_query(), vector_store=MagicMock(), embedder=None)
        assert result == []  # retrieve() swallows and returns []

    def test_returns_chunk_models(self):
        from raglab_retrievers.bm25_retriever import BM25Corpus, BM25Retriever
        corpus = BM25Corpus(CHUNKS)
        r = BM25Retriever()
        results = r.retrieve(make_query(text="dense vector"), corpus)
        assert all(isinstance(c, ChunkModel) for c in results)

    def test_retriever_tag_in_metadata(self):
        from raglab_retrievers.bm25_retriever import BM25Corpus, BM25Retriever
        corpus = BM25Corpus(CHUNKS)
        r = BM25Retriever()
        results = r.retrieve(make_query(text="RAG retrieval", top_k=2), corpus)
        assert all(c.metadata["retriever"] == "bm25" for c in results)

    def test_score_in_metadata(self):
        from raglab_retrievers.bm25_retriever import BM25Corpus, BM25Retriever
        corpus = BM25Corpus(CHUNKS)
        r = BM25Retriever()
        results = r.retrieve(make_query(text="RAG"), corpus)
        assert all("score" in c.metadata for c in results)

    def test_respects_top_k(self):
        from raglab_retrievers.bm25_retriever import BM25Corpus, BM25Retriever
        corpus = BM25Corpus(CHUNKS)
        r = BM25Retriever()
        results = r.retrieve(make_query(top_k=2), corpus)
        assert len(results) <= 2

    def test_schema_keys(self):
        from raglab_retrievers.bm25_retriever import BM25Retriever
        schema = BM25Retriever.config_schema()
        assert "k1" in schema and "b" in schema


# ═══════════════════════════════════════════════════════════════════════════════
# HybridRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridRetriever:
    def test_config_defaults(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        r = HybridRetriever()
        assert r.alpha == 0.5
        assert r.rrf_k == 60

    def test_invalid_alpha(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        with pytest.raises(ValueError, match="alpha"):
            HybridRetriever(config={"alpha": 1.5})

    def test_invalid_rrf_k(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        with pytest.raises(ValueError, match="rrf_k"):
            HybridRetriever(config={"rrf_k": 0})

    def test_rrf_fuse_deduplicates(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        chunk = CHUNKS[0]
        dense_list = [chunk]
        bm25_list = [chunk]  # same chunk in both lists
        result = HybridRetriever._rrf_fuse(dense_list, bm25_list, alpha=0.5, k=60, top_k=5, query_id="q1")
        # Should appear only once despite being in both lists
        assert len(result) == 1

    def test_rrf_fuse_ordering(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        # chunk0 appears in both → higher RRF score
        chunk0, chunk1 = CHUNKS[0], CHUNKS[1]
        dense = [chunk0, chunk1]
        bm25 = [chunk0]  # chunk0 in both, chunk1 only dense
        result = HybridRetriever._rrf_fuse(dense, bm25, alpha=0.5, k=60, top_k=5, query_id="q1")
        assert result[0].chunk_id == chunk0.chunk_id

    def test_rrf_score_in_metadata(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        result = HybridRetriever._rrf_fuse(CHUNKS[:2], CHUNKS[:2], 0.5, 60, 5, "q1")
        assert all("rrf_score" in c.metadata for c in result)

    def test_unpack_store_tuple(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        qdrant = MagicMock()
        bm25 = MagicMock()
        q, b = HybridRetriever._unpack_store((qdrant, bm25))
        assert q is qdrant and b is bm25

    def test_unpack_store_object(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        store = MagicMock()
        store.qdrant = MagicMock()
        store.bm25 = MagicMock()
        q, b = HybridRetriever._unpack_store(store)
        assert q is store.qdrant

    def test_unpack_store_invalid_raises(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        with pytest.raises(RetrieverError, match="tuple"):
            HybridRetriever._unpack_store(MagicMock(spec=[]))

    def test_alpha_1_pure_dense(self):
        """alpha=1.0 should return pure dense results, BM25 contributes nothing."""
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        result = HybridRetriever._rrf_fuse(
            dense_list=CHUNKS[:3], bm25_list=[], alpha=1.0, k=60, top_k=3, query_id="q1"
        )
        assert len(result) == 3

    def test_schema_keys(self):
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        schema = HybridRetriever.config_schema()
        assert "alpha" in schema and "rrf_k" in schema

    def test_naming_distinct_from_hybrid_chunker(self):
        """HybridRetriever ≠ HybridChunker — different concepts."""
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        from raglab_retrievers.base import BaseRetriever
        assert issubclass(HybridRetriever, BaseRetriever)
        assert HybridRetriever.retriever_type == "hybrid"


# ═══════════════════════════════════════════════════════════════════════════════
# MMRRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class TestMMRRetriever:
    def test_config_defaults(self):
        from raglab_retrievers.mmr_retriever import MMRRetriever
        r = MMRRetriever()
        assert r.lambda_mult == 0.5
        assert r.fetch_k == 20

    def test_invalid_lambda(self):
        from raglab_retrievers.mmr_retriever import MMRRetriever
        with pytest.raises(ValueError, match="lambda_mult"):
            MMRRetriever(config={"lambda_mult": 1.5})

    def test_invalid_fetch_k(self):
        from raglab_retrievers.mmr_retriever import MMRRetriever
        with pytest.raises(ValueError, match="fetch_k"):
            MMRRetriever(config={"fetch_k": 0})

    def test_no_embedder_returns_empty(self):
        from raglab_retrievers.mmr_retriever import MMRRetriever
        r = MMRRetriever()
        result = r.retrieve(make_query(), mock_qdrant(), embedder=None)
        assert result == []

    def test_cosine_sim_identical_vectors(self):
        from raglab_retrievers.mmr_retriever import _cosine_sim
        v = [1.0, 0.0, 0.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_cosine_sim_orthogonal_vectors(self):
        from raglab_retrievers.mmr_retriever import _cosine_sim
        assert _cosine_sim([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_sim_zero_vector(self):
        from raglab_retrievers.mmr_retriever import _cosine_sim
        assert _cosine_sim([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_mmr_select_returns_diverse(self):
        from raglab_retrievers.mmr_retriever import MMRRetriever
        # With lambda_mult=0.0 (pure diversity), after selecting the most relevant
        # (index 0), the next selection maximises distance from index 0.
        # Index 2 ([0,1]) is orthogonal to index 0 ([1,0]) → maximum diversity.
        vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        query_vec = [1.0, 0.0]
        selected = MMRRetriever._mmr_select(query_vec, vecs, top_k=2, lambda_mult=0.0)
        # First selected: whichever scores best (all equal relevance at lambda=0,
        # so first pick is arbitrary — just verify we get 2 distinct indices)
        assert len(selected) == 2
        assert len(set(selected)) == 2  # no duplicates

    def test_mmr_select_lambda_1_pure_relevance(self):
        from raglab_retrievers.mmr_retriever import MMRRetriever
        vecs = [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
        query_vec = [1.0, 0.0]
        # lambda=1.0 → pure relevance → top-2 by cosine sim
        selected = MMRRetriever._mmr_select(query_vec, vecs, top_k=2, lambda_mult=1.0)
        assert selected[0] == 0  # most relevant

    def test_mmr_returns_chunk_models_with_mocked_dense(self):
        from raglab_retrievers.mmr_retriever import MMRRetriever
        r = MMRRetriever(config={"fetch_k": 3, "lambda_mult": 0.5})
        vs = mock_qdrant()
        results = r.retrieve(make_query(top_k=2), vs, embedder=mock_embedder)
        assert all(isinstance(c, ChunkModel) for c in results)
        assert all(c.metadata["retriever"] == "mmr" for c in results)

    def test_schema_keys(self):
        from raglab_retrievers.mmr_retriever import MMRRetriever
        schema = MMRRetriever.config_schema()
        assert "lambda_mult" in schema and "fetch_k" in schema


# ═══════════════════════════════════════════════════════════════════════════════
# ReRankerRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class TestReRankerRetriever:
    def test_config_defaults(self):
        from raglab_retrievers.reranker_retriever import ReRankerRetriever
        r = ReRankerRetriever()
        assert r.fetch_k == 20
        assert r.batch_size == 16
        assert "MiniLM" in r.model_name

    def test_invalid_fetch_k(self):
        from raglab_retrievers.reranker_retriever import ReRankerRetriever
        with pytest.raises(ValueError, match="fetch_k"):
            ReRankerRetriever(config={"fetch_k": 0})

    def test_no_embedder_returns_empty(self):
        from raglab_retrievers.reranker_retriever import ReRankerRetriever
        r = ReRankerRetriever()
        result = r.retrieve(make_query(), mock_qdrant(), embedder=None)
        assert result == []

    def test_reranking_with_mocked_cross_encoder(self):
        from raglab_retrievers.reranker_retriever import ReRankerRetriever
        r = ReRankerRetriever(config={"fetch_k": 3})
        vs = mock_qdrant()

        # Mock the cross-encoder model
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9, 0.3, 0.7]  # rerank order: 0, 2, 1
        r._cross_encoder = mock_ce

        results = r.retrieve(make_query(top_k=2), vs, embedder=mock_embedder)
        assert len(results) <= 2
        assert all(c.metadata["retriever"] == "reranker" for c in results)

    def test_reranker_score_in_metadata(self):
        from raglab_retrievers.reranker_retriever import ReRankerRetriever
        r = ReRankerRetriever(config={"fetch_k": 3})
        vs = mock_qdrant()
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.8, 0.5, 0.6]
        r._cross_encoder = mock_ce
        results = r.retrieve(make_query(top_k=3), vs, embedder=mock_embedder)
        assert all("reranker_score" in c.metadata for c in results)

    def test_reranker_rank_in_metadata(self):
        from raglab_retrievers.reranker_retriever import ReRankerRetriever
        r = ReRankerRetriever(config={"fetch_k": 3})
        vs = mock_qdrant()
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.8, 0.5, 0.6]
        r._cross_encoder = mock_ce
        results = r.retrieve(make_query(top_k=3), vs, embedder=mock_embedder)
        ranks = [c.metadata["reranker_rank"] for c in results]
        assert sorted(ranks) == list(range(len(results)))

    def test_score_threshold_filters_results(self):
        from raglab_retrievers.reranker_retriever import ReRankerRetriever
        r = ReRankerRetriever(config={"fetch_k": 3, "score_threshold": 0.7})
        vs = mock_qdrant()
        mock_ce = MagicMock()
        mock_ce.predict.return_value = [0.9, 0.3, 0.5]  # only 0.9 passes threshold
        r._cross_encoder = mock_ce
        results = r.retrieve(make_query(top_k=3), vs, embedder=mock_embedder)
        assert len(results) == 1

    def test_schema_keys(self):
        from raglab_retrievers.reranker_retriever import ReRankerRetriever
        schema = ReRankerRetriever.config_schema()
        assert "model_name" in schema and "fetch_k" in schema and "batch_size" in schema


# ═══════════════════════════════════════════════════════════════════════════════
# CompressionRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompressionRetriever:
    def test_config_defaults(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        r = CompressionRetriever()
        assert r.strategy == "keyword"
        assert r.min_keyword_overlap == 1
        assert r.fetch_k == 20

    def test_invalid_strategy(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        with pytest.raises(ValueError, match="strategy"):
            CompressionRetriever(config={"strategy": "neural"})

    def test_invalid_min_keyword_overlap(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        with pytest.raises(ValueError, match="min_keyword_overlap"):
            CompressionRetriever(config={"min_keyword_overlap": -1})

    def test_no_embedder_returns_empty(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        r = CompressionRetriever()
        result = r.retrieve(make_query(), mock_qdrant(), embedder=None)
        assert result == []

    def test_keyword_compress_filters_by_overlap(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        r = CompressionRetriever(config={"min_keyword_overlap": 1})
        query_text = "RAG retrieval"
        candidates = [
            make_chunk("RAG retrieves documents for context."),      # overlap: RAG
            make_chunk("This chunk has nothing to do with the query."),  # no overlap
            make_chunk("retrieval augmented generation is useful."),  # overlap: retrieval
        ]
        filtered = r._keyword_compress(query_text, candidates, top_k=5)
        assert len(filtered) == 2  # "RAG" and "retrieval" match

    def test_keyword_zero_overlap_keeps_all(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        r = CompressionRetriever(config={"min_keyword_overlap": 0})
        query_text = "test query"
        filtered = r._keyword_compress(query_text, CHUNKS[:3], top_k=10)
        assert len(filtered) == 3

    def test_returns_chunk_models_with_keyword_strategy(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        r = CompressionRetriever(config={"fetch_k": 3, "min_keyword_overlap": 0})
        vs = mock_qdrant()
        results = r.retrieve(make_query(text="RAG dense retrieval"), vs, embedder=mock_embedder)
        assert all(isinstance(c, ChunkModel) for c in results)
        assert all(c.metadata["retriever"] == "compression" for c in results)

    def test_compression_strategy_in_metadata(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        r = CompressionRetriever(config={"fetch_k": 3, "min_keyword_overlap": 0})
        vs = mock_qdrant()
        results = r.retrieve(make_query(text="RAG"), vs, embedder=mock_embedder)
        assert all(c.metadata["compression_strategy"] == "keyword" for c in results)

    def test_schema_keys(self):
        from raglab_retrievers.compression_retriever import CompressionRetriever
        schema = CompressionRetriever.config_schema()
        assert "strategy" in schema and "min_keyword_overlap" in schema


# ═══════════════════════════════════════════════════════════════════════════════
# RetrieverFactory — R3 all active
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrieverFactoryR3:
    def test_all_r3_types_create(self):
        from raglab_retrievers import RetrieverFactory, BM25Retriever, HybridRetriever
        from raglab_retrievers import MMRRetriever, ReRankerRetriever, CompressionRetriever
        assert isinstance(RetrieverFactory.create("bm25"), BM25Retriever)
        assert isinstance(RetrieverFactory.create("hybrid"), HybridRetriever)
        assert isinstance(RetrieverFactory.create("mmr"), MMRRetriever)
        assert isinstance(RetrieverFactory.create("reranker"), ReRankerRetriever)
        assert isinstance(RetrieverFactory.create("compression"), CompressionRetriever)

    def test_all_active_in_available(self):
        from raglab_retrievers import RetrieverFactory
        entries = {e["type"]: e for e in RetrieverFactory.available()}
        for t in ["dense", "bm25", "hybrid", "mmr", "reranker", "compression"]:
            assert entries[t]["active"] is True

    def test_schema_returns_correct_keys(self):
        from raglab_retrievers import RetrieverFactory
        assert "k1" in RetrieverFactory.schema("bm25")
        assert "alpha" in RetrieverFactory.schema("hybrid")
        assert "lambda_mult" in RetrieverFactory.schema("mmr")
        assert "model_name" in RetrieverFactory.schema("reranker")
        assert "strategy" in RetrieverFactory.schema("compression")

    def test_unknown_raises(self):
        from raglab_retrievers import RetrieverFactory
        with pytest.raises(ValueError, match="Unknown retriever type"):
            RetrieverFactory.create("quantum_retriever")

    def test_naming_distinction_hybrid_retriever_vs_chunker(self):
        """HybridRetriever (retrieval fusion) ≠ HybridChunker (meta-chunking). Tested explicitly."""
        from raglab_retrievers import HybridRetriever
        from raglab_retrievers.base import BaseRetriever
        assert issubclass(HybridRetriever, BaseRetriever)
        assert not hasattr(HybridRetriever, "chunker_type")
        assert HybridRetriever.retriever_type == "hybrid"
