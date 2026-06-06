"""
Unit tests for GraphRetriever (R4 Phase 6).

All external calls (dense retrieval, graph traversal) use controlled fakes.
Zero infra required.

Covers:
- Config validation (mode, traversal_depth, graph_weight, top_k_multiplier)
- _find_entry_nodes: name in text, case-insensitive, word boundary, no match
- _traverse_graph: BFS from entry nodes, depth=0/1/2, chunk_id collection,
  visited-node deduplication, cycles handled
- _merge_results: dedup by chunk_id, graph_mode tags, top_k cap
- _classical_retrieve: delegates to DenseRetriever, tags graph_mode='classical'
- _graph_retrieve: no graph → empty, entry nodes found, traversal called
- _hybrid_retrieve: classical + graph chunks merged, graph_weight=0 → classical only,
  graph_weight=1 → graph only, both paths combined
- retrieve(): error → empty list, calls _retrieve correctly
- Factory: GraphRetriever created, active, schema
- RetrieverType.GRAPH exists
- Naming distinction: GraphRetriever ≠ HybridRetriever
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from raglab_retrievers.graph_retriever import GraphRetriever, _VALID_MODES
from raglab_common.models import ChunkModel, LLMProvider, QueryModel, RetrieverType


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_query(text="How does RAG work?", top_k=4) -> QueryModel:
    return QueryModel(
        text=text, collection="raglab", top_k=top_k,
        retriever_type=RetrieverType.GRAPH,
        llm_provider=LLMProvider.AZURE_OPENAI,
    )


def make_chunk(text: str, chunk_id: str | None = None, score: float = 0.9) -> ChunkModel:
    return ChunkModel(
        chunk_id=chunk_id or str(uuid.uuid4()),
        doc_id="doc-001", text=text,
        chunk_index=0, token_count=len(text.split()),
        metadata={"score": score},
    )


def make_graph_with_entities(entities: list[tuple[str, str, str]], edges: list[tuple[int, int, str, str]]) -> nx.DiGraph:
    """
    entities: [(node_id, name, entity_type), ...]
    edges:    [(src_idx, tgt_idx, relation_type, chunk_id), ...]
    """
    g = nx.DiGraph()
    for node_id, name, etype in entities:
        g.add_node(node_id, name=name, entity_type=etype, community_id=None)
    for src_idx, tgt_idx, rel_type, chunk_id in edges:
        src_id = entities[src_idx][0]
        tgt_id = entities[tgt_idx][0]
        g.add_edge(src_id, tgt_id, relation_type=rel_type, weight=1.0, chunk_id=chunk_id)
    return g


def make_qdrant(chunks: list[ChunkModel] | None = None):
    vs = MagicMock()
    hits = [
        {
            "payload": {
                "chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                "chunk_index": c.chunk_index, "token_count": c.token_count,
            },
            "score": c.metadata.get("score", 0.9),
        }
        for c in (chunks or [])
    ]
    vs.search.return_value = hits
    return vs


def embedder(text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


# ── Sample graph ──────────────────────────────────────────────────────────────

RAG_ID  = str(uuid.uuid4())
QDRANT_ID = str(uuid.uuid4())
LLM_ID  = str(uuid.uuid4())

SAMPLE_ENTITIES = [
    (RAG_ID,    "RAG",    "CONCEPT"),
    (QDRANT_ID, "Qdrant", "TECHNOLOGY"),
    (LLM_ID,    "LLM",    "TECHNOLOGY"),
]

SAMPLE_EDGES = [
    (0, 1, "USES", "chunk-rag-qdrant"),
    (0, 2, "USES", "chunk-rag-llm"),
    (2, 1, "RELATED_TO", "chunk-llm-qdrant"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Config validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphRetrieverConfig:
    def test_defaults(self):
        r = GraphRetriever()
        assert r.mode == "hybrid"
        assert r.traversal_depth == 2
        assert r.graph_weight == 0.5
        assert r.top_k_multiplier == 3
        assert r.score_threshold == 0.0
        assert r.ef == 128

    def test_custom_config(self):
        r = GraphRetriever(config={
            "mode": "classical",
            "traversal_depth": 3,
            "graph_weight": 0.7,
            "top_k_multiplier": 2,
        })
        assert r.mode == "classical"
        assert r.traversal_depth == 3
        assert r.graph_weight == 0.7

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            GraphRetriever(config={"mode": "quantum"})

    def test_invalid_traversal_depth(self):
        with pytest.raises(ValueError, match="traversal_depth"):
            GraphRetriever(config={"traversal_depth": -1})

    def test_invalid_graph_weight_high(self):
        with pytest.raises(ValueError, match="graph_weight"):
            GraphRetriever(config={"graph_weight": 1.5})

    def test_invalid_top_k_multiplier(self):
        with pytest.raises(ValueError, match="top_k_multiplier"):
            GraphRetriever(config={"top_k_multiplier": 0})

    def test_all_valid_modes(self):
        for mode in _VALID_MODES:
            r = GraphRetriever(config={"mode": mode})
            assert r.mode == mode


# ═══════════════════════════════════════════════════════════════════════════════
# _find_entry_nodes
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindEntryNodes:
    def setup_method(self):
        self.r = GraphRetriever()
        self.graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)

    def test_exact_match_found(self):
        nodes = self.r._find_entry_nodes("How does RAG work?", self.graph)
        assert RAG_ID in nodes

    def test_case_insensitive_match(self):
        nodes = self.r._find_entry_nodes("rag retrieves documents", self.graph)
        assert RAG_ID in nodes

    def test_multiple_entities_found(self):
        nodes = self.r._find_entry_nodes("RAG uses Qdrant for search", self.graph)
        assert RAG_ID in nodes
        assert QDRANT_ID in nodes

    def test_no_match_returns_empty(self):
        nodes = self.r._find_entry_nodes("What is the weather today?", self.graph)
        assert len(nodes) == 0

    def test_word_boundary_prevents_partial_match(self):
        # "RAGNAR" should not match "RAG"
        nodes = self.r._find_entry_nodes("RAGNAR was a viking", self.graph)
        assert RAG_ID not in nodes

    def test_empty_text_returns_empty(self):
        nodes = self.r._find_entry_nodes("", self.graph)
        assert len(nodes) == 0

    def test_empty_graph_returns_empty(self):
        nodes = self.r._find_entry_nodes("RAG uses Qdrant", nx.DiGraph())
        assert len(nodes) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# _traverse_graph
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraverseGraph:
    def setup_method(self):
        self.r = GraphRetriever()
        self.graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)

    def test_depth_0_returns_entry_edge_chunks(self):
        chunk_ids = self.r._traverse_graph({RAG_ID}, self.graph, depth=0)
        assert "chunk-rag-qdrant" in chunk_ids or "chunk-rag-llm" in chunk_ids

    def test_depth_1_expands_to_neighbours(self):
        chunk_ids = self.r._traverse_graph({RAG_ID}, self.graph, depth=1)
        # Should include chunks from RAG→Qdrant, RAG→LLM, and LLM→Qdrant (via LLM expansion)
        assert "chunk-rag-qdrant" in chunk_ids
        assert "chunk-rag-llm" in chunk_ids

    def test_no_duplicate_chunk_ids(self):
        chunk_ids = self.r._traverse_graph({RAG_ID}, self.graph, depth=2)
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_empty_entry_nodes_returns_empty(self):
        assert self.r._traverse_graph(set(), self.graph, depth=2) == []

    def test_cycle_safe(self):
        """Graph with a cycle should not cause infinite loop."""
        g = nx.DiGraph()
        n1, n2 = str(uuid.uuid4()), str(uuid.uuid4())
        g.add_node(n1, name="A")
        g.add_node(n2, name="B")
        g.add_edge(n1, n2, chunk_id="c1", relation_type="R", weight=1.0)
        g.add_edge(n2, n1, chunk_id="c2", relation_type="R", weight=1.0)  # cycle
        chunk_ids = self.r._traverse_graph({n1}, g, depth=3)
        assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates despite cycle

    def test_nodes_without_edges_return_empty(self):
        g = nx.DiGraph()
        isolated = str(uuid.uuid4())
        g.add_node(isolated, name="X")
        chunk_ids = self.r._traverse_graph({isolated}, g, depth=2)
        assert chunk_ids == []


# ═══════════════════════════════════════════════════════════════════════════════
# _merge_results
# ═══════════════════════════════════════════════════════════════════════════════

class TestMergeResults:
    def setup_method(self):
        self.r = GraphRetriever()

    def test_classical_chunks_come_first(self):
        c1 = make_chunk("classical text", "cid-classic")
        g1 = make_chunk("graph text", "cid-graph")
        merged = self.r._merge_results([c1], [g1], top_k=5, query_id="q1")
        assert merged[0].chunk_id == "cid-classic"
        assert merged[1].chunk_id == "cid-graph"

    def test_dedup_by_chunk_id(self):
        shared = make_chunk("shared", "cid-shared")
        merged = self.r._merge_results([shared], [shared], top_k=5, query_id="q1")
        assert len(merged) == 1

    def test_top_k_capped(self):
        classical = [make_chunk(f"c{i}") for i in range(4)]
        graph_c = [make_chunk(f"g{i}") for i in range(4)]
        merged = self.r._merge_results(classical, graph_c, top_k=3, query_id="q1")
        assert len(merged) == 3

    def test_classical_tagged_hybrid_classical(self):
        c = make_chunk("classical")
        merged = self.r._merge_results([c], [], top_k=5, query_id="q1")
        assert merged[0].metadata["graph_mode"] == "hybrid_classical"

    def test_graph_chunks_tagged_hybrid_graph(self):
        g = make_chunk("graph")
        g.metadata["graph_mode"] = "graph_expansion"
        merged = self.r._merge_results([], [g], top_k=5, query_id="q1")
        assert merged[0].metadata["graph_mode"] == "hybrid_graph"

    def test_empty_both_returns_empty(self):
        assert self.r._merge_results([], [], top_k=5, query_id="q1") == []


# ═══════════════════════════════════════════════════════════════════════════════
# Classical mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestClassicalMode:
    def test_classical_delegates_to_dense(self):
        r = GraphRetriever(config={"mode": "classical"})
        classical_chunk = make_chunk("RAG reduces hallucinations.")
        vs = make_qdrant([classical_chunk])
        result = r.retrieve(make_query(), vs, embedder=embedder)
        assert len(result) >= 1

    def test_classical_tags_graph_mode(self):
        r = GraphRetriever(config={"mode": "classical"})
        c = make_chunk("Dense retrieval uses vectors.")
        vs = make_qdrant([c])
        result = r.retrieve(make_query(), vs, embedder=embedder)
        assert all(ch.metadata.get("graph_mode") == "classical" for ch in result)

    def test_classical_no_embedder_returns_empty(self):
        r = GraphRetriever(config={"mode": "classical"})
        result = r.retrieve(make_query(), make_qdrant([make_chunk("text")]), embedder=None)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# Graph-only mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphOnlyMode:
    def test_no_graph_returns_empty(self):
        r = GraphRetriever(config={"mode": "graph"})
        result = r.retrieve(make_query("RAG uses Qdrant"), MagicMock(), embedder=None, graph=None)
        assert result == []

    def test_no_matching_entities_returns_empty(self):
        r = GraphRetriever(config={"mode": "graph"})
        graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)
        result = r.retrieve(
            make_query("What is the weather today?"),
            MagicMock(), embedder=None, graph=graph,
        )
        assert result == []

    def test_entity_match_returns_chunk_ids(self):
        r = GraphRetriever(config={"mode": "graph", "traversal_depth": 1})
        graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)
        result = r.retrieve(
            make_query("RAG uses Qdrant for retrieval"),
            MagicMock(), embedder=None, graph=graph,
        )
        assert len(result) >= 1

    def test_graph_mode_tagged_in_metadata(self):
        r = GraphRetriever(config={"mode": "graph", "traversal_depth": 1})
        graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)
        result = r.retrieve(
            make_query("RAG uses Qdrant"),
            MagicMock(), embedder=None, graph=graph,
        )
        assert all(ch.metadata.get("graph_mode") == "graph" for ch in result)

    def test_top_k_respected(self):
        r = GraphRetriever(config={"mode": "graph", "traversal_depth": 2})
        graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)
        result = r.retrieve(
            make_query("RAG LLM Qdrant", top_k=1),
            MagicMock(), embedder=None, graph=graph,
        )
        assert len(result) <= 1


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridMode:
    def test_hybrid_with_graph_returns_merged(self):
        r = GraphRetriever(config={"mode": "hybrid", "graph_weight": 0.5})
        graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)
        classical_chunk = make_chunk("RAG reduces hallucinations via retrieved context.")
        vs = make_qdrant([classical_chunk])
        result = r.retrieve(
            make_query("RAG uses Qdrant"), vs, embedder=embedder, graph=graph
        )
        assert len(result) >= 1

    def test_graph_weight_0_classical_only(self):
        """graph_weight=0.0 → only classical chunks, no graph traversal."""
        r = GraphRetriever(config={"mode": "hybrid", "graph_weight": 0.0})
        graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)
        c = make_chunk("Classical chunk.", "classical-id")
        vs = make_qdrant([c])
        result = r.retrieve(make_query("RAG"), vs, embedder=embedder, graph=graph)
        modes = {ch.metadata.get("graph_mode") for ch in result}
        assert "hybrid_graph" not in modes

    def test_graph_weight_1_graph_preferred(self):
        """graph_weight=1.0 → classical fetches 0 chunks (skip), graph dominates."""
        r = GraphRetriever(config={"mode": "hybrid", "graph_weight": 1.0, "traversal_depth": 1})
        graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)
        result = r.retrieve(
            make_query("RAG uses Qdrant", top_k=2),
            MagicMock(), embedder=None, graph=graph,
        )
        # Graph chunks should be in result
        assert all(
            ch.metadata.get("graph_mode") in ("hybrid_graph", "graph_expansion")
            for ch in result
        )

    def test_no_graph_hybrid_falls_back_to_classical(self):
        r = GraphRetriever(config={"mode": "hybrid", "graph_weight": 0.5})
        c = make_chunk("Classical only.")
        vs = make_qdrant([c])
        result = r.retrieve(make_query("test"), vs, embedder=embedder, graph=None)
        assert len(result) >= 1

    def test_dedup_across_classical_and_graph(self):
        """Chunk appearing in both classical and graph should appear only once."""
        r = GraphRetriever(config={"mode": "hybrid", "graph_weight": 0.5})
        shared_id = "shared-chunk-id"
        c = make_chunk("Shared content.", shared_id)
        vs = make_qdrant([c])
        graph = make_graph_with_entities(SAMPLE_ENTITIES, SAMPLE_EDGES)
        # Inject same chunk_id into a graph edge
        graph.edges[RAG_ID, QDRANT_ID]["chunk_id"] = shared_id
        result = r.retrieve(
            make_query("RAG uses Qdrant", top_k=5),
            vs, embedder=embedder, graph=graph,
        )
        chunk_ids = [ch.chunk_id for ch in result]
        assert len(chunk_ids) == len(set(chunk_ids))


# ═══════════════════════════════════════════════════════════════════════════════
# retrieve() error handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrieveErrorHandling:
    def test_vector_store_error_returns_empty(self):
        r = GraphRetriever(config={"mode": "classical"})
        vs = MagicMock()
        vs.search.side_effect = Exception("Qdrant down")
        result = r.retrieve(make_query(), vs, embedder=embedder)
        assert result == []

    def test_graph_traversal_error_returns_empty(self):
        r = GraphRetriever(config={"mode": "graph"})
        bad_graph = MagicMock()
        bad_graph.nodes = MagicMock(side_effect=Exception("Graph error"))
        result = r.retrieve(make_query(), MagicMock(), embedder=None, graph=bad_graph)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# config_schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigSchema:
    def test_schema_returns_dict(self):
        assert isinstance(GraphRetriever.config_schema(), dict)

    def test_schema_has_required_keys(self):
        schema = GraphRetriever.config_schema()
        for key in ["mode", "traversal_depth", "graph_weight",
                    "graph_service_url", "score_threshold", "ef", "top_k_multiplier"]:
            assert key in schema

    def test_schema_mode_options(self):
        schema = GraphRetriever.config_schema()
        for mode in _VALID_MODES:
            assert mode in schema["mode"]["options"]


# ═══════════════════════════════════════════════════════════════════════════════
# Factory + naming
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphRetrieverFactory:
    def test_factory_creates_graph_retriever(self):
        from raglab_retrievers import RetrieverFactory
        r = RetrieverFactory.create("graph", config={"mode": "classical"})
        assert isinstance(r, GraphRetriever)

    def test_graph_active_in_available(self):
        from raglab_retrievers import RetrieverFactory
        entries = {e["type"]: e for e in RetrieverFactory.available()}
        assert entries["graph"]["active"] is True

    def test_schema_via_factory(self):
        from raglab_retrievers import RetrieverFactory
        schema = RetrieverFactory.schema("graph")
        assert "mode" in schema and "traversal_depth" in schema

    def test_retriever_type_graph_enum_exists(self):
        assert RetrieverType.GRAPH.value == "graph"

    def test_graph_retriever_is_base_retriever(self):
        from raglab_retrievers.base import BaseRetriever
        assert issubclass(GraphRetriever, BaseRetriever)

    def test_naming_distinct_from_hybrid_retriever(self):
        """GraphRetriever ≠ HybridRetriever — different concepts, tested explicitly."""
        from raglab_retrievers.hybrid_retriever import HybridRetriever
        assert GraphRetriever.retriever_type == "graph"
        assert HybridRetriever.retriever_type == "hybrid"
        assert GraphRetriever is not HybridRetriever

    def test_all_r4_retrievers_active(self):
        from raglab_retrievers import RetrieverFactory
        entries = {e["type"]: e for e in RetrieverFactory.available()}
        for t in ["dense", "bm25", "hybrid", "mmr", "reranker", "compression", "graph"]:
            assert entries[t]["active"] is True
