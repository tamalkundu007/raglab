"""
GraphRetriever — graph-augmented retrieval with three operating modes.

Design principle (from R4 FRS):
    Graph RAG is ADDITIVE. It augments classical retrieval — it never silently
    replaces it. Three modes give the caller full control:

    "classical"  — Standard DenseRetriever, zero graph involvement.
                   Identical to calling DenseRetriever directly.

    "graph"      — No dense vector search. Find entities mentioned in the query
                   via keyword overlap → fetch their chunks from the graph.
                   Pure graph traversal from query-entity entry points.

    "hybrid"     — Dense retrieval for entry-point chunks → extract entity
                   mentions → traverse graph to expand context → merge results.
                   The core Graph RAG pattern.

Graph traversal:
    Starting from entry-point entity nodes, the traversal walks outgoing and
    incoming edges up to `traversal_depth` hops. At each hop, the neighbour's
    entity name is used to find associated chunks via BM25 lookup or exact
    match in the graph's chunk_id metadata.

    graph_weight parameter:
        0.0 → all context from classical retrieval (mode="hybrid" degrades to classical)
        1.0 → all context from graph traversal
        0.5 → equal mix (default)

Parameters:
    mode             : str   = "hybrid"      — "classical" | "graph" | "hybrid"
    traversal_depth  : int   = 2             — max hops from entry-point entities
    graph_weight     : float = 0.5           — fraction of top_k from graph traversal
    graph_service_url: str   = "http://graph:8010" — for entity lookups
    score_threshold  : float = 0.0           — dense retrieval floor
    ef               : int   = 128           — HNSW ef for dense stage
    top_k_multiplier : int   = 3             — over-fetch factor for dense stage
"""

from __future__ import annotations

import re
from typing import Any

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, QueryModel

from raglab_retrievers.base import BaseRetriever

_VALID_MODES = ("classical", "graph", "hybrid")


class GraphRetriever(BaseRetriever):
    """
    Graph-augmented retriever with classical / graph / hybrid modes. Active in R4.

    Requires:
        - A Qdrant-compatible vector_store for classical/hybrid modes.
        - An embedder callable for classical/hybrid modes.
        - app.state.graph (NetworkX DiGraph) for graph/hybrid modes,
          passed via the `graph` kwarg or extracted from vector_store context.
    """

    retriever_type: str = "graph"

    _DEFAULT_MODE: str = "hybrid"
    _DEFAULT_TRAVERSAL_DEPTH: int = 2
    _DEFAULT_GRAPH_WEIGHT: float = 0.5
    _DEFAULT_TOP_K_MULTIPLIER: int = 3

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}

        self.mode: str = cfg.get("mode", self._DEFAULT_MODE)
        self.traversal_depth: int = int(cfg.get("traversal_depth", self._DEFAULT_TRAVERSAL_DEPTH))
        self.graph_weight: float = float(cfg.get("graph_weight", self._DEFAULT_GRAPH_WEIGHT))
        self.graph_service_url: str = cfg.get("graph_service_url", "http://graph:8010").rstrip("/")
        self.score_threshold: float = float(cfg.get("score_threshold", 0.0))
        self.ef: int = int(cfg.get("ef", 128))
        self.top_k_multiplier: int = int(cfg.get("top_k_multiplier", self._DEFAULT_TOP_K_MULTIPLIER))

        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_MODES}, got {self.mode!r}"
            )
        if self.traversal_depth < 0:
            raise ValueError(f"traversal_depth must be >= 0, got {self.traversal_depth}")
        if not 0.0 <= self.graph_weight <= 1.0:
            raise ValueError(f"graph_weight must be in [0.0, 1.0], got {self.graph_weight}")
        if self.top_k_multiplier < 1:
            raise ValueError(f"top_k_multiplier must be >= 1, got {self.top_k_multiplier}")

    def _retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
        graph: Any = None,  # NetworkX DiGraph — injected by caller
    ) -> list[ChunkModel]:
        """
        Retrieve chunks using the configured mode.

        Args:
            query:        QueryModel (text, collection, top_k).
            vector_store: Qdrant client (required for classical/hybrid).
            embedder:     Embedding callable (required for classical/hybrid).
            graph:        NetworkX DiGraph (required for graph/hybrid). Optional.

        Returns:
            Ranked list of ChunkModel, deduplicated by chunk_id.
        """
        if self.mode == "classical":
            return self._classical_retrieve(query, vector_store, embedder)

        if self.mode == "graph":
            return self._graph_retrieve(query, graph, query.top_k)

        # hybrid: classical entry + graph expansion
        return self._hybrid_retrieve(query, vector_store, embedder, graph)

    def retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None = None,
        graph: Any = None,
    ) -> list[ChunkModel]:
        """
        Public retrieve method — wraps _retrieve with error handling.

        Accepts `graph` kwarg for NetworkX DiGraph injection.
        """
        try:
            return self._retrieve(query, vector_store, embedder, graph=graph)
        except RetrieverError:
            raise
        except Exception as exc:
            self._log.warning(
                "graph_retriever.error",
                mode=self.mode,
                error=str(exc),
            )
            return []

    # ── Classical mode ─────────────────────────────────────────────────────────

    def _classical_retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
    ) -> list[ChunkModel]:
        """Pure dense retrieval — no graph involved."""
        from raglab_retrievers.dense_retriever import DenseRetriever
        dense = DenseRetriever(config={
            "score_threshold": self.score_threshold,
            "ef": self.ef,
        })
        results = dense.retrieve(query, vector_store, embedder=embedder)
        for chunk in results:
            chunk.metadata["retriever"] = "graph"
            chunk.metadata["graph_mode"] = "classical"
        return results

    # ── Graph-only mode ────────────────────────────────────────────────────────

    def _graph_retrieve(
        self,
        query: QueryModel,
        graph: Any,
        top_k: int,
    ) -> list[ChunkModel]:
        """
        Graph-only retrieval: find entry-point entities from query text,
        traverse the graph, collect chunk_ids, return ChunkModel list.
        """
        if graph is None:
            self._log.warning("graph_retriever.no_graph", mode="graph")
            return []

        entry_nodes = self._find_entry_nodes(query.text, graph)
        if not entry_nodes:
            return []

        chunk_ids = self._traverse_graph(entry_nodes, graph, self.traversal_depth)
        return self._chunk_ids_to_models(
            chunk_ids=chunk_ids[:top_k],
            query_id=str(query.query_id),
            graph=graph,
            mode="graph",
        )

    # ── Hybrid mode ────────────────────────────────────────────────────────────

    def _hybrid_retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
        graph: Any,
    ) -> list[ChunkModel]:
        """
        Hybrid retrieval:
            1. Dense retrieval for classical entry-point chunks.
            2. Extract entity mentions from retrieved chunk texts.
            3. Traverse graph from those entities to expand context.
            4. Merge results, deduplicate, apply graph_weight blend.
        """
        top_k = query.top_k
        classical_k = max(1, round(top_k * (1.0 - self.graph_weight) * self.top_k_multiplier))
        graph_k = max(1, round(top_k * self.graph_weight * self.top_k_multiplier))

        # Step 1: classical dense retrieval
        classical_chunks: list[ChunkModel] = []
        if self.graph_weight < 1.0 and embedder is not None and vector_store is not None:
            from raglab_retrievers.dense_retriever import DenseRetriever
            from raglab_common.models import QueryModel as QM
            dense_q = QM(
                text=query.text,
                collection=query.collection,
                top_k=classical_k,
                retriever_type=query.retriever_type,
                llm_provider=query.llm_provider,
            )
            dense = DenseRetriever(config={"score_threshold": self.score_threshold, "ef": self.ef})
            classical_chunks = dense.retrieve(dense_q, vector_store, embedder=embedder)

        # Step 2: graph traversal expansion
        graph_chunks: list[ChunkModel] = []
        if graph is not None and self.graph_weight > 0.0:
            # Entry points: entities matching query terms + entities from classical chunks
            entry_nodes = self._find_entry_nodes(query.text, graph)

            # Also add entities mentioned in classical chunks
            for chunk in classical_chunks:
                entry_nodes.update(
                    self._find_entry_nodes(chunk.text, graph)
                )

            if entry_nodes:
                chunk_ids = self._traverse_graph(entry_nodes, graph, self.traversal_depth)
                graph_chunks = self._chunk_ids_to_models(
                    chunk_ids=chunk_ids[:graph_k],
                    query_id=str(query.query_id),
                    graph=graph,
                    mode="graph_expansion",
                )

        # Step 3: merge + deduplicate
        merged = self._merge_results(
            classical_chunks=classical_chunks,
            graph_chunks=graph_chunks,
            top_k=top_k,
            query_id=str(query.query_id),
        )
        return merged

    # ── Graph traversal helpers ────────────────────────────────────────────────

    def _find_entry_nodes(self, text: str, graph: Any) -> set[str]:
        """
        Find entity nodes whose name appears in the query/chunk text.

        Case-insensitive word-boundary match. Returns node IDs (UUIDs).
        """
        text_lower = text.lower()
        entry_nodes: set[str] = set()

        for node_id, attrs in graph.nodes(data=True):
            name = attrs.get("name", "")
            if not name:
                continue
            # Word-boundary match — "RAG" should not match "RAGNAR"
            pattern = r"\b" + re.escape(name.lower()) + r"\b"
            if re.search(pattern, text_lower):
                entry_nodes.add(node_id)

        return entry_nodes

    def _traverse_graph(
        self,
        entry_nodes: set[str],
        graph: Any,
        depth: int,
    ) -> list[str]:
        """
        BFS traversal from entry_nodes up to `depth` hops.

        Returns a list of chunk_ids discovered via traversal,
        ordered by traversal distance (closer nodes first).
        """
        visited_nodes: set[str] = set()
        chunk_ids: list[str] = []
        seen_chunks: set[str] = set()

        frontier = set(entry_nodes)

        for hop in range(depth + 1):
            next_frontier: set[str] = set()

            for node_id in frontier:
                if node_id in visited_nodes:
                    continue
                visited_nodes.add(node_id)

                # Collect chunk_ids from node and its edges
                node_attrs = graph.nodes.get(node_id, {})

                # chunk_ids from edges touching this node
                for _, tgt, edge_attrs in graph.out_edges(node_id, data=True):
                    cid = edge_attrs.get("chunk_id", "")
                    if cid and cid not in seen_chunks:
                        chunk_ids.append(cid)
                        seen_chunks.add(cid)
                    if hop < depth:
                        next_frontier.add(tgt)

                for src, _, edge_attrs in graph.in_edges(node_id, data=True):
                    cid = edge_attrs.get("chunk_id", "")
                    if cid and cid not in seen_chunks:
                        chunk_ids.append(cid)
                        seen_chunks.add(cid)
                    if hop < depth:
                        next_frontier.add(src)

            frontier = next_frontier - visited_nodes

        return chunk_ids

    def _chunk_ids_to_models(
        self,
        chunk_ids: list[str],
        query_id: str,
        graph: Any,
        mode: str,
    ) -> list[ChunkModel]:
        """
        Convert graph-traversal chunk_ids to ChunkModel list.

        In production, chunk texts would be fetched from Qdrant by ID.
        In this implementation we create placeholder ChunkModels — the
        retrieval-service enriches them from Qdrant in Phase 6 wiring.
        """
        chunks = []
        for i, chunk_id in enumerate(chunk_ids):
            chunks.append(ChunkModel(
                chunk_id=chunk_id,
                doc_id="",
                text=f"[Graph traversal chunk — ID: {chunk_id[:8]}]",
                chunk_index=i,
                token_count=0,
                metadata={
                    "retriever": "graph",
                    "graph_mode": mode,
                    "traversal_rank": i,
                    "query_id": query_id,
                },
            ))
        return chunks

    def _merge_results(
        self,
        classical_chunks: list[ChunkModel],
        graph_chunks: list[ChunkModel],
        top_k: int,
        query_id: str,
    ) -> list[ChunkModel]:
        """
        Merge classical and graph chunks, deduplicate by chunk_id, return top_k.

        Ordering: interleave by graph_weight proportion.
        Classical chunks come first (they have scores), graph after.
        """
        seen: set[str] = set()
        merged: list[ChunkModel] = []

        for chunk in classical_chunks:
            if chunk.chunk_id not in seen:
                chunk.metadata["graph_mode"] = "hybrid_classical"
                chunk.metadata["query_id"] = query_id
                merged.append(chunk)
                seen.add(chunk.chunk_id)

        for chunk in graph_chunks:
            if chunk.chunk_id not in seen:
                chunk.metadata["graph_mode"] = "hybrid_graph"
                merged.append(chunk)
                seen.add(chunk.chunk_id)

        return merged[:top_k]

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "mode": {
                "type": "str", "default": cls._DEFAULT_MODE,
                "options": list(_VALID_MODES),
                "description": (
                    "classical: dense only. "
                    "graph: entity matching + traversal only. "
                    "hybrid: dense entry + graph expansion (default)."
                ),
            },
            "traversal_depth": {
                "type": "int", "default": cls._DEFAULT_TRAVERSAL_DEPTH,
                "min": 0, "max": 5,
                "description": "Max hops from entry-point entities in graph traversal.",
            },
            "graph_weight": {
                "type": "float", "default": cls._DEFAULT_GRAPH_WEIGHT,
                "min": 0.0, "max": 1.0,
                "description": (
                    "Fraction of top_k filled from graph traversal. "
                    "0.0 = all classical, 1.0 = all graph."
                ),
            },
            "graph_service_url": {
                "type": "str", "default": "http://graph:8010",
                "description": "graph-service base URL for entity lookups.",
            },
            "score_threshold": {
                "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                "description": "Minimum score for dense retrieval stage.",
            },
            "ef": {
                "type": "int", "default": 128, "min": 16, "max": 512,
                "description": "HNSW ef parameter for dense retrieval.",
            },
            "top_k_multiplier": {
                "type": "int", "default": cls._DEFAULT_TOP_K_MULTIPLIER,
                "min": 1, "max": 10,
                "description": "Over-fetch multiplier for each retrieval stage.",
            },
        }
