"""
GraphBuilder — constructs and caches a NetworkX DiGraph from Postgres data.

Architecture (Phase 0 decision: Postgres + NetworkX):
    - Entities and relationships live in Postgres (source of truth).
    - NetworkX DiGraph is built in-memory for traversal queries.
    - The graph is cached on app.state (rebuilt on /graph/build or cache TTL).
    - No Neo4j, no new infra. Traversal depth >3 can revisit this choice.

Graph structure:
    Nodes: entity UUIDs (str) — attributes: name, entity_type, collection
    Edges: (source_id, target_id) — attributes: relation_type, weight, chunk_id

Community detection (optional):
    Requires leidenalg + igraph. If not installed, community detection is skipped
    gracefully. Communities are stored as node attributes: community_id (int).

    The Leiden algorithm (Traag, Waltman, van Eck 2019) is a refinement of Louvain
    that guarantees well-connected communities. It is the state-of-the-art for
    knowledge graph community detection in GraphRAG pipelines.

    Community summaries:
        After detection, each community gets a summary node (virtual entity)
        listing the entities in that cluster. This enables summary-level retrieval
        in Phase 6 — a query can hit a community summary and expand from there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from raglab_common.logging import get_logger

log = get_logger(__name__)

# Module-level optional imports for test patchability
try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    nx = None  # type: ignore[assignment]
    _NX_AVAILABLE = False

try:
    import leidenalg
    import igraph as ig
    _LEIDEN_AVAILABLE = True
except ImportError:
    leidenalg = None  # type: ignore[assignment]
    ig = None  # type: ignore[assignment]
    _LEIDEN_AVAILABLE = False


@dataclass
class GraphBuildResult:
    """Result of building the in-memory graph."""
    node_count: int = 0
    edge_count: int = 0
    community_count: int = 0
    communities_detected: bool = False
    build_time_ms: float = 0.0
    collection: str = "raglab"
    error: str | None = None


@dataclass
class CommunityInfo:
    """Metadata for a detected community cluster."""
    community_id: int
    entity_ids: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    entity_types: list[str] = field(default_factory=list)
    size: int = 0


class GraphBuilder:
    """
    Builds and caches a NetworkX DiGraph from Postgres graph data.

    Usage:
        builder = GraphBuilder()
        graph, result = await builder.build(session, collection="raglab")

    The returned graph is a networkx.DiGraph. Nodes are entity UUIDs.
    Each node has attributes: name, entity_type, collection, community_id.
    Each edge has attributes: relation_type, weight, chunk_id.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enable_community_detection: bool = bool(
            cfg.get("enable_community_detection", True)
        )
        self.leiden_resolution: float = float(cfg.get("leiden_resolution", 1.0))
        self.leiden_n_iterations: int = int(cfg.get("leiden_n_iterations", 10))
        self.min_community_size: int = int(cfg.get("min_community_size", 2))
        self.cache_ttl_seconds: float = float(cfg.get("cache_ttl_seconds", 300.0))

        # In-memory cache: collection → (graph, build_time)
        self._cache: dict[str, tuple[Any, float]] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def build(
        self,
        session: Any,
        collection: str = "raglab",
        force_rebuild: bool = False,
    ) -> tuple[Any, GraphBuildResult]:
        """
        Build or return cached NetworkX DiGraph for a collection.

        Args:
            session:       AsyncSession for Postgres queries.
            collection:    Collection name filter.
            force_rebuild: Bypass cache and rebuild.

        Returns:
            (nx.DiGraph, GraphBuildResult)
        """
        if not _NX_AVAILABLE:
            raise RuntimeError("networkx not installed. Run: pip install networkx")

        # Check cache
        if not force_rebuild and collection in self._cache:
            graph, cached_at = self._cache[collection]
            if time.time() - cached_at < self.cache_ttl_seconds:
                log.info("graph_builder.cache_hit", collection=collection)
                nodes = graph.number_of_nodes()
                edges = graph.number_of_edges()
                return graph, GraphBuildResult(
                    node_count=nodes, edge_count=edges, collection=collection
                )

        t0 = time.perf_counter()
        try:
            graph, result = await self._build_from_db(session, collection)
        except Exception as exc:
            log.error("graph_builder.build_failed", collection=collection, error=str(exc))
            return nx.DiGraph(), GraphBuildResult(
                collection=collection, error=str(exc)
            )

        # Community detection
        if self.enable_community_detection and graph.number_of_nodes() > 0:
            communities = self._detect_communities(graph)
            result.community_count = len(communities)
            result.communities_detected = True
            self._annotate_communities(graph, communities)

        result.build_time_ms = (time.perf_counter() - t0) * 1000
        self._cache[collection] = (graph, time.time())

        log.info(
            "graph_builder.built",
            collection=collection,
            nodes=result.node_count,
            edges=result.edge_count,
            communities=result.community_count,
            ms=round(result.build_time_ms, 1),
        )
        return graph, result

    def get_cached(self, collection: str) -> Any | None:
        """Return cached graph for collection, or None if not cached / expired."""
        if collection not in self._cache:
            return None
        graph, cached_at = self._cache[collection]
        if time.time() - cached_at > self.cache_ttl_seconds:
            del self._cache[collection]
            return None
        return graph

    def invalidate(self, collection: str) -> None:
        """Remove collection from cache (call after new extraction run)."""
        self._cache.pop(collection, None)
        log.info("graph_builder.cache_invalidated", collection=collection)

    def get_communities(self, graph: Any) -> list[CommunityInfo]:
        """
        Extract community metadata from an annotated graph.

        Returns:
            List of CommunityInfo, sorted by size descending.
        """
        if graph is None or not _NX_AVAILABLE:
            return []

        community_map: dict[int, CommunityInfo] = {}
        for node_id, attrs in graph.nodes(data=True):
            cid = attrs.get("community_id")
            if cid is None:
                continue
            if cid not in community_map:
                community_map[cid] = CommunityInfo(community_id=cid)
            info = community_map[cid]
            info.entity_ids.append(node_id)
            info.entity_names.append(attrs.get("name", ""))
            info.entity_types.append(attrs.get("entity_type", ""))
            info.size += 1

        return sorted(community_map.values(), key=lambda c: c.size, reverse=True)

    # ── Internal build ─────────────────────────────────────────────────────────

    async def _build_from_db(
        self, session: Any, collection: str
    ) -> tuple[Any, GraphBuildResult]:
        """Load entities and relationships from Postgres, build nx.DiGraph."""
        from sqlalchemy import select
        from graph.models.orm import GraphEntity, GraphRelationship

        graph = nx.DiGraph()

        # Load entities
        entity_stmt = select(GraphEntity).where(GraphEntity.collection == collection)
        entity_result = await session.execute(entity_stmt)
        entities = list(entity_result.scalars().all())

        for entity in entities:
            graph.add_node(
                str(entity.id),
                name=entity.name,
                entity_type=entity.entity_type,
                collection=entity.collection,
                description=entity.description or "",
                doc_id=entity.doc_id or "",
                community_id=None,
            )

        # Load relationships
        from sqlalchemy.orm import joinedload
        rel_stmt = (
            select(GraphRelationship)
            .where(GraphRelationship.collection == collection)
        )
        rel_result = await session.execute(rel_stmt)
        relationships = list(rel_result.scalars().all())

        for rel in relationships:
            src = str(rel.source_id)
            tgt = str(rel.target_id)
            if graph.has_node(src) and graph.has_node(tgt):
                graph.add_edge(
                    src, tgt,
                    relation_type=rel.relation_type,
                    weight=rel.weight,
                    chunk_id=rel.source_chunk_id or "",
                )

        return graph, GraphBuildResult(
            node_count=graph.number_of_nodes(),
            edge_count=graph.number_of_edges(),
            collection=collection,
        )

    # ── Community detection ────────────────────────────────────────────────────

    def _detect_communities(self, graph: Any) -> list[CommunityInfo]:
        """
        Run Leiden community detection on the NetworkX graph.

        Falls back to a trivial single-community partition if leidenalg
        is unavailable (no crash — community detection is optional).

        Returns:
            List of CommunityInfo objects (one per detected community).
        """
        if not _LEIDEN_AVAILABLE:
            log.info("graph_builder.leiden_unavailable", reason="leidenalg not installed")
            return self._fallback_communities(graph)

        try:
            return self._leiden_communities(graph)
        except Exception as exc:
            log.warning("graph_builder.leiden_failed", error=str(exc))
            return self._fallback_communities(graph)

    def _leiden_communities(self, graph: Any) -> list[CommunityInfo]:
        """Run Leiden algorithm via leidenalg + igraph."""
        nodes = list(graph.nodes())
        if not nodes:
            return []

        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Build igraph from NetworkX edges
        edges_idx = [
            (node_to_idx[u], node_to_idx[v])
            for u, v in graph.edges()
            if u in node_to_idx and v in node_to_idx
        ]

        ig_graph = ig.Graph(n=len(nodes), edges=edges_idx, directed=False)

        # Run Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.leiden_resolution,
            n_iterations=self.leiden_n_iterations,
        )

        # Map partition to CommunityInfo
        communities: dict[int, CommunityInfo] = {}
        for community_id, member_indices in enumerate(partition):
            if len(member_indices) < self.min_community_size:
                continue
            info = CommunityInfo(community_id=community_id)
            for idx in member_indices:
                node_id = nodes[idx]
                attrs = graph.nodes[node_id]
                info.entity_ids.append(node_id)
                info.entity_names.append(attrs.get("name", ""))
                info.entity_types.append(attrs.get("entity_type", ""))
                info.size += 1
            communities[community_id] = info

        return list(communities.values())

    def _fallback_communities(self, graph: Any) -> list[CommunityInfo]:
        """
        Simple connected-component fallback when Leiden is unavailable.

        Each weakly connected component becomes a community.
        """
        if not _NX_AVAILABLE or graph is None:
            return []

        communities = []
        for cid, component in enumerate(nx.weakly_connected_components(graph)):
            if len(component) < self.min_community_size:
                continue
            info = CommunityInfo(community_id=cid)
            for node_id in component:
                attrs = graph.nodes[node_id]
                info.entity_ids.append(node_id)
                info.entity_names.append(attrs.get("name", ""))
                info.entity_types.append(attrs.get("entity_type", ""))
                info.size += 1
            communities.append(info)

        return communities

    @staticmethod
    def _annotate_communities(graph: Any, communities: list[CommunityInfo]) -> None:
        """Write community_id back onto graph node attributes."""
        for community in communities:
            for node_id in community.entity_ids:
                if graph.has_node(node_id):
                    graph.nodes[node_id]["community_id"] = community.community_id

    # ── Schema ────────────────────────────────────────────────────────────────

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "enable_community_detection": {
                "type": "bool", "default": True,
                "description": "Run Leiden community detection after graph build.",
            },
            "leiden_resolution": {
                "type": "float", "default": 1.0, "min": 0.1, "max": 10.0,
                "description": "Leiden resolution parameter. Higher = more, smaller communities.",
            },
            "leiden_n_iterations": {
                "type": "int", "default": 10, "min": 1, "max": 100,
                "description": "Leiden algorithm iterations.",
            },
            "min_community_size": {
                "type": "int", "default": 2, "min": 1, "max": 100,
                "description": "Minimum entities per community to retain.",
            },
            "cache_ttl_seconds": {
                "type": "float", "default": 300.0, "min": 0.0,
                "description": "Seconds before cached graph is considered stale.",
            },
        }
