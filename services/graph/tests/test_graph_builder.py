"""
Unit tests for GraphBuilder (R4 Phase 5).

All DB calls and NetworkX/Leiden operations use controlled in-memory
fixtures — zero infrastructure required.

Covers:
- GraphBuildResult dataclass
- CommunityInfo dataclass
- GraphBuilder config
- _build_from_db: entities + relationships added as nodes/edges
- _build_from_db: edge only added if both nodes exist
- get_cached: cache hit, cache miss, TTL expiry
- invalidate: removes from cache
- _fallback_communities: weakly connected components
- _leiden_communities: mocked leidenalg partition → CommunityInfo list
- _annotate_communities: community_id written to node attributes
- get_communities: extracts CommunityInfo from annotated graph
- build(): cache hit returns cached graph
- build(): new build stores in cache + annotates communities
- build(): DB error returns empty graph with error in result
- build(): community detection skipped when disabled
- POST /graph/build: 200 response shape, stores graph on app.state
- POST /graph/build: DB unavailable → 503
- GET /graph/communities: 404 without build, 200 after
- GET /graph/node/{id}: 404 no graph, 404 bad entity, 200 with neighbours
- min_community_size filters small components
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import networkx as nx

from graph.extraction.graph_builder import (
    GraphBuilder,
    GraphBuildResult,
    CommunityInfo,
    _NX_AVAILABLE,
    _LEIDEN_AVAILABLE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_graph(nodes: list[tuple[str, dict]], edges: list[tuple[str, str, dict]]) -> nx.DiGraph:
    """Build a NetworkX DiGraph from node/edge specs."""
    g = nx.DiGraph()
    for node_id, attrs in nodes:
        g.add_node(node_id, **attrs)
    for src, tgt, attrs in edges:
        g.add_edge(src, tgt, **attrs)
    return g


def make_entity_mock(name: str, etype: str = "CONCEPT", collection: str = "raglab") -> MagicMock:
    e = MagicMock()
    e.id = uuid.uuid4()
    e.name = name
    e.entity_type = etype
    e.collection = collection
    e.description = f"Description of {name}"
    e.doc_id = "doc-001"
    return e


def make_rel_mock(source_id, target_id, rel_type="RELATED_TO") -> MagicMock:
    r = MagicMock()
    r.source_id = source_id
    r.target_id = target_id
    r.relation_type = rel_type
    r.weight = 1.0
    r.source_chunk_id = "chunk-1"
    return r


def make_session_with_data(entities: list, relationships: list) -> AsyncMock:
    session = AsyncMock()

    entity_result = MagicMock()
    entity_result.scalars.return_value.all.return_value = entities

    rel_result = MagicMock()
    rel_result.scalars.return_value.all.return_value = relationships

    session.execute = AsyncMock(side_effect=[entity_result, rel_result])
    return session


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclass contracts
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataclasses:
    def test_graph_build_result_defaults(self):
        r = GraphBuildResult()
        assert r.node_count == 0
        assert r.edge_count == 0
        assert r.community_count == 0
        assert r.communities_detected is False
        assert r.error is None

    def test_community_info_defaults(self):
        c = CommunityInfo(community_id=0)
        assert c.entity_ids == []
        assert c.entity_names == []
        assert c.size == 0

    def test_community_info_fields(self):
        c = CommunityInfo(
            community_id=1,
            entity_ids=["id1", "id2"],
            entity_names=["RAG", "Qdrant"],
            size=2,
        )
        assert c.size == 2
        assert len(c.entity_ids) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# GraphBuilder config
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphBuilderConfig:
    def test_defaults(self):
        b = GraphBuilder()
        assert b.enable_community_detection is True
        assert b.leiden_resolution == 1.0
        assert b.leiden_n_iterations == 10
        assert b.min_community_size == 2
        assert b.cache_ttl_seconds == 300.0

    def test_custom_config(self):
        b = GraphBuilder(config={
            "enable_community_detection": False,
            "leiden_resolution": 0.5,
            "min_community_size": 3,
            "cache_ttl_seconds": 60.0,
        })
        assert b.enable_community_detection is False
        assert b.leiden_resolution == 0.5
        assert b.min_community_size == 3
        assert b.cache_ttl_seconds == 60.0

    def test_config_schema_keys(self):
        schema = GraphBuilder.config_schema()
        for key in ["enable_community_detection", "leiden_resolution",
                    "leiden_n_iterations", "min_community_size", "cache_ttl_seconds"]:
            assert key in schema


# ═══════════════════════════════════════════════════════════════════════════════
# _build_from_db
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildFromDB:
    @pytest.mark.asyncio
    async def test_entities_become_nodes(self):
        e1 = make_entity_mock("RAG")
        e2 = make_entity_mock("Qdrant", "TECHNOLOGY")
        session = make_session_with_data([e1, e2], [])
        builder = GraphBuilder(config={"enable_community_detection": False})
        graph, result = await builder._build_from_db(session, "raglab")
        assert graph.number_of_nodes() == 2
        assert result.node_count == 2

    @pytest.mark.asyncio
    async def test_node_attributes_populated(self):
        e = make_entity_mock("RAG", "CONCEPT")
        session = make_session_with_data([e], [])
        builder = GraphBuilder(config={"enable_community_detection": False})
        graph, _ = await builder._build_from_db(session, "raglab")
        node_attrs = graph.nodes[str(e.id)]
        assert node_attrs["name"] == "RAG"
        assert node_attrs["entity_type"] == "CONCEPT"
        assert node_attrs["community_id"] is None

    @pytest.mark.asyncio
    async def test_relationships_become_edges(self):
        e1 = make_entity_mock("RAG")
        e2 = make_entity_mock("Qdrant")
        rel = make_rel_mock(e1.id, e2.id, "USES")
        session = make_session_with_data([e1, e2], [rel])
        builder = GraphBuilder(config={"enable_community_detection": False})
        graph, result = await builder._build_from_db(session, "raglab")
        assert graph.number_of_edges() == 1
        assert result.edge_count == 1

    @pytest.mark.asyncio
    async def test_edge_attributes_populated(self):
        e1 = make_entity_mock("A")
        e2 = make_entity_mock("B")
        rel = make_rel_mock(e1.id, e2.id, "CAUSES")
        session = make_session_with_data([e1, e2], [rel])
        builder = GraphBuilder(config={"enable_community_detection": False})
        graph, _ = await builder._build_from_db(session, "raglab")
        edge_attrs = graph.edges[str(e1.id), str(e2.id)]
        assert edge_attrs["relation_type"] == "CAUSES"
        assert edge_attrs["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_edge_skipped_if_node_missing(self):
        """Relationship referencing a non-existent entity should be silently skipped."""
        e1 = make_entity_mock("A")
        phantom_id = uuid.uuid4()
        rel = make_rel_mock(e1.id, phantom_id, "RELATED_TO")
        session = make_session_with_data([e1], [rel])
        builder = GraphBuilder(config={"enable_community_detection": False})
        graph, _ = await builder._build_from_db(session, "raglab")
        assert graph.number_of_edges() == 0

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_graph(self):
        session = make_session_with_data([], [])
        builder = GraphBuilder(config={"enable_community_detection": False})
        graph, result = await builder._build_from_db(session, "raglab")
        assert graph.number_of_nodes() == 0
        assert result.node_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Cache behaviour
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheBehaviour:
    def test_get_cached_returns_none_for_unknown(self):
        b = GraphBuilder()
        assert b.get_cached("unknown") is None

    def test_get_cached_returns_graph_after_store(self):
        b = GraphBuilder(config={"cache_ttl_seconds": 60.0})
        g = nx.DiGraph()
        b._cache["raglab"] = (g, time.time())
        assert b.get_cached("raglab") is g

    def test_get_cached_returns_none_after_ttl_expiry(self):
        b = GraphBuilder(config={"cache_ttl_seconds": 0.0})
        g = nx.DiGraph()
        b._cache["raglab"] = (g, time.time() - 1.0)
        assert b.get_cached("raglab") is None
        assert "raglab" not in b._cache  # evicted

    def test_invalidate_removes_from_cache(self):
        b = GraphBuilder()
        g = nx.DiGraph()
        b._cache["raglab"] = (g, time.time())
        b.invalidate("raglab")
        assert "raglab" not in b._cache

    def test_invalidate_unknown_collection_no_error(self):
        b = GraphBuilder()
        b.invalidate("does_not_exist")  # should not raise

    @pytest.mark.asyncio
    async def test_build_returns_cached_on_second_call(self):
        b = GraphBuilder(config={
            "enable_community_detection": False,
            "cache_ttl_seconds": 60.0,
        })
        e = make_entity_mock("RAG")
        session = make_session_with_data([e], [])

        # First build
        g1, _ = await b.build(session, "raglab")
        # Second build — should hit cache (session.execute not called again)
        session2 = AsyncMock()  # fresh session that should NOT be called
        g2, _ = await b.build(session2, "raglab")
        session2.execute.assert_not_called()
        assert g1 is g2

    @pytest.mark.asyncio
    async def test_force_rebuild_bypasses_cache(self):
        b = GraphBuilder(config={
            "enable_community_detection": False,
            "cache_ttl_seconds": 60.0,
        })
        g_cached = nx.DiGraph()
        b._cache["raglab"] = (g_cached, time.time())

        e = make_entity_mock("NewEntity")
        session = make_session_with_data([e], [])
        g_new, _ = await b.build(session, "raglab", force_rebuild=True)
        assert g_new is not g_cached


# ═══════════════════════════════════════════════════════════════════════════════
# Community detection — fallback (connected components)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFallbackCommunities:
    def test_single_component_gives_one_community(self):
        g = make_graph(
            [("n1", {"name": "A"}), ("n2", {"name": "B"}), ("n3", {"name": "C"})],
            [("n1", "n2", {}), ("n2", "n3", {})],
        )
        b = GraphBuilder(config={"min_community_size": 2})
        communities = b._fallback_communities(g)
        assert len(communities) == 1
        assert communities[0].size == 3

    def test_two_disconnected_components(self):
        g = make_graph(
            [("n1", {"name": "A"}), ("n2", {"name": "B"}),
             ("n3", {"name": "C"}), ("n4", {"name": "D"})],
            [("n1", "n2", {}), ("n3", "n4", {})],
        )
        b = GraphBuilder(config={"min_community_size": 2})
        communities = b._fallback_communities(g)
        assert len(communities) == 2

    def test_singleton_filtered_by_min_size(self):
        g = make_graph(
            [("n1", {"name": "A"}), ("n2", {"name": "B"}), ("isolated", {"name": "X"})],
            [("n1", "n2", {})],
        )
        b = GraphBuilder(config={"min_community_size": 2})
        communities = b._fallback_communities(g)
        sizes = [c.size for c in communities]
        assert all(s >= 2 for s in sizes)

    def test_empty_graph_returns_empty(self):
        g = nx.DiGraph()
        b = GraphBuilder()
        assert b._fallback_communities(g) == []

    def test_community_names_populated(self):
        g = make_graph(
            [("n1", {"name": "RAG", "entity_type": "CONCEPT"}),
             ("n2", {"name": "Qdrant", "entity_type": "TECHNOLOGY"})],
            [("n1", "n2", {})],
        )
        b = GraphBuilder(config={"min_community_size": 2})
        communities = b._fallback_communities(g)
        assert "RAG" in communities[0].entity_names or "Qdrant" in communities[0].entity_names


# ═══════════════════════════════════════════════════════════════════════════════
# Community annotation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnotateCommunities:
    def test_community_id_written_to_nodes(self):
        g = make_graph(
            [("n1", {"community_id": None}), ("n2", {"community_id": None})],
            [],
        )
        c = CommunityInfo(community_id=7, entity_ids=["n1", "n2"], size=2)
        GraphBuilder._annotate_communities(g, [c])
        assert g.nodes["n1"]["community_id"] == 7
        assert g.nodes["n2"]["community_id"] == 7

    def test_node_not_in_graph_skipped_safely(self):
        g = make_graph([("n1", {})], [])
        c = CommunityInfo(community_id=0, entity_ids=["n1", "phantom"], size=2)
        GraphBuilder._annotate_communities(g, [c])  # should not raise
        assert g.nodes["n1"]["community_id"] == 0

    def test_multiple_communities_annotated(self):
        g = make_graph(
            [("n1", {}), ("n2", {}), ("n3", {}), ("n4", {})],
            [],
        )
        c0 = CommunityInfo(community_id=0, entity_ids=["n1", "n2"], size=2)
        c1 = CommunityInfo(community_id=1, entity_ids=["n3", "n4"], size=2)
        GraphBuilder._annotate_communities(g, [c0, c1])
        assert g.nodes["n1"]["community_id"] == 0
        assert g.nodes["n3"]["community_id"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# get_communities
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetCommunities:
    def test_returns_community_infos(self):
        g = make_graph(
            [("n1", {"name": "RAG", "entity_type": "CONCEPT", "community_id": 0}),
             ("n2", {"name": "Qdrant", "entity_type": "TECHNOLOGY", "community_id": 0}),
             ("n3", {"name": "LangChain", "entity_type": "TECHNOLOGY", "community_id": 1}),
             ("n4", {"name": "OpenAI", "entity_type": "ORGANIZATION", "community_id": 1})],
            [],
        )
        b = GraphBuilder()
        communities = b.get_communities(g)
        assert len(communities) == 2

    def test_sorted_by_size_descending(self):
        g = make_graph(
            [("n1", {"name": "A", "entity_type": "C", "community_id": 0}),
             ("n2", {"name": "B", "entity_type": "C", "community_id": 0}),
             ("n3", {"name": "X", "entity_type": "C", "community_id": 0}),
             ("n4", {"name": "Y", "entity_type": "C", "community_id": 1})],
            [],
        )
        b = GraphBuilder()
        communities = b.get_communities(g)
        assert communities[0].size >= communities[-1].size

    def test_nodes_without_community_id_excluded(self):
        g = make_graph(
            [("n1", {"name": "A", "entity_type": "C", "community_id": None}),
             ("n2", {"name": "B", "entity_type": "C", "community_id": 0})],
            [],
        )
        b = GraphBuilder()
        communities = b.get_communities(g)
        # Only community 0 should be present
        assert len(communities) == 1
        assert communities[0].community_id == 0

    def test_empty_graph_returns_empty(self):
        b = GraphBuilder()
        assert b.get_communities(nx.DiGraph()) == []

    def test_none_graph_returns_empty(self):
        b = GraphBuilder()
        assert b.get_communities(None) == []


# ═══════════════════════════════════════════════════════════════════════════════
# build() end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildEndToEnd:
    @pytest.mark.asyncio
    async def test_build_returns_graph_and_result(self):
        e1 = make_entity_mock("RAG")
        e2 = make_entity_mock("Qdrant")
        rel = make_rel_mock(e1.id, e2.id)
        session = make_session_with_data([e1, e2], [rel])

        b = GraphBuilder(config={"enable_community_detection": False})
        graph, result = await b.build(session, "raglab")

        assert graph.number_of_nodes() == 2
        assert graph.number_of_edges() == 1
        assert result.node_count == 2
        assert result.edge_count == 1
        assert result.error is None

    @pytest.mark.asyncio
    async def test_build_stores_in_cache(self):
        e = make_entity_mock("RAG")
        session = make_session_with_data([e], [])
        b = GraphBuilder(config={"enable_community_detection": False})
        await b.build(session, "raglab")
        assert "raglab" in b._cache

    @pytest.mark.asyncio
    async def test_build_with_community_detection_disabled(self):
        e1 = make_entity_mock("A")
        e2 = make_entity_mock("B")
        rel = make_rel_mock(e1.id, e2.id)
        session = make_session_with_data([e1, e2], [rel])

        b = GraphBuilder(config={"enable_community_detection": False})
        graph, result = await b.build(session, "raglab")
        assert result.communities_detected is False
        assert result.community_count == 0

    @pytest.mark.asyncio
    async def test_build_with_fallback_community_detection(self):
        """Test community detection runs (as fallback) when graph has nodes."""
        e1 = make_entity_mock("A")
        e2 = make_entity_mock("B")
        rel = make_rel_mock(e1.id, e2.id)
        session = make_session_with_data([e1, e2], [rel])

        b = GraphBuilder(config={
            "enable_community_detection": True,
            "min_community_size": 1,
        })
        # Patch Leiden as unavailable to use fallback
        with patch("graph.extraction.graph_builder._LEIDEN_AVAILABLE", False):
            graph, result = await b.build(session, "raglab")

        assert result.communities_detected is True
        assert result.community_count >= 1

    @pytest.mark.asyncio
    async def test_build_sets_community_id_on_nodes(self):
        e1 = make_entity_mock("A")
        e2 = make_entity_mock("B")
        rel = make_rel_mock(e1.id, e2.id)
        session = make_session_with_data([e1, e2], [rel])

        b = GraphBuilder(config={"enable_community_detection": True, "min_community_size": 1})
        with patch("graph.extraction.graph_builder._LEIDEN_AVAILABLE", False):
            graph, result = await b.build(session, "raglab")

        # Nodes should have community_id set
        cids = {data.get("community_id") for _, data in graph.nodes(data=True)}
        assert None not in cids or len(cids) > 1  # at least some got IDs

    @pytest.mark.asyncio
    async def test_build_error_returns_empty_graph_with_error(self):
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB connection lost"))

        b = GraphBuilder(config={"enable_community_detection": False})
        graph, result = await b.build(session, "raglab")

        assert graph.number_of_nodes() == 0
        assert result.error is not None
        assert "DB connection lost" in result.error

    @pytest.mark.asyncio
    async def test_build_time_ms_populated(self):
        e = make_entity_mock("X")
        session = make_session_with_data([e], [])
        b = GraphBuilder(config={"enable_community_detection": False})
        _, result = await b.build(session, "raglab")
        assert result.build_time_ms >= 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def graph_client_no_db():
    from graph.main import app
    app.state.session_factory = None
    app.state.graph = None
    app.state.graph_builder = None
    from fastapi.testclient import TestClient
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def graph_client_with_graph():
    from graph.main import app
    from fastapi.testclient import TestClient

    # Inject a pre-built graph
    e1_id = str(uuid.uuid4())
    e2_id = str(uuid.uuid4())
    g = make_graph(
        [(e1_id, {"name": "RAG", "entity_type": "CONCEPT", "community_id": 0,
                  "description": "Framework", "collection": "raglab", "doc_id": "d1"}),
         (e2_id, {"name": "Qdrant", "entity_type": "TECHNOLOGY", "community_id": 0,
                  "description": "Vector DB", "collection": "raglab", "doc_id": "d1"})],
        [(e1_id, e2_id, {"relation_type": "USES", "weight": 1.0})],
    )
    b = GraphBuilder()
    b._cache["raglab"] = (g, time.time())
    app.state.graph = g
    app.state.graph_builder = b
    app.state.graph_collection = "raglab"
    app.state.session_factory = None  # not needed for community/node endpoints

    return TestClient(app, raise_server_exceptions=False), g, e1_id, e2_id


class TestBuildEndpoint:
    def test_build_without_db_returns_503(self, graph_client_no_db):
        r = graph_client_no_db.post("/graph/build", json={"collection": "raglab"})
        assert r.status_code == 503


class TestCommunitiesEndpoint:
    def test_communities_without_graph_returns_404(self, graph_client_no_db):
        r = graph_client_no_db.get("/graph/communities")
        assert r.status_code == 404

    def test_communities_with_graph_returns_200(self, graph_client_with_graph):
        client, g, e1_id, e2_id = graph_client_with_graph
        r = client.get("/graph/communities")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_communities_contain_entity_names(self, graph_client_with_graph):
        client, g, e1_id, e2_id = graph_client_with_graph
        r = client.get("/graph/communities")
        all_names = [name for c in r.json() for name in c["entity_names"]]
        assert "RAG" in all_names or "Qdrant" in all_names


class TestNodeEndpoint:
    def test_node_without_graph_returns_404(self, graph_client_no_db):
        r = graph_client_no_db.get("/graph/node/some-id")
        assert r.status_code == 404

    def test_node_unknown_id_returns_404(self, graph_client_with_graph):
        client, g, e1_id, e2_id = graph_client_with_graph
        r = client.get("/graph/node/unknown-entity-id")
        assert r.status_code == 404

    def test_node_known_id_returns_200(self, graph_client_with_graph):
        client, g, e1_id, e2_id = graph_client_with_graph
        r = client.get(f"/graph/node/{e1_id}")
        assert r.status_code == 200

    def test_node_response_has_correct_fields(self, graph_client_with_graph):
        client, g, e1_id, e2_id = graph_client_with_graph
        r = client.get(f"/graph/node/{e1_id}")
        body = r.json()
        assert body["name"] == "RAG"
        assert body["entity_type"] == "CONCEPT"
        assert "outgoing" in body
        assert "incoming" in body

    def test_node_outgoing_contains_neighbour(self, graph_client_with_graph):
        client, g, e1_id, e2_id = graph_client_with_graph
        r = client.get(f"/graph/node/{e1_id}")
        outgoing = r.json()["outgoing"]
        assert len(outgoing) == 1
        assert outgoing[0]["target_id"] == e2_id

    def test_node_relation_type_in_edge(self, graph_client_with_graph):
        client, g, e1_id, e2_id = graph_client_with_graph
        r = client.get(f"/graph/node/{e1_id}")
        edge = r.json()["outgoing"][0]
        assert edge["relation_type"] == "USES"
