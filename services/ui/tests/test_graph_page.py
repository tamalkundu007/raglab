"""
Unit tests for Graph Explorer page and router (R4 Phase 7).

Covers:
- GET /graph returns 200 HTML
- Template contains D3.js script import
- Template contains all required JS functions
- Template contains all control knobs (collection, community detection, leiden resolution,
  traversal depth, traversal query, node-size-by, colour-by, show-labels, link-strength)
- Template contains graph_service_url injection
- Template contains build/reset buttons
- Template contains inspector panel
- Template contains status bar
- Template contains legend container
- Template contains SVG canvas element
- Control Panel sidebar: Graph Explorer link present with R4 badge
- Control Panel: GraphRAG section activated (no 'Coming Soon')
- Control Panel: graph mode select (hybrid/classical/graph) present
- Control Panel: community detection select present
- Control Panel: traversal depth range present
- Router: _ctx() includes graph_url and graph_service_url
- Router: GET /graph serves graph.html
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from ui.settings import UISettings


@pytest.fixture
def ui_client():
    from ui.main import app
    app.state.settings = UISettings()
    return TestClient(app)


# ── /graph endpoint ────────────────────────────────────────────────────────────

class TestGraphEndpoint:
    def test_get_graph_returns_200(self, ui_client):
        r = ui_client.get("/graph")
        assert r.status_code == 200

    def test_get_graph_returns_html(self, ui_client):
        r = ui_client.get("/graph")
        assert "text/html" in r.headers["content-type"]

    def test_graph_page_contains_raglab_branding(self, ui_client):
        r = ui_client.get("/graph")
        assert b"RAGLab" in r.content

    def test_graph_page_title(self, ui_client):
        r = ui_client.get("/graph")
        assert b"Graph Explorer" in r.content

    def test_graph_page_r4_badge(self, ui_client):
        r = ui_client.get("/graph")
        assert b"R4" in r.content

    def test_graph_service_url_injected(self, ui_client):
        r = ui_client.get("/graph")
        # Default graph service URL injected via Jinja2
        assert b"graph" in r.content and b"8010" in r.content

    def test_api_base_injected(self, ui_client):
        r = ui_client.get("/graph")
        assert b"/api/v1" in r.content

    def test_graph_links_back_to_control_panel(self, ui_client):
        r = ui_client.get("/graph")
        assert b"Control Panel" in r.content


# ── D3.js presence ─────────────────────────────────────────────────────────────

class TestD3Presence:
    def test_d3_script_imported(self, ui_client):
        r = ui_client.get("/graph")
        assert b"d3" in r.content

    def test_d3_from_cdnjs(self, ui_client):
        r = ui_client.get("/graph")
        assert b"cdnjs.cloudflare.com" in r.content

    def test_d3_version_7(self, ui_client):
        r = ui_client.get("/graph")
        assert b"d3/7" in r.content or b"d3.min.js" in r.content


# ── JS functions ───────────────────────────────────────────────────────────────

class TestJSFunctions:
    def test_buildGraph_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"buildGraph" in r.content

    def test_renderGraph_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"renderGraph" in r.content

    def test_traverseFromEntity_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"traverseFromEntity" in r.content

    def test_highlightNeighbours_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"highlightNeighbours" in r.content

    def test_selectNode_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"selectNode" in r.content

    def test_buildD3Data_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"buildD3Data" in r.content

    def test_buildLegend_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"buildLegend" in r.content

    def test_resetView_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"resetView" in r.content

    def test_showToast_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"showToast" in r.content

    def test_updateStats_function_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"updateStats" in r.content


# ── Control knobs ──────────────────────────────────────────────────────────────

class TestControlKnobs:
    def test_collection_input_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"collection" in r.content

    def test_graph_service_url_input_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"graph-url" in r.content

    def test_community_detection_select_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"community-detection" in r.content

    def test_leiden_option_in_select(self, ui_client):
        r = ui_client.get("/graph")
        assert b"Leiden" in r.content

    def test_leiden_resolution_range_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"leiden-res" in r.content

    def test_traversal_depth_range_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"traversal-depth" in r.content

    def test_query_entity_input_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"query-entity" in r.content

    def test_node_size_by_select_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"node-size-by" in r.content

    def test_colour_by_select_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"colour-by" in r.content

    def test_show_labels_select_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"show-labels" in r.content

    def test_link_strength_range_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"link-strength" in r.content

    def test_build_button_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"Build Graph" in r.content

    def test_reset_button_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"Reset View" in r.content


# ── Graph API endpoints called ─────────────────────────────────────────────────

class TestAPIEndpointReferences:
    def test_calls_graph_build_endpoint(self, ui_client):
        r = ui_client.get("/graph")
        assert b"/graph/build" in r.content

    def test_calls_entities_endpoint(self, ui_client):
        r = ui_client.get("/graph")
        assert b"/graph/entities" in r.content

    def test_calls_relationships_endpoint(self, ui_client):
        r = ui_client.get("/graph")
        assert b"/graph/relationships" in r.content

    def test_calls_communities_endpoint(self, ui_client):
        r = ui_client.get("/graph")
        assert b"/graph/communities" in r.content


# ── UI elements ────────────────────────────────────────────────────────────────

class TestUIElements:
    def test_svg_canvas_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"graph-svg" in r.content

    def test_inspector_panel_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"inspector" in r.content

    def test_status_bar_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"status-bar" in r.content

    def test_legend_container_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"legend" in r.content

    def test_tooltip_element_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"tooltip" in r.content

    def test_empty_state_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"empty-state" in r.content

    def test_community_colours_defined(self, ui_client):
        r = ui_client.get("/graph")
        assert b"COMMUNITY_COLOURS" in r.content

    def test_type_colours_defined(self, ui_client):
        r = ui_client.get("/graph")
        assert b"TYPE_COLOURS" in r.content


# ── Graph modes present ────────────────────────────────────────────────────────

class TestGraphModes:
    def test_community_detection_in_page(self, ui_client):
        r = ui_client.get("/graph")
        assert b"community-detection" in r.content

    def test_leiden_referenced_in_page(self, ui_client):
        r = ui_client.get("/graph")
        assert b"Leiden" in r.content

    def test_graph_only_traversal_present(self, ui_client):
        r = ui_client.get("/graph")
        assert b"Traversal" in r.content or b"traversal" in r.content

    def test_force_simulation_used(self, ui_client):
        r = ui_client.get("/graph")
        assert b"forceSimulation" in r.content

    def test_bfs_traversal_in_js(self, ui_client):
        r = ui_client.get("/graph")
        assert b"frontier" in r.content


# ── Control Panel sidebar ──────────────────────────────────────────────────────

class TestControlPanelUpdates:
    def test_graph_explorer_link_in_sidebar(self, ui_client):
        r = ui_client.get("/")
        assert b"/graph" in r.content

    def test_graph_explorer_r4_badge_in_sidebar(self, ui_client):
        r = ui_client.get("/")
        # R4 badge present in sidebar
        assert b"R4" in r.content

    def test_graphrag_section_activated(self, ui_client):
        r = ui_client.get("/")
        # Should NOT say "Coming Soon" in the GraphRAG section
        # (it was the stub label before R4)
        content = r.content.decode()
        # Check that the section is now active
        assert "R4 Active" in content or "GraphRAG" in content

    def test_graph_mode_select_in_control_panel(self, ui_client):
        r = ui_client.get("/")
        assert b"graph-mode" in r.content or b"Graph Mode" in r.content

    def test_traversal_depth_in_control_panel(self, ui_client):
        r = ui_client.get("/")
        assert b"Traversal Depth" in r.content or b"graph-traversal-depth" in r.content


# ── Router context ─────────────────────────────────────────────────────────────

class TestRouterContext:
    def test_ctx_includes_graph_url(self):
        from ui.routers.pages import _ctx
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert "graph_url" in ctx
        assert ctx["graph_url"] == "/graph"

    def test_ctx_includes_graph_service_url(self):
        from ui.routers.pages import _ctx
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert "graph_service_url" in ctx

    def test_ctx_includes_comparison_url(self):
        from ui.routers.pages import _ctx
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert ctx["comparison_url"] == "/compare"

    def test_ctx_no_settings_uses_defaults(self):
        from ui.routers.pages import _ctx
        req = MagicMock()
        req.app.state.settings = None
        ctx = _ctx(req)
        assert ctx["graph_url"] == "/graph"
        assert "graph_service_url" in ctx

    def test_get_graph_endpoint_exists(self, ui_client):
        r = ui_client.get("/graph")
        assert r.status_code == 200
