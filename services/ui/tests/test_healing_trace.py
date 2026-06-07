"""
Unit tests for Self-Healing Trace page (R5 Phase 6).

Covers:
- GET /healing-trace returns 200 HTML
- Template contains all three gate names
- Template contains detect/score/remediate language
- Template contains JS functions: buildTrace, renderTrace, clearTrace, runWithHealing
- Template contains query input, run button, gate-bar, empty-state
- Template contains gate icon elements
- Template contains trace-header, trace-body, trace-action classes
- Template contains chunk-list rendering
- Template contains stat elements: query-time, chunks-returned, gates-fired, heals
- Control Panel: self-healing section present and active (not stub)
- Control Panel: chunk-quality-enabled toggle present
- Control Panel: retrieval-healing-enabled toggle present
- Control Panel: groundedness-enabled toggle present
- Control Panel: quarantine-strategy select present
- Control Panel: escalation-order select present
- Control Panel: groundedness-action select present
- Control Panel: updateHealConfig JS function present
- Control Panel: link to /healing-trace present
- Router: _ctx includes healing_trace_url
- Router: GET /healing-trace serves healing_trace.html
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


# ── /healing-trace endpoint ────────────────────────────────────────────────────

class TestHealingTraceEndpoint:
    def test_get_healing_trace_200(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert r.status_code == 200

    def test_returns_html(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert "text/html" in r.headers["content-type"]

    def test_page_title_present(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"Healing Trace" in r.content

    def test_r5_badge_present(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"R5" in r.content

    def test_raglab_branding_present(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"RAGLab" in r.content

    def test_links_back_to_control_panel(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"Control Panel" in r.content


# ── Gate names ────────────────────────────────────────────────────────────────

class TestGateNames:
    def test_chunk_quality_gate_named(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"CHUNK QUALITY" in r.content or b"Chunk Quality" in r.content

    def test_retrieval_feedback_gate_named(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"RETRIEVAL FEEDBACK" in r.content or b"Retrieval Feedback" in r.content

    def test_groundedness_gate_named(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"GROUNDEDNESS" in r.content or b"Groundedness" in r.content


# ── Detect→score→remediate language ───────────────────────────────────────────

class TestTraceLanguage:
    def test_not_a_black_box_message(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"black box" in r.content or b"auditable" in r.content

    def test_detect_score_remediate_language(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"detect" in r.content.lower() or b"score" in r.content.lower()


# ── JS functions ──────────────────────────────────────────────────────────────

class TestJSFunctions:
    def test_buildTrace_function(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"buildTrace" in r.content

    def test_renderTrace_function(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"renderTrace" in r.content

    def test_clearTrace_function(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"clearTrace" in r.content

    def test_runWithHealing_function(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"runWithHealing" in r.content

    def test_toggleBody_function(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"toggleBody" in r.content

    def test_showToast_function(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"showToast" in r.content


# ── UI elements ───────────────────────────────────────────────────────────────

class TestUIElements:
    def test_query_textarea_present(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"query-input" in r.content

    def test_run_button_present(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"run-btn" in r.content or b"Run" in r.content

    def test_gate_bar_present(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"gate-bar" in r.content

    def test_empty_state_present(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"empty-state" in r.content or b"empty" in r.content

    def test_stat_query_time_element(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"stat-query-time" in r.content

    def test_stat_chunks_returned_element(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"stat-chunks-returned" in r.content

    def test_stat_gates_fired_element(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"stat-gates-fired" in r.content

    def test_stat_heals_element(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"stat-heals" in r.content

    def test_trace_entry_class(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"trace-entry" in r.content

    def test_gate_icon_classes(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"gate-pass" in r.content
        assert b"gate-fail" in r.content
        assert b"gate-heal" in r.content

    def test_action_badge_classes(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"action-accepted" in r.content or b"action-healed" in r.content

    def test_enable_healing_toggle(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"enable-healing" in r.content

    def test_chunk_list_rendering(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"chunk-list" in r.content or b"renderChunkList" in r.content

    def test_collection_select(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"collection-select" in r.content

    def test_strategy_select(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert b"strategy-select" in r.content


# ── Control Panel self-healing section ────────────────────────────────────────

class TestControlPanelHealing:
    def test_self_healing_section_exists(self, ui_client):
        r = ui_client.get("/")
        assert b"Self-Healing" in r.content

    def test_r5_active_badge(self, ui_client):
        r = ui_client.get("/")
        assert b"R5 Active" in r.content or b"R5" in r.content

    def test_chunk_quality_toggle_present(self, ui_client):
        r = ui_client.get("/")
        assert b"chunk-quality-enabled" in r.content

    def test_retrieval_healing_toggle_present(self, ui_client):
        r = ui_client.get("/")
        assert b"retrieval-healing-enabled" in r.content

    def test_groundedness_toggle_present(self, ui_client):
        r = ui_client.get("/")
        assert b"groundedness-enabled" in r.content

    def test_quarantine_strategy_select(self, ui_client):
        r = ui_client.get("/")
        assert b"quarantine-strategy" in r.content

    def test_escalation_order_select(self, ui_client):
        r = ui_client.get("/")
        assert b"escalation-order" in r.content

    def test_groundedness_action_select(self, ui_client):
        r = ui_client.get("/")
        assert b"groundedness-action" in r.content

    def test_updateHealConfig_js_function(self, ui_client):
        r = ui_client.get("/")
        assert b"updateHealConfig" in r.content

    def test_healing_trace_link(self, ui_client):
        r = ui_client.get("/")
        assert b"/healing-trace" in r.content

    def test_view_healing_trace_text(self, ui_client):
        r = ui_client.get("/")
        assert b"Healing Trace" in r.content

    def test_min_quality_score_range(self, ui_client):
        r = ui_client.get("/")
        assert b"min-quality-score" in r.content

    def test_score_floor_range(self, ui_client):
        r = ui_client.get("/")
        assert b"score-floor" in r.content

    def test_groundedness_threshold_range(self, ui_client):
        r = ui_client.get("/")
        assert b"groundedness-threshold" in r.content


# ── Router context ─────────────────────────────────────────────────────────────

class TestRouterContext:
    def test_ctx_includes_healing_trace_url(self):
        from ui.routers.pages import _ctx
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert "healing_trace_url" in ctx
        assert ctx["healing_trace_url"] == "/healing-trace"

    def test_ctx_still_has_graph_url(self):
        from ui.routers.pages import _ctx
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert ctx["graph_url"] == "/graph"

    def test_ctx_still_has_comparison_url(self):
        from ui.routers.pages import _ctx
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert ctx["comparison_url"] == "/compare"

    def test_healing_trace_endpoint_exists(self, ui_client):
        r = ui_client.get("/healing-trace")
        assert r.status_code == 200

    def test_all_ui_routes_return_200(self, ui_client):
        for route in ["/", "/compare", "/graph", "/healing-trace"]:
            r = ui_client.get(route)
            assert r.status_code == 200, f"Route {route} returned {r.status_code}"
