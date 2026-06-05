"""
Unit tests for the Retrieval Comparison page (R3).

Covers:
- GET /compare returns 200 HTML
- Template contains all 6 strategy sections with correct labels
- Template contains per-strategy config knobs
- Template contains all required JS functions
- Template injects api_base correctly
- Control Panel sidebar contains Compare link
- Router: _ctx() builds correct context dict
- Overlap detection logic (computeOverlaps equivalent in Python)
- Strategy colour assignments present in template
- Keyboard shortcut (Ctrl+Enter) wired in template
- Summary bar elements present
- View controls (side-by-side / overlap) present
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ui.settings import UISettings


@pytest.fixture
def ui_client():
    from ui.main import app
    app.state.settings = UISettings()
    return TestClient(app)


# ── /compare endpoint ──────────────────────────────────────────────────────────


class TestComparisonEndpoint:
    def test_get_compare_returns_200(self, ui_client):
        r = ui_client.get("/compare")
        assert r.status_code == 200

    def test_get_compare_returns_html(self, ui_client):
        r = ui_client.get("/compare")
        assert "text/html" in r.headers["content-type"]

    def test_compare_contains_raglab_branding(self, ui_client):
        r = ui_client.get("/compare")
        assert b"RAGLab" in r.content

    def test_compare_has_page_title(self, ui_client):
        r = ui_client.get("/compare")
        assert b"Retrieval Comparison" in r.content

    def test_api_base_injected(self, ui_client):
        r = ui_client.get("/compare")
        assert b"/api/v1" in r.content

    def test_compare_links_back_to_control_panel(self, ui_client):
        r = ui_client.get("/compare")
        assert b"Control Panel" in r.content


# ── Strategy presence ──────────────────────────────────────────────────────────


class TestStrategyPresence:
    def test_dense_strategy_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"DENSE" in r.content or b"dense" in r.content

    def test_bm25_strategy_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"BM25" in r.content or b"bm25" in r.content

    def test_hybrid_strategy_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"HYBRID" in r.content or b"hybrid" in r.content

    def test_mmr_strategy_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"MMR" in r.content or b"mmr" in r.content

    def test_reranker_strategy_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"RE-RANKER" in r.content or b"reranker" in r.content

    def test_compression_strategy_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"COMPRESSION" in r.content or b"compression" in r.content

    def test_all_six_checkboxes_present(self, ui_client):
        r = ui_client.get("/compare")
        content = r.content
        for s in [b"chk-dense", b"chk-bm25", b"chk-hybrid", b"chk-mmr", b"chk-reranker", b"chk-compression"]:
            assert s in content, f"Missing checkbox: {s}"


# ── Strategy config knobs ──────────────────────────────────────────────────────


class TestStrategyKnobs:
    def test_dense_score_threshold_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"dense-score-threshold" in r.content

    def test_dense_ef_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"dense-ef" in r.content

    def test_bm25_k1_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"bm25-k1" in r.content

    def test_bm25_b_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"bm25-b" in r.content

    def test_hybrid_alpha_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"hybrid-alpha" in r.content

    def test_hybrid_rrf_k_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"hybrid-rrf-k" in r.content

    def test_mmr_lambda_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"mmr-lambda" in r.content

    def test_mmr_fetch_k_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"mmr-fetch-k" in r.content

    def test_reranker_model_select(self, ui_client):
        r = ui_client.get("/compare")
        assert b"reranker-model" in r.content
        assert b"MiniLM" in r.content

    def test_compression_strategy_select(self, ui_client):
        r = ui_client.get("/compare")
        assert b"compression-strategy" in r.content

    def test_compression_overlap_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"compression-overlap" in r.content

    def test_common_top_k_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"top-k" in r.content

    def test_common_collection_knob(self, ui_client):
        r = ui_client.get("/compare")
        assert b"collection" in r.content

    def test_common_llm_provider_select(self, ui_client):
        r = ui_client.get("/compare")
        assert b"llm-provider" in r.content


# ── JS functions ───────────────────────────────────────────────────────────────


class TestJSFunctions:
    def test_runComparison_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"runComparison" in r.content

    def test_fetchStrategy_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"fetchStrategy" in r.content

    def test_computeOverlaps_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"computeOverlaps" in r.content

    def test_renderColumn_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"renderColumn" in r.content

    def test_setView_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"setView" in r.content

    def test_clearResults_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"clearResults" in r.content

    def test_toggleStrategy_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"toggleStrategy" in r.content

    def test_getStrategyConfig_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"getStrategyConfig" in r.content

    def test_updateSummary_function_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"updateSummary" in r.content

    def test_keyboard_shortcut_ctrl_enter(self, ui_client):
        r = ui_client.get("/compare")
        assert b"ctrlKey" in r.content or b"metaKey" in r.content
        assert b"Enter" in r.content


# ── UI elements ────────────────────────────────────────────────────────────────


class TestUIElements:
    def test_compare_button_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"Compare" in r.content

    def test_clear_button_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"Clear" in r.content

    def test_results_grid_container_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"results-grid" in r.content

    def test_summary_bar_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"summary-bar" in r.content

    def test_view_controls_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"view-controls" in r.content

    def test_side_by_side_view_button(self, ui_client):
        r = ui_client.get("/compare")
        assert b"Side by Side" in r.content

    def test_overlap_highlight_button(self, ui_client):
        r = ui_client.get("/compare")
        assert b"Highlight Overlaps" in r.content

    def test_overlap_css_class_present(self, ui_client):
        r = ui_client.get("/compare")
        assert b"overlap" in r.content

    def test_strategy_colours_defined(self, ui_client):
        r = ui_client.get("/compare")
        # At least the known strategy colours are referenced
        assert b"#d4a843" in r.content  # dense gold
        assert b"#3ecf8e" in r.content  # bm25 green
        assert b"#5e9cf5" in r.content  # hybrid blue
        assert b"#a78bfa" in r.content  # mmr purple


# ── RRF label ─────────────────────────────────────────────────────────────────


class TestRRFDocumentation:
    def test_rrf_mentioned_in_template(self, ui_client):
        r = ui_client.get("/compare")
        assert b"RRF" in r.content or b"rrf" in r.content

    def test_rrf_k_param_exposed(self, ui_client):
        r = ui_client.get("/compare")
        assert b"rrf_k" in r.content or b"rrf-k" in r.content


# ── Control Panel sidebar ──────────────────────────────────────────────────────


class TestControlPanelSidebar:
    def test_compare_link_in_sidebar(self, ui_client):
        r = ui_client.get("/")
        assert b"/compare" in r.content

    def test_compare_link_has_r3_badge(self, ui_client):
        r = ui_client.get("/")
        # R3 badge near compare link
        assert b"R3" in r.content


# ── Router context builder ─────────────────────────────────────────────────────


class TestRouterContext:
    def test_ctx_includes_api_base(self):
        from ui.routers.pages import _ctx
        from unittest.mock import MagicMock
        from ui.settings import UISettings
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert "api_base" in ctx
        assert ctx["api_base"] == "/api/v1"

    def test_ctx_includes_control_panel_url(self):
        from ui.routers.pages import _ctx
        from unittest.mock import MagicMock
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert "control_panel_url" in ctx
        assert ctx["control_panel_url"] == "/"

    def test_ctx_includes_comparison_url(self):
        from ui.routers.pages import _ctx
        from unittest.mock import MagicMock
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert "comparison_url" in ctx
        assert ctx["comparison_url"] == "/compare"

    def test_ctx_includes_gateway_url(self):
        from ui.routers.pages import _ctx
        from unittest.mock import MagicMock
        req = MagicMock()
        req.app.state.settings = UISettings()
        ctx = _ctx(req)
        assert "gateway_url" in ctx

    def test_ctx_no_settings_uses_defaults(self):
        from ui.routers.pages import _ctx
        from unittest.mock import MagicMock
        req = MagicMock()
        req.app.state.settings = None
        ctx = _ctx(req)
        assert ctx["api_base"] == "/api/v1"
        assert ctx["control_panel_url"] == "/"


# ── Overlap computation (Python port of JS computeOverlaps) ───────────────────


class TestOverlapDetectionLogic:
    """
    Python-side tests of the overlap detection algorithm.
    The JS function computeOverlaps(resultLists) finds chunk_ids
    appearing in more than one strategy's result list.
    We test the equivalent Python logic here.
    """

    def _compute_overlaps(self, result_lists: list[list[str]]) -> set[str]:
        """Python port of the JS computeOverlaps function."""
        id_counts: dict[str, int] = {}
        for lst in result_lists:
            seen: set[str] = set()
            for chunk_id in lst:
                if chunk_id not in seen:
                    id_counts[chunk_id] = id_counts.get(chunk_id, 0) + 1
                    seen.add(chunk_id)
        return {cid for cid, count in id_counts.items() if count > 1}

    def test_no_overlap_empty_set(self):
        result = self._compute_overlaps([["a", "b"], ["c", "d"]])
        assert result == set()

    def test_full_overlap_all_ids(self):
        result = self._compute_overlaps([["a", "b"], ["a", "b"]])
        assert result == {"a", "b"}

    def test_partial_overlap(self):
        result = self._compute_overlaps([["a", "b", "c"], ["c", "d", "e"]])
        assert result == {"c"}

    def test_three_strategies_overlap(self):
        result = self._compute_overlaps([["a", "b"], ["b", "c"], ["c", "d"]])
        assert result == {"b", "c"}

    def test_duplicate_within_single_list_not_counted(self):
        """Same chunk_id twice in one list should count as 1 occurrence."""
        result = self._compute_overlaps([["a", "a"], ["b", "c"]])
        assert "a" not in result  # a appears only in one list (deduplicated)

    def test_empty_lists_no_overlap(self):
        result = self._compute_overlaps([[], []])
        assert result == set()

    def test_single_strategy_no_overlap(self):
        result = self._compute_overlaps([["x", "y", "z"]])
        assert result == set()

    def test_all_six_strategies_overlap(self):
        shared = "shared-chunk-id"
        lists = [[shared, f"unique-{i}"] for i in range(6)]
        result = self._compute_overlaps(lists)
        assert shared in result
