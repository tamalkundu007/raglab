"""
Unit tests for observability-service (R6 Phase 2).

All DB calls mocked — zero infra required.

Covers:
- GET /health: 200, db status in dependencies
- GET /: version 0.2.0, release R6, endpoints list
- GET /obs/traces: 200, DB unavailable → 503
- GET /obs/traces/{id}: 200, not found → 404, DB unavailable → 503
- GET /obs/traces/{id}/timeline: 200, not found → 404, DB unavailable → 503
- GET /obs/services/stats: 200, DB unavailable → 503
- GET /obs/viewer: 200, returns HTML
- trace_viewer.html: D3.js present, service colour map, waterfall rendering functions
- get_trace_timeline: builds timeline structure from spans, depth assignment
- list_recent_traces: returns empty list on DB error (no raise)
- get_trace: returns empty list on DB error (no raise)
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def obs_client_no_db():
    from observability.main import app
    app.state.session_factory = None
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def obs_client_with_db():
    from observability.main import app

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock()

    mock_factory = MagicMock()
    mock_factory.return_value = mock_session
    app.state.session_factory = mock_factory
    return TestClient(app, raise_server_exceptions=False), mock_session


# ═══════════════════════════════════════════════════════════════════════════════
# Health + root
# ═══════════════════════════════════════════════════════════════════════════════

class TestObservabilityHealth:
    def test_health_returns_200(self, obs_client_no_db):
        r = obs_client_no_db.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, obs_client_no_db):
        assert obs_client_no_db.get("/health").json()["status"] == "ok"

    def test_health_shows_db_status(self, obs_client_no_db):
        r = obs_client_no_db.get("/health")
        assert "database" in r.json().get("dependencies", {})

    def test_health_db_unavailable_without_session(self, obs_client_no_db):
        r = obs_client_no_db.get("/health")
        assert r.json()["dependencies"]["database"] == "unavailable"

    def test_root_version_02(self, obs_client_no_db):
        r = obs_client_no_db.get("/")
        assert r.json()["version"] == "0.2.0"

    def test_root_release_r6(self, obs_client_no_db):
        r = obs_client_no_db.get("/")
        assert r.json()["release"] == "R6"

    def test_root_endpoints_listed(self, obs_client_no_db):
        r = obs_client_no_db.get("/")
        endpoints = r.json()["endpoints"]
        assert any("/obs/traces" in e for e in endpoints)
        assert any("/obs/viewer" in e for e in endpoints)


# ═══════════════════════════════════════════════════════════════════════════════
# /obs/traces — list
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceListEndpoint:
    def test_no_db_returns_503(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/traces")
        assert r.status_code == 503

    def test_with_db_returns_200(self, obs_client_with_db):
        client, session = obs_client_with_db
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        session.execute.return_value = mock_result
        r = client.get("/obs/traces")
        assert r.status_code == 200

    def test_with_db_returns_list(self, obs_client_with_db):
        client, session = obs_client_with_db
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        session.execute.return_value = mock_result
        r = client.get("/obs/traces")
        assert isinstance(r.json(), list)

    def test_service_filter_accepted(self, obs_client_with_db):
        client, session = obs_client_with_db
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        session.execute.return_value = mock_result
        r = client.get("/obs/traces?service=pipeline")
        assert r.status_code == 200

    def test_status_filter_accepted(self, obs_client_with_db):
        client, session = obs_client_with_db
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        session.execute.return_value = mock_result
        r = client.get("/obs/traces?status=error")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# /obs/traces/{id} — single trace
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceSingleEndpoint:
    def test_no_db_returns_503(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/traces/abc123")
        assert r.status_code == 503

    def test_not_found_returns_404(self, obs_client_with_db):
        client, session = obs_client_with_db
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        session.execute.return_value = mock_result
        r = client.get("/obs/traces/nonexistent-trace")
        assert r.status_code == 404

    def test_found_returns_200(self, obs_client_with_db):
        client, session = obs_client_with_db
        tid = str(uuid.uuid4()).replace("-", "")
        sid = str(uuid.uuid4()).replace("-", "")[:16]
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [{
            "trace_id": tid, "span_id": sid, "parent_span_id": None,
            "service_name": "pipeline", "operation_name": "run_pipeline",
            "start_time_ms": 1000, "duration_ms": 150,
            "status": "ok", "attributes": {}, "events": [], "created_at": None,
        }]
        session.execute.return_value = mock_result
        r = client.get(f"/obs/traces/{tid}")
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# /obs/traces/{id}/timeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimelineEndpoint:
    def test_no_db_returns_503(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/traces/abc/timeline")
        assert r.status_code == 503

    def test_not_found_returns_404(self, obs_client_with_db):
        client, session = obs_client_with_db
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        session.execute.return_value = mock_result
        r = client.get("/obs/traces/bad-id/timeline")
        assert r.status_code == 404

    def test_found_returns_timeline_structure(self, obs_client_with_db):
        client, session = obs_client_with_db
        tid = "a" * 32
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [{
            "trace_id": tid, "span_id": "b" * 16, "parent_span_id": None,
            "service_name": "api-gateway", "operation_name": "POST /query",
            "start_time_ms": 1000, "duration_ms": 80,
            "status": "ok", "attributes": json.dumps({}), "events": json.dumps([]),
            "created_at": None,
        }]
        session.execute.return_value = mock_result
        r = client.get(f"/obs/traces/{tid}/timeline")
        assert r.status_code == 200
        body = r.json()
        assert "trace_id" in body
        assert "spans" in body
        assert "total_duration_ms" in body

    def test_timeline_spans_have_start_offset(self, obs_client_with_db):
        client, session = obs_client_with_db
        tid = "c" * 32
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [{
            "trace_id": tid, "span_id": "d" * 16, "parent_span_id": None,
            "service_name": "pipeline", "operation_name": "chunk",
            "start_time_ms": 5000, "duration_ms": 50,
            "status": "ok", "attributes": {}, "events": [], "created_at": None,
        }]
        session.execute.return_value = mock_result
        r = client.get(f"/obs/traces/{tid}/timeline")
        spans = r.json()["spans"]
        assert "start_offset_ms" in spans[0]
        assert spans[0]["start_offset_ms"] == 0  # only span, offset is 0

    def test_timeline_spans_have_depth(self, obs_client_with_db):
        client, session = obs_client_with_db
        tid = "e" * 32
        sid_parent = "f" * 16
        sid_child  = "0" * 16
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [
            {
                "trace_id": tid, "span_id": sid_parent, "parent_span_id": None,
                "service_name": "api-gateway", "operation_name": "root",
                "start_time_ms": 1000, "duration_ms": 200,
                "status": "ok", "attributes": {}, "events": [], "created_at": None,
            },
            {
                "trace_id": tid, "span_id": sid_child, "parent_span_id": sid_parent,
                "service_name": "pipeline", "operation_name": "child",
                "start_time_ms": 1010, "duration_ms": 100,
                "status": "ok", "attributes": {}, "events": [], "created_at": None,
            },
        ]
        session.execute.return_value = mock_result
        r = client.get(f"/obs/traces/{tid}/timeline")
        spans = r.json()["spans"]
        depths = {s["span_id"]: s["depth"] for s in spans}
        assert depths[sid_parent] == 0
        assert depths[sid_child] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# /obs/services/stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestServiceStatsEndpoint:
    def test_no_db_returns_503(self, obs_client_no_db):
        assert obs_client_no_db.get("/obs/services/stats").status_code == 503

    def test_with_db_returns_200(self, obs_client_with_db):
        client, session = obs_client_with_db
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        session.execute.return_value = mock_result
        assert client.get("/obs/services/stats").status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# /obs/viewer — HTML page
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraceViewerPage:
    def test_viewer_returns_200(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert r.status_code == 200

    def test_viewer_returns_html(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert "text/html" in r.headers["content-type"]

    def test_viewer_has_d3(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"d3" in r.content

    def test_viewer_has_raglab_branding(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"RAGLab" in r.content

    def test_viewer_has_trace_viewer_title(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"Trace Viewer" in r.content

    def test_viewer_has_service_colours(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"SERVICE_COLOURS" in r.content

    def test_viewer_has_waterfall_function(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"renderWaterfall" in r.content

    def test_viewer_has_load_traces_function(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"loadTraces" in r.content

    def test_viewer_has_span_detail_function(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"showSpanDetail" in r.content

    def test_viewer_has_timeline_api_call(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"timeline" in r.content

    def test_viewer_r6_badge(self, obs_client_no_db):
        r = obs_client_no_db.get("/obs/viewer")
        assert b"R6" in r.content


# ═══════════════════════════════════════════════════════════════════════════════
# DB query functions — error handling
# ═══════════════════════════════════════════════════════════════════════════════

class TestQueryFunctions:
    @pytest.mark.asyncio
    async def test_list_recent_traces_db_error_returns_empty(self):
        from observability.db.queries import list_recent_traces
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("connection lost"))
        result = await list_recent_traces(session)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_trace_db_error_returns_empty(self):
        from observability.db.queries import get_trace
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("timeout"))
        result = await get_trace(session, "abc123")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_trace_timeline_empty_trace(self):
        from observability.db.queries import get_trace_timeline
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = []
        session.execute.return_value = mock_result
        result = await get_trace_timeline(session, "bad-trace")
        assert result["total_duration_ms"] == 0
        assert result["spans"] == []

    @pytest.mark.asyncio
    async def test_get_service_stats_db_error_returns_empty(self):
        from observability.db.queries import get_service_stats
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db down"))
        result = await get_service_stats(session)
        assert result == []
