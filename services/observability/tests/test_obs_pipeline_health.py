"""Unit tests for pipeline health + self-healing trace (R6 Phase 6)."""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def obs_no_db():
    from observability.main import app
    app.state.session_factory = None
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def obs_db():
    from observability.main import app
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_factory = MagicMock(return_value=mock_session)
    app.state.session_factory = mock_factory
    return TestClient(app, raise_server_exceptions=False), mock_session


class TestHealthEndpoints:
    def test_pipeline_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/health/pipeline").status_code == 503

    def test_failed_jobs_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/health/failed-jobs").status_code == 503

    def test_gates_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/health/gates").status_code == 503

    def test_pipeline_with_db_200(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.one_or_none.return_value = None
        session.execute.return_value = mr
        r = client.get("/obs/health/pipeline")
        assert r.status_code == 200
        assert r.json()["total_jobs"] == 0

    def test_failed_jobs_returns_list(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert isinstance(client.get("/obs/health/failed-jobs").json(), list)

    def test_gates_returns_list(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert isinstance(client.get("/obs/health/gates").json(), list)


class TestHealthDashboardPage:
    def test_dashboard_200(self, obs_no_db):
        assert obs_no_db.get("/obs/health/dashboard").status_code == 200

    def test_returns_html(self, obs_no_db):
        assert "text/html" in obs_no_db.get("/obs/health/dashboard").headers["content-type"]

    def test_has_raglab_branding(self, obs_no_db):
        assert b"RAGLab" in obs_no_db.get("/obs/health/dashboard").content

    def test_has_pipeline_health_title(self, obs_no_db):
        assert b"Pipeline Health" in obs_no_db.get("/obs/health/dashboard").content

    def test_has_heal_gate_section(self, obs_no_db):
        assert b"gate-card" in obs_no_db.get("/obs/health/dashboard").content

    def test_has_failed_jobs_section(self, obs_no_db):
        assert b"failed-table" in obs_no_db.get("/obs/health/dashboard").content

    def test_has_gate_api_call(self, obs_no_db):
        assert b"health/gates" in obs_no_db.get("/obs/health/dashboard").content

    def test_r6_badge(self, obs_no_db):
        assert b"R6" in obs_no_db.get("/obs/health/dashboard").content


class TestHealthQueryFunctions:
    @pytest.mark.asyncio
    async def test_pipeline_health_error_returns_empty(self):
        from observability.db.health_queries import get_pipeline_health
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await get_pipeline_health(session) == {}

    @pytest.mark.asyncio
    async def test_failed_jobs_error_returns_empty(self):
        from observability.db.health_queries import get_failed_jobs
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await get_failed_jobs(session) == []

    @pytest.mark.asyncio
    async def test_gate_summary_error_returns_empty(self):
        from observability.db.health_queries import get_heal_gate_summary
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await get_heal_gate_summary(session) == []
