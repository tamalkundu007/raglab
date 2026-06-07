"""Unit tests for cost dashboard (R6 Phase 5)."""
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


class TestCostEndpoints:
    def test_summary_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/cost/summary").status_code == 503

    def test_by_provider_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/cost/by-provider").status_code == 503

    def test_trend_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/cost/trend").status_code == 503

    def test_cache_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/cost/cache").status_code == 503

    def test_summary_with_db_200(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.one_or_none.return_value = None
        session.execute.return_value = mr
        assert client.get("/obs/cost/summary").status_code == 200

    def test_trend_with_db_returns_list(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert isinstance(client.get("/obs/cost/trend").json(), list)

    def test_cache_empty_returns_defaults(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.one_or_none.return_value = None
        session.execute.return_value = mr
        r = client.get("/obs/cost/cache")
        assert r.status_code == 200
        body = r.json()
        assert "total_hits" in body
        assert "avg_hit_rate_pct" in body


class TestCostDashboardPage:
    def test_dashboard_200(self, obs_no_db):
        assert obs_no_db.get("/obs/cost/dashboard").status_code == 200

    def test_returns_html(self, obs_no_db):
        r = obs_no_db.get("/obs/cost/dashboard")
        assert "text/html" in r.headers["content-type"]

    def test_has_d3(self, obs_no_db):
        assert b"d3" in obs_no_db.get("/obs/cost/dashboard").content

    def test_has_raglab_branding(self, obs_no_db):
        assert b"RAGLab" in obs_no_db.get("/obs/cost/dashboard").content

    def test_has_cost_dashboard_title(self, obs_no_db):
        assert b"Cost Dashboard" in obs_no_db.get("/obs/cost/dashboard").content

    def test_has_token_stats(self, obs_no_db):
        assert b"token-stats" in obs_no_db.get("/obs/cost/dashboard").content

    def test_has_trend_chart(self, obs_no_db):
        assert b"renderTrend" in obs_no_db.get("/obs/cost/dashboard").content

    def test_has_provider_table(self, obs_no_db):
        assert b"provider-pill" in obs_no_db.get("/obs/cost/dashboard").content

    def test_has_cache_hit_rate(self, obs_no_db):
        assert b"cache-bar" in obs_no_db.get("/obs/cost/dashboard").content

    def test_has_roi_message(self, obs_no_db):
        assert b"re-ingestion" in obs_no_db.get("/obs/cost/dashboard").content

    def test_r6_badge(self, obs_no_db):
        assert b"R6" in obs_no_db.get("/obs/cost/dashboard").content


class TestCostQueryFunctions:
    @pytest.mark.asyncio
    async def test_token_summary_error_returns_empty(self):
        from observability.db.cost_queries import get_token_summary
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await get_token_summary(session) == {}

    @pytest.mark.asyncio
    async def test_by_provider_error_returns_empty(self):
        from observability.db.cost_queries import get_tokens_by_provider
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await get_tokens_by_provider(session) == []

    @pytest.mark.asyncio
    async def test_daily_trend_error_returns_empty(self):
        from observability.db.cost_queries import get_daily_token_trend
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await get_daily_token_trend(session) == []

    @pytest.mark.asyncio
    async def test_cache_stats_error_returns_empty(self):
        from observability.db.cost_queries import get_cache_stats_summary
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await get_cache_stats_summary(session) == {}
