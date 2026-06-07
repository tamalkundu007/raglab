"""
Unit tests for retrieval scorer (R6 Phase 4).

Covers:
- GET /obs/retrieval/queries: 200 with DB, 503 without
- GET /obs/retrieval/queries/{id}: 200 found, 404 not found, 503 no DB
- GET /obs/retrieval/distribution: 200, 503
- GET /obs/retrieval/healing: 200, 503, empty → default dict
- GET /obs/retrieval/scorer: 200, HTML, D3 present
- retrieval_scorer.html: stat-row, score distribution, strategy-pill, healed-badge
- DB queries: all return empty on error (no raise)
"""

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


class TestRetrievalQueriesEndpoint:
    def test_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/retrieval/queries").status_code == 503

    def test_with_db_200(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert client.get("/obs/retrieval/queries").status_code == 200

    def test_returns_list(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert isinstance(client.get("/obs/retrieval/queries").json(), list)


class TestQueryDetailEndpoint:
    def test_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/retrieval/queries/abc").status_code == 503

    def test_not_found_404(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.one_or_none.return_value = None
        session.execute.return_value = mr
        assert client.get("/obs/retrieval/queries/bad-id").status_code == 404

    def test_found_200(self, obs_db):
        client, session = obs_db
        mr = MagicMock()
        mr.mappings.return_value.one_or_none.return_value = {
            "trace_id": "abc123", "span_id": "def456",
            "operation_name": "retrieve", "start_time_ms": 1000,
            "duration_ms": 50, "status": "ok",
            "attributes": {"retriever_type": "dense"},
            "events": [],
        }
        session.execute.return_value = mr
        assert client.get("/obs/retrieval/queries/abc123").status_code == 200


class TestDistributionEndpoint:
    def test_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/retrieval/distribution").status_code == 503

    def test_with_db_200(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert client.get("/obs/retrieval/distribution").status_code == 200


class TestHealingStatsEndpoint:
    def test_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/retrieval/healing").status_code == 503

    def test_with_db_200(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.one_or_none.return_value = None
        session.execute.return_value = mr
        r = client.get("/obs/retrieval/healing")
        assert r.status_code == 200
        assert r.json()["total_queries"] == 0

    def test_returns_defaults_when_no_data(self, obs_db):
        client, session = obs_db
        mr = MagicMock(); mr.mappings.return_value.one_or_none.return_value = None
        session.execute.return_value = mr
        r = client.get("/obs/retrieval/healing")
        body = r.json()
        assert "healed_count" in body
        assert "avg_top_score" in body


class TestRetrievalScorerPage:
    def test_scorer_200(self, obs_no_db):
        assert obs_no_db.get("/obs/retrieval/scorer").status_code == 200

    def test_returns_html(self, obs_no_db):
        r = obs_no_db.get("/obs/retrieval/scorer")
        assert "text/html" in r.headers["content-type"]

    def test_has_d3(self, obs_no_db):
        assert b"d3" in obs_no_db.get("/obs/retrieval/scorer").content

    def test_has_raglab_branding(self, obs_no_db):
        assert b"RAGLab" in obs_no_db.get("/obs/retrieval/scorer").content

    def test_has_retrieval_scorer_title(self, obs_no_db):
        assert b"Retrieval Scorer" in obs_no_db.get("/obs/retrieval/scorer").content

    def test_has_stat_row(self, obs_no_db):
        assert b"stat-row" in obs_no_db.get("/obs/retrieval/scorer").content

    def test_has_score_distribution_chart(self, obs_no_db):
        assert b"renderDistChart" in obs_no_db.get("/obs/retrieval/scorer").content

    def test_has_strategy_pill(self, obs_no_db):
        assert b"strategy-pill" in obs_no_db.get("/obs/retrieval/scorer").content

    def test_has_healed_badge(self, obs_no_db):
        assert b"healed-badge" in obs_no_db.get("/obs/retrieval/scorer").content

    def test_has_healing_stats_api_call(self, obs_no_db):
        assert b"retrieval/healing" in obs_no_db.get("/obs/retrieval/scorer").content

    def test_r6_badge(self, obs_no_db):
        assert b"R6" in obs_no_db.get("/obs/retrieval/scorer").content


class TestRetrievalQueryFunctions:
    @pytest.mark.asyncio
    async def test_list_recent_queries_error_returns_empty(self):
        from observability.db.retrieval_queries import list_recent_queries
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await list_recent_queries(session) == []

    @pytest.mark.asyncio
    async def test_get_query_detail_error_returns_empty(self):
        from observability.db.retrieval_queries import get_query_detail
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("timeout"))
        assert await get_query_detail(session, "abc") == {}

    @pytest.mark.asyncio
    async def test_get_score_distribution_error_returns_empty(self):
        from observability.db.retrieval_queries import get_score_distribution
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("gone"))
        assert await get_score_distribution(session) == []

    @pytest.mark.asyncio
    async def test_get_healing_stats_error_returns_empty(self):
        from observability.db.retrieval_queries import get_healing_stats
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("gone"))
        assert await get_healing_stats(session) == {}
