"""
Unit tests for chunk inspector (R6 Phase 3).

Covers:
- GET /obs/chunks/docs: 200 with DB, 503 without
- GET /obs/chunks/{doc_id}: 200 with results, 404 no chunks, 503 no DB
- GET /obs/chunks/{doc_id}/summary: 200 with data, 404 empty, 503 no DB
- GET /obs/inspector: 200, HTML, has quality elements
- chunk_inspector.html: quality-bar, chunk-card classes, score display, filter-row
- DB query: list_recent_docs — empty list on error
- DB query: get_chunks_for_doc — empty list on error
- DB query: get_doc_quality_summary — empty dict on error
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
def obs_with_db():
    from observability.main import app
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_factory = MagicMock(return_value=mock_session)
    app.state.session_factory = mock_factory
    return TestClient(app, raise_server_exceptions=False), mock_session


class TestChunkDocsEndpoint:
    def test_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/chunks/docs").status_code == 503

    def test_with_db_200(self, obs_with_db):
        client, session = obs_with_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert client.get("/obs/chunks/docs").status_code == 200

    def test_returns_list(self, obs_with_db):
        client, session = obs_with_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert isinstance(client.get("/obs/chunks/docs").json(), list)


class TestChunksByDocEndpoint:
    def test_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/chunks/doc-001").status_code == 503

    def test_not_found_404(self, obs_with_db):
        client, session = obs_with_db
        mr = MagicMock(); mr.mappings.return_value.all.return_value = []
        session.execute.return_value = mr
        assert client.get("/obs/chunks/nonexistent").status_code == 404

    def test_found_200(self, obs_with_db):
        client, session = obs_with_db
        mr = MagicMock()
        mr.mappings.return_value.all.return_value = [{
            "chunk_id": "c1", "doc_id": "doc-001", "collection": "raglab",
            "chunk_index": 0, "text": "Test content here.", "token_count": 3,
            "chunker_type": "text", "metadata": {}, "created_at": None,
            "quality_score": 0.85, "quality_passed": True,
            "quality_action": "accepted", "quality_reason": None,
        }]
        session.execute.return_value = mr
        assert client.get("/obs/chunks/doc-001").status_code == 200

    def test_returns_quality_fields(self, obs_with_db):
        client, session = obs_with_db
        mr = MagicMock()
        mr.mappings.return_value.all.return_value = [{
            "chunk_id": "c1", "doc_id": "doc-001", "collection": "raglab",
            "chunk_index": 0, "text": "Text.", "token_count": 1,
            "chunker_type": "text", "metadata": {}, "created_at": None,
            "quality_score": 0.72, "quality_passed": True,
            "quality_action": "accepted", "quality_reason": None,
        }]
        session.execute.return_value = mr
        r = client.get("/obs/chunks/doc-001")
        chunk = r.json()[0]
        assert "quality_score" in chunk
        assert "quality_action" in chunk


class TestQualitySummaryEndpoint:
    def test_no_db_503(self, obs_no_db):
        assert obs_no_db.get("/obs/chunks/doc-001/summary").status_code == 503

    def test_not_found_404(self, obs_with_db):
        client, session = obs_with_db
        mr = MagicMock(); mr.mappings.return_value.one_or_none.return_value = None
        session.execute.return_value = mr
        assert client.get("/obs/chunks/doc-001/summary").status_code == 404

    def test_found_200(self, obs_with_db):
        client, session = obs_with_db
        mr = MagicMock()
        mr.mappings.return_value.one_or_none.return_value = {
            "total": 10, "accepted": 8, "flagged": 1, "excluded": 1,
            "avg_quality_score": 0.78, "min_quality_score": 0.3,
            "chunker_type": "text",
        }
        session.execute.return_value = mr
        r = client.get("/obs/chunks/doc-001/summary")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 10
        assert body["accepted"] == 8


class TestChunkInspectorPage:
    def test_inspector_200(self, obs_no_db):
        assert obs_no_db.get("/obs/inspector").status_code == 200

    def test_returns_html(self, obs_no_db):
        r = obs_no_db.get("/obs/inspector")
        assert "text/html" in r.headers["content-type"]

    def test_has_raglab_branding(self, obs_no_db):
        assert b"RAGLab" in obs_no_db.get("/obs/inspector").content

    def test_has_chunk_inspector_title(self, obs_no_db):
        assert b"Chunk Inspector" in obs_no_db.get("/obs/inspector").content

    def test_has_quality_bar_element(self, obs_no_db):
        assert b"quality-bar" in obs_no_db.get("/obs/inspector").content

    def test_has_chunk_card_class(self, obs_no_db):
        assert b"chunk-card" in obs_no_db.get("/obs/inspector").content

    def test_has_load_chunks_function(self, obs_no_db):
        assert b"loadChunks" in obs_no_db.get("/obs/inspector").content

    def test_has_filter_row(self, obs_no_db):
        assert b"filter-row" in obs_no_db.get("/obs/inspector").content

    def test_has_score_display(self, obs_no_db):
        assert b"chunk-score" in obs_no_db.get("/obs/inspector").content

    def test_r6_badge(self, obs_no_db):
        assert b"R6" in obs_no_db.get("/obs/inspector").content


class TestChunkQueryFunctions:
    @pytest.mark.asyncio
    async def test_list_recent_docs_error_returns_empty(self):
        from observability.db.chunk_queries import list_recent_docs
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db error"))
        assert await list_recent_docs(session) == []

    @pytest.mark.asyncio
    async def test_get_chunks_for_doc_error_returns_empty(self):
        from observability.db.chunk_queries import get_chunks_for_doc
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("timeout"))
        assert await get_chunks_for_doc(session, "doc-001") == []

    @pytest.mark.asyncio
    async def test_get_doc_quality_summary_error_returns_empty(self):
        from observability.db.chunk_queries import get_doc_quality_summary
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("gone"))
        result = await get_doc_quality_summary(session, "doc-001")
        assert result == {}
