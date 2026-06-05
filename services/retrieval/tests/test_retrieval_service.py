"""
Tests for the retrieval-service.

Covers:
- Settings defaults
- /retrieve endpoint: success (mocked qdrant + embedding-service), 
  missing qdrant (503), embedding failure (502), stub retriever (400)
- /health and / endpoints
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from raglab_common.models import ChunkModel


def make_hit(text: str = "Some result.", score: float = 0.9) -> dict:
    return {
        "payload": {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": "doc-001",
            "text": text,
            "chunk_index": 0,
            "token_count": len(text.split()),
        },
        "score": score,
    }


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestRetrievalSettings:
    def test_defaults(self):
        from retrieval.settings import RetrievalSettings
        s = RetrievalSettings()
        assert s.service_name == "retrieval"
        assert s.port == 8004
        assert s.default_top_k == 5


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def retrieval_client():
    from retrieval.main import app
    from retrieval.settings import RetrievalSettings

    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = [make_hit("Result about RAG.", 0.92)]

    app.state.qdrant_client = mock_qdrant
    app.state.settings = RetrievalSettings()
    return TestClient(app), mock_qdrant


class TestRetrievalEndpoints:
    def test_health_returns_200(self, retrieval_client):
        client, _ = retrieval_client
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "retrieval"

    def test_root_returns_service_info(self, retrieval_client):
        client, _ = retrieval_client
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "retrieval"

    def test_retrieve_success(self, retrieval_client):
        client, qdrant = retrieval_client
        with patch("retrieval.routers.retrieve.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vector": [0.1] * 1536}
            mock_resp.raise_for_status = MagicMock()
            mock_http.post = AsyncMock(return_value=mock_resp)

            r = client.post("/retrieve", json={
                "query": "What is RAG?",
                "collection": "raglab",
                "top_k": 3,
            })

        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "What is RAG?"
        assert body["result_count"] == 1
        assert len(body["results"]) == 1
        assert body["results"][0]["text"] == "Result about RAG."

    def test_retrieve_no_qdrant_returns_503(self):
        from retrieval.main import app
        app.state.qdrant_client = None
        client = TestClient(app)
        r = client.post("/retrieve", json={"query": "q", "collection": "raglab"})
        assert r.status_code == 503

    def test_retrieve_embedding_failure_returns_502(self, retrieval_client):
        client, _ = retrieval_client
        with patch("retrieval.routers.retrieve.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_http.post = AsyncMock(side_effect=Exception("embedding down"))

            r = client.post("/retrieve", json={
                "query": "What is RAG?",
                "collection": "raglab",
            })

        assert r.status_code == 502

    def test_retrieve_stub_retriever_returns_400(self, retrieval_client):
        client, _ = retrieval_client
        with patch("retrieval.routers.retrieve.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vector": [0.1] * 1536}
            mock_resp.raise_for_status = MagicMock()
            mock_http.post = AsyncMock(return_value=mock_resp)

            # bm25 is a stub — factory raises NotImplementedFeatureError
            r = client.post("/retrieve", json={
                "query": "q",
                "collection": "raglab",
                "retriever_type": "bm25",
            })

        assert r.status_code == 400

    def test_retrieve_returns_multiple_results(self, retrieval_client):
        client, qdrant = retrieval_client
        qdrant.search.return_value = [
            make_hit(f"Result {i}.", 0.9 - i * 0.1) for i in range(5)
        ]
        with patch("retrieval.routers.retrieve.httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"vector": [0.1] * 1536}
            mock_resp.raise_for_status = MagicMock()
            mock_http.post = AsyncMock(return_value=mock_resp)

            r = client.post("/retrieve", json={
                "query": "multi-result query",
                "collection": "raglab",
                "top_k": 5,
            })

        assert r.status_code == 200
        assert r.json()["result_count"] == 5
