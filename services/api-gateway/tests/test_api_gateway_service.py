"""
Tests for the api-gateway.

Covers:
- GatewaySettings defaults
- HealthRegistry: poll success/failure, is_available, aggregate_status, all_statuses
- proxy_request: success, ProxyError on connect failure, hop-by-hop header stripping
- _require_service: health-aware 503 when service unavailable
- All gateway routes: success proxy, 503 when service down, 502 on upstream error
- /health and /api/v1/health/services endpoints
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from api_gateway.health_registry import HealthRegistry, ServiceStatus
from api_gateway.settings import GatewaySettings


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestGatewaySettings:
    def test_defaults(self):
        s = GatewaySettings()
        assert s.service_name == "api-gateway"
        assert s.port == 8000
        assert s.health_check_timeout == 3.0
        assert s.proxy_timeout == 120.0
        assert s.health_cache_ttl == 10.0


# ---------------------------------------------------------------------------
# HealthRegistry
# ---------------------------------------------------------------------------


class TestHealthRegistry:
    def _registry(self) -> HealthRegistry:
        r = HealthRegistry(timeout=1.0, ttl=5.0)
        r.configure_urls({"svc-a": "http://a:8001", "svc-b": "http://b:8002"})
        return r

    def test_configure_urls_creates_statuses(self):
        r = self._registry()
        assert "svc-a" in r._services
        assert "svc-b" in r._services
        assert r._services["svc-a"].status == "unknown"

    def test_is_available_unknown_returns_false(self):
        r = self._registry()
        assert r.is_available("svc-a") is False

    def test_is_available_ok_returns_true(self):
        r = self._registry()
        r._services["svc-a"].status = "ok"
        assert r.is_available("svc-a") is True

    def test_is_available_degraded_returns_true(self):
        r = self._registry()
        r._services["svc-a"].status = "degraded"
        assert r.is_available("svc-a") is True

    def test_is_available_unavailable_returns_false(self):
        r = self._registry()
        r._services["svc-a"].status = "unavailable"
        assert r.is_available("svc-a") is False

    def test_is_available_unknown_service_returns_false(self):
        r = self._registry()
        assert r.is_available("nonexistent") is False

    def test_get_status_returns_service_status(self):
        r = self._registry()
        svc = r.get_status("svc-a")
        assert isinstance(svc, ServiceStatus)
        assert svc.name == "svc-a"

    def test_get_status_unknown_returns_none(self):
        r = self._registry()
        assert r.get_status("ghost") is None

    def test_all_statuses_returns_list(self):
        r = self._registry()
        statuses = r.all_statuses()
        assert isinstance(statuses, list)
        assert len(statuses) == 2
        names = {s["name"] for s in statuses}
        assert "svc-a" in names

    def test_aggregate_status_all_ok(self):
        r = HealthRegistry()
        r.configure_urls({
            "ingestion": "http://i:1", "embedding": "http://e:2",
            "indexing": "http://x:3", "retrieval": "http://r:4",
            "llm": "http://l:5", "pipeline": "http://p:6",
        })
        for svc in r._services.values():
            svc.status = "ok"
        assert r.aggregate_status() == "ok"

    def test_aggregate_status_one_degraded(self):
        r = HealthRegistry()
        r.configure_urls({
            "ingestion": "http://i:1", "embedding": "http://e:2",
            "indexing": "http://x:3", "retrieval": "http://r:4",
            "llm": "http://l:5", "pipeline": "http://p:6",
        })
        for svc in r._services.values():
            svc.status = "ok"
        r._services["llm"].status = "unavailable"
        assert r.aggregate_status() == "degraded"

    @pytest.mark.asyncio
    async def test_poll_one_success(self):
        r = self._registry()
        svc = r._services["svc-a"]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok", "dependencies": {"db": "ok"}}

        with patch("api_gateway.health_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            await r._poll_one(svc)

        assert svc.status == "ok"
        assert svc.detail == {"db": "ok"}
        assert svc.last_checked > 0

    @pytest.mark.asyncio
    async def test_poll_one_connection_failure(self):
        r = self._registry()
        svc = r._services["svc-a"]

        with patch("api_gateway.health_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
            await r._poll_one(svc)

        assert svc.status == "unavailable"
        assert "error" in svc.detail

    @pytest.mark.asyncio
    async def test_poll_one_non_200_marks_unavailable(self):
        r = self._registry()
        svc = r._services["svc-a"]
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_resp.json.return_value = {}

        with patch("api_gateway.health_registry.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            await r._poll_one(svc)

        assert svc.status == "unavailable"


# ---------------------------------------------------------------------------
# proxy_request
# ---------------------------------------------------------------------------


class TestProxyRequest:
    @pytest.mark.asyncio
    async def test_forwards_request_and_returns_response(self):
        from api_gateway.proxy import proxy_request
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b'{"result": "ok"}'
        mock_resp.headers = {"content-type": "application/json"}

        with patch("api_gateway.proxy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(return_value=mock_resp)

            scope = {
                "type": "http", "method": "POST",
                "path": "/test", "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
            }
            request = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))
            response = await proxy_request(request, "http://downstream:8001/test")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_connect_error_raises_proxy_error(self):
        from api_gateway.proxy import proxy_request, ProxyError
        import httpx

        with patch("api_gateway.proxy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))

            scope = {
                "type": "http", "method": "GET",
                "path": "/test", "query_string": b"",
                "headers": [],
            }
            request = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))
            with pytest.raises(ProxyError, match="Cannot connect"):
                await proxy_request(request, "http://bad:9999/test")

    @pytest.mark.asyncio
    async def test_strips_hop_by_hop_headers(self):
        from api_gateway.proxy import proxy_request

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"{}"
        mock_resp.headers = {"content-type": "application/json"}
        captured_headers = {}

        with patch("api_gateway.proxy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            async def capture_request(**kwargs):
                captured_headers.update(kwargs.get("headers", {}))
                return mock_resp

            mock_client.request = capture_request

            scope = {
                "type": "http", "method": "GET",
                "path": "/test", "query_string": b"",
                "headers": [
                    (b"host", b"gateway:8000"),
                    (b"connection", b"keep-alive"),
                    (b"x-request-id", b"abc123"),
                ],
            }
            request = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))
            await proxy_request(request, "http://downstream:8001/test")

        assert "host" not in captured_headers
        assert "connection" not in captured_headers
        assert "x-request-id" in captured_headers


# ---------------------------------------------------------------------------
# Gateway HTTP endpoints
# ---------------------------------------------------------------------------


def _make_client(registry_available: bool = True) -> tuple[TestClient, MagicMock]:
    from api_gateway.main import app

    mock_registry = MagicMock()
    mock_registry.aggregate_status.return_value = "ok" if registry_available else "degraded"
    mock_registry.all_statuses.return_value = [
        {"name": "ingestion", "url": "http://ingestion:8001",
         "status": "ok", "response_ms": 5.0, "last_checked": 1234567890.0, "detail": {}},
        {"name": "llm", "url": "http://llm:8005",
         "status": "ok", "response_ms": 8.0, "last_checked": 1234567890.0, "detail": {}},
    ]
    mock_registry.is_available.return_value = registry_available
    mock_registry.get_status.return_value = None

    app.state.registry = mock_registry
    app.state.settings = GatewaySettings()
    return TestClient(app), mock_registry


class TestGatewayEndpoints:
    def test_health_ok(self):
        client, _ = _make_client()
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "api-gateway"

    def test_root_returns_gateway_info(self):
        client, _ = _make_client()
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "api-gateway"
        assert body["api_base"] == "/api/v1"

    def test_services_health_endpoint(self):
        client, _ = _make_client()
        r = client.get("/api/v1/health/services")
        assert r.status_code == 200
        body = r.json()
        assert "gateway_status" in body
        assert "services" in body
        assert len(body["services"]) == 2

    def test_ingest_route_proxied(self):
        client, _ = _make_client()
        with patch("api_gateway.routers.gateway.proxy_request") as mock_proxy:
            from fastapi.responses import JSONResponse
            mock_proxy.return_value = JSONResponse({"doc_id": "d-001", "status": "pending"})
            r = client.post("/api/v1/ingest", json={"filename": "f.txt", "storage_path": "/tmp/f.txt"})
        assert r.status_code == 200
        mock_proxy.assert_called_once()
        # Verify target URL contains ingestion service
        target_url = mock_proxy.call_args[0][1]
        assert "ingestion" in target_url or "8001" in target_url

    def test_retrieve_route_proxied(self):
        client, _ = _make_client()
        with patch("api_gateway.routers.gateway.proxy_request") as mock_proxy:
            from fastapi.responses import JSONResponse
            mock_proxy.return_value = JSONResponse({"results": [], "result_count": 0})
            r = client.post("/api/v1/retrieve", json={"query": "test"})
        assert r.status_code == 200
        mock_proxy.assert_called_once()

    def test_generate_route_proxied(self):
        client, _ = _make_client()
        with patch("api_gateway.routers.gateway.proxy_request") as mock_proxy:
            from fastapi.responses import JSONResponse
            mock_proxy.return_value = JSONResponse({"answer": "42", "model": "gpt-4o"})
            r = client.post("/api/v1/generate", json={"query": "q", "chunks": []})
        assert r.status_code == 200

    def test_service_unavailable_returns_503(self):
        client, _ = _make_client(registry_available=False)
        r = client.post("/api/v1/ingest", json={"filename": "f.txt", "storage_path": "/tmp/f.txt"})
        assert r.status_code == 503

    def test_proxy_error_returns_502(self):
        client, _ = _make_client()
        from api_gateway.proxy import ProxyError
        with patch("api_gateway.routers.gateway.proxy_request", side_effect=ProxyError("upstream down")):
            r = client.post("/api/v1/ingest", json={"filename": "f.txt", "storage_path": "/tmp/f.txt"})
        assert r.status_code == 502

    def test_collection_info_route(self):
        client, _ = _make_client()
        with patch("api_gateway.routers.gateway.proxy_request") as mock_proxy:
            from fastapi.responses import JSONResponse
            mock_proxy.return_value = JSONResponse({"name": "raglab", "vectors_count": 42})
            r = client.get("/api/v1/collections/raglab")
        assert r.status_code == 200
        target_url = mock_proxy.call_args[0][1]
        assert "raglab" in target_url

    def test_ensure_collection_route(self):
        client, _ = _make_client()
        with patch("api_gateway.routers.gateway.proxy_request") as mock_proxy:
            from fastapi.responses import JSONResponse
            mock_proxy.return_value = JSONResponse({"collection": "raglab", "created": True})
            r = client.post("/api/v1/collections/raglab/ensure")
        assert r.status_code == 200

    def test_providers_route(self):
        client, _ = _make_client()
        with patch("api_gateway.routers.gateway.proxy_request") as mock_proxy:
            from fastapi.responses import JSONResponse
            mock_proxy.return_value = JSONResponse([{"provider": "azure_openai", "active": True}])
            r = client.get("/api/v1/providers")
        assert r.status_code == 200

    def test_pipeline_run_route(self):
        client, _ = _make_client()
        with patch("api_gateway.routers.gateway.proxy_request") as mock_proxy:
            from fastapi.responses import JSONResponse
            mock_proxy.return_value = JSONResponse({"doc_id": "d-001", "status": "completed"})
            r = client.post("/api/v1/pipeline/run", json={"filename": "f.txt", "storage_path": "/tmp/f.txt"})
        assert r.status_code == 200

    def test_ingest_status_route(self):
        client, _ = _make_client()
        with patch("api_gateway.routers.gateway.proxy_request") as mock_proxy:
            from fastapi.responses import JSONResponse
            mock_proxy.return_value = JSONResponse({"doc_id": "d-001", "status": "completed"})
            r = client.get("/api/v1/ingest/d-001")
        assert r.status_code == 200
        target_url = mock_proxy.call_args[0][1]
        assert "d-001" in target_url
