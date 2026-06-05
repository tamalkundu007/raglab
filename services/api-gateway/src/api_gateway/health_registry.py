"""
Health registry — polls all downstream services and caches results.

The registry runs a background task that checks each service's /health
endpoint every `ttl` seconds. The gateway uses cached results for:
  - Its own /health response (aggregate view)
  - Routing decisions: refuse to proxy to a service marked unavailable

Design:
  - Non-blocking: /health of the gateway always responds even if pollers lag.
  - Best-effort: a service that fails its health check is marked "unavailable"
    but not removed from the registry — it will recover on the next poll.
  - Per-service status: ok | degraded | unavailable
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from raglab_common.logging import get_logger

log = get_logger(__name__)


# All internal services the gateway knows about
DOWNSTREAM_SERVICES: dict[str, str] = {
    "ingestion":     "http://ingestion:8001",
    "embedding":     "http://embedding:8002",
    "indexing":      "http://indexing:8003",
    "retrieval":     "http://retrieval:8004",
    "llm":           "http://llm:8005",
    "pipeline":      "http://pipeline:8006",
    "config":        "http://config:8007",
    "storage":       "http://storage:8008",
    "ui":            "http://ui:8009",
    "graph":         "http://graph:8010",
    "observability": "http://observability:8011",
    "auth":          "http://auth:8012",
}


class ServiceStatus:
    """Snapshot of a single downstream service's health."""

    def __init__(self, name: str, base_url: str) -> None:
        self.name = name
        self.base_url = base_url
        self.status: str = "unknown"          # ok | degraded | unavailable | unknown
        self.last_checked: float = 0.0        # epoch seconds
        self.response_ms: float = 0.0
        self.detail: dict[str, Any] = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.base_url,
            "status": self.status,
            "response_ms": round(self.response_ms, 1),
            "last_checked": self.last_checked,
            "detail": self.detail,
        }


class HealthRegistry:
    """
    Polls all downstream services and caches their health status.

    Usage (in lifespan):
        registry = HealthRegistry(timeout=3.0, ttl=10.0)
        registry.configure_urls({"ingestion": "http://ingestion:8001", ...})
        task = asyncio.create_task(registry.run())
        yield
        task.cancel()
    """

    def __init__(self, timeout: float = 3.0, ttl: float = 10.0) -> None:
        self._timeout = timeout
        self._ttl = ttl
        self._services: dict[str, ServiceStatus] = {}

    def configure_urls(self, urls: dict[str, str]) -> None:
        """Set downstream service URLs (called once at startup)."""
        for name, url in urls.items():
            self._services[name] = ServiceStatus(name, url)

    async def run(self) -> None:
        """Background polling loop — runs until cancelled."""
        log.info("health_registry.started", services=list(self._services.keys()))
        while True:
            await self._poll_all()
            await asyncio.sleep(self._ttl)

    async def _poll_all(self) -> None:
        """Poll all services concurrently."""
        tasks = [self._poll_one(svc) for svc in self._services.values()]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_one(self, svc: ServiceStatus) -> None:
        """Poll a single service /health endpoint."""
        start = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{svc.base_url}/health")
                elapsed_ms = (time.monotonic() - start) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    svc.status = data.get("status", "ok")
                    svc.detail = data.get("dependencies", {})
                else:
                    svc.status = "unavailable"
                    svc.detail = {"http_status": resp.status_code}

                svc.response_ms = elapsed_ms
        except Exception as exc:
            svc.status = "unavailable"
            svc.detail = {"error": str(exc)[:128]}
            svc.response_ms = (time.monotonic() - start) * 1000

        svc.last_checked = time.time()
        log.debug(
            "health_registry.polled",
            service=svc.name,
            status=svc.status,
            response_ms=round(svc.response_ms, 1),
        )

    def get_status(self, service: str) -> ServiceStatus | None:
        return self._services.get(service)

    def all_statuses(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self._services.values()]

    def is_available(self, service: str) -> bool:
        """Return True if service is ok or degraded (not unavailable/unknown)."""
        svc = self._services.get(service)
        if svc is None:
            return False
        return svc.status in ("ok", "degraded")

    def aggregate_status(self) -> str:
        """
        Roll up all service statuses into a single gateway status.

        ok        → all services ok
        degraded  → at least one degraded or unavailable (non-stub)
        """
        core_services = {"ingestion", "embedding", "indexing", "retrieval", "llm", "pipeline"}
        statuses = {
            name: svc.status
            for name, svc in self._services.items()
            if name in core_services
        }
        if all(s == "ok" for s in statuses.values()):
            return "ok"
        return "degraded"
