"""
API Gateway routers — proxy all RAGLab service endpoints.

Route map (gateway path → downstream service):

  Ingestion:
    POST /api/v1/ingest              → ingestion:8001/ingest
    GET  /api/v1/ingest/{doc_id}     → ingestion:8001/ingest/{doc_id}

  Retrieval:
    POST /api/v1/retrieve            → retrieval:8004/retrieve

  Generation:
    POST /api/v1/generate            → llm:8005/generate

  Query (combined retrieve + generate in one call):
    POST /api/v1/query               → pipeline:8006/pipeline/run
                                       (orchestrated end-to-end via pipeline)

  Pipeline:
    POST /api/v1/pipeline/run        → pipeline:8006/pipeline/run
    GET  /api/v1/pipeline/status     → pipeline:8006/pipeline/status

  Collections:
    GET  /api/v1/collections/{name}  → indexing:8003/collections/{name}
    POST /api/v1/collections/{name}/ensure → indexing:8003/...

  LLM Providers:
    GET  /api/v1/providers           → llm:8005/providers

  System health (aggregate):
    GET  /api/v1/health/services     → returns HealthRegistry snapshot
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from raglab_common.logging import get_logger
from api_gateway.proxy import ProxyError, proxy_request

log = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["gateway"])


# ---------------------------------------------------------------------------
# Dependency: get downstream URL or 503
# ---------------------------------------------------------------------------


def _require_service(request: Request, service: str) -> str:
    """Return the base URL for a service or raise 503 if unavailable."""
    registry = getattr(request.app.state, "registry", None)
    settings = getattr(request.app.state, "settings", None)

    # URL map (mirrors docker-compose service names)
    url_map: dict[str, str] = {
        "ingestion":  getattr(settings, "ingestion_url", "http://ingestion:8001"),
        "embedding":  getattr(settings, "embedding_url", "http://embedding:8002"),
        "indexing":   getattr(settings, "indexing_url",  "http://indexing:8003"),
        "retrieval":  getattr(settings, "retrieval_url", "http://retrieval:8004"),
        "llm":        getattr(settings, "llm_url",       "http://llm:8005"),
        "pipeline":   getattr(settings, "pipeline_url",  "http://pipeline:8006"),
        "config":     getattr(settings, "config_url",    "http://config:8007"),
        "storage":    getattr(settings, "storage_url",   "http://storage:8008"),
    }

    base_url = url_map.get(service, "")
    if not base_url:
        raise HTTPException(status_code=503, detail=f"Service '{service}' not configured.")

    # Health-aware routing: refuse if registry knows the service is down
    if registry is not None and not registry.is_available(service):
        raise HTTPException(
            status_code=503,
            detail=f"Service '{service}' is currently unavailable.",
        )

    return base_url


async def _proxy(request: Request, service: str, path: str) -> Response:
    """Get service URL, check health, proxy request."""
    base = _require_service(request, service)
    target = f"{base}{path}"
    settings = getattr(request.app.state, "settings", None)
    timeout = getattr(settings, "proxy_timeout", 120.0)
    try:
        return await proxy_request(request, target, timeout=timeout)
    except ProxyError as exc:
        log.error("gateway.proxy_error", service=service, target=target, error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------------------------------------------------------------------
# Ingestion routes
# ---------------------------------------------------------------------------


@router.post("/ingest")
async def ingest(request: Request) -> Response:
    """Submit a document for async ingestion."""
    return await _proxy(request, "ingestion", "/ingest")


@router.get("/ingest/{doc_id}")
async def ingest_status(doc_id: str, request: Request) -> Response:
    """Check ingestion status for a document."""
    return await _proxy(request, "ingestion", f"/ingest/{doc_id}")


# ---------------------------------------------------------------------------
# Retrieval routes
# ---------------------------------------------------------------------------


@router.post("/retrieve")
async def retrieve(request: Request) -> Response:
    """Execute vector retrieval for a query."""
    return await _proxy(request, "retrieval", "/retrieve")


# ---------------------------------------------------------------------------
# LLM generation routes
# ---------------------------------------------------------------------------


@router.post("/generate")
async def generate(request: Request) -> Response:
    """RAG generation: chunks + query → answer."""
    return await _proxy(request, "llm", "/generate")


@router.get("/providers")
async def providers(request: Request) -> Response:
    """List available LLM providers and their load status."""
    return await _proxy(request, "llm", "/providers")


# ---------------------------------------------------------------------------
# Pipeline routes
# ---------------------------------------------------------------------------


@router.post("/pipeline/run")
async def pipeline_run(request: Request) -> Response:
    """Direct pipeline invocation (dev/test — bypasses RabbitMQ)."""
    return await _proxy(request, "pipeline", "/pipeline/run")


@router.get("/pipeline/status")
async def pipeline_status(request: Request) -> Response:
    """Consumer health and queue stats."""
    return await _proxy(request, "pipeline", "/pipeline/status")


# ---------------------------------------------------------------------------
# Collection management routes
# ---------------------------------------------------------------------------


@router.get("/collections/{name}")
async def collection_info(name: str, request: Request) -> Response:
    """Return vector count and status for a Qdrant collection."""
    return await _proxy(request, "indexing", f"/collections/{name}")


@router.post("/collections/{name}/ensure")
async def ensure_collection(name: str, request: Request) -> Response:
    """Idempotent collection creation."""
    return await _proxy(request, "indexing", f"/collections/{name}/ensure")


@router.delete("/collections/{name}")
async def delete_collection(name: str, request: Request) -> Response:
    """Delete a Qdrant collection (dev/test only)."""
    return await _proxy(request, "indexing", f"/collections/{name}")


# ---------------------------------------------------------------------------
# Aggregate health snapshot
# ---------------------------------------------------------------------------


@router.get("/health/services")
async def services_health(request: Request) -> JSONResponse:
    """
    Return a health snapshot of all downstream services.

    Pulled from the HealthRegistry cache — never blocks on live checks.
    """
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        return JSONResponse({"status": "unknown", "services": []})

    return JSONResponse({
        "gateway_status": registry.aggregate_status(),
        "services": registry.all_statuses(),
    })
