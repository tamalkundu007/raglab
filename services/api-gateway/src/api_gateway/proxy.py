"""
Async reverse proxy helper for the api-gateway.

`proxy_request()` forwards an incoming FastAPI Request to a downstream
service URL, streams the response body back, and preserves status code
and content-type headers.

Design:
  - Uses httpx.AsyncClient per request (connection pooling handled by httpx).
  - Strips hop-by-hop headers (host, connection, transfer-encoding) before
    forwarding — standard reverse-proxy behaviour.
  - Raises ProxyError on connection failure so the router can return 502/503.
"""

from __future__ import annotations

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from raglab_common.exceptions import RAGLabError
from raglab_common.logging import get_logger

log = get_logger(__name__)

# Headers that must not be forwarded (hop-by-hop)
_HOP_BY_HOP = frozenset({
    "host", "connection", "transfer-encoding", "te",
    "trailers", "upgrade", "proxy-authorization", "keep-alive",
})


class ProxyError(RAGLabError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="PROXY_ERROR")


async def proxy_request(
    request: Request,
    target_url: str,
    timeout: float = 120.0,
) -> Response:
    """
    Forward `request` to `target_url` and return the downstream response.

    Args:
        request:    Incoming FastAPI Request.
        target_url: Full URL of the downstream endpoint (already constructed).
        timeout:    HTTP client timeout in seconds.

    Returns:
        FastAPI Response with downstream status code, headers, and body.

    Raises:
        ProxyError: On connection error or timeout.
    """
    # Filter forwarding headers
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    # Read body (may be empty for GET)
    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            upstream_resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                params=dict(request.query_params),
            )
    except httpx.ConnectError as exc:
        raise ProxyError(f"Cannot connect to {target_url}: {exc}") from exc
    except httpx.TimeoutException as exc:
        raise ProxyError(f"Timeout proxying to {target_url}: {exc}") from exc
    except Exception as exc:
        raise ProxyError(f"Proxy error for {target_url}: {exc}") from exc

    # Filter response headers
    response_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=response_headers,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
    )
