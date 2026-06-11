"""
Authorization enforcement — role guards and identity context propagation (R7 Phase 3).

This module provides:
    1. RoleEnforcementMiddleware — injects IdentityContext from trusted headers
       into request.state on every downstream service (not the gateway).
       Downstream services call get_identity(request) to access it.

    2. require_role(*roles) — FastAPI Depends() decorator.
       Raises 403 if the authenticated user lacks the required role.

    3. require_admin / require_member / require_viewer — convenience shortcuts.

    4. IdentityContext.from_request(request) — extract from request.state or
       reconstruct from headers. Raises 401 if neither is present.

    5. propagate_identity(identity, headers) — injects identity headers into
       an outbound httpx request dict (used by gateway when proxying).

Design:
    The gateway validates JWTs via JWTValidatorMiddleware (auth/middleware/jwt_validator.py).
    Downstream services use RoleEnforcementMiddleware — they never validate JWTs.
    Both share the same IdentityContext type — the contract is the header set.

    Identity header set (injected by gateway, trusted by downstream):
        X-User-Id:       user_id
        X-Tenant-Id:     tenant_id
        X-User-Email:    email
        X-User-Name:     name
        X-User-Roles:    comma-separated roles (e.g. "admin,member")
        X-Auth-Provider: provider name

    Tenant isolation: every authenticated request carries tenant_id.
    The TenantScope layer (Phase 4) reads it from IdentityContext.
    The authorization layer (this module) enforces it at the API level.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import Depends, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from raglab_common.logging import get_logger
from auth.models import AuthError, IdentityContext, UserRole

log = get_logger(__name__)


# ── Downstream service middleware (not gateway) ────────────────────────────────

class RoleEnforcementMiddleware(BaseHTTPMiddleware):
    """
    Middleware for downstream services (not the gateway).

    Reconstructs IdentityContext from gateway-injected headers and stores
    it in request.state.identity. Does NOT validate JWTs.

    Public paths (health checks, docs) bypass identity injection.
    """

    PUBLIC_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc", "/")

    def __init__(self, app: Any, require_auth: bool = True) -> None:
        super().__init__(app)
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Public paths — skip
        if any(path == p or path.startswith(p + "/")
               for p in self.PUBLIC_PREFIXES if p != "/") or path == "/":
            return await call_next(request)

        try:
            identity = IdentityContext.from_headers(dict(request.headers))
            request.state.identity = identity
            import structlog
            structlog.contextvars.bind_contextvars(
                user_id=identity.user_id,
                tenant_id=identity.tenant_id,
            )
        except ValueError:
            if self.require_auth:
                return Response(
                    content='{"detail":"Identity headers missing. Route through gateway."}',
                    status_code=401,
                    media_type="application/json",
                )
            # Auth not required (dev/internal mode) — continue without identity
            request.state.identity = None

        return await call_next(request)


# ── FastAPI dependencies ───────────────────────────────────────────────────────

def get_identity(request: Request) -> IdentityContext:
    """
    FastAPI dependency: extract IdentityContext from request.state.

    Raises 401 if identity not present (request didn't come through gateway).
    """
    identity = getattr(request.state, "identity", None)
    if identity is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return identity


def require_role(*roles: UserRole) -> Any:
    """
    FastAPI Depends() factory: enforce one or more roles.

    Usage:
        @router.post("/ingest", dependencies=[Depends(require_role(UserRole.MEMBER))])
        async def ingest(...): ...

        @router.delete("/admin/purge",
                       dependencies=[Depends(require_role(UserRole.ADMIN))])
        async def purge(...): ...

    Admins implicitly satisfy any role requirement.
    """
    async def _check(identity: IdentityContext = Depends(get_identity)) -> IdentityContext:
        if identity.is_admin:
            return identity
        for role in roles:
            if role not in identity.roles:
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required role: {role.value}",
                )
        return identity

    return Depends(_check)


# Convenience shortcuts
require_admin  = require_role(UserRole.ADMIN)
require_member = require_role(UserRole.MEMBER)
require_viewer = require_role(UserRole.VIEWER)


# ── Outbound identity propagation (gateway → downstream) ──────────────────────

def propagate_identity(identity: IdentityContext) -> dict[str, str]:
    """
    Return headers dict to inject into outbound proxy/httpx calls.

    Gateway uses this when proxying requests to downstream services.
    Ensures identity context travels with every forwarded request.

    Usage (gateway proxy):
        headers = propagate_identity(request.state.identity)
        resp = await client.post(url, json=body, headers=headers)
    """
    return identity.to_headers()


def propagate_identity_from_request(request: Any) -> dict[str, str]:
    """
    Extract identity from request.state and return as propagation headers.
    Falls back to empty dict if no identity (dev/internal mode).
    """
    try:
        identity = getattr(getattr(request, "state", None), "identity", None)
    except Exception:
        identity = None
    if identity is None:
        return {}
    return identity.to_headers()


# ── Auth router additions — /auth/permissions ─────────────────────────────────

def get_permissions_summary(identity: IdentityContext) -> dict:
    """
    Return a permissions summary for the current user.
    Used by /auth/permissions endpoint and UI permission checks.
    """
    return {
        "user_id":   identity.user_id,
        "tenant_id": identity.tenant_id,
        "roles":     [r.value for r in identity.roles],
        "is_admin":  identity.is_admin,
        "can_write": identity.can_write,
        "can_read":  identity.can_read,
        "permissions": {
            "ingest":         identity.can_write,
            "query":          identity.can_read,
            "manage_tenants": identity.is_admin,
            "view_all_traces":identity.is_admin,
            "delete_docs":    identity.can_write,
            "manage_users":   identity.is_admin,
        },
    }
