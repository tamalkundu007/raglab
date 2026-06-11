"""
JWT validator middleware — used by the API Gateway (Phase 1).

The gateway is the single JWT validation point.
Downstream services NEVER validate JWTs — they trust gateway-injected headers.

Middleware flow per request:
    1. Extract Bearer token from Authorization header.
    2. Determine provider (from token header `azp` / issuer claim).
    3. Validate token via the appropriate OIDCProvider.
    4. Build IdentityContext from validated claims.
    5. Inject identity headers (X-User-Id, X-Tenant-Id, X-User-Roles, ...).
    6. If validation fails → 401 immediately, request never reaches downstream.

Public paths (bypass auth):
    /health, /docs, /openapi.json, /redoc, /auth/* (login/callback routes)

Design:
    JWTValidatorMiddleware is added to the gateway app.
    It wraps the request before any route handler runs.
    No route handler ever sees an unauthenticated request (except public paths).
"""

from __future__ import annotations

import re
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from raglab_common.logging import get_logger
from auth.models import (
    AuthError,
    IdentityContext,
    OIDCProviderConfig,
    TokenMissingError,
    UserRole,
)
from auth.providers.base import OIDCProviderFactory

log = get_logger(__name__)

# Paths that bypass authentication entirely
_PUBLIC_PATH_PATTERNS = [
    re.compile(r"^/health$"),
    re.compile(r"^/docs.*"),
    re.compile(r"^/openapi\.json$"),
    re.compile(r"^/redoc.*"),
    re.compile(r"^/auth/.*"),      # login, callback, logout routes
    re.compile(r"^/$"),            # root info endpoint
]


def _is_public_path(path: str) -> bool:
    return any(p.match(path) for p in _PUBLIC_PATH_PATTERNS)


class JWTValidatorMiddleware(BaseHTTPMiddleware):
    """
    FastAPI/Starlette middleware that validates JWTs at the gateway.

    Constructor args:
        providers:        dict[str, OIDCProviderBase] — provider_name → provider instance
        default_provider: which provider to try when issuer can't be determined
        bypass_auth:      if True, skip validation (dev/test mode only)
    """

    def __init__(
        self,
        app: Any,
        providers: dict,
        default_provider: str = "entra_id",
        bypass_auth: bool = False,
    ) -> None:
        super().__init__(app)
        self.providers = providers
        self.default_provider = default_provider
        self.bypass_auth = bypass_auth

    async def dispatch(self, request: Request, call_next) -> Response:
        # Public paths bypass auth
        if _is_public_path(request.url.path):
            return await call_next(request)

        # Dev/test bypass
        if self.bypass_auth:
            request.state.identity = _dev_identity()
            return await call_next(request)

        # Extract token
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return _auth_error_response(TokenMissingError())

        token = auth_header[len("Bearer "):]

        # Validate
        try:
            provider = self._select_provider(token)
            identity = provider.validate_token(token)
        except AuthError as exc:
            log.warning("gateway.auth_failed",
                        path=request.url.path,
                        error=str(exc),
                        status=exc.status_code)
            return _auth_error_response(exc)

        # Inject identity context into request state + headers for downstream
        request.state.identity = identity

        # Call next, then inject headers into response
        response = await call_next(request)
        for key, value in identity.to_headers().items():
            response.headers[key] = value

        log.info("gateway.authenticated",
                 user_id=identity.user_id,
                 tenant_id=identity.tenant_id,
                 provider=identity.provider,
                 path=request.url.path)

        return response

    def _select_provider(self, token: str) -> Any:
        """
        Select the correct OIDC provider for a token.

        Strategy:
            1. Peek at the JWT issuer claim (no signature check yet).
            2. Match issuer to known providers.
            3. Fall back to default_provider.
        """
        try:
            # Decode without verification just to read the issuer
            import base64, json as _json
            header_b64 = token.split(".")[1]
            # Add padding
            padded = header_b64 + "=" * (4 - len(header_b64) % 4)
            payload_bytes = base64.urlsafe_b64decode(padded)
            payload = _json.loads(payload_bytes)
            iss = payload.get("iss", "")

            if "microsoftonline.com" in iss:
                return self.providers.get("entra_id", self._default_provider())
            if "accounts.google.com" in iss:
                return self.providers.get("google", self._default_provider())
            if "cognito" in iss or "amazonaws.com" in iss:
                return self.providers.get("cognito", self._default_provider())
        except Exception:
            pass  # fall through to default

        return self._default_provider()

    def _default_provider(self) -> Any:
        p = self.providers.get(self.default_provider)
        if p is None:
            raise AuthError(f"No provider configured for '{self.default_provider}'")
        return p


def _auth_error_response(exc: AuthError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc), "type": type(exc).__name__},
    )


def _dev_identity() -> IdentityContext:
    """
    Dev/test identity — used when bypass_auth=True.
    Always tenant 'dev', user 'dev-user', role admin.
    Never used in production.
    """
    return IdentityContext(
        user_id="dev-user-001",
        tenant_id="dev",
        email="dev@raglab.local",
        name="Dev User",
        roles=[UserRole.ADMIN],
        provider="dev",
    )


# ── Role guard dependency ──────────────────────────────────────────────────────

def require_role(*roles: UserRole):
    """
    FastAPI dependency that enforces role requirements.

    Usage:
        @app.get("/admin/things", dependencies=[Depends(require_role(UserRole.ADMIN))])
        async def admin_endpoint():
            ...
    """
    from fastapi import Depends, HTTPException

    async def _guard(request: Request) -> IdentityContext:
        identity: IdentityContext | None = getattr(request.state, "identity", None)
        if identity is None:
            raise HTTPException(status_code=401, detail="Not authenticated.")
        for required_role in roles:
            if required_role not in identity.roles and not identity.is_admin:
                from auth.models import InsufficientRoleError
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {required_role.value}",
                )
        return identity

    return Depends(_guard)


def get_identity(request: Request) -> IdentityContext:
    """
    FastAPI dependency to extract IdentityContext from request state.
    Raises 401 if not authenticated.
    """
    from fastapi import HTTPException
    identity = getattr(request.state, "identity", None)
    if identity is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return identity
