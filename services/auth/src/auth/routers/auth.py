"""
Auth router — login, callback, logout, /me endpoints.

GET  /auth/login/{provider}          — redirect to provider authorization URL
GET  /auth/callback/{provider}       — handle OIDC callback, issue session JWT
GET  /auth/me                        — return current user identity
POST /auth/logout                    — invalidate session
GET  /auth/providers                 — list available providers
"""

from __future__ import annotations

import secrets
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from raglab_common.logging import get_logger
from auth.models import IdentityContext, UserRole
from auth.middleware.jwt_validator import get_identity

log = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/providers")
async def list_providers(request: Request) -> dict:
    """List available OIDC providers."""
    providers = getattr(request.app.state, "providers", {})
    return {
        "providers": list(providers.keys()),
        "count": len(providers),
    }


@router.get("/login/{provider}")
async def login(provider: str, request: Request) -> RedirectResponse:
    """
    Initiate OIDC login flow.
    Redirects to the provider's authorization endpoint.
    """
    providers = getattr(request.app.state, "providers", {})
    if provider not in providers:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    state = secrets.token_urlsafe(16)
    nonce = secrets.token_urlsafe(16)

    # Store state in session cookie (stateless — just echo it back in callback)
    auth_url = providers[provider].get_authorization_url(state, nonce)

    log.info("auth.login_initiated", provider=provider)
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/callback/{provider}")
async def callback(
    provider: str,
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    error_description: str = "",
) -> JSONResponse:
    """
    Handle OIDC authorization code callback.
    Exchanges code for tokens, validates, returns identity.
    """
    if error:
        log.warning("auth.callback_error", provider=provider,
                    error=error, description=error_description)
        raise HTTPException(
            status_code=400,
            detail=f"Authorization failed: {error_description or error}",
        )

    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")

    providers = getattr(request.app.state, "providers", {})
    if provider not in providers:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    try:
        token_response = providers[provider].exchange_code(code, state)
        id_token = token_response.get("id_token", "")
        if not id_token:
            raise HTTPException(status_code=400, detail="No id_token in response.")

        identity = providers[provider].validate_token(id_token)

        log.info("auth.login_success",
                 provider=provider,
                 user_id=identity.user_id,
                 tenant_id=identity.tenant_id)

        # Return identity context + the original tokens
        # In production: issue a signed session JWT here instead
        return JSONResponse(content={
            "user_id":   identity.user_id,
            "tenant_id": identity.tenant_id,
            "email":     identity.email,
            "name":      identity.name,
            "roles":     [r.value for r in identity.roles],
            "provider":  identity.provider,
            "access_token": token_response.get("access_token", ""),
            "id_token":     id_token,
            "expires_in":   token_response.get("expires_in", 3600),
        })
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("auth.callback_failed", provider=provider, error=str(exc))
        raise HTTPException(status_code=401, detail=f"Authentication failed: {exc}")


@router.get("/me")
async def me(request: Request) -> dict:
    """
    Return current user identity.
    Requires authenticated request (gateway injects X-User-Id / X-Tenant-Id headers).
    """
    # Reconstruct from gateway-injected headers
    try:
        identity = IdentityContext.from_headers(dict(request.headers))
    except ValueError:
        # Also check request.state (set by JWTValidatorMiddleware on gateway)
        identity = getattr(request.state, "identity", None)
        if identity is None:
            raise HTTPException(status_code=401, detail="Not authenticated.")

    return {
        "user_id":   identity.user_id,
        "tenant_id": identity.tenant_id,
        "email":     identity.email,
        "name":      identity.name,
        "roles":     [r.value for r in identity.roles],
        "provider":  identity.provider,
        "is_admin":  identity.is_admin,
        "can_write": identity.can_write,
    }




@router.get("/permissions")
async def permissions(request: Request) -> dict:
    """
    Return detailed permission summary for the current user.
    Useful for UI permission checks and debugging.
    """
    try:
        identity = IdentityContext.from_headers(dict(request.headers))
    except ValueError:
        identity = getattr(request.state, "identity", None)
        if identity is None:
            raise HTTPException(status_code=401, detail="Not authenticated.")

    from auth.middleware.role_enforcement import get_permissions_summary
    return get_permissions_summary(identity)

@router.post("/logout")
async def logout(request: Request) -> dict:
    """
    Invalidate session.
    Stateless JWT: client discards token; no server-side state to clear.
    """
    identity = getattr(request.state, "identity", None)
    if identity:
        log.info("auth.logout", user_id=identity.user_id, tenant_id=identity.tenant_id)
    return {"message": "Logged out. Discard your token."}
