"""
Identity models — shared types for auth-service and gateway middleware.

These types flow through every authenticated request:

    Client → Gateway (JWT validation) → IdentityContext injected
    → Services receive X-User-Id / X-Tenant-Id / X-User-Roles headers
    → Services extract IdentityContext from headers (trusted, never re-validate)

Design principle: JWT validated once at the gateway.
Downstream services trust gateway-injected headers — no re-validation,
no per-service JWT libraries, no per-service mistakes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Roles ──────────────────────────────────────────────────────────────────────

class UserRole(str, Enum):
    """
    Role hierarchy (highest → lowest privilege):
        admin   — full access across all tenants (platform operator)
        member  — full access within their tenant
        viewer  — read-only within their tenant
    """
    ADMIN  = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


# ── Identity context ───────────────────────────────────────────────────────────

class IdentityContext(BaseModel):
    """
    Verified identity context for an authenticated request.

    Injected by gateway after JWT validation.
    Propagated to all downstream services as trusted headers.
    Never constructed from unverified sources.
    """
    user_id:   str = Field(..., description="Unique user identifier from OIDC provider")
    tenant_id: str = Field(..., description="Tenant this user belongs to")
    email:     str = Field(default="", description="User email from OIDC claims")
    name:      str = Field(default="", description="Display name from OIDC claims")
    roles:     list[UserRole] = Field(
        default_factory=lambda: [UserRole.MEMBER],
        description="User roles within the tenant",
    )
    provider:  str = Field(default="", description="OIDC provider: entra_id | google | cognito")

    @property
    def is_admin(self) -> bool:
        return UserRole.ADMIN in self.roles

    @property
    def is_member(self) -> bool:
        return UserRole.MEMBER in self.roles or self.is_admin

    @property
    def can_write(self) -> bool:
        return UserRole.ADMIN in self.roles or UserRole.MEMBER in self.roles

    @property
    def can_read(self) -> bool:
        return True  # all roles can read

    def to_headers(self) -> dict[str, str]:
        """
        Serialize to gateway-injected trusted headers.
        Downstream services call IdentityContext.from_headers() to reconstruct.
        """
        return {
            "X-User-Id":    self.user_id,
            "X-Tenant-Id":  self.tenant_id,
            "X-User-Email": self.email,
            "X-User-Name":  self.name,
            "X-User-Roles": ",".join(r.value for r in self.roles),
            "X-Auth-Provider": self.provider,
        }

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "IdentityContext":
        """
        Reconstruct from trusted gateway-injected headers.
        Raises ValueError if required headers are missing.
        """
        user_id   = headers.get("x-user-id") or headers.get("X-User-Id", "")
        tenant_id = headers.get("x-tenant-id") or headers.get("X-Tenant-Id", "")
        if not user_id or not tenant_id:
            raise ValueError(
                "Missing required identity headers X-User-Id / X-Tenant-Id. "
                "All requests must be authenticated via the gateway."
            )
        roles_raw = (
            headers.get("x-user-roles") or headers.get("X-User-Roles", "member")
        )
        roles = [UserRole(r.strip()) for r in roles_raw.split(",") if r.strip()]
        return cls(
            user_id=user_id,
            tenant_id=tenant_id,
            email=headers.get("x-user-email") or headers.get("X-User-Email", ""),
            name=headers.get("x-user-name") or headers.get("X-User-Name", ""),
            roles=roles or [UserRole.MEMBER],
            provider=headers.get("x-auth-provider") or headers.get("X-Auth-Provider", ""),
        )


# ── JWT token claims ───────────────────────────────────────────────────────────

class TokenClaims(BaseModel):
    """
    Parsed JWT claims. Provider-agnostic.

    After parsing, gateway maps these to IdentityContext.
    """
    sub:         str            # subject — unique user ID in the provider
    iss:         str            # issuer URL
    aud:         list[str] | str  # audience
    exp:         int            # expiry (Unix timestamp)
    iat:         int            # issued at
    email:       str = ""
    name:        str = ""
    tenant_id:   str = ""       # mapped from provider-specific claim
    roles:       list[str] = Field(default_factory=list)
    raw:         dict[str, Any] = Field(default_factory=dict)


# ── Provider config ────────────────────────────────────────────────────────────

class OIDCProviderConfig(BaseModel):
    """Configuration for a single OIDC provider."""
    provider_name:  str          # "entra_id" | "google" | "cognito"
    client_id:      str
    client_secret:  str = Field(default="", repr=False)  # not logged
    tenant_id:      str = ""     # Entra ID: Azure AD tenant ID
    authority:      str = ""     # full issuer URL
    redirect_uri:   str = ""
    scopes:         list[str] = Field(default_factory=lambda: ["openid", "email", "profile"])
    jwks_uri:       str = ""     # if empty, discovered from well-known endpoint
    audience:       str = ""     # expected audience claim
    # Claim mapping — which JWT claim holds the tenant_id
    tenant_claim:   str = "tid"  # Entra: "tid"; Google: "hd"; Cognito: custom
    roles_claim:    str = ""     # claim that holds role list (if provider supports)


# ── Auth errors ────────────────────────────────────────────────────────────────

class AuthError(Exception):
    """Base authentication error — raised by JWT validator, caught by gateway."""
    def __init__(self, message: str, status_code: int = 401) -> None:
        super().__init__(message)
        self.status_code = status_code


class TokenExpiredError(AuthError):
    def __init__(self) -> None:
        super().__init__("Token has expired.", status_code=401)


class TokenInvalidError(AuthError):
    def __init__(self, reason: str = "") -> None:
        super().__init__(f"Token is invalid.{' ' + reason if reason else ''}", status_code=401)


class TokenMissingError(AuthError):
    def __init__(self) -> None:
        super().__init__("Authorization token required.", status_code=401)


class InsufficientRoleError(AuthError):
    def __init__(self, required: str) -> None:
        super().__init__(
            f"Insufficient permissions. Required role: {required}.",
            status_code=403,
        )
