"""
OIDC provider abstraction + Microsoft Entra ID implementation.

Design:
    OIDCProviderBase — abstract interface all providers implement.
    EntraIDProvider  — Microsoft Entra ID (Azure AD) implementation.
    OIDCProviderFactory — creates providers by name.

JWT validation flow (Entra ID):
    1. Fetch JWKS from well-known endpoint (cached in memory with TTL).
    2. Decode and validate JWT signature using the matching key.
    3. Verify: issuer, audience, expiry, not-before.
    4. Extract claims → TokenClaims → IdentityContext.

JWKS caching:
    Keys are fetched lazily and cached for 1 hour.
    On validation failure with a cached key, keys are re-fetched once
    (handles key rotation without service restart).

This module is imported by:
    - auth-service (for token exchange + user info)
    - api-gateway JWT middleware (for token validation only)

The gateway only needs validate_token() — it never redirects to login.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from raglab_common.logging import get_logger
from auth.models import (
    AuthError,
    IdentityContext,
    OIDCProviderConfig,
    TokenClaims,
    TokenExpiredError,
    TokenInvalidError,
    UserRole,
)

log = get_logger(__name__)

# Optional imports — patchable in tests
try:
    import jwt as pyjwt
    from jwt import PyJWKClient, PyJWKClientError
    _JWT_AVAILABLE = True
except ImportError:
    pyjwt = None  # type: ignore[assignment]
    PyJWKClient = None  # type: ignore[assignment]
    _JWT_AVAILABLE = False

try:
    import httpx as _httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HTTPX_AVAILABLE = False


# ── Base provider ─────────────────────────────────────────────────────────────

class OIDCProviderBase(ABC):
    """
    Abstract OIDC provider.

    All providers implement:
        validate_token(token) → IdentityContext
        get_authorization_url(state, nonce) → str
        exchange_code(code, state) → dict (token response)
        get_user_info(access_token) → dict
    """

    def __init__(self, config: OIDCProviderConfig) -> None:
        self.config = config
        self._jwks_client: Any = None
        self._jwks_cache_ts: float = 0
        self._jwks_ttl: int = 3600  # 1 hour

    @abstractmethod
    def validate_token(self, token: str) -> IdentityContext:
        """Validate a JWT and return IdentityContext. Raises AuthError on failure."""
        ...

    @abstractmethod
    def get_authorization_url(self, state: str, nonce: str) -> str:
        """Return the OIDC authorization redirect URL."""
        ...

    @abstractmethod
    def exchange_code(self, code: str, state: str) -> dict:
        """Exchange authorization code for tokens."""
        ...

    def _claims_to_identity(
        self,
        claims: TokenClaims,
        provider: str,
    ) -> IdentityContext:
        """Map TokenClaims → IdentityContext. Shared across providers."""
        tenant_id = claims.tenant_id or claims.sub.split("@")[0] if "@" in claims.sub else claims.sub[:8]
        roles = self._map_roles(claims.roles)
        return IdentityContext(
            user_id=claims.sub,
            tenant_id=tenant_id,
            email=claims.email,
            name=claims.name,
            roles=roles,
            provider=provider,
        )

    @staticmethod
    def _map_roles(raw_roles: list[str]) -> list[UserRole]:
        """Map provider role strings to UserRole enum. Default to MEMBER."""
        role_map = {
            "admin":         UserRole.ADMIN,
            "raglab_admin":  UserRole.ADMIN,
            "owner":         UserRole.ADMIN,
            "member":        UserRole.MEMBER,
            "raglab_member": UserRole.MEMBER,
            "user":          UserRole.MEMBER,
            "viewer":        UserRole.VIEWER,
            "raglab_viewer": UserRole.VIEWER,
            "readonly":      UserRole.VIEWER,
        }
        mapped = [role_map[r.lower()] for r in raw_roles if r.lower() in role_map]
        return mapped if mapped else [UserRole.MEMBER]


# ── Microsoft Entra ID (Azure AD) ─────────────────────────────────────────────

class EntraIDProvider(OIDCProviderBase):
    """
    Microsoft Entra ID (Azure AD) OIDC provider.

    Supports both single-tenant (specific tenant_id) and
    multi-tenant (common endpoint) configurations.

    JWT validation:
        Issuer:   https://login.microsoftonline.com/{tenant_id}/v2.0
        JWKS:     https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys
        Audience: config.client_id (the application client ID)
    """

    AUTHORITY_BASE = "https://login.microsoftonline.com"
    TOKEN_URL_TMPL = "{authority}/{tenant}/oauth2/v2.0/token"
    AUTH_URL_TMPL  = "{authority}/{tenant}/oauth2/v2.0/authorize"
    JWKS_URL_TMPL  = "{authority}/{tenant}/discovery/v2.0/keys"

    def __init__(self, config: OIDCProviderConfig) -> None:
        super().__init__(config)
        tenant = config.tenant_id or "common"
        self._tenant      = tenant
        self._authority   = config.authority or self.AUTHORITY_BASE
        self._jwks_url    = (
            config.jwks_uri
            or self.JWKS_URL_TMPL.format(authority=self._authority, tenant=tenant)
        )
        self._token_url   = self.TOKEN_URL_TMPL.format(
            authority=self._authority, tenant=tenant
        )
        self._auth_url    = self.AUTH_URL_TMPL.format(
            authority=self._authority, tenant=tenant
        )
        # Expected issuers — single-tenant and multi-tenant
        self._valid_issuers = {
            f"{self._authority}/{tenant}/v2.0",
            f"{self._authority}/common/v2.0",
        }

    # ── JWT validation ────────────────────────────────────────────────────────

    def validate_token(self, token: str) -> IdentityContext:
        """
        Validate a Microsoft Entra ID JWT.

        Steps:
            1. Fetch/cache JWKS.
            2. Decode header to get kid.
            3. Verify signature + claims.
            4. Map to IdentityContext.

        Raises:
            TokenExpiredError — token has expired
            TokenInvalidError — signature/claims invalid
        """
        if not _JWT_AVAILABLE:
            raise TokenInvalidError("pyjwt not installed")

        claims_dict = self._decode_jwt(token)
        claims = self._parse_entra_claims(claims_dict)
        return self._claims_to_identity(claims, "entra_id")

    def _decode_jwt(self, token: str) -> dict:
        """Decode + verify JWT using JWKS. Retries once on key rotation."""
        jwks_client = self._get_jwks_client()
        try:
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.config.client_id or self.config.audience,
                options={"verify_iss": False},  # issuer verified manually
            )
            self._verify_issuer(payload.get("iss", ""))
            return payload
        except pyjwt.ExpiredSignatureError:
            raise TokenExpiredError()
        except PyJWKClientError:
            # Key may have rotated — refresh and retry once
            self._jwks_client = None
            self._jwks_cache_ts = 0
            jwks_client = self._get_jwks_client()
            try:
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                payload = pyjwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    audience=self.config.client_id or self.config.audience,
                    options={"verify_iss": False},
                )
                self._verify_issuer(payload.get("iss", ""))
                return payload
            except Exception as exc:
                raise TokenInvalidError(str(exc))
        except pyjwt.InvalidTokenError as exc:
            raise TokenInvalidError(str(exc))
        except Exception as exc:
            raise TokenInvalidError(f"Unexpected error: {exc}")

    def _get_jwks_client(self) -> Any:
        """Return cached JWKS client, refreshing if TTL expired."""
        now = time.time()
        if self._jwks_client is None or (now - self._jwks_cache_ts) > self._jwks_ttl:
            if PyJWKClient is None:
                raise TokenInvalidError("pyjwt[cryptography] not installed")
            self._jwks_client = PyJWKClient(self._jwks_url)
            self._jwks_cache_ts = now
            log.info("auth.jwks_refreshed", provider="entra_id", url=self._jwks_url)
        return self._jwks_client

    def _verify_issuer(self, iss: str) -> None:
        """Verify token issuer is a known Entra ID endpoint."""
        # Also allow tenant-specific issuer from the actual token
        if iss in self._valid_issuers:
            return
        # Accept any valid Entra ID tenant issuer
        if iss.startswith(self._authority) and iss.endswith("/v2.0"):
            return
        raise TokenInvalidError(f"Untrusted issuer: {iss}")

    def _parse_entra_claims(self, payload: dict) -> TokenClaims:
        """Parse Entra ID specific claims into TokenClaims."""
        sub   = payload.get("oid") or payload.get("sub", "")
        email = payload.get("email") or payload.get("preferred_username") or payload.get("upn", "")
        name  = payload.get("name", "")
        # tid = tenant ID in Entra tokens
        tenant_id = payload.get("tid", "")
        # Roles from app roles claim
        roles = payload.get("roles", [])
        # Also check "groups" for group-based role mapping
        groups = payload.get("groups", [])

        aud = payload.get("aud", "")
        if isinstance(aud, str):
            aud = [aud]

        return TokenClaims(
            sub=sub,
            iss=payload.get("iss", ""),
            aud=aud,
            exp=payload.get("exp", 0),
            iat=payload.get("iat", 0),
            email=email,
            name=name,
            tenant_id=tenant_id,
            roles=roles,
            raw=payload,
        )

    # ── Authorization URL ─────────────────────────────────────────────────────

    def get_authorization_url(self, state: str, nonce: str) -> str:
        """Build Entra ID OIDC authorization redirect URL."""
        from urllib.parse import urlencode
        params = {
            "client_id":     self.config.client_id,
            "response_type": "code",
            "redirect_uri":  self.config.redirect_uri,
            "response_mode": "query",
            "scope":         " ".join(self.config.scopes),
            "state":         state,
            "nonce":         nonce,
        }
        return f"{self._auth_url}?{urlencode(params)}"

    # ── Code exchange ─────────────────────────────────────────────────────────

    def exchange_code(self, code: str, state: str) -> dict:
        """Exchange authorization code for tokens via Entra ID token endpoint."""
        if not _HTTPX_AVAILABLE:
            raise AuthError("httpx not installed")
        try:
            resp = _httpx.post(
                self._token_url,
                data={
                    "client_id":     self.config.client_id,
                    "client_secret": self.config.client_secret,
                    "code":          code,
                    "redirect_uri":  self.config.redirect_uri,
                    "grant_type":    "authorization_code",
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("auth.code_exchange_failed", provider="entra_id", error=str(exc))
            raise AuthError(f"Code exchange failed: {exc}")


# ── Provider factory (Phase 1 — Entra ID only; Phase 2 adds Google + Cognito) ─

class OIDCProviderFactory:
    """
    Creates OIDC provider instances by name.

    Phase 1: EntraIDProvider only.
    Phase 2: GoogleOIDCProvider, CognitoOIDCProvider added.
    """

    _registry: dict[str, type[OIDCProviderBase]] = {
        "entra_id": EntraIDProvider,
    }

    @classmethod
    def create(cls, provider_name: str, config: OIDCProviderConfig) -> OIDCProviderBase:
        """Create a provider instance. Raises ValueError for unknown providers."""
        provider_class = cls._registry.get(provider_name)
        if provider_class is None:
            known = list(cls._registry.keys())
            raise ValueError(
                f"Unknown OIDC provider: '{provider_name}'. Known: {known}"
            )
        return provider_class(config)

    @classmethod
    def register(cls, name: str, provider_class: type[OIDCProviderBase]) -> None:
        """Register a new provider (used in Phase 2 for Google + Cognito)."""
        cls._registry[name] = provider_class

    @classmethod
    def available_providers(cls) -> list[str]:
        return list(cls._registry.keys())
