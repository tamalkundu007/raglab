"""
Google OIDC and AWS Cognito OIDC provider implementations (R7 Phase 2).

GoogleOIDCProvider:
    Issuer:   https://accounts.google.com
    JWKS:     https://www.googleapis.com/oauth2/v3/certs
    Audience: client_id
    Tenant:   hd (hosted domain) claim — Google Workspace tenant
    User ID:  sub claim

CognitoOIDCProvider:
    Issuer:   https://cognito-idp.{region}.amazonaws.com/{user_pool_id}
    JWKS:     {issuer}/.well-known/jwks.json
    Audience: client_id (app client)
    Tenant:   custom:tenant_id claim (standard Cognito custom attribute)
              Falls back to user pool ID as tenant if not set.
    User ID:  sub claim

Both follow the same pattern as EntraIDProvider:
    1. Fetch/cache JWKS.
    2. Verify signature + claims.
    3. Map to IdentityContext via _claims_to_identity().

OIDCProviderFactory.register() is called at module load time,
making both providers available for the factory by name.
"""

from __future__ import annotations

import time
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
from auth.providers.base import OIDCProviderBase, OIDCProviderFactory

log = get_logger(__name__)

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


# ── Google OIDC ────────────────────────────────────────────────────────────────

class GoogleOIDCProvider(OIDCProviderBase):
    """
    Google OIDC provider.

    Supports Google Workspace (G Suite) accounts where `hd` claim = hosted domain.
    Personal Google accounts have no `hd` — they get tenant_id derived from email.
    """

    ISSUER    = "https://accounts.google.com"
    JWKS_URL  = "https://www.googleapis.com/oauth2/v3/certs"
    AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    def __init__(self, config: OIDCProviderConfig) -> None:
        super().__init__(config)
        self._jwks_url = config.jwks_uri or self.JWKS_URL

    def validate_token(self, token: str) -> IdentityContext:
        """Validate a Google ID token."""
        if not _JWT_AVAILABLE:
            raise TokenInvalidError("pyjwt not installed")
        claims_dict = self._decode_jwt(token)
        claims = self._parse_google_claims(claims_dict)
        return self._claims_to_identity(claims, "google")

    def _decode_jwt(self, token: str) -> dict:
        jwks_client = self._get_jwks_client()
        try:
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.config.client_id or self.config.audience,
                issuer=self.ISSUER,
            )
            return payload
        except pyjwt.ExpiredSignatureError:
            raise TokenExpiredError()
        except PyJWKClientError:
            self._jwks_client = None
            self._jwks_cache_ts = 0
            jwks_client = self._get_jwks_client()
            try:
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                return pyjwt.decode(
                    token, signing_key.key, algorithms=["RS256"],
                    audience=self.config.client_id,
                    issuer=self.ISSUER,
                )
            except Exception as exc:
                raise TokenInvalidError(str(exc))
        except pyjwt.InvalidTokenError as exc:
            raise TokenInvalidError(str(exc))

    def _get_jwks_client(self) -> Any:
        now = time.time()
        if self._jwks_client is None or (now - self._jwks_cache_ts) > self._jwks_ttl:
            if PyJWKClient is None:
                raise TokenInvalidError("pyjwt[cryptography] not installed")
            self._jwks_client = PyJWKClient(self._jwks_url)
            self._jwks_cache_ts = now
            log.info("auth.jwks_refreshed", provider="google", url=self._jwks_url)
        return self._jwks_client

    def _parse_google_claims(self, payload: dict) -> TokenClaims:
        """Parse Google ID token claims into TokenClaims."""
        sub     = payload.get("sub", "")
        email   = payload.get("email", "")
        name    = payload.get("name", "")
        # hd = hosted domain (Google Workspace tenant)
        hd      = payload.get("hd", "")
        # If no hosted domain, derive tenant from email domain
        if not hd and email and "@" in email:
            hd = email.split("@")[1]
        # Roles from custom claim (Google doesn't have native roles)
        roles   = payload.get("raglab_roles", [])

        aud = payload.get("aud", "")
        if isinstance(aud, str):
            aud = [aud]

        return TokenClaims(
            sub=sub, iss=payload.get("iss", ""), aud=aud,
            exp=payload.get("exp", 0), iat=payload.get("iat", 0),
            email=email, name=name, tenant_id=hd, roles=roles, raw=payload,
        )

    def get_authorization_url(self, state: str, nonce: str) -> str:
        from urllib.parse import urlencode
        params = {
            "client_id":     self.config.client_id,
            "response_type": "code",
            "redirect_uri":  self.config.redirect_uri,
            "scope":         " ".join(self.config.scopes),
            "state":         state,
            "nonce":         nonce,
            "access_type":   "offline",  # for refresh token
            "prompt":        "select_account",
        }
        if self.config.tenant_id:  # hosted domain restriction
            params["hd"] = self.config.tenant_id
        return f"{self.AUTH_URL}?{urlencode(params)}"

    def exchange_code(self, code: str, state: str) -> dict:
        if not _HTTPX_AVAILABLE:
            raise AuthError("httpx not installed")
        try:
            resp = _httpx.post(
                self.TOKEN_URL,
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
            raise AuthError(f"Google code exchange failed: {exc}")


# ── AWS Cognito ────────────────────────────────────────────────────────────────

class CognitoOIDCProvider(OIDCProviderBase):
    """
    AWS Cognito OIDC provider.

    User pool ID format: {region}_{PoolId}  (e.g. us-east-1_AbCdEfGhI)
    Issuer:  https://cognito-idp.{region}.amazonaws.com/{user_pool_id}
    JWKS:    {issuer}/.well-known/jwks.json
    Tenant:  custom:tenant_id attribute (set at user creation)
    """

    def __init__(self, config: OIDCProviderConfig) -> None:
        super().__init__(config)
        # user_pool_id stored in config.tenant_id field (overloaded for Cognito)
        self._user_pool_id = config.tenant_id
        if self._user_pool_id:
            region = self._user_pool_id.split("_")[0]
            self._issuer   = (
                config.authority
                or f"https://cognito-idp.{region}.amazonaws.com/{self._user_pool_id}"
            )
            self._jwks_url = config.jwks_uri or f"{self._issuer}/.well-known/jwks.json"
        else:
            self._issuer   = config.authority or ""
            self._jwks_url = config.jwks_uri  or ""

    def validate_token(self, token: str) -> IdentityContext:
        """Validate a Cognito JWT (access token or ID token)."""
        if not _JWT_AVAILABLE:
            raise TokenInvalidError("pyjwt not installed")
        if not self._jwks_url:
            raise TokenInvalidError("Cognito user_pool_id not configured.")
        claims_dict = self._decode_jwt(token)
        claims = self._parse_cognito_claims(claims_dict)
        return self._claims_to_identity(claims, "cognito")

    def _decode_jwt(self, token: str) -> dict:
        jwks_client = self._get_jwks_client()
        try:
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.config.client_id or self.config.audience,
                issuer=self._issuer,
                options={"verify_iss": bool(self._issuer)},
            )
            return payload
        except pyjwt.ExpiredSignatureError:
            raise TokenExpiredError()
        except PyJWKClientError:
            self._jwks_client = None
            self._jwks_cache_ts = 0
            jwks_client = self._get_jwks_client()
            try:
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                return pyjwt.decode(
                    token, signing_key.key, algorithms=["RS256"],
                    audience=self.config.client_id,
                    issuer=self._issuer,
                )
            except Exception as exc:
                raise TokenInvalidError(str(exc))
        except pyjwt.InvalidTokenError as exc:
            raise TokenInvalidError(str(exc))

    def _get_jwks_client(self) -> Any:
        now = time.time()
        if self._jwks_client is None or (now - self._jwks_cache_ts) > self._jwks_ttl:
            if PyJWKClient is None:
                raise TokenInvalidError("pyjwt[cryptography] not installed")
            self._jwks_client = PyJWKClient(self._jwks_url)
            self._jwks_cache_ts = now
            log.info("auth.jwks_refreshed", provider="cognito", url=self._jwks_url)
        return self._jwks_client

    def _parse_cognito_claims(self, payload: dict) -> TokenClaims:
        """Parse Cognito JWT claims into TokenClaims."""
        sub       = payload.get("sub", "")
        email     = payload.get("email", "")
        name      = payload.get("name", "") or payload.get("cognito:username", "")
        # Custom attribute for tenant_id
        tenant_id = (
            payload.get("custom:tenant_id", "")
            or payload.get("custom:tenantId", "")
            or self._user_pool_id  # fallback: pool-level tenant
        )
        # Cognito groups map to roles
        groups    = payload.get("cognito:groups", [])
        roles     = [g for g in groups if g in ("admin","member","viewer",
                                                  "raglab_admin","raglab_member")]

        aud = payload.get("aud", "")
        if isinstance(aud, str):
            aud = [aud]

        return TokenClaims(
            sub=sub, iss=payload.get("iss", ""), aud=aud,
            exp=payload.get("exp", 0), iat=payload.get("iat", 0),
            email=email, name=name, tenant_id=tenant_id, roles=roles, raw=payload,
        )

    def get_authorization_url(self, state: str, nonce: str) -> str:
        from urllib.parse import urlencode
        domain = getattr(self.config, "cognito_domain", "")
        base   = f"https://{domain}/oauth2/authorize" if domain else self._issuer
        params = {
            "client_id":     self.config.client_id,
            "response_type": "code",
            "redirect_uri":  self.config.redirect_uri,
            "scope":         " ".join(self.config.scopes),
            "state":         state,
        }
        return f"{base}?{urlencode(params)}"

    def exchange_code(self, code: str, state: str) -> dict:
        if not _HTTPX_AVAILABLE:
            raise AuthError("httpx not installed")
        domain = getattr(self.config, "cognito_domain", "")
        token_url = (
            f"https://{domain}/oauth2/token"
            if domain
            else f"{self._issuer}/oauth2/token"
        )
        try:
            resp = _httpx.post(
                token_url,
                data={
                    "client_id":    self.config.client_id,
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
            raise AuthError(f"Cognito code exchange failed: {exc}")


# ── Register both providers with the factory ──────────────────────────────────

OIDCProviderFactory.register("google",   GoogleOIDCProvider)
OIDCProviderFactory.register("cognito",  CognitoOIDCProvider)
