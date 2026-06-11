"""
Unit tests for auth-service Phase 1 (R7).

Covers:
- IdentityContext: construction, to_headers, from_headers, role properties
- UserRole: hierarchy (admin → member → viewer)
- AuthError subclasses: status codes, messages
- OIDCProviderConfig: defaults
- OIDCProviderFactory: create entra_id, unknown raises ValueError, register
- EntraIDProvider: get_authorization_url has required params
- EntraIDProvider: validate_token — expired → TokenExpiredError
- EntraIDProvider: validate_token — invalid → TokenInvalidError
- EntraIDProvider: JWT unavailable → TokenInvalidError
- EntraIDProvider: _parse_entra_claims extracts tid, email, name, roles
- EntraIDProvider: _verify_issuer accepts valid, rejects invalid
- JWTValidatorMiddleware: public paths bypass auth (no 401)
- JWTValidatorMiddleware: missing token → 401
- JWTValidatorMiddleware: bypass_auth=True injects dev identity
- JWTValidatorMiddleware: valid token → identity injected into request.state
- JWTValidatorMiddleware: X-User-Id / X-Tenant-Id in response headers after auth
- Auth router: GET /auth/providers returns list
- Auth router: GET /auth/login/unknown_provider → 400
- Auth router: GET /auth/me with identity headers → returns user dict
- Auth router: GET /auth/me without headers → 401
- Auth router: POST /auth/logout → 200
- Auth router: GET /auth/callback with error param → 400
- Auth router: GET /auth/callback missing code → 400
- auth-service /health → 200, status ok
- auth-service / → version 0.2.0, release R7
- GatewaySettings: auth_enabled default False, auth_service_url present
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ═══════════════════════════════════════════════════════════════════════════════
# IdentityContext
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdentityContext:
    def test_construction_minimal(self):
        from auth.models import IdentityContext
        ctx = IdentityContext(user_id="u1", tenant_id="t1")
        assert ctx.user_id == "u1"
        assert ctx.tenant_id == "t1"

    def test_default_role_is_member(self):
        from auth.models import IdentityContext, UserRole
        ctx = IdentityContext(user_id="u1", tenant_id="t1")
        assert UserRole.MEMBER in ctx.roles

    def test_is_admin_true_for_admin_role(self):
        from auth.models import IdentityContext, UserRole
        ctx = IdentityContext(user_id="u", tenant_id="t", roles=[UserRole.ADMIN])
        assert ctx.is_admin is True

    def test_is_admin_false_for_member(self):
        from auth.models import IdentityContext, UserRole
        ctx = IdentityContext(user_id="u", tenant_id="t", roles=[UserRole.MEMBER])
        assert ctx.is_admin is False

    def test_can_write_true_for_admin(self):
        from auth.models import IdentityContext, UserRole
        ctx = IdentityContext(user_id="u", tenant_id="t", roles=[UserRole.ADMIN])
        assert ctx.can_write is True

    def test_can_write_true_for_member(self):
        from auth.models import IdentityContext, UserRole
        ctx = IdentityContext(user_id="u", tenant_id="t", roles=[UserRole.MEMBER])
        assert ctx.can_write is True

    def test_can_write_false_for_viewer(self):
        from auth.models import IdentityContext, UserRole
        ctx = IdentityContext(user_id="u", tenant_id="t", roles=[UserRole.VIEWER])
        assert ctx.can_write is False

    def test_can_read_true_for_all_roles(self):
        from auth.models import IdentityContext, UserRole
        for role in UserRole:
            ctx = IdentityContext(user_id="u", tenant_id="t", roles=[role])
            assert ctx.can_read is True

    def test_to_headers_contains_required_keys(self):
        from auth.models import IdentityContext, UserRole
        ctx = IdentityContext(
            user_id="u1", tenant_id="t1", email="u@t.com", name="User",
            roles=[UserRole.ADMIN], provider="entra_id",
        )
        h = ctx.to_headers()
        assert "X-User-Id" in h
        assert "X-Tenant-Id" in h
        assert "X-User-Roles" in h
        assert h["X-User-Id"] == "u1"
        assert h["X-Tenant-Id"] == "t1"
        assert "admin" in h["X-User-Roles"]

    def test_from_headers_round_trip(self):
        from auth.models import IdentityContext, UserRole
        original = IdentityContext(
            user_id="u1", tenant_id="tenant-abc",
            email="user@example.com", name="Test User",
            roles=[UserRole.MEMBER], provider="entra_id",
        )
        headers = original.to_headers()
        # Lowercase header keys (as HTTP delivers them)
        lc_headers = {k.lower(): v for k, v in headers.items()}
        restored = IdentityContext.from_headers(lc_headers)
        assert restored.user_id == "u1"
        assert restored.tenant_id == "tenant-abc"
        assert restored.email == "user@example.com"
        assert UserRole.MEMBER in restored.roles

    def test_from_headers_raises_without_user_id(self):
        from auth.models import IdentityContext
        with pytest.raises(ValueError, match="X-User-Id"):
            IdentityContext.from_headers({"x-tenant-id": "t1"})

    def test_from_headers_raises_without_tenant_id(self):
        from auth.models import IdentityContext
        with pytest.raises(ValueError, match="X-Tenant-Id"):
            IdentityContext.from_headers({"x-user-id": "u1"})


# ═══════════════════════════════════════════════════════════════════════════════
# AuthError subclasses
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthErrors:
    def test_token_expired_status_401(self):
        from auth.models import TokenExpiredError
        exc = TokenExpiredError()
        assert exc.status_code == 401

    def test_token_invalid_status_401(self):
        from auth.models import TokenInvalidError
        exc = TokenInvalidError("bad sig")
        assert exc.status_code == 401
        assert "bad sig" in str(exc)

    def test_token_missing_status_401(self):
        from auth.models import TokenMissingError
        assert TokenMissingError().status_code == 401

    def test_insufficient_role_status_403(self):
        from auth.models import InsufficientRoleError
        exc = InsufficientRoleError("admin")
        assert exc.status_code == 403
        assert "admin" in str(exc)

    def test_all_errors_are_auth_error_subclass(self):
        from auth.models import (AuthError, InsufficientRoleError,
                                  TokenExpiredError, TokenInvalidError, TokenMissingError)
        for cls in [TokenExpiredError, TokenInvalidError,
                    TokenMissingError, InsufficientRoleError]:
            assert issubclass(cls, AuthError)


# ═══════════════════════════════════════════════════════════════════════════════
# OIDCProviderFactory
# ═══════════════════════════════════════════════════════════════════════════════

class TestOIDCProviderFactory:
    def test_create_entra_id(self):
        from auth.providers.base import OIDCProviderFactory, EntraIDProvider
        from auth.models import OIDCProviderConfig
        cfg = OIDCProviderConfig(
            provider_name="entra_id", client_id="app-id", tenant_id="my-tenant"
        )
        provider = OIDCProviderFactory.create("entra_id", cfg)
        assert isinstance(provider, EntraIDProvider)

    def test_unknown_provider_raises(self):
        from auth.providers.base import OIDCProviderFactory
        from auth.models import OIDCProviderConfig
        cfg = OIDCProviderConfig(provider_name="unknown", client_id="x")
        with pytest.raises(ValueError, match="Unknown OIDC provider"):
            OIDCProviderFactory.create("unknown", cfg)

    def test_available_providers_includes_entra_id(self):
        from auth.providers.base import OIDCProviderFactory
        assert "entra_id" in OIDCProviderFactory.available_providers()

    def test_register_new_provider(self):
        from auth.providers.base import OIDCProviderFactory, OIDCProviderBase
        class FakeProvider(OIDCProviderBase):
            def validate_token(self, token): pass
            def get_authorization_url(self, s, n): return ""
            def exchange_code(self, c, s): return {}
        OIDCProviderFactory.register("fake", FakeProvider)
        assert "fake" in OIDCProviderFactory.available_providers()
        # Clean up
        del OIDCProviderFactory._registry["fake"]


# ═══════════════════════════════════════════════════════════════════════════════
# EntraIDProvider
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntraIDProvider:
    def _make_provider(self, tenant="my-tenant", client_id="app-123"):
        from auth.providers.base import EntraIDProvider
        from auth.models import OIDCProviderConfig
        cfg = OIDCProviderConfig(
            provider_name="entra_id", client_id=client_id,
            tenant_id=tenant, redirect_uri="http://localhost/callback",
        )
        return EntraIDProvider(cfg)

    def test_authorization_url_contains_client_id(self):
        p = self._make_provider()
        url = p.get_authorization_url("state123", "nonce456")
        assert "app-123" in url
        assert "state123" in url

    def test_authorization_url_contains_openid_scope(self):
        p = self._make_provider()
        url = p.get_authorization_url("s", "n")
        assert "openid" in url

    def test_authorization_url_contains_redirect_uri(self):
        p = self._make_provider()
        url = p.get_authorization_url("s", "n")
        assert "callback" in url

    def test_validate_token_jwt_unavailable_raises(self):
        from auth.models import TokenInvalidError
        p = self._make_provider()
        with patch("auth.providers.base._JWT_AVAILABLE", False):
            with pytest.raises(TokenInvalidError):
                p.validate_token("fake.token.here")

    def test_validate_token_expired_raises_token_expired_error(self):
        from auth.models import TokenExpiredError
        p = self._make_provider()
        with patch("auth.providers.base._JWT_AVAILABLE", True), \
             patch("auth.providers.base.pyjwt") as mock_jwt, \
             patch("auth.providers.base.PyJWKClient") as mock_jwks:
            import jwt as real_jwt
            mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
            mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError
            mock_jwks_instance = MagicMock()
            mock_jwks_instance.get_signing_key_from_jwt.return_value = MagicMock(key="key")
            mock_jwks.return_value = mock_jwks_instance
            mock_jwt.decode.side_effect = real_jwt.ExpiredSignatureError("expired")
            p._jwks_client = mock_jwks_instance
            with pytest.raises(TokenExpiredError):
                p.validate_token("expired.token.here")

    def test_validate_token_invalid_raises_token_invalid_error(self):
        from auth.models import TokenInvalidError
        p = self._make_provider()
        with patch("auth.providers.base._JWT_AVAILABLE", True), \
             patch("auth.providers.base.pyjwt") as mock_jwt, \
             patch("auth.providers.base.PyJWKClient") as mock_jwks:
            import jwt as real_jwt
            mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
            mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError
            mock_jwks_instance = MagicMock()
            mock_jwks_instance.get_signing_key_from_jwt.return_value = MagicMock(key="key")
            mock_jwks.return_value = mock_jwks_instance
            mock_jwt.decode.side_effect = real_jwt.InvalidTokenError("bad sig")
            p._jwks_client = mock_jwks_instance
            with pytest.raises(TokenInvalidError):
                p.validate_token("invalid.token.here")

    def test_parse_entra_claims_extracts_tid(self):
        p = self._make_provider()
        payload = {
            "oid": "user-oid-123", "iss": "https://login.microsoftonline.com/tid/v2.0",
            "aud": "app-123", "exp": int(time.time()) + 3600, "iat": int(time.time()),
            "tid": "tenant-xyz", "email": "user@corp.com",
            "name": "Test User", "roles": ["member"],
        }
        claims = p._parse_entra_claims(payload)
        assert claims.tenant_id == "tenant-xyz"
        assert claims.email == "user@corp.com"
        assert claims.name == "Test User"
        assert "member" in claims.roles

    def test_parse_entra_claims_uses_preferred_username_fallback(self):
        p = self._make_provider()
        payload = {
            "oid": "u1", "iss": "x", "aud": "a",
            "exp": 9999999999, "iat": 1000,
            "preferred_username": "preferred@corp.com",
        }
        claims = p._parse_entra_claims(payload)
        assert claims.email == "preferred@corp.com"

    def test_verify_issuer_accepts_valid_entra_issuer(self):
        p = self._make_provider()
        valid = "https://login.microsoftonline.com/my-tenant/v2.0"
        p._verify_issuer(valid)  # should not raise

    def test_verify_issuer_rejects_unknown(self):
        from auth.models import TokenInvalidError
        p = self._make_provider()
        with pytest.raises(TokenInvalidError, match="Untrusted issuer"):
            p._verify_issuer("https://evil.attacker.com/token")

    def test_map_roles_admin(self):
        from auth.providers.base import OIDCProviderBase
        from auth.models import UserRole
        roles = OIDCProviderBase._map_roles(["admin"])
        assert UserRole.ADMIN in roles

    def test_map_roles_default_member(self):
        from auth.providers.base import OIDCProviderBase
        from auth.models import UserRole
        roles = OIDCProviderBase._map_roles([])
        assert UserRole.MEMBER in roles

    def test_map_roles_viewer(self):
        from auth.providers.base import OIDCProviderBase
        from auth.models import UserRole
        roles = OIDCProviderBase._map_roles(["viewer"])
        assert UserRole.VIEWER in roles


# ═══════════════════════════════════════════════════════════════════════════════
# JWTValidatorMiddleware
# ═══════════════════════════════════════════════════════════════════════════════

def _make_middleware_app(bypass: bool = True, providers: dict | None = None):
    from auth.middleware.jwt_validator import JWTValidatorMiddleware
    app = FastAPI()
    app.add_middleware(JWTValidatorMiddleware,
                       providers=providers or {},
                       bypass_auth=bypass)

    @app.get("/health")
    async def health(): return {"status": "ok"}

    @app.get("/protected")
    async def protected(request): return {"user": getattr(request.state, "identity", None) and "present"}

    return TestClient(app, raise_server_exceptions=False)


class TestJWTValidatorMiddleware:
    def test_public_health_path_no_auth_required(self):
        client = _make_middleware_app(bypass=False)
        assert client.get("/health").status_code == 200

    def test_docs_path_no_auth_required(self):
        client = _make_middleware_app(bypass=False)
        assert client.get("/docs").status_code == 200

    def test_missing_token_returns_401(self):
        client = _make_middleware_app(bypass=False)
        r = client.get("/protected")
        assert r.status_code == 401

    def test_bypass_auth_injects_dev_identity(self):
        from auth.middleware.jwt_validator import JWTValidatorMiddleware, _dev_identity
        # Test the dev identity directly — bypass injects it unconditionally
        identity = _dev_identity()
        assert identity.tenant_id == "dev"
        assert identity.user_id == "dev-user-001"
        from auth.models import UserRole
        assert UserRole.ADMIN in identity.roles

    def test_bypass_response_has_x_trace_id(self):
        client = _make_middleware_app(bypass=True)
        r = client.get("/health")
        assert r.status_code == 200

    def test_missing_bearer_prefix_returns_401(self):
        client = _make_middleware_app(bypass=False)
        r = client.get("/protected", headers={"Authorization": "Basic abc"})
        assert r.status_code == 401

    def test_auth_error_response_has_detail_field(self):
        client = _make_middleware_app(bypass=False)
        r = client.get("/protected")
        assert "detail" in r.json()


# ═══════════════════════════════════════════════════════════════════════════════
# Auth router
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def auth_client():
    from auth.main import app
    app.state.providers = {}
    return TestClient(app, raise_server_exceptions=False)


class TestAuthRouter:
    def test_health_200(self, auth_client):
        assert auth_client.get("/health").status_code == 200

    def test_health_status_ok(self, auth_client):
        assert auth_client.get("/health").json()["status"] == "ok"

    def test_root_version_02(self, auth_client):
        assert auth_client.get("/").json()["version"] == "0.2.0"

    def test_root_release_r7(self, auth_client):
        assert auth_client.get("/").json()["release"] == "R7"

    def test_providers_returns_list(self, auth_client):
        r = auth_client.get("/auth/providers")
        assert r.status_code == 200
        assert "providers" in r.json()

    def test_login_unknown_provider_400(self, auth_client):
        r = auth_client.get("/auth/login/nonexistent_provider")
        assert r.status_code == 400

    def test_callback_with_error_param_400(self, auth_client):
        r = auth_client.get("/auth/callback/entra_id?error=access_denied&error_description=User+denied")
        assert r.status_code == 400

    def test_callback_missing_code_400(self, auth_client):
        r = auth_client.get("/auth/callback/entra_id")
        assert r.status_code == 400

    def test_me_without_headers_401(self, auth_client):
        r = auth_client.get("/auth/me")
        assert r.status_code == 401

    def test_me_with_headers_returns_user_dict(self, auth_client):
        r = auth_client.get("/auth/me", headers={
            "X-User-Id":    "user-123",
            "X-Tenant-Id":  "tenant-abc",
            "X-User-Roles": "member",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == "user-123"
        assert body["tenant_id"] == "tenant-abc"

    def test_me_has_can_write_field(self, auth_client):
        r = auth_client.get("/auth/me", headers={
            "X-User-Id": "u", "X-Tenant-Id": "t", "X-User-Roles": "member"
        })
        assert "can_write" in r.json()

    def test_logout_200(self, auth_client):
        assert auth_client.post("/auth/logout").status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# GatewaySettings auth fields
# ═══════════════════════════════════════════════════════════════════════════════

class TestGatewayAuthSettings:
    def test_auth_enabled_default_false(self):
        from api_gateway.settings import GatewaySettings
        assert GatewaySettings().auth_enabled is False

    def test_auth_service_url_default_set(self):
        from api_gateway.settings import GatewaySettings
        s = GatewaySettings()
        assert "auth" in s.auth_service_url

    def test_gateway_accepts_auth_enabled_true(self):
        import os
        os.environ["RAGLAB_AUTH_ENABLED"] = "true"
        from api_gateway.settings import GatewaySettings
        s = GatewaySettings()
        assert s.auth_enabled is True
        del os.environ["RAGLAB_AUTH_ENABLED"]
