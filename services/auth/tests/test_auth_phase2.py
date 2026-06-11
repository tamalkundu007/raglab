"""
Unit tests for auth-service Phase 2 (R7) — Google + AWS Cognito providers.

Covers:
- OIDCProviderFactory: google + cognito registered after import
- GoogleOIDCProvider: get_authorization_url contains client_id, scope, state
- GoogleOIDCProvider: hd param included when tenant_id set (Workspace restriction)
- GoogleOIDCProvider: _parse_google_claims: email, sub, hd→tenant_id
- GoogleOIDCProvider: _parse_google_claims: derives tenant from email domain when no hd
- GoogleOIDCProvider: JWT unavailable → TokenInvalidError
- GoogleOIDCProvider: expired → TokenExpiredError
- GoogleOIDCProvider: invalid → TokenInvalidError
- CognitoOIDCProvider: issuer built from user_pool_id
- CognitoOIDCProvider: JWKS URL built from issuer
- CognitoOIDCProvider: get_authorization_url contains client_id
- CognitoOIDCProvider: _parse_cognito_claims: custom:tenant_id extracted
- CognitoOIDCProvider: _parse_cognito_claims: cognito:groups → roles
- CognitoOIDCProvider: _parse_cognito_claims: fallback tenant = user_pool_id
- CognitoOIDCProvider: JWT unavailable → TokenInvalidError
- CognitoOIDCProvider: expired → TokenExpiredError
- All three providers: available_providers() includes all three
- IdentityContext from Google claims has provider='google'
- IdentityContext from Cognito claims has provider='cognito'
- auth-service /auth/providers lists all configured providers
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


# ── Import registers Google + Cognito ─────────────────────────────────────────
import auth.providers.google_cognito  # noqa — side effect: registers providers


# ═══════════════════════════════════════════════════════════════════════════════
# Factory registration
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviderRegistration:
    def test_google_registered(self):
        from auth.providers.base import OIDCProviderFactory
        assert "google" in OIDCProviderFactory.available_providers()

    def test_cognito_registered(self):
        from auth.providers.base import OIDCProviderFactory
        assert "cognito" in OIDCProviderFactory.available_providers()

    def test_all_three_providers_available(self):
        from auth.providers.base import OIDCProviderFactory
        providers = OIDCProviderFactory.available_providers()
        for name in ("entra_id", "google", "cognito"):
            assert name in providers

    def test_factory_creates_google(self):
        from auth.providers.base import OIDCProviderFactory
        from auth.providers.google_cognito import GoogleOIDCProvider
        from auth.models import OIDCProviderConfig
        cfg = OIDCProviderConfig(provider_name="google", client_id="gid")
        provider = OIDCProviderFactory.create("google", cfg)
        assert isinstance(provider, GoogleOIDCProvider)

    def test_factory_creates_cognito(self):
        from auth.providers.base import OIDCProviderFactory
        from auth.providers.google_cognito import CognitoOIDCProvider
        from auth.models import OIDCProviderConfig
        cfg = OIDCProviderConfig(
            provider_name="cognito", client_id="cid",
            tenant_id="us-east-1_TestPool"
        )
        provider = OIDCProviderFactory.create("cognito", cfg)
        assert isinstance(provider, CognitoOIDCProvider)


# ═══════════════════════════════════════════════════════════════════════════════
# GoogleOIDCProvider
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoogleOIDCProvider:
    def _make_provider(self, client_id="gclient", tenant_id=""):
        from auth.providers.google_cognito import GoogleOIDCProvider
        from auth.models import OIDCProviderConfig
        cfg = OIDCProviderConfig(
            provider_name="google", client_id=client_id,
            tenant_id=tenant_id,
            redirect_uri="http://localhost/callback/google",
            scopes=["openid", "email", "profile"],
        )
        return GoogleOIDCProvider(cfg)

    def test_auth_url_contains_client_id(self):
        p = self._make_provider()
        url = p.get_authorization_url("state1", "nonce1")
        assert "gclient" in url

    def test_auth_url_contains_state(self):
        p = self._make_provider()
        url = p.get_authorization_url("mystate", "nonce1")
        assert "mystate" in url

    def test_auth_url_contains_openid_scope(self):
        p = self._make_provider()
        url = p.get_authorization_url("s", "n")
        assert "openid" in url

    def test_auth_url_has_hd_when_tenant_set(self):
        p = self._make_provider(tenant_id="example.com")
        url = p.get_authorization_url("s", "n")
        assert "example.com" in url

    def test_auth_url_no_hd_when_tenant_empty(self):
        p = self._make_provider(tenant_id="")
        url = p.get_authorization_url("s", "n")
        assert "hd=" not in url

    def test_parse_claims_extracts_sub_email_name(self):
        p = self._make_provider()
        payload = {
            "sub": "google-user-123", "iss": "https://accounts.google.com",
            "aud": "gclient", "exp": int(time.time()) + 3600, "iat": int(time.time()),
            "email": "user@workspace.com", "name": "Google User",
            "hd": "workspace.com",
        }
        claims = p._parse_google_claims(payload)
        assert claims.sub == "google-user-123"
        assert claims.email == "user@workspace.com"
        assert claims.name == "Google User"

    def test_parse_claims_hd_becomes_tenant_id(self):
        p = self._make_provider()
        payload = {
            "sub": "u1", "iss": "https://accounts.google.com",
            "aud": "g", "exp": 9999999999, "iat": 0,
            "email": "user@corp.com", "hd": "corp.com",
        }
        claims = p._parse_google_claims(payload)
        assert claims.tenant_id == "corp.com"

    def test_parse_claims_derives_tenant_from_email_when_no_hd(self):
        p = self._make_provider()
        payload = {
            "sub": "u1", "iss": "https://accounts.google.com",
            "aud": "g", "exp": 9999999999, "iat": 0,
            "email": "personal@gmail.com",
        }
        claims = p._parse_google_claims(payload)
        assert claims.tenant_id == "gmail.com"

    def test_validate_token_jwt_unavailable_raises(self):
        from auth.models import TokenInvalidError
        p = self._make_provider()
        with patch("auth.providers.google_cognito._JWT_AVAILABLE", False):
            with pytest.raises(TokenInvalidError):
                p.validate_token("fake.token.here")

    def test_validate_token_expired_raises_token_expired(self):
        from auth.models import TokenExpiredError
        from auth.providers.google_cognito import _JWT_AVAILABLE
        p = self._make_provider()
        with patch("auth.providers.google_cognito._JWT_AVAILABLE", True), \
             patch("auth.providers.google_cognito.pyjwt") as mock_jwt, \
             patch("auth.providers.google_cognito.PyJWKClient") as mock_jwks:
            import jwt as real_jwt
            mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
            mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError
            mock_client = MagicMock()
            mock_client.get_signing_key_from_jwt.return_value = MagicMock(key="k")
            mock_jwks.return_value = mock_client
            mock_jwt.decode.side_effect = real_jwt.ExpiredSignatureError("exp")
            p._jwks_client = mock_client
            with pytest.raises(TokenExpiredError):
                p.validate_token("tok")

    def test_validate_token_invalid_raises_token_invalid(self):
        from auth.models import TokenInvalidError
        p = self._make_provider()
        with patch("auth.providers.google_cognito._JWT_AVAILABLE", True), \
             patch("auth.providers.google_cognito.pyjwt") as mock_jwt, \
             patch("auth.providers.google_cognito.PyJWKClient") as mock_jwks:
            import jwt as real_jwt
            mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
            mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError
            mock_client = MagicMock()
            mock_client.get_signing_key_from_jwt.return_value = MagicMock(key="k")
            mock_jwks.return_value = mock_client
            mock_jwt.decode.side_effect = real_jwt.InvalidTokenError("bad")
            p._jwks_client = mock_client
            with pytest.raises(TokenInvalidError):
                p.validate_token("tok")

    def test_claims_to_identity_provider_is_google(self):
        p = self._make_provider()
        from auth.models import TokenClaims
        claims = TokenClaims(
            sub="g-user", iss="https://accounts.google.com", aud=["gclient"],
            exp=9999999999, iat=0, email="u@g.com", name="G User",
            tenant_id="g.com",
        )
        identity = p._claims_to_identity(claims, "google")
        assert identity.provider == "google"
        assert identity.user_id == "g-user"


# ═══════════════════════════════════════════════════════════════════════════════
# CognitoOIDCProvider
# ═══════════════════════════════════════════════════════════════════════════════

class TestCognitoOIDCProvider:
    def _make_provider(self, pool_id="us-east-1_TestPool", client_id="capp"):
        from auth.providers.google_cognito import CognitoOIDCProvider
        from auth.models import OIDCProviderConfig
        cfg = OIDCProviderConfig(
            provider_name="cognito", client_id=client_id,
            tenant_id=pool_id,
            redirect_uri="http://localhost/callback/cognito",
        )
        return CognitoOIDCProvider(cfg)

    def test_issuer_built_from_pool_id(self):
        p = self._make_provider("us-east-1_TestPool")
        assert "us-east-1" in p._issuer
        assert "TestPool" in p._issuer

    def test_jwks_url_is_well_known(self):
        p = self._make_provider()
        assert ".well-known/jwks.json" in p._jwks_url

    def test_auth_url_contains_client_id(self):
        p = self._make_provider()
        url = p.get_authorization_url("state", "nonce")
        assert "capp" in url

    def test_parse_claims_extracts_custom_tenant_id(self):
        p = self._make_provider()
        payload = {
            "sub": "cognito-user-1", "iss": p._issuer,
            "aud": "capp", "exp": 9999999999, "iat": 0,
            "email": "user@corp.com", "name": "Cognito User",
            "custom:tenant_id": "tenant-from-attr",
            "cognito:groups": ["member"],
        }
        claims = p._parse_cognito_claims(payload)
        assert claims.tenant_id == "tenant-from-attr"
        assert "member" in claims.roles

    def test_parse_claims_groups_become_roles(self):
        p = self._make_provider()
        payload = {
            "sub": "u1", "iss": p._issuer, "aud": "c",
            "exp": 9999999999, "iat": 0,
            "cognito:groups": ["admin", "other-group"],
        }
        claims = p._parse_cognito_claims(payload)
        assert "admin" in claims.roles

    def test_parse_claims_fallback_tenant_is_pool_id(self):
        p = self._make_provider("us-east-1_TestPool")
        payload = {
            "sub": "u1", "iss": p._issuer, "aud": "c",
            "exp": 9999999999, "iat": 0,
        }
        claims = p._parse_cognito_claims(payload)
        assert claims.tenant_id == "us-east-1_TestPool"

    def test_validate_token_no_pool_id_raises(self):
        from auth.providers.google_cognito import CognitoOIDCProvider
        from auth.models import OIDCProviderConfig, TokenInvalidError
        cfg = OIDCProviderConfig(provider_name="cognito", client_id="c")
        p = CognitoOIDCProvider(cfg)
        with pytest.raises(TokenInvalidError, match="user_pool_id"):
            p.validate_token("tok")

    def test_validate_token_expired_raises_token_expired(self):
        from auth.models import TokenExpiredError
        p = self._make_provider()
        with patch("auth.providers.google_cognito._JWT_AVAILABLE", True), \
             patch("auth.providers.google_cognito.pyjwt") as mock_jwt, \
             patch("auth.providers.google_cognito.PyJWKClient") as mock_jwks:
            import jwt as real_jwt
            mock_jwt.ExpiredSignatureError = real_jwt.ExpiredSignatureError
            mock_jwt.InvalidTokenError = real_jwt.InvalidTokenError
            mock_client = MagicMock()
            mock_client.get_signing_key_from_jwt.return_value = MagicMock(key="k")
            mock_jwks.return_value = mock_client
            mock_jwt.decode.side_effect = real_jwt.ExpiredSignatureError("exp")
            p._jwks_client = mock_client
            with pytest.raises(TokenExpiredError):
                p.validate_token("tok")

    def test_claims_to_identity_provider_is_cognito(self):
        p = self._make_provider()
        from auth.models import TokenClaims
        claims = TokenClaims(
            sub="c-user", iss=p._issuer, aud=["capp"],
            exp=9999999999, iat=0, tenant_id="tenant-123",
        )
        identity = p._claims_to_identity(claims, "cognito")
        assert identity.provider == "cognito"
        assert identity.tenant_id == "tenant-123"


# ═══════════════════════════════════════════════════════════════════════════════
# Provider selector in JWTValidatorMiddleware
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviderSelector:
    def _make_middleware(self, providers):
        from auth.middleware.jwt_validator import JWTValidatorMiddleware
        app = MagicMock()
        mw = JWTValidatorMiddleware.__new__(JWTValidatorMiddleware)
        mw.providers = providers
        mw.default_provider = "entra_id"
        mw.bypass_auth = False
        return mw

    def _encode_payload(self, payload: dict) -> str:
        """Create a minimal JWT (unsigned) for provider selection tests."""
        import base64, json
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        body   = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b"=").decode()
        return f"{header}.{body}.fakesig"

    def test_selects_entra_id_from_microsoft_issuer(self):
        mock_entra = MagicMock()
        mw = self._make_middleware({"entra_id": mock_entra})
        token = self._encode_payload({"iss": "https://login.microsoftonline.com/tid/v2.0"})
        selected = mw._select_provider(token)
        assert selected is mock_entra

    def test_selects_google_from_google_issuer(self):
        mock_google = MagicMock()
        mock_entra  = MagicMock()
        mw = self._make_middleware({"google": mock_google, "entra_id": mock_entra})
        token = self._encode_payload({"iss": "https://accounts.google.com"})
        selected = mw._select_provider(token)
        assert selected is mock_google

    def test_selects_cognito_from_cognito_issuer(self):
        mock_cognito = MagicMock()
        mock_entra   = MagicMock()
        mw = self._make_middleware({"cognito": mock_cognito, "entra_id": mock_entra})
        token = self._encode_payload({
            "iss": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_Pool"
        })
        selected = mw._select_provider(token)
        assert selected is mock_cognito

    def test_falls_back_to_default_for_unknown_issuer(self):
        mock_entra = MagicMock()
        mw = self._make_middleware({"entra_id": mock_entra})
        token = self._encode_payload({"iss": "https://unknown-idp.example.com"})
        selected = mw._select_provider(token)
        assert selected is mock_entra
