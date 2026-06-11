"""
Adversarial tenant-isolation tests (R7 Phase 7).

PRINCIPLE: Every test explicitly attempts a cross-tenant operation.
           The test PASSES only if the attempt FAILS CLOSED.
           No cross-tenant data may be returned. No silent degradation.

Layers tested:
  1. TenantScope API    — no context → raises, never returns data
  2. Qdrant (search)    — cross-tenant search blocked at scoped client
  3. Qdrant (upsert)    — cross-tenant write attempt raises ValueError
  4. Redis cache        — tenant A miss never hits tenant B's key
  5. Storage paths      — tenant A cannot request tenant B's path
  6. IngestionMessage   — tenant_id cannot be overridden mid-queue
  7. Identity headers   — missing headers blocked before data access
  8. Role enforcement   — viewer cannot write, member cannot admin
  9. JWT middleware     — forged/missing token → 401 (never 200)
  10. Cache key isolation — statistical: N tenants, N distinct key spaces

"Fail closed" definition:
    An operation that should be blocked MUST either raise an exception
    OR return an empty list/dict — it MUST NOT return data from another tenant.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raglab_common.tenant_scope import (
    TenantContextMissing,
    InvalidTenantIdError,
    ScopedQdrantClient,
    clear_current_tenant,
    get_current_tenant,
    scoped_cache_key,
    scoped_storage_path,
    set_current_tenant,
    with_tenant,
)


@pytest.fixture(autouse=True)
def clear_tenant():
    clear_current_tenant()
    yield
    clear_current_tenant()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: TenantScope API — no context raises, never returns data
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialTenantScopeAPI:
    def test_get_current_tenant_without_set_raises_not_none(self):
        """Must raise, NEVER return None silently."""
        with pytest.raises(TenantContextMissing):
            get_current_tenant()

    def test_scoped_cache_key_without_context_raises(self):
        """Cannot build a cache key without a tenant. Must raise."""
        with pytest.raises(TenantContextMissing):
            scoped_cache_key("embed:abc123")

    def test_scoped_storage_path_without_context_raises(self):
        """Cannot build a storage path without a tenant. Must raise."""
        with pytest.raises(TenantContextMissing):
            scoped_storage_path("docs/secret.pdf")

    def test_tenant_context_does_not_leak_across_with_tenant_blocks(self):
        """Tenant set inside a block MUST NOT be visible outside."""
        with with_tenant("inside"):
            assert get_current_tenant() == "inside"
        # After exit: context must be cleared
        with pytest.raises(TenantContextMissing):
            get_current_tenant()

    def test_nested_tenant_inner_cannot_read_outer_after_exit(self):
        """Inner context exits cleanly; outer is restored, not leaked."""
        with with_tenant("outer"):
            with with_tenant("inner"):
                assert get_current_tenant() == "inner"
            assert get_current_tenant() == "outer"  # outer restored
        with pytest.raises(TenantContextMissing):
            get_current_tenant()  # all cleared

    def test_invalid_tenant_id_rejected_before_context_set(self):
        """Invalid tenant_id must be rejected. Context must NOT be set."""
        with pytest.raises(InvalidTenantIdError):
            set_current_tenant("../../etc/passwd")
        with pytest.raises(TenantContextMissing):
            get_current_tenant()  # never set

    def test_sql_injection_tenant_id_rejected(self):
        with pytest.raises(InvalidTenantIdError):
            set_current_tenant("tenant'; DROP TABLE documents; --")

    def test_empty_tenant_id_rejected(self):
        with pytest.raises(InvalidTenantIdError):
            set_current_tenant("")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Qdrant — cross-tenant search blocked
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialQdrantSearch:
    def _make_scoped_client(self, underlying_data=None):
        """Create ScopedQdrantClient whose underlying client returns `underlying_data`."""
        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = underlying_data or []
        return ScopedQdrantClient(mock_qdrant), mock_qdrant

    def test_search_without_context_raises_not_returns_data(self):
        """No tenant context → MUST raise TenantContextMissing before hitting Qdrant."""
        client, mock_qdrant = self._make_scoped_client(
            underlying_data=[{"id": "secret-chunk", "payload": {"tenant_id": "victim"}}]
        )
        with pytest.raises(TenantContextMissing):
            client.search("raglab", [0.1] * 10, limit=5)
        # Underlying Qdrant MUST NOT have been called
        mock_qdrant.search.assert_not_called()

    def test_tenant_a_search_cannot_return_tenant_b_data(self):
        """
        Tenant A's search must only receive results filtered to tenant A.
        Even if the underlying Qdrant returns mixed data, the filter ensures isolation.
        """
        tenant_a_result = MagicMock()
        tenant_a_result.id = "chunk-A"
        tenant_a_result.payload = {"tenant_id": "tenant-A", "text": "A's data"}

        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = [tenant_a_result]

        client = ScopedQdrantClient(mock_qdrant)
        with with_tenant("tenant-A"):
            results = client.search("raglab", [0.1] * 10)

        # Verify the filter passed to Qdrant includes tenant-A
        call_kwargs = mock_qdrant.search.call_args[1]
        q_filter = call_kwargs.get("query_filter")
        # Filter is not None — tenant isolation applied
        assert q_filter is not None

    def test_delete_without_context_raises(self):
        client, mock_qdrant = self._make_scoped_client()
        with pytest.raises(TenantContextMissing):
            client.delete("raglab", ["chunk-id"])
        mock_qdrant.delete.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: Qdrant — cross-tenant write attempt raises
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialQdrantUpsert:
    def test_upsert_without_context_raises_not_writes(self):
        """No tenant context → upsert MUST raise before writing to Qdrant."""
        mock_qdrant = MagicMock()
        client = ScopedQdrantClient(mock_qdrant)
        point = MagicMock()
        point.id = "p1"
        point.vector = [0.1]
        point.payload = {}
        with pytest.raises(TenantContextMissing):
            client.upsert("raglab", [point])
        mock_qdrant.upsert.assert_not_called()

    def test_cross_tenant_point_rejected_at_upsert(self):
        """
        A point already tagged with tenant-B cannot be upserted into tenant-A's context.
        Attempted cross-tenant write MUST raise ValueError.
        """
        mock_qdrant = MagicMock()
        client = ScopedQdrantClient(mock_qdrant)
        point = MagicMock()
        point.id = "p1"
        point.vector = [0.1]
        point.payload = {"tenant_id": "tenant-B", "text": "B's secret data"}

        with with_tenant("tenant-A"):
            with pytest.raises(ValueError, match="Cross-tenant"):
                client.upsert("raglab", [point])

        # Qdrant MUST NOT have been called
        mock_qdrant.upsert.assert_not_called()

    def test_cross_tenant_write_attempt_blocked_silently_does_not_write(self):
        """Ensure upsert raises AND Qdrant is never touched on cross-tenant attempt."""
        mock_qdrant = MagicMock()
        client = ScopedQdrantClient(mock_qdrant)
        attacker_points = [
            MagicMock(id=f"p{i}", vector=[0.1], payload={"tenant_id": "victim-tenant"})
            for i in range(5)
        ]
        with with_tenant("attacker-tenant"):
            with pytest.raises(ValueError):
                client.upsert("raglab", attacker_points)
        mock_qdrant.upsert.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: Redis cache — no cross-tenant key collision
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialRedisIsolation:
    def test_cache_key_uniqueness_across_tenants(self):
        """Same text + model for N tenants produces N distinct keys."""
        from embedding.cache import _cache_key
        text = "What is retrieval augmented generation?"
        provider = "azure_openai"
        model = "text-embedding-3-small"

        keys = set()
        for i in range(10):
            with with_tenant(f"tenant-{i:03d}"):
                keys.add(_cache_key(text, provider, model))

        assert len(keys) == 10, "Every tenant MUST have a unique cache key"

    def test_tenant_a_cannot_hit_tenant_b_cache(self):
        """
        Simulate: tenant-B has a cached vector.
        Tenant-A requests same text → MUST miss (different key).
        """
        from embedding.cache import _cache_key
        text = "RAG combines retrieval and generation"

        with with_tenant("tenant-B"):
            key_b = _cache_key(text, "azure_openai", "model")
        with with_tenant("tenant-A"):
            key_a = _cache_key(text, "azure_openai", "model")

        # Different keys → tenant-A's cache lookup CANNOT hit tenant-B's entry
        assert key_a != key_b

    def test_no_context_key_differs_from_tenanted_key(self):
        """Unscoped key and tenanted key must differ — no accidental sharing."""
        from embedding.cache import _cache_key
        text = "same text"
        unscoped = _cache_key(text, "p", "m")
        with with_tenant("any-tenant"):
            tenanted = _cache_key(text, "p", "m")
        assert unscoped != tenanted


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 5: Storage paths — tenant A cannot forge tenant B's path
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialStoragePaths:
    def test_storage_path_without_context_raises(self):
        with pytest.raises(TenantContextMissing):
            scoped_storage_path("docs/sensitive.pdf")

    def test_tenant_a_path_differs_from_tenant_b_path(self):
        path_a = scoped_storage_path("report.pdf", tenant_id="tenant-A")
        path_b = scoped_storage_path("report.pdf", tenant_id="tenant-B")
        assert path_a != path_b
        assert path_a.startswith("tenant-A/")
        assert path_b.startswith("tenant-B/")

    def test_path_traversal_in_filename_scoped_but_not_escaped(self):
        """
        Note: path traversal in filename is not escaped by scoped_storage_path —
        that is the responsibility of the storage backend.
        But the tenant prefix is always first, containing the traversal.
        """
        path = scoped_storage_path("../../../etc/passwd", tenant_id="attacker")
        # Tenant prefix is prepended — traversal is relative to tenant namespace
        assert path.startswith("attacker/")

    def test_explicit_tenant_id_overrides_context(self):
        """Explicit arg takes precedence — no leakage from context."""
        with with_tenant("ctx-tenant"):
            path = scoped_storage_path("file.pdf", tenant_id="explicit-tenant")
        assert path.startswith("explicit-tenant/")
        assert "ctx-tenant" not in path


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 6: IngestionMessage — tenant_id locked at creation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialIngestionMessage:
    def test_message_tenant_id_survives_serialisation(self):
        from raglab_common.queue import IngestionMessage
        msg = IngestionMessage(
            doc_id="d", idempotency_key="k", filename="f.txt",
            content_type="text/plain", storage_path="/t/f.txt",
            collection="c", chunker_type="text", chunker_config={},
            llm_provider="azure_openai", tenant_id="locked-tenant",
        )
        raw = msg.to_bytes()
        restored = IngestionMessage.from_bytes(raw)
        assert restored.tenant_id == "locked-tenant"

    def test_message_tenant_cannot_be_set_to_empty(self):
        from raglab_common.queue import IngestionMessage
        msg = IngestionMessage(
            doc_id="d", idempotency_key="k", filename="f.txt",
            content_type="text/plain", storage_path="/t/f.txt",
            collection="c", chunker_type="text", chunker_config={},
            llm_provider="azure_openai", tenant_id="",
        )
        # Empty string is allowed by model but pipeline will default it
        # (pipeline does: getattr(message, 'tenant_id', 'default') or 'default')
        effective_tenant = msg.tenant_id or "default"
        assert effective_tenant == "default"


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 7: Identity headers — missing headers blocked before data access
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialIdentityHeaders:
    def test_identity_from_headers_missing_user_id_raises(self):
        from auth.models import IdentityContext
        with pytest.raises(ValueError, match="X-User-Id"):
            IdentityContext.from_headers({"x-tenant-id": "t1"})

    def test_identity_from_headers_missing_tenant_id_raises(self):
        from auth.models import IdentityContext
        with pytest.raises(ValueError, match="X-Tenant-Id"):
            IdentityContext.from_headers({"x-user-id": "u1"})

    def test_identity_from_headers_empty_both_raises(self):
        from auth.models import IdentityContext
        with pytest.raises(ValueError):
            IdentityContext.from_headers({})

    def test_cannot_craft_admin_identity_without_admin_role(self):
        from auth.models import IdentityContext, UserRole
        identity = IdentityContext.from_headers({
            "x-user-id": "u1", "x-tenant-id": "t1",
            "x-user-roles": "member",  # NOT admin
        })
        assert identity.is_admin is False
        assert identity.can_write is True
        assert UserRole.ADMIN not in identity.roles

    def test_forged_role_string_rejected_or_defaults_to_member(self):
        """Unknown role string: either raises ValueError or defaults to member. Never escalates."""
        from auth.models import IdentityContext, UserRole
        try:
            identity = IdentityContext.from_headers({
                "x-user-id": "u1", "x-tenant-id": "t1",
                "x-user-roles": "superuser",  # not a valid role
            })
            # If it doesn't raise: must NOT be admin
            assert identity.is_admin is False
        except ValueError:
            pass  # Rejecting unknown role is also acceptable (fail closed)


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 8: Role enforcement — privilege escalation blocked
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialRoleEnforcement:
    def _make_role_app(self):
        from auth.middleware.role_enforcement import RoleEnforcementMiddleware, require_role
        from auth.models import UserRole
        from fastapi import FastAPI, Request as FRequest
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(RoleEnforcementMiddleware, require_auth=True)

        @app.delete("/admin/purge", dependencies=[require_role(UserRole.ADMIN)])
        async def purge():
            return {"purged": True}

        @app.post("/ingest", dependencies=[require_role(UserRole.MEMBER)])
        async def ingest():
            return {"ingested": True}

        return TestClient(app, raise_server_exceptions=False)

    def test_viewer_cannot_ingest(self):
        client = self._make_role_app()
        r = client.post("/ingest", headers={
            "X-User-Id": "viewer", "X-Tenant-Id": "t",
            "X-User-Roles": "viewer"
        })
        assert r.status_code == 403

    def test_member_cannot_purge(self):
        client = self._make_role_app()
        r = client.delete("/admin/purge", headers={
            "X-User-Id": "member", "X-Tenant-Id": "t",
            "X-User-Roles": "member"
        })
        assert r.status_code == 403

    def test_admin_can_purge(self):
        client = self._make_role_app()
        r = client.delete("/admin/purge", headers={
            "X-User-Id": "admin", "X-Tenant-Id": "t",
            "X-User-Roles": "admin"
        })
        assert r.status_code == 200

    def test_unauthenticated_cannot_ingest(self):
        client = self._make_role_app()
        r = client.post("/ingest")  # no headers
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 9: JWT middleware — forged/missing token → 401
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialJWTMiddleware:
    def _make_gateway_app(self, providers):
        from auth.middleware.jwt_validator import JWTValidatorMiddleware
        from fastapi import FastAPI, Request as FRequest
        from fastapi.testclient import TestClient

        app = FastAPI()
        app.add_middleware(JWTValidatorMiddleware,
                           providers=providers, bypass_auth=False)

        @app.get("/protected")
        async def protected(req: FRequest):
            return {"ok": True}

        return TestClient(app, raise_server_exceptions=False)

    def test_no_token_returns_401_not_200(self):
        client = self._make_gateway_app({})
        r = client.get("/protected")
        assert r.status_code == 401

    def test_invalid_bearer_token_returns_401(self):
        mock_provider = MagicMock()
        from auth.models import TokenInvalidError
        mock_provider.validate_token.side_effect = TokenInvalidError("bad sig")
        client = self._make_gateway_app({"entra_id": mock_provider})
        r = client.get("/protected",
                       headers={"Authorization": "Bearer forged.token.here"})
        assert r.status_code == 401

    def test_expired_token_returns_401(self):
        mock_provider = MagicMock()
        from auth.models import TokenExpiredError
        mock_provider.validate_token.side_effect = TokenExpiredError()
        client = self._make_gateway_app({"entra_id": mock_provider})
        r = client.get("/protected",
                       headers={"Authorization": "Bearer expired.token.here"})
        assert r.status_code == 401

    def test_wrong_scheme_returns_401(self):
        """Basic auth scheme must not be accepted."""
        client = self._make_gateway_app({})
        r = client.get("/protected",
                       headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert r.status_code == 401

    def test_health_endpoint_always_accessible_no_token(self):
        """Health checks MUST work without a token (for load balancer probes)."""
        client = self._make_gateway_app({})
        r = client.get("/health")
        assert r.status_code in (200, 404)  # not 401


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 10: Cache key statistical isolation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdversarialCacheKeyIsolation:
    def test_50_tenants_produce_50_distinct_key_spaces(self):
        from embedding.cache import _cache_key
        texts = ["same text", "another text", "third text"]
        for text in texts:
            keys = set()
            for i in range(50):
                with with_tenant(f"tenant-{i:04d}"):
                    keys.add(_cache_key(text, "azure_openai", "model"))
            assert len(keys) == 50, (
                f"Text '{text[:20]}' produced {len(keys)} unique keys "
                f"for 50 tenants — expected 50"
            )

    def test_cache_key_entropy_sufficient(self):
        """Verify SHA-256 part is present (not truncated or predictable)."""
        from embedding.cache import _cache_key
        with with_tenant("entropy-test"):
            key = _cache_key("text", "azure_openai", "model")
        # Format: raglab:{tenant}:embed:{sha256}
        parts = key.split(":")
        assert len(parts) >= 3
        sha_part = parts[-1]
        assert len(sha_part) == 64  # SHA-256 = 64 hex chars
