"""
Unit tests for auth Phase 3 (R7) — Authorization, roles, identity propagation.

Covers:
- RoleEnforcementMiddleware: public paths bypass (no 401)
- RoleEnforcementMiddleware: missing headers → 401 when require_auth=True
- RoleEnforcementMiddleware: missing headers → continues when require_auth=False
- RoleEnforcementMiddleware: valid headers → identity in request.state
- RoleEnforcementMiddleware: /health bypasses regardless of headers
- get_identity: returns identity from request.state
- get_identity: raises 401 when state has no identity
- require_role: admin passes any role check
- require_role: member passes MEMBER check
- require_role: viewer fails MEMBER check → 403
- require_role: member fails ADMIN check → 403
- require_admin / require_member / require_viewer shortcuts exist
- propagate_identity: returns X-User-Id, X-Tenant-Id, X-User-Roles headers
- propagate_identity_from_request: returns headers from request.state.identity
- propagate_identity_from_request: returns {} when no identity
- get_permissions_summary: admin has manage_tenants=True
- get_permissions_summary: member has ingest=True, manage_tenants=False
- get_permissions_summary: viewer has query=True, ingest=False
- GET /auth/permissions: 200 with headers, 401 without
- GET /auth/permissions: returns permissions dict with all keys
- IdentityContext.from_headers: case-insensitive header lookup
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import Depends, FastAPI, Request as FastAPIRequest
from fastapi.testclient import TestClient
from starlette.requests import Request


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_identity(role="member", user_id="u1", tenant_id="t1"):
    from auth.models import IdentityContext, UserRole
    role_map = {"admin": UserRole.ADMIN, "member": UserRole.MEMBER, "viewer": UserRole.VIEWER}
    return IdentityContext(
        user_id=user_id, tenant_id=tenant_id,
        email=f"{user_id}@test.com", name="Test User",
        roles=[role_map[role]], provider="entra_id",
    )


IDENTITY_HEADERS = {
    "X-User-Id":    "user-123",
    "X-Tenant-Id":  "tenant-abc",
    "X-User-Email": "user@corp.com",
    "X-User-Name":  "Test User",
    "X-User-Roles": "member",
    "X-Auth-Provider": "entra_id",
}

ADMIN_HEADERS = {**IDENTITY_HEADERS, "X-User-Roles": "admin"}


# ═══════════════════════════════════════════════════════════════════════════════
# RoleEnforcementMiddleware
# ═══════════════════════════════════════════════════════════════════════════════

def _make_enforcement_app(require_auth: bool = True):
    from auth.middleware.role_enforcement import RoleEnforcementMiddleware
    
    app = FastAPI()
    app.add_middleware(RoleEnforcementMiddleware, require_auth=require_auth)

    @app.get("/health")
    async def health(): return {"status": "ok"}

    @app.get("/protected")
    async def protected(req: FastAPIRequest):
        identity = getattr(req.state, "identity", None)
        return {"tenant": identity.tenant_id if identity else None}

    return TestClient(app, raise_server_exceptions=False)


class TestRoleEnforcementMiddleware:
    def test_health_bypasses_auth(self):
        client = _make_enforcement_app(require_auth=True)
        assert client.get("/health").status_code == 200

    def test_missing_headers_401_when_require_auth(self):
        client = _make_enforcement_app(require_auth=True)
        r = client.get("/protected")
        assert r.status_code == 401

    def test_missing_headers_continues_when_auth_not_required(self):
        client = _make_enforcement_app(require_auth=False)
        r = client.get("/protected")
        assert r.status_code == 200
        assert r.json()["tenant"] is None

    def test_valid_headers_inject_identity(self):
        client = _make_enforcement_app(require_auth=True)
        r = client.get("/protected", headers=IDENTITY_HEADERS)
        assert r.status_code == 200
        assert r.json()["tenant"] == "tenant-abc"

    def test_identity_tenant_correct(self):
        client = _make_enforcement_app(require_auth=True)
        r = client.get("/protected", headers={
            **IDENTITY_HEADERS, "X-Tenant-Id": "my-specific-tenant"
        })
        assert r.json()["tenant"] == "my-specific-tenant"

    def test_root_path_bypasses(self):
        client = _make_enforcement_app(require_auth=True)
        r = client.get("/")
        assert r.status_code in (200, 404)  # not 401


# ═══════════════════════════════════════════════════════════════════════════════
# get_identity dependency
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetIdentityDependency:
    def test_returns_identity_from_state(self):
        from auth.middleware.role_enforcement import get_identity, RoleEnforcementMiddleware
        
        app = FastAPI()
        app.add_middleware(RoleEnforcementMiddleware, require_auth=True)

        @app.get("/me")
        async def me(req: FastAPIRequest, identity=Depends(get_identity)):
            return {"user": identity.user_id}

        client = TestClient(app)
        r = client.get("/me", headers=IDENTITY_HEADERS)
        assert r.status_code == 200
        assert r.json()["user"] == "user-123"

    def test_raises_401_without_identity(self):
        from auth.middleware.role_enforcement import get_identity
        
        app = FastAPI()  # no enforcement middleware

        @app.get("/me")
        async def me(identity=Depends(get_identity)):
            return {"user": identity.user_id}

        client = TestClient(app, raise_server_exceptions=False)
        r = client.get("/me")
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# require_role dependency
# ═══════════════════════════════════════════════════════════════════════════════

def _make_role_app():
    from auth.middleware.role_enforcement import (
        RoleEnforcementMiddleware, require_role
    )
    from auth.models import UserRole
    
    app = FastAPI()
    app.add_middleware(RoleEnforcementMiddleware, require_auth=True)

    @app.get("/admin-only", dependencies=[require_role(UserRole.ADMIN)])
    async def admin_only(): return {"ok": True}

    @app.get("/member-only", dependencies=[require_role(UserRole.MEMBER)])
    async def member_only(): return {"ok": True}

    @app.get("/viewer-ok", dependencies=[require_role(UserRole.VIEWER)])
    async def viewer_ok(): return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


class TestRequireRole:
    def test_admin_passes_admin_check(self):
        client = _make_role_app()
        assert client.get("/admin-only", headers=ADMIN_HEADERS).status_code == 200

    def test_member_fails_admin_check(self):
        client = _make_role_app()
        r = client.get("/admin-only", headers=IDENTITY_HEADERS)
        assert r.status_code == 403

    def test_member_passes_member_check(self):
        client = _make_role_app()
        assert client.get("/member-only", headers=IDENTITY_HEADERS).status_code == 200

    def test_viewer_fails_member_check(self):
        client = _make_role_app()
        viewer_headers = {**IDENTITY_HEADERS, "X-User-Roles": "viewer"}
        r = client.get("/member-only", headers=viewer_headers)
        assert r.status_code == 403

    def test_admin_passes_member_check(self):
        """Admin implicitly satisfies any lower role requirement."""
        client = _make_role_app()
        assert client.get("/member-only", headers=ADMIN_HEADERS).status_code == 200

    def test_viewer_passes_viewer_check(self):
        client = _make_role_app()
        viewer_headers = {**IDENTITY_HEADERS, "X-User-Roles": "viewer"}
        assert client.get("/viewer-ok", headers=viewer_headers).status_code == 200

    def test_403_response_has_detail_field(self):
        client = _make_role_app()
        r = client.get("/admin-only", headers=IDENTITY_HEADERS)
        assert "detail" in r.json()
        assert "admin" in r.json()["detail"].lower()


class TestRoleShortcuts:
    def test_require_admin_exists(self):
        from auth.middleware.role_enforcement import require_admin
        assert require_admin is not None

    def test_require_member_exists(self):
        from auth.middleware.role_enforcement import require_member
        assert require_member is not None

    def test_require_viewer_exists(self):
        from auth.middleware.role_enforcement import require_viewer
        assert require_viewer is not None


# ═══════════════════════════════════════════════════════════════════════════════
# propagate_identity
# ═══════════════════════════════════════════════════════════════════════════════

class TestPropagateIdentity:
    def test_returns_x_user_id(self):
        from auth.middleware.role_enforcement import propagate_identity
        identity = make_identity(user_id="user-xyz")
        headers = propagate_identity(identity)
        assert headers.get("X-User-Id") == "user-xyz"

    def test_returns_x_tenant_id(self):
        from auth.middleware.role_enforcement import propagate_identity
        identity = make_identity(tenant_id="my-tenant")
        headers = propagate_identity(identity)
        assert headers.get("X-Tenant-Id") == "my-tenant"

    def test_returns_x_user_roles(self):
        from auth.middleware.role_enforcement import propagate_identity
        identity = make_identity(role="admin")
        headers = propagate_identity(identity)
        assert "admin" in headers.get("X-User-Roles", "")

    def test_propagate_from_request_with_identity(self):
        from auth.middleware.role_enforcement import propagate_identity_from_request
        req = MagicMock()
        req.state.identity = make_identity(user_id="req-user", tenant_id="req-tenant")
        headers = propagate_identity_from_request(req)
        assert headers["X-User-Id"] == "req-user"
        assert headers["X-Tenant-Id"] == "req-tenant"

    def test_propagate_from_request_no_identity_returns_empty(self):
        from auth.middleware.role_enforcement import propagate_identity_from_request
        req = MagicMock()
        req.state.identity = None
        headers = propagate_identity_from_request(req)
        assert headers == {}

    def test_propagate_from_request_missing_state_returns_empty(self):
        from auth.middleware.role_enforcement import propagate_identity_from_request
        # Use object() instead of MagicMock — no state attribute at all
        class FakeReq:
            pass
        headers = propagate_identity_from_request(FakeReq())
        assert headers == {}


# ═══════════════════════════════════════════════════════════════════════════════
# get_permissions_summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestPermissionsSummary:
    def test_admin_has_manage_tenants(self):
        from auth.middleware.role_enforcement import get_permissions_summary
        summary = get_permissions_summary(make_identity("admin"))
        assert summary["permissions"]["manage_tenants"] is True

    def test_admin_has_view_all_traces(self):
        from auth.middleware.role_enforcement import get_permissions_summary
        summary = get_permissions_summary(make_identity("admin"))
        assert summary["permissions"]["view_all_traces"] is True

    def test_member_has_ingest_true(self):
        from auth.middleware.role_enforcement import get_permissions_summary
        summary = get_permissions_summary(make_identity("member"))
        assert summary["permissions"]["ingest"] is True

    def test_member_has_manage_tenants_false(self):
        from auth.middleware.role_enforcement import get_permissions_summary
        summary = get_permissions_summary(make_identity("member"))
        assert summary["permissions"]["manage_tenants"] is False

    def test_viewer_has_query_true(self):
        from auth.middleware.role_enforcement import get_permissions_summary
        summary = get_permissions_summary(make_identity("viewer"))
        assert summary["permissions"]["query"] is True

    def test_viewer_has_ingest_false(self):
        from auth.middleware.role_enforcement import get_permissions_summary
        summary = get_permissions_summary(make_identity("viewer"))
        assert summary["permissions"]["ingest"] is False

    def test_summary_has_all_required_keys(self):
        from auth.middleware.role_enforcement import get_permissions_summary
        summary = get_permissions_summary(make_identity("member"))
        for key in ["user_id", "tenant_id", "roles", "is_admin", "can_write", "permissions"]:
            assert key in summary


# ═══════════════════════════════════════════════════════════════════════════════
# GET /auth/permissions endpoint
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def auth_client():
    from auth.main import app
    app.state.providers = {}
    return TestClient(app, raise_server_exceptions=False)


class TestPermissionsEndpoint:
    def test_permissions_without_headers_401(self, auth_client):
        r = auth_client.get("/auth/permissions")
        assert r.status_code == 401

    def test_permissions_with_headers_200(self, auth_client):
        r = auth_client.get("/auth/permissions", headers=IDENTITY_HEADERS)
        assert r.status_code == 200

    def test_permissions_returns_user_id(self, auth_client):
        r = auth_client.get("/auth/permissions", headers=IDENTITY_HEADERS)
        assert r.json()["user_id"] == "user-123"

    def test_permissions_returns_tenant_id(self, auth_client):
        r = auth_client.get("/auth/permissions", headers=IDENTITY_HEADERS)
        assert r.json()["tenant_id"] == "tenant-abc"

    def test_permissions_has_permissions_dict(self, auth_client):
        r = auth_client.get("/auth/permissions", headers=IDENTITY_HEADERS)
        assert "permissions" in r.json()

    def test_permissions_member_cannot_manage_tenants(self, auth_client):
        r = auth_client.get("/auth/permissions", headers=IDENTITY_HEADERS)
        assert r.json()["permissions"]["manage_tenants"] is False

    def test_permissions_admin_can_manage_tenants(self, auth_client):
        r = auth_client.get("/auth/permissions", headers=ADMIN_HEADERS)
        assert r.json()["permissions"]["manage_tenants"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# IdentityContext header case-insensitivity
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdentityContextCaseInsensitive:
    def test_lowercase_headers_parsed(self):
        from auth.models import IdentityContext, UserRole
        lc_headers = {k.lower(): v for k, v in IDENTITY_HEADERS.items()}
        identity = IdentityContext.from_headers(lc_headers)
        assert identity.user_id == "user-123"
        assert identity.tenant_id == "tenant-abc"

    def test_uppercase_headers_parsed(self):
        from auth.models import IdentityContext
        identity = IdentityContext.from_headers(IDENTITY_HEADERS)
        assert identity.user_id == "user-123"

    def test_roles_parsed_correctly(self):
        from auth.models import IdentityContext, UserRole
        identity = IdentityContext.from_headers({
            "x-user-id": "u", "x-tenant-id": "t", "x-user-roles": "admin,member"
        })
        assert UserRole.ADMIN in identity.roles
        assert UserRole.MEMBER in identity.roles
