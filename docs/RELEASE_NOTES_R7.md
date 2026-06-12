# RAGLab R7 — Release Notes

**Version:** 1.0.0 · **Theme:** Auth + Multi-tenancy + GCP (Enterprise-ready) · **Date:** June 2026  
**Builds on:** R1–R6 · **This is the final planned release.**

---

## Summary

Release 7 makes RAGLab enterprise-ready. Three identity providers (Microsoft Entra ID, Google, AWS Cognito) authenticate users via OIDC, with JWT validated once at the gateway and identity context propagated to all services. Tenant isolation is enforced at every data layer — Qdrant, Postgres, Redis, storage, queue — via a single centralized scoping dependency. 39 adversarial tests explicitly attempt cross-tenant attacks; all fail closed. GCP CI/CD and Terraform are activated, making RAGLab genuinely tri-cloud.

After R7: 13 services, 4 internal packages, tri-cloud, multi-tenant, observable, self-healing, authenticated. Platform tagged **v1.0.0**.

---

## Stats

| Metric | Value |
|--------|-------|
| New tests (R7) | 255 |
| Total tests passing | 2,082 |
| raglab-common version | 0.3.0 |
| raglab-eval version | 0.2.0 |
| raglab-chunkers version | 0.6.0 |
| raglab-retrievers version | 0.6.0 |
| auth-service version | 0.2.0 |
| OIDC providers | 3 (Entra ID, Google, Cognito) |
| Adversarial tests | 39 |
| Layers enforced | 10 |
| Clouds | 3 (Azure + AWS + GCP) |

---

## What Shipped

### Phase 1 — auth-service: Entra ID + JWT gateway

`auth-service v0.2.0` (was a stub since R1).

`OIDCProviderBase` — abstract interface: `validate_token()`, `get_authorization_url()`, `exchange_code()`.

`EntraIDProvider` (Microsoft Entra ID):
- JWKS fetched from `login.microsoftonline.com/{tenant}/discovery/v2.0/keys`, cached 1h
- Key rotation: re-fetched once on `PyJWKClientError` without service restart
- Issuer verification: accepts `login.microsoftonline.com/*/v2.0`
- Claims: `oid` → user_id, `tid` → tenant_id, `roles` → roles

`JWTValidatorMiddleware` at API Gateway:
- **Single validation point.** Downstream services never re-validate JWTs.
- Public paths bypass: `/health`, `/docs`, `/openapi.json`, `/redoc`, `/auth/*`
- Peek JWT issuer (no sig check) → route to correct provider
- On success: `request.state.identity = IdentityContext`; identity headers injected into response
- On failure: 401/403 JSON with `detail` + `type` fields

`IdentityContext`: `user_id`, `tenant_id`, `email`, `name`, `roles`, `provider`. `to_headers()` / `from_headers()`.

---

### Phase 2 — Google + AWS Cognito

`GoogleOIDCProvider`: JWKS from `googleapis.com/oauth2/v3/certs`. `hd` claim → tenant_id (hosted domain). Falls back to email domain for personal accounts.

`CognitoOIDCProvider`: Issuer built from `user_pool_id`. JWKS at `{issuer}/.well-known/jwks.json`. `custom:tenant_id` attribute → tenant_id. `cognito:groups` → roles.

`OIDCProviderFactory.register()` — extensible for future providers.

`JWTValidatorMiddleware._select_provider()`: peek issuer → route: `microsoftonline.com` → Entra, `accounts.google.com` → Google, `amazonaws.com/cognito` → Cognito.

---

### Phase 3 — Authorization + role enforcement + identity propagation

`UserRole`: `ADMIN > MEMBER > VIEWER`. Admins implicitly pass any lower role check.

`RoleEnforcementMiddleware` (downstream services):
- Reconstructs `IdentityContext` from gateway-injected headers
- Never validates JWTs — trusts gateway
- `require_auth=False` for internal/dev paths

`require_role(*roles)` → `FastAPI Depends()`: raises 403 with role name in detail.

`propagate_identity(identity)` → headers dict for outbound proxy calls.

`GET /auth/permissions`: full permissions summary — `ingest`, `query`, `manage_tenants`, `view_all_traces`, `delete_docs`, `manage_users`.

---

### Phase 4 — Centralized tenant-scoping layer

`raglab_common/tenant_scope.py` — the single enforcement point.

**The cardinal rule:** if each service writes its own filter, one missed filter = a data leak. Centralize the enforcement, test it adversarially.

`TenantContextMissing` — raised when a scoped operation has no tenant context. Fail closed, never silently degrade.

`with_tenant(tenant_id)` — context manager: set tenant for a block, restore on exit. Nested contexts restore correctly.

`ScopedQdrantClient` — wraps Qdrant; injects `tenant_id` filter on every `search()`, injects `tenant_id` into every point payload on `upsert()`. Cross-tenant write attempt (point.tenant_id != current) → `ValueError`.

`scoped_cache_key(suffix)` → `raglab:{tenant_id}:{suffix}` — tenant-isolated Redis keys.

`scoped_storage_path(path)` → `{tenant_id}/{path}` — tenant-prefixed S3/Blob/GCS paths.

---

### Phase 5 — Tenancy across Postgres + Qdrant + storage

- `ChunkModel`, `QueryModel`, `IngestionMessage`: `tenant_id` + `user_id` fields (default `"default"` for backward compat)
- `DocumentRecord`, `ChunkRecord`: `tenant_id` column (non-nullable, indexed)
- `QdrantIndexer.upsert_chunks()`: tenant_id injected into every Qdrant payload
- Storage upload: key prefixed with `{tenant_id}/` when identity present

---

### Phase 6 — Tenancy across RabbitMQ + Redis + observability

- `pipeline/runner.py`: `set_current_tenant(message.tenant_id)` at start of every job
- Embedding cache keys: `raglab:{tenant_id}:embed:{sha256}` — no cross-tenant cache hits
- `IngestionMessage` tenant_id survives AMQP `to_bytes()/from_bytes()` round-trip
- Observability: admin sees all traces; members scoped to their tenant

---

### Phase 7 — Adversarial tenant-isolation tests (39 tests)

Every test explicitly attempts a cross-tenant attack. Passes only if the attack fails closed.

**10 layers tested:**

| Layer | What was attempted | Result |
|-------|-------------------|--------|
| TenantScope API | Call without context, SQL injection, path traversal in tenant_id | All raise |
| Qdrant search | Search without context; filter not injected | Raises before Qdrant called |
| Qdrant upsert | Cross-tenant point (payload.tenant_id = victim) | ValueError; Qdrant never called |
| Redis cache | tenant-A attempts to hit tenant-B's cache entry | Different keys → guaranteed miss |
| Storage paths | tenant-A requests tenant-B's file path | Different prefixed paths |
| IngestionMessage | tenant_id override attempt mid-queue | Locked at serialisation |
| Identity headers | Missing X-User-Id / X-Tenant-Id | ValueError before data access |
| Role enforcement | viewer→ingest, member→admin, no-auth→any | 403/401 as appropriate |
| JWT middleware | No token, invalid token, expired, wrong scheme | 401; health always accessible |
| Cache isolation | 50 tenants × 3 texts = 150 distinct key spaces | 50 unique keys per text, 0 collisions |

---

### Phase 8 — GCP CI/CD + Terraform

**cd-gcp.yml** (activated from R4 stub):
- Workload Identity Federation — no static service account keys
- Matrix build → Google Artifact Registry (`{region}-docker.pkg.dev/{project}/raglab/{service}:{sha}`)
- Terraform plan + apply with GCS remote state backend
- `kubectl set image` for all 12 services + rollout status wait
- Smoke test after deploy

**infra/terraform/gcp/main.tf** (activated from R4 stub):
- GKE Autopilot (serverless node management)
- Cloud SQL PostgreSQL 15 (REGIONAL HA in prod, private IP, IAM auth)
- Memorystore Redis 7.0 (STANDARD_HA in prod, `allkeys-lru`)
- Workload Identity bindings (no static keys in pods)
- Secret Manager for Redis URL and DB credentials
- GCS Terraform state backend (remote, not local)

RAGLab is now tri-cloud: **Azure AKS + AWS EKS + GCP GKE**.

---

## Interview Angles

**"How do you guarantee tenant isolation?"** Enforce at the data layer (Qdrant filter, Postgres WHERE, storage prefix, Redis key prefix, queue field), not the UI. Centralize in one shared dependency — one missed filter per service becomes zero missed filters when enforcement is in the library. Test adversarially: 39 tests explicitly attempt cross-tenant attacks; if any pass, the test fails.

**"Collection-per-tenant vs shared collection?"** Hard isolation vs operational simplicity. At 800+ tenants, collection-per-tenant means 800+ Qdrant collections to manage, backup, and monitor. Shared collection + payload filter enforced centrally scales cleanly. The risk (filter never omitted) is mitigated by centralizing in `ScopedQdrantClient` — services can't forget a filter they never write.

**"JWT validated at gateway?"** Single enforcement point. Downstream services trust verified, injected identity headers instead of each re-validating — less duplication, fewer ways to get it wrong. Same "centralize the cross-cutting concern" principle as tracing (R6) and tenancy (R7).

**"Carrying nullable tenant fields from R1?"** Forward-compatible schema design. Adding `tenant_id` as nullable from day one meant R7 activated isolation without a schema migration nightmare. The `"default"` backward-compat value means existing data keeps working.

**"OIDC across three providers?"** One abstraction (`OIDCProviderBase`), three implementations (Entra ID, Google, Cognito). Provider-specific config, common token handling. Cloud-agnostic identity mirrors the storage/vector/LLM abstractions throughout.

---

## 7-Release Summary

| Release | Theme | Tests |
|---------|-------|-------|
| R1 | Full Shell + Core Pipeline | 369 |
| R2 | Advanced Chunking + Cloud Storage | 564 |
| R3 | Retrieval Power + CI/CD | 796 |
| R4 | Graph RAG + Advanced Document Types | 1,287 |
| R5 | Self-Healing RAG + Cost Efficiency | 1,558 |
| R6 | Observability / LLMOps | 1,811 |
| **R7** | **Auth + Multi-tenancy + GCP** | **2,082** |

**Platform complete: v1.0.0**

13 services · 4 internal packages · tri-cloud · multi-tenant · observable · self-healing · authenticated

*Built by [Tamal Kundu](https://tamalkundu.com) · Kundu Corp · June 2026*
