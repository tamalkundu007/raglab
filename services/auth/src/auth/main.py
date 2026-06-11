"""
auth-service — R7 activated.

Provides OIDC authentication for RAGLab:
  - Microsoft Entra ID (Phase 1)
  - Google + AWS Cognito (Phase 2)

Endpoints:
  GET  /health                       — liveness
  GET  /                             — service info
  GET  /auth/providers               — list available providers
  GET  /auth/login/{provider}        — initiate OIDC login
  GET  /auth/callback/{provider}     — handle OIDC callback
  GET  /auth/me                      — current user identity
  POST /auth/logout                  — logout

JWT validation middleware lives in auth.middleware.jwt_validator.
The gateway imports JWTValidatorMiddleware directly — it does not
proxy through auth-service for each request (that would add latency).
Auth-service handles: login flows, token exchange, /me, provider management.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel
from raglab_common.tracing import configure_tracing, make_trace_middleware
from auth.routers.auth import router as auth_router

log = get_logger(__name__)
configure_logging(level="info", json_logs=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load provider configs from environment
    app.state.providers = {}

    configure_tracing(
        service_name="auth",
        postgres_dsn=os.getenv("RAGLAB_TRACING_POSTGRES_DSN", ""),
        enabled=True,
    )

    # Entra ID — activated if env vars present
    entra_client_id = os.getenv("RAGLAB_ENTRA_CLIENT_ID", "")
    entra_tenant    = os.getenv("RAGLAB_ENTRA_TENANT_ID", "common")
    if entra_client_id:
        from auth.providers.base import OIDCProviderFactory
        from auth.models import OIDCProviderConfig
        cfg = OIDCProviderConfig(
            provider_name="entra_id",
            client_id=entra_client_id,
            client_secret=os.getenv("RAGLAB_ENTRA_CLIENT_SECRET", ""),
            tenant_id=entra_tenant,
            redirect_uri=os.getenv("RAGLAB_ENTRA_REDIRECT_URI", ""),
            audience=entra_client_id,
        )
        app.state.providers["entra_id"] = OIDCProviderFactory.create("entra_id", cfg)
        log.info("auth.provider_loaded", provider="entra_id", tenant=entra_tenant)
    else:
        log.info("auth.provider_not_configured", provider="entra_id",
                 hint="Set RAGLAB_ENTRA_CLIENT_ID to activate")

    log.info("service.started", service="auth",
             providers=list(app.state.providers.keys()))
    yield
    log.info("service.shutdown", service="auth")


app = FastAPI(
    title="raglab-auth",
    description="RAGLab authentication — OIDC/OAuth2 with Entra ID, Google, Cognito",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("auth"))
app.include_router(auth_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    providers = list(getattr(app.state, "providers", {}).keys())
    return HealthModel(
        service="auth",
        status="ok",
        dependencies={"providers_loaded": str(len(providers))},
    )


@app.get("/")
async def root() -> dict:
    providers = list(getattr(app.state, "providers", {}).keys())
    return {
        "service":  "auth",
        "version":  "0.2.0",
        "release":  "R7",
        "providers": providers,
        "endpoints": [
            "GET  /auth/providers",
            "GET  /auth/login/{provider}",
            "GET  /auth/callback/{provider}",
            "GET  /auth/me",
            "POST /auth/logout",
        ],
    }
