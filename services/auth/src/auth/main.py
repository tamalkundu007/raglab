"""
auth — Stub service placeholder.

This service activates in R7. The /health endpoint is live
so docker-compose health checks pass from R1 onward.
"""

from fastapi import FastAPI
from raglab_common.models import HealthModel

app = FastAPI(
    title="raglab-auth",
    description="Authentication and multi-tenancy",
    version="0.1.0",
)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    """Health check — always returns ok for stub services."""
    return HealthModel(service="auth", status="ok")


@app.get("/")
async def root() -> dict:
    return {
        "service": "auth",
        "status": "stub",
        "activates_in": "R7",
        "message": "This service is not yet active. See /health.",
    }
