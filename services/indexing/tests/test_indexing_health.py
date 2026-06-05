"""Smoke tests for indexing — /health and / endpoints."""
import pytest
from fastapi.testclient import TestClient

from indexing.main import app

client = TestClient(app)


def test_health_returns_200():
    """Health endpoint must always return 200 — ok or degraded, never 5xx."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "indexing"
    assert data["status"] in ("ok", "degraded")  # degraded when infra not wired


def test_root_returns_service_info():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()
    assert response.json()["service"] == "indexing"
