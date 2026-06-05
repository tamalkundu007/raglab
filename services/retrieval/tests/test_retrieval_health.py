"""Smoke tests for retrieval — /health and /."""
import pytest
from fastapi.testclient import TestClient
from retrieval.main import app

client = TestClient(app)

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "retrieval"
    assert data["status"] in ("ok", "degraded")

def test_root_returns_service_info():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "retrieval"
