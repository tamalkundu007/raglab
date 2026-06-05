"""Smoke tests for api-gateway — /health and /."""
import pytest
from fastapi.testclient import TestClient
from api_gateway.main import app

client = TestClient(app)

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "api-gateway"
    assert data["status"] in ("ok", "degraded", "unknown")

def test_root_returns_service_info():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "api-gateway"
