"""Smoke tests for api-gateway — /health and / endpoints."""
import pytest
from fastapi.testclient import TestClient

from api_gateway.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "api-gateway"


def test_root_returns_service_info():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "service" in body
    assert body["service"] == "api-gateway"
