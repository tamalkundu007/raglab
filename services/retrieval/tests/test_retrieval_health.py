"""Smoke tests for retrieval — /health and / endpoints."""
import pytest
from fastapi.testclient import TestClient

from retrieval.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "retrieval"


def test_root_returns_service_info():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "service" in body
    assert body["service"] == "retrieval"
