"""Smoke tests for ingestion — /health and /."""
import pytest
from fastapi.testclient import TestClient
from ingestion.main import app

client = TestClient(app)

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ingestion"
    assert data["status"] in ("ok", "degraded")

def test_root_returns_service_info():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "ingestion"
