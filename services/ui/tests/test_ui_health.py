"""Smoke tests for ui — /health and /."""
import pytest
from fastapi.testclient import TestClient
from ui.main import app

client = TestClient(app)

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ui"

def test_root_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert b"RAGLab" in response.content
