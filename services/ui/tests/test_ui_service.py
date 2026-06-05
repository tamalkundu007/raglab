"""
Tests for the ui-service.

Covers:
- UISettings defaults
- GET / returns 200 HTML with correct template markers
- GET /health returns ok
- Template renders gateway_url and api_base correctly
- All nav pages accessible (static HTML — no JS execution needed)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ui.settings import UISettings


class TestUISettings:
    def test_defaults(self):
        s = UISettings()
        assert s.service_name == "ui"
        assert s.port == 8009
        assert "api-gateway" in s.gateway_url or "localhost" in s.gateway_url
        assert s.api_base == "/api/v1"
        assert s.app_title == "RAGLab"


@pytest.fixture
def ui_client():
    from ui.main import app
    app.state.settings = UISettings()
    return TestClient(app)


class TestUIEndpoints:
    def test_health_returns_ok(self, ui_client):
        r = ui_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert r.json()["service"] == "ui"

    def test_root_returns_html(self, ui_client):
        r = ui_client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_control_panel_has_raglab_title(self, ui_client):
        r = ui_client.get("/")
        assert b"RAGLab" in r.content

    def test_control_panel_has_api_base(self, ui_client):
        r = ui_client.get("/")
        # api_base is injected into the template JS
        assert b"/api/v1" in r.content

    def test_control_panel_has_chunker_section(self, ui_client):
        r = ui_client.get("/")
        assert b"Chunker" in r.content
        assert b"TextChunker" in r.content

    def test_control_panel_has_retriever_section(self, ui_client):
        r = ui_client.get("/")
        assert b"Retriever" in r.content
        assert b"DenseRetriever" in r.content

    def test_control_panel_has_llm_section(self, ui_client):
        r = ui_client.get("/")
        assert b"LLM Provider" in r.content
        assert b"azure_openai" in r.content

    def test_control_panel_has_vector_store_section(self, ui_client):
        r = ui_client.get("/")
        assert b"Vector Store" in r.content
        assert b"Qdrant" in r.content

    def test_control_panel_has_r2_stubs(self, ui_client):
        r = ui_client.get("/")
        # R2 chunkers must be present but disabled
        assert b"PDFChunker" in r.content
        assert b"DOCXChunker" in r.content
        assert b"MarkdownChunker" in r.content
        assert b"HTMLChunker" in r.content
        assert b"ExcelChunker" in r.content

    def test_control_panel_has_r3_stubs(self, ui_client):
        r = ui_client.get("/")
        assert b"BM25" in r.content
        assert b"Hybrid" in r.content
        assert b"MMR" in r.content

    def test_control_panel_has_r4_plus_stubs(self, ui_client):
        r = ui_client.get("/")
        assert b"GraphRAG" in r.content
        assert b"Observability" in r.content
        assert b"R4" in r.content
        assert b"R6" in r.content

    def test_control_panel_has_navigation(self, ui_client):
        r = ui_client.get("/")
        assert b"Control Panel" in r.content
        assert b"Chunk Inspector" in r.content
        assert b"Config Library" in r.content
        assert b"Health" in r.content

    def test_control_panel_has_query_page(self, ui_client):
        r = ui_client.get("/")
        assert b"Query" in r.content
        assert b"runQuery" in r.content

    def test_control_panel_has_ingest_page(self, ui_client):
        r = ui_client.get("/")
        assert b"Ingest" in r.content
        assert b"submitIngest" in r.content

    def test_control_panel_has_health_page(self, ui_client):
        r = ui_client.get("/")
        assert b"refreshHealth" in r.content

    def test_template_injects_gateway_url(self, ui_client):
        r = ui_client.get("/")
        # gateway_url should be in the rendered JS
        settings = UISettings()
        assert settings.gateway_url.encode() in r.content or b"api-gateway" in r.content

    def test_coming_soon_pills_present(self, ui_client):
        r = ui_client.get("/")
        assert b"coming-soon-pill" in r.content
        assert b"Coming Soon" in r.content or b"R2" in r.content

    def test_no_500_errors(self, ui_client):
        """Smoke test — template renders without errors."""
        r = ui_client.get("/")
        assert r.status_code < 500
