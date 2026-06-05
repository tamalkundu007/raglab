"""Settings for the ui-service."""

from raglab_common.settings import BaseServiceSettings


class UISettings(BaseServiceSettings):
    service_name: str = "ui"
    port: int = 8009

    # API gateway URL — all UI calls go through the gateway
    gateway_url: str = "http://api-gateway:8000"
    api_base: str = "/api/v1"

    # UI metadata
    app_title: str = "RAGLab"
    app_version: str = "R1"
    app_tagline: str = "Configurable RAG Platform"
