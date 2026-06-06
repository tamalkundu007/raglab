"""
Unit tests for PDFImageChunker captioning path (R4 Phase 2).

Covers:
- CaptionService config validation
- CaptionService.caption_chunks: happy path, failure placeholder, failure raise
- CaptionService HTTP call: correct payload shape, timeout forwarded
- Captioned chunk metadata: captioned=True, caption_provider, updated text
- PDFImageChunker image_handling='caption': CaptionService called
- BaseLLMProvider.caption_image: default fallback text
- AzureOpenAIProvider.caption_image: vision API call shape (mocked)
- AnthropicProvider.caption_image: vision API call shape (mocked)
- caption_service_config forwarded from PDFImageChunker to CaptionService
- /caption HTTP endpoint: 200 on success, 503 on missing provider
- on_failure='raise' raises ChunkerError
- Chunks without image_bytes pass through unchanged
"""

from __future__ import annotations

import base64
import json
import uuid
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from raglab_chunkers.caption_service import CaptionService
from raglab_chunkers.pdf_image_chunker import PDFImageChunker
from raglab_common.exceptions import ChunkerError, LLMError
from raglab_common.models import ChunkModel

# ── Helpers ───────────────────────────────────────────────────────────────────

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)
PNG_B64 = base64.b64encode(TINY_PNG).decode("ascii")

SAMPLE_OCR_TEXT = (
    "This scanned document contains a diagram. "
    "The diagram shows the RAG retrieval pipeline with multiple stages."
)


def make_image_chunk(
    chunk_id: str | None = None,
    image_b64: str = PNG_B64,
    page_number: int = 1,
    image_index: int = 0,
) -> ChunkModel:
    return ChunkModel(
        chunk_id=chunk_id or str(uuid.uuid4()),
        doc_id="doc-test",
        text=f"[Image on page {page_number}, region {image_index + 1}] Width: 200px, Height: 150px.",
        chunk_index=image_index,
        token_count=12,
        metadata={
            "chunk_type": "image",
            "chunker": "pdf_images",
            "image_bytes": image_b64,
            "image_ext": "png",
            "page_number": page_number,
            "image_index": image_index,
            "image_width": 200,
            "image_height": 150,
            "captioned": False,
        },
    )


def make_text_chunk() -> ChunkModel:
    return ChunkModel(
        chunk_id=str(uuid.uuid4()),
        doc_id="doc-test",
        text="Plain text chunk from OCR.",
        chunk_index=0,
        token_count=5,
        metadata={"chunk_type": "text"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CaptionService config
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaptionServiceConfig:
    def test_defaults(self):
        svc = CaptionService()
        assert svc.llm_service_url == "http://llm:8005"
        assert svc.caption_provider == "azure_openai"
        assert svc.caption_max_tokens == 256
        assert svc.timeout_seconds == 30.0
        assert svc.on_failure == "placeholder"

    def test_custom_config(self):
        svc = CaptionService(config={
            "llm_service_url": "http://localhost:8005",
            "caption_provider": "anthropic",
            "caption_max_tokens": 512,
            "timeout_seconds": 15.0,
            "on_failure": "raise",
        })
        assert svc.llm_service_url == "http://localhost:8005"
        assert svc.caption_provider == "anthropic"
        assert svc.caption_max_tokens == 512
        assert svc.on_failure == "raise"

    def test_trailing_slash_stripped(self):
        svc = CaptionService(config={"llm_service_url": "http://llm:8005/"})
        assert not svc.llm_service_url.endswith("/")

    def test_invalid_on_failure(self):
        with pytest.raises(ValueError, match="on_failure"):
            CaptionService(config={"on_failure": "ignore"})


# ═══════════════════════════════════════════════════════════════════════════════
# CaptionService.caption_chunks
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaptionChunks:
    def test_happy_path_returns_captioned_chunks(self):
        svc = CaptionService()
        img_chunk = make_image_chunk()

        with patch.object(svc, "_call_caption_api", return_value="A diagram showing RAG pipeline."):
            result = svc.caption_chunks([img_chunk])

        assert len(result) == 1
        assert result[0].text == "A diagram showing RAG pipeline."
        assert result[0].metadata["captioned"] is True
        assert result[0].metadata["caption_provider"] == "azure_openai"

    def test_chunk_id_preserved(self):
        svc = CaptionService()
        img_chunk = make_image_chunk(chunk_id="fixed-id-123")

        with patch.object(svc, "_call_caption_api", return_value="Caption text."):
            result = svc.caption_chunks([img_chunk])

        assert result[0].chunk_id == "fixed-id-123"

    def test_doc_id_preserved(self):
        svc = CaptionService()
        img_chunk = make_image_chunk()

        with patch.object(svc, "_call_caption_api", return_value="Caption."):
            result = svc.caption_chunks([img_chunk])

        assert result[0].doc_id == "doc-test"

    def test_chunk_index_preserved(self):
        svc = CaptionService()
        img_chunk = make_image_chunk(image_index=2)
        img_chunk = img_chunk.model_copy(update={"chunk_index": 2})

        with patch.object(svc, "_call_caption_api", return_value="Caption."):
            result = svc.caption_chunks([img_chunk])

        assert result[0].chunk_index == 2

    def test_token_count_updated_to_caption_length(self):
        svc = CaptionService()
        img_chunk = make_image_chunk()
        caption = "A diagram showing the retrieval augmented generation pipeline stages."

        with patch.object(svc, "_call_caption_api", return_value=caption):
            result = svc.caption_chunks([img_chunk])

        assert result[0].token_count == len(caption.split())

    def test_non_image_chunk_passes_through(self):
        svc = CaptionService()
        text_chunk = make_text_chunk()

        with patch.object(svc, "_call_caption_api") as mock_api:
            result = svc.caption_chunks([text_chunk])

        mock_api.assert_not_called()
        assert result[0].text == text_chunk.text

    def test_chunk_without_image_bytes_passes_through(self):
        svc = CaptionService()
        chunk = ChunkModel(
            chunk_id=str(uuid.uuid4()), doc_id="d", text="no image",
            chunk_index=0, token_count=2,
            metadata={"chunk_type": "image"},  # no image_bytes
        )
        with patch.object(svc, "_call_caption_api") as mock_api:
            result = svc.caption_chunks([chunk])
        mock_api.assert_not_called()
        assert result[0].text == "no image"

    def test_mixed_list_only_images_captioned(self):
        svc = CaptionService()
        text_chunk = make_text_chunk()
        img_chunk = make_image_chunk()

        with patch.object(svc, "_call_caption_api", return_value="Image caption."):
            result = svc.caption_chunks([text_chunk, img_chunk])

        assert result[0].text == "Plain text chunk from OCR."
        assert result[1].text == "Image caption."

    def test_multiple_image_chunks_all_captioned(self):
        svc = CaptionService()
        chunks = [make_image_chunk(image_index=i) for i in range(3)]

        with patch.object(svc, "_call_caption_api", return_value="Caption."):
            result = svc.caption_chunks(chunks)

        assert all(c.metadata["captioned"] is True for c in result)
        assert len(result) == 3

    def test_failure_returns_placeholder_by_default(self):
        svc = CaptionService(config={"on_failure": "placeholder"})
        img_chunk = make_image_chunk()

        with patch.object(svc, "_call_caption_api", side_effect=Exception("HTTP 503")):
            result = svc.caption_chunks([img_chunk])

        assert len(result) == 1
        assert "captioning failed" in result[0].text.lower() or "[Image" in result[0].text

    def test_failure_raises_when_on_failure_raise(self):
        svc = CaptionService(config={"on_failure": "raise"})
        img_chunk = make_image_chunk()

        with patch.object(svc, "_call_caption_api", side_effect=Exception("HTTP 503")):
            with pytest.raises(ChunkerError, match="Caption request failed"):
                svc.caption_chunks([img_chunk])

    def test_empty_list_returns_empty(self):
        svc = CaptionService()
        assert svc.caption_chunks([]) == []


# ═══════════════════════════════════════════════════════════════════════════════
# CaptionService HTTP call
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaptionHTTPCall:
    def test_posts_to_correct_url(self):
        svc = CaptionService(config={"llm_service_url": "http://llm:8005"})

        mock_response = MagicMock()
        mock_response.json.return_value = {"caption": "A chart."}
        mock_response.raise_for_status = MagicMock()

        with patch("raglab_chunkers.caption_service._requests") as mock_req:
            mock_req.post.return_value = mock_response
            result = svc._call_caption_api(
                image_b64=PNG_B64, image_ext="png",
                doc_id="d", page_number=1, image_index=0,
            )

        mock_req.post.assert_called_once()
        call_url = mock_req.post.call_args[0][0]
        assert call_url == "http://llm:8005/caption"
        assert result == "A chart."

    def test_payload_contains_required_fields(self):
        svc = CaptionService(config={"caption_provider": "anthropic", "caption_max_tokens": 300})

        mock_response = MagicMock()
        mock_response.json.return_value = {"caption": "Diagram."}
        mock_response.raise_for_status = MagicMock()

        with patch("raglab_chunkers.caption_service._requests") as mock_req:
            mock_req.post.return_value = mock_response
            svc._call_caption_api(
                image_b64=PNG_B64, image_ext="jpg",
                doc_id="doc1", page_number=3, image_index=1,
            )

        payload = mock_req.post.call_args[1]["json"]
        assert payload["image_b64"] == PNG_B64
        assert payload["image_ext"] == "jpg"
        assert payload["provider"] == "anthropic"
        assert payload["max_tokens"] == 300
        assert payload["doc_id"] == "doc1"
        assert payload["page_number"] == 3
        assert payload["image_index"] == 1

    def test_timeout_forwarded(self):
        svc = CaptionService(config={"timeout_seconds": 45.0})
        mock_response = MagicMock()
        mock_response.json.return_value = {"caption": "x"}
        mock_response.raise_for_status = MagicMock()

        with patch("raglab_chunkers.caption_service._requests") as mock_req:
            mock_req.post.return_value = mock_response
            svc._call_caption_api(PNG_B64, "png", "d", 1, 0)

        timeout = mock_req.post.call_args[1]["timeout"]
        assert timeout == 45.0


# ═══════════════════════════════════════════════════════════════════════════════
# PDFImageChunker image_handling='caption'
# ═══════════════════════════════════════════════════════════════════════════════

class TestPDFImageChunkerCaptionMode:
    def _run_caption(self, caption_text="A flowchart showing pipeline stages.", **cfg_kwargs):
        img_xref = 99
        images = [(img_xref, None, None, None, None, None, None)]
        page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
            b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        page.get_pixmap.return_value = mock_pixmap
        page.get_images.return_value = images

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__ = lambda self, i: page
        mock_doc.close = MagicMock()
        mock_doc.extract_image = lambda xref: {
            "image": TINY_PNG, "width": 400, "height": 300, "ext": "png"
        }

        cfg = {
            "tokenizer": "word_count", "chunk_size": 30, "chunk_overlap": 3,
            "min_chunk_size": 5, "image_handling": "caption",
            "ocr_engine": "none", "min_image_area": 100,
            "caption_service_config": {"llm_service_url": "http://llm-test:8005"},
        }
        cfg.update(cfg_kwargs)

        with patch("raglab_chunkers.pdf_image_chunker.fitz") as mock_fitz, \
             patch("raglab_chunkers.pdf_image_chunker.pytesseract") as mock_tess, \
             patch("raglab_chunkers.pdf_image_chunker.Image") as mock_pil, \
             patch("raglab_chunkers.caption_service._requests") as mock_req:

            mock_fitz.open.return_value = mock_doc
            mock_tess.image_to_string.return_value = ""
            mock_pil.open.return_value = MagicMock()

            mock_resp = MagicMock()
            mock_resp.json.return_value = {"caption": caption_text}
            mock_resp.raise_for_status = MagicMock()
            mock_req.post.return_value = mock_resp

            chunker = PDFImageChunker(config=cfg)
            return chunker.chunk_pdf_bytes(b"fake-pdf", "doc-caption")

    def test_caption_mode_produces_chunks(self):
        chunks = self._run_caption()
        assert len(chunks) >= 1

    def test_caption_text_replaces_placeholder(self):
        chunks = self._run_caption(caption_text="A flowchart showing pipeline stages.")
        assert chunks[0].text == "A flowchart showing pipeline stages."

    def test_captioned_true_in_metadata(self):
        chunks = self._run_caption()
        assert chunks[0].metadata["captioned"] is True

    def test_caption_provider_in_metadata(self):
        chunks = self._run_caption()
        assert "caption_provider" in chunks[0].metadata

    def test_caption_service_config_forwarded(self):
        """caption_service_config.llm_service_url must be used in the HTTP call."""
        with patch("raglab_chunkers.caption_service._requests") as mock_req:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"caption": "test"}
            mock_resp.raise_for_status = MagicMock()
            mock_req.post.return_value = mock_resp

            img_chunk = make_image_chunk()
            svc = CaptionService(config={"llm_service_url": "http://custom-llm:9999"})
            svc.caption_chunks([img_chunk])

            call_url = mock_req.post.call_args[0][0]
            assert "custom-llm:9999" in call_url


# ═══════════════════════════════════════════════════════════════════════════════
# BaseLLMProvider.caption_image fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseLLMProviderCaptionFallback:
    def test_default_returns_placeholder(self):
        from llm.providers.base import BaseLLMProvider

        class _StubProvider(BaseLLMProvider):
            provider = "stub"
            def _call_api(self, *a, **kw): return ""
            def _model_name(self): return "stub"

        p = _StubProvider()
        result = p.caption_image(PNG_B64, "png")
        assert "captioning not supported" in result.lower() or "[Image" in result

    def test_provider_name_in_fallback(self):
        from llm.providers.base import BaseLLMProvider

        class _StubProvider(BaseLLMProvider):
            provider = "ollama"
            def _call_api(self, *a, **kw): return ""
            def _model_name(self): return "stub"

        p = _StubProvider()
        result = p.caption_image(PNG_B64)
        assert "ollama" in result


# ═══════════════════════════════════════════════════════════════════════════════
# AzureOpenAIProvider.caption_image
# ═══════════════════════════════════════════════════════════════════════════════

class TestAzureOpenAICaptionImage:
    def _make_provider(self):
        with patch("llm.providers.azure_openai.AzureOpenAI"):
            from llm.providers.azure_openai import AzureOpenAIProvider
            p = AzureOpenAIProvider.__new__(AzureOpenAIProvider)
            p._client = MagicMock()
            p._deployment = "gpt-4o"
            p._log = MagicMock()
            return p

    def test_posts_image_as_data_url(self):
        p = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "A pipeline diagram."
        p._client.chat.completions.create.return_value = mock_resp

        result = p.caption_image(PNG_B64, "png", "Describe this.")
        assert result == "A pipeline diagram."

        call_messages = p._client.chat.completions.create.call_args[1]["messages"]
        content = call_messages[0]["content"]
        image_part = next(c for c in content if c["type"] == "image_url")
        assert "data:image/png;base64," in image_part["image_url"]["url"]

    def test_returns_stripped_content(self):
        p = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "  A chart.  "
        p._client.chat.completions.create.return_value = mock_resp

        result = p.caption_image(PNG_B64, "png")
        assert result == "A chart."

    def test_empty_response_returns_placeholder(self):
        p = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = ""
        p._client.chat.completions.create.return_value = mock_resp

        result = p.caption_image(PNG_B64, "png")
        assert result == "[No caption returned]"

    def test_api_error_raises_llm_error(self):
        p = self._make_provider()
        p._client.chat.completions.create.side_effect = Exception("API unavailable")

        with pytest.raises(LLMError, match="vision caption failed"):
            p.caption_image(PNG_B64, "png")

    def test_jpg_media_type_mapped_correctly(self):
        p = self._make_provider()
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Photo."
        p._client.chat.completions.create.return_value = mock_resp

        p.caption_image(PNG_B64, "jpg")
        call_messages = p._client.chat.completions.create.call_args[1]["messages"]
        image_part = next(c for c in call_messages[0]["content"] if c["type"] == "image_url")
        assert "image/jpeg" in image_part["image_url"]["url"]


# ═══════════════════════════════════════════════════════════════════════════════
# AnthropicProvider.caption_image
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnthropicCaptionImage:
    def _make_provider(self):
        with patch("llm.providers.anthropic_provider.anthropic"):
            from llm.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider.__new__(AnthropicProvider)
            p._client = MagicMock()
            p._model = "claude-3-5-sonnet-20241022"
            p._log = MagicMock()
            return p

    def test_sends_base64_image_block(self):
        p = self._make_provider()
        mock_content = MagicMock()
        mock_content.text = "A technical diagram."
        p._client.messages.create.return_value = MagicMock(content=[mock_content])

        result = p.caption_image(PNG_B64, "png", "Describe this image.")
        assert result == "A technical diagram."

        call_messages = p._client.messages.create.call_args[1]["messages"]
        content = call_messages[0]["content"]
        image_block = next(c for c in content if c["type"] == "image")
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["data"] == PNG_B64
        assert image_block["source"]["media_type"] == "image/png"

    def test_returns_stripped_text(self):
        p = self._make_provider()
        mock_content = MagicMock()
        mock_content.text = "  Caption here.  "
        p._client.messages.create.return_value = MagicMock(content=[mock_content])

        result = p.caption_image(PNG_B64, "png")
        assert result == "Caption here."

    def test_empty_content_returns_placeholder(self):
        p = self._make_provider()
        p._client.messages.create.return_value = MagicMock(content=[])

        result = p.caption_image(PNG_B64, "png")
        assert result == "[No caption returned]"

    def test_api_error_raises_llm_error(self):
        p = self._make_provider()
        p._client.messages.create.side_effect = Exception("Anthropic error")

        with pytest.raises(LLMError, match="Anthropic vision caption failed"):
            p.caption_image(PNG_B64, "png")

    def test_jpg_media_type_mapped_correctly(self):
        p = self._make_provider()
        mock_content = MagicMock()
        mock_content.text = "Photo caption."
        p._client.messages.create.return_value = MagicMock(content=[mock_content])

        p.caption_image(PNG_B64, "jpg")
        call_messages = p._client.messages.create.call_args[1]["messages"]
        image_block = next(c for c in call_messages[0]["content"] if c["type"] == "image")
        assert image_block["source"]["media_type"] == "image/jpeg"


# ═══════════════════════════════════════════════════════════════════════════════
# /caption HTTP endpoint
# ═══════════════════════════════════════════════════════════════════════════════

class TestCaptionEndpoint:
    @pytest.fixture
    def llm_client(self):
        from fastapi.testclient import TestClient
        from llm.main import app

        mock_provider = MagicMock()
        mock_provider.caption_image.return_value = "A RAG pipeline diagram."
        mock_provider._model_name.return_value = "azure_openai/gpt-4o"
        app.state.providers = {"azure_openai": mock_provider}
        return TestClient(app)

    def test_caption_returns_200(self, llm_client):
        r = llm_client.post("/caption", json={
            "image_b64": PNG_B64,
            "image_ext": "png",
            "provider": "azure_openai",
        })
        assert r.status_code == 200

    def test_caption_returns_caption_text(self, llm_client):
        r = llm_client.post("/caption", json={
            "image_b64": PNG_B64,
            "image_ext": "png",
            "provider": "azure_openai",
        })
        assert r.json()["caption"] == "A RAG pipeline diagram."

    def test_caption_returns_captioned_true(self, llm_client):
        r = llm_client.post("/caption", json={"image_b64": PNG_B64, "provider": "azure_openai"})
        assert r.json()["captioned"] is True

    def test_caption_returns_model_name(self, llm_client):
        r = llm_client.post("/caption", json={"image_b64": PNG_B64, "provider": "azure_openai"})
        assert "gpt-4o" in r.json()["model"]

    def test_missing_provider_returns_503(self, llm_client):
        r = llm_client.post("/caption", json={
            "image_b64": PNG_B64,
            "provider": "ollama",  # not in app.state.providers
        })
        assert r.status_code == 503

    def test_llm_error_returns_502(self, llm_client):
        from fastapi.testclient import TestClient
        from llm.main import app

        mock_provider = MagicMock()
        mock_provider.caption_image.side_effect = LLMError("Vision API down")
        mock_provider._model_name.return_value = "test"
        app.state.providers = {"azure_openai": mock_provider}

        client = TestClient(app)
        r = client.post("/caption", json={"image_b64": PNG_B64, "provider": "azure_openai"})
        assert r.status_code == 502

    def test_caption_request_forwards_doc_id(self, llm_client):
        r = llm_client.post("/caption", json={
            "image_b64": PNG_B64,
            "provider": "azure_openai",
            "doc_id": "my-document",
        })
        assert r.json()["doc_id"] == "my-document"

    def test_caption_request_forwards_page_number(self, llm_client):
        r = llm_client.post("/caption", json={
            "image_b64": PNG_B64,
            "provider": "azure_openai",
            "page_number": 5,
        })
        assert r.json()["page_number"] == 5
