"""Test fixtures for raglab-common."""
import pytest


@pytest.fixture
def sample_chunk_data() -> dict:
    return {
        "doc_id": "doc-001",
        "text": "This is a sample chunk for testing.",
        "chunk_index": 0,
        "token_count": 8,
    }


@pytest.fixture
def sample_document_data() -> dict:
    return {
        "filename": "test.txt",
        "content_type": "text/plain",
        "storage_path": "/tmp/test.txt",
        "idempotency_key": "test-idem-001",
    }
