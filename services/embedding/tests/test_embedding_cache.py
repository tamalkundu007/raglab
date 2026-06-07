"""
Unit tests for EmbeddingCache (R5 Phase 1).

All Redis calls mocked — zero infra required.

Covers:
- EmbeddingCache config: enabled/disabled, TTL, URL
- _cache_key: deterministic, provider-scoped, model-scoped, text-scoped
- get(): HIT returns vector + increments hit counter
- get(): MISS returns None + increments miss counter
- get(): Redis error returns None + increments error counter (no raise)
- get(): disabled cache returns None without Redis call
- set(): stores JSON-serialised vector with TTL
- set(): Redis error is silent (no raise)
- set(): disabled cache is no-op
- invalidate(): deletes key, returns True/False
- flush(): deletes all raglab:embed:* keys
- stats(): hit_rate calculation, hit_rate_pct, all fields
- reset_stats(): zeroes counters
- hit_rate: 0.0 when no requests yet
- Redis unavailable: gracefully degrades, enabled→False
- POST /embed: cache hit path — provider not called
- POST /embed: cache miss path — provider called, vector cached
- POST /embed: cache_hit field in response
- POST /embed/batch: partial cache — only uncached texts sent to provider
- POST /embed/batch: cache_hits + cache_misses counts in response
- POST /embed/batch: all cached — provider not called at all
- GET /embed/cache/stats: returns stats dict
- GET /embed/cache/stats: returns defaults when cache not configured
- DELETE /embed/cache/flush: returns deleted count
"""

from __future__ import annotations

import hashlib
import json
import uuid
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi.testclient import TestClient

from embedding.cache import EmbeddingCache, _cache_key, _KEY_PREFIX


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_cache(enabled=True, ttl=3600) -> EmbeddingCache:
    """Create a cache with mocked Redis client."""
    with patch("embedding.cache._REDIS_AVAILABLE", True), \
         patch("embedding.cache._redis_module") as mock_redis:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.Redis.from_url.return_value = mock_client
        cache = EmbeddingCache(
            redis_url="redis://localhost:6379/0",
            ttl_seconds=ttl,
            enabled=enabled,
        )
        cache._client = mock_client  # hold reference
        return cache


SAMPLE_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5]
SAMPLE_TEXT = "RAG retrieves relevant chunks for context."
PROVIDER = "azure_openai"
MODEL = "text-embedding-3-small"


# ═══════════════════════════════════════════════════════════════════════════════
# _cache_key
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheKey:
    def test_returns_string_with_prefix(self):
        key = _cache_key("text", "provider", "model")
        assert key.startswith(_KEY_PREFIX)

    def test_deterministic(self):
        k1 = _cache_key("hello", "azure_openai", "model-A")
        k2 = _cache_key("hello", "azure_openai", "model-A")
        assert k1 == k2

    def test_different_text_different_key(self):
        assert _cache_key("text A", PROVIDER, MODEL) != _cache_key("text B", PROVIDER, MODEL)

    def test_different_provider_different_key(self):
        assert _cache_key("text", "azure_openai", MODEL) != _cache_key("text", "openai", MODEL)

    def test_different_model_different_key(self):
        assert _cache_key("text", PROVIDER, "model-A") != _cache_key("text", PROVIDER, "model-B")

    def test_key_is_sha256_hex(self):
        key = _cache_key("t", "p", "m")
        hex_part = key[len(_KEY_PREFIX):]
        assert len(hex_part) == 64  # SHA-256 = 32 bytes = 64 hex chars

    def test_special_characters_in_text(self):
        key = _cache_key("こんにちは 🌸", PROVIDER, MODEL)
        assert key.startswith(_KEY_PREFIX)
        assert len(key) == len(_KEY_PREFIX) + 64


# ═══════════════════════════════════════════════════════════════════════════════
# EmbeddingCache.get()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheGet:
    def test_hit_returns_vector(self):
        cache = make_cache()
        cache._client.get.return_value = json.dumps(SAMPLE_VECTOR).encode()
        result = cache.get(SAMPLE_TEXT, PROVIDER, MODEL)
        assert result == SAMPLE_VECTOR

    def test_hit_increments_hit_counter(self):
        cache = make_cache()
        cache._client.get.return_value = json.dumps(SAMPLE_VECTOR).encode()
        cache.get(SAMPLE_TEXT, PROVIDER, MODEL)
        assert cache._hits == 1
        assert cache._misses == 0

    def test_miss_returns_none(self):
        cache = make_cache()
        cache._client.get.return_value = None
        result = cache.get(SAMPLE_TEXT, PROVIDER, MODEL)
        assert result is None

    def test_miss_increments_miss_counter(self):
        cache = make_cache()
        cache._client.get.return_value = None
        cache.get(SAMPLE_TEXT, PROVIDER, MODEL)
        assert cache._misses == 1
        assert cache._hits == 0

    def test_redis_error_returns_none(self):
        cache = make_cache()
        cache._client.get.side_effect = Exception("Redis connection lost")
        result = cache.get(SAMPLE_TEXT, PROVIDER, MODEL)
        assert result is None

    def test_redis_error_increments_error_counter(self):
        cache = make_cache()
        cache._client.get.side_effect = Exception("timeout")
        cache.get(SAMPLE_TEXT, PROVIDER, MODEL)
        assert cache._errors == 1

    def test_redis_error_does_not_raise(self):
        cache = make_cache()
        cache._client.get.side_effect = RuntimeError("connection refused")
        # Should not raise
        cache.get(SAMPLE_TEXT, PROVIDER, MODEL)

    def test_disabled_cache_returns_none_without_redis(self):
        cache = make_cache(enabled=False)
        result = cache.get(SAMPLE_TEXT, PROVIDER, MODEL)
        assert result is None
        cache._client.get.assert_not_called()

    def test_uses_correct_cache_key(self):
        cache = make_cache()
        cache._client.get.return_value = None
        cache.get(SAMPLE_TEXT, PROVIDER, MODEL)
        expected_key = _cache_key(SAMPLE_TEXT, PROVIDER, MODEL)
        cache._client.get.assert_called_once_with(expected_key)


# ═══════════════════════════════════════════════════════════════════════════════
# EmbeddingCache.set()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheSet:
    def test_stores_json_vector_with_ttl(self):
        cache = make_cache(ttl=7200)
        cache.set(SAMPLE_TEXT, PROVIDER, MODEL, SAMPLE_VECTOR)
        expected_key = _cache_key(SAMPLE_TEXT, PROVIDER, MODEL)
        cache._client.setex.assert_called_once_with(
            expected_key,
            7200,
            json.dumps(SAMPLE_VECTOR),
        )

    def test_redis_error_is_silent(self):
        cache = make_cache()
        cache._client.setex.side_effect = Exception("write failed")
        cache.set(SAMPLE_TEXT, PROVIDER, MODEL, SAMPLE_VECTOR)  # should not raise
        assert cache._errors == 1

    def test_disabled_is_noop(self):
        cache = make_cache(enabled=False)
        cache.set(SAMPLE_TEXT, PROVIDER, MODEL, SAMPLE_VECTOR)
        cache._client.setex.assert_not_called()

    def test_large_vector_stored_correctly(self):
        cache = make_cache()
        large_vector = [float(i) / 1000 for i in range(1536)]
        cache.set("text", PROVIDER, MODEL, large_vector)
        args = cache._client.setex.call_args[0]
        stored_vector = json.loads(args[2])
        assert len(stored_vector) == 1536
        assert stored_vector[0] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# EmbeddingCache.invalidate() + flush()
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheInvalidateFlush:
    def test_invalidate_returns_true_when_deleted(self):
        cache = make_cache()
        cache._client.delete.return_value = 1
        assert cache.invalidate(SAMPLE_TEXT, PROVIDER, MODEL) is True

    def test_invalidate_returns_false_when_not_found(self):
        cache = make_cache()
        cache._client.delete.return_value = 0
        assert cache.invalidate(SAMPLE_TEXT, PROVIDER, MODEL) is False

    def test_invalidate_disabled_returns_false(self):
        cache = make_cache(enabled=False)
        assert cache.invalidate(SAMPLE_TEXT, PROVIDER, MODEL) is False

    def test_flush_calls_delete_on_all_keys(self):
        cache = make_cache()
        key1 = _cache_key("text1", PROVIDER, MODEL)
        key2 = _cache_key("text2", PROVIDER, MODEL)
        cache._client.keys.return_value = [key1.encode(), key2.encode()]
        cache._client.delete.return_value = 2
        deleted = cache.flush()
        assert deleted == 2
        cache._client.keys.assert_called_once_with(f"{_KEY_PREFIX}*")

    def test_flush_empty_store_returns_zero(self):
        cache = make_cache()
        cache._client.keys.return_value = []
        assert cache.flush() == 0

    def test_flush_disabled_returns_zero(self):
        cache = make_cache(enabled=False)
        assert cache.flush() == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Stats + hit_rate
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheStats:
    def test_hit_rate_zero_when_no_requests(self):
        cache = make_cache()
        assert cache.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        cache = make_cache()
        cache._hits = 7
        cache._misses = 3
        assert cache.hit_rate == pytest.approx(0.7)

    def test_hit_rate_100_percent(self):
        cache = make_cache()
        cache._hits = 5
        cache._misses = 0
        assert cache.hit_rate == 1.0

    def test_stats_dict_has_required_keys(self):
        cache = make_cache()
        s = cache.stats()
        for key in ["enabled","connected","hits","misses","errors",
                    "total_requests","hit_rate","hit_rate_pct","ttl_seconds","redis_url"]:
            assert key in s, f"Missing stats key: {key}"

    def test_stats_hit_rate_pct_correct(self):
        cache = make_cache()
        cache._hits = 3
        cache._misses = 1
        s = cache.stats()
        assert s["hit_rate_pct"] == pytest.approx(75.0)

    def test_stats_total_requests(self):
        cache = make_cache()
        cache._hits = 4
        cache._misses = 6
        assert cache.stats()["total_requests"] == 10

    def test_stats_redis_url_strips_credentials(self):
        with patch("embedding.cache._REDIS_AVAILABLE", True), \
             patch("embedding.cache._redis_module") as mock_redis:
            mock_redis.Redis.from_url.return_value = MagicMock(ping=MagicMock())
            cache = EmbeddingCache(redis_url="redis://:secret@host:6379/0", enabled=True)
            assert "secret" not in cache.stats()["redis_url"]

    def test_reset_stats_zeroes_counters(self):
        cache = make_cache()
        cache._hits = 10
        cache._misses = 5
        cache._errors = 2
        cache.reset_stats()
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._errors == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Redis unavailable graceful degradation
# ═══════════════════════════════════════════════════════════════════════════════

class TestGracefulDegradation:
    def test_redis_package_not_installed(self):
        with patch("embedding.cache._REDIS_AVAILABLE", False):
            cache = EmbeddingCache(enabled=True)
            assert cache.enabled is False

    def test_redis_connect_failure_disables_cache(self):
        with patch("embedding.cache._REDIS_AVAILABLE", True), \
             patch("embedding.cache._redis_module") as mock_redis:
            mock_redis.Redis.from_url.side_effect = Exception("connection refused")
            cache = EmbeddingCache(enabled=True)
            assert cache.enabled is False
            assert cache._client is None

    def test_disabled_cache_get_returns_none(self):
        cache = EmbeddingCache(enabled=False)
        assert cache.get("text", PROVIDER, MODEL) is None

    def test_no_client_get_returns_none(self):
        cache = make_cache()
        cache._client = None
        assert cache.get("text", PROVIDER, MODEL) is None


# ═══════════════════════════════════════════════════════════════════════════════
# POST /embed — cache integration
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def embed_client():
    from embedding.main import app
    from embedding.settings import EmbeddingSettings

    # Mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = SAMPLE_VECTOR
    mock_embedder.embed_batch.return_value = [SAMPLE_VECTOR]
    mock_embedder.model_name = MODEL

    # Mock cache (starts with no hits)
    mock_cache = MagicMock(spec=EmbeddingCache)
    mock_cache.get.return_value = None  # default: miss
    mock_cache.set.return_value = None
    mock_cache.stats.return_value = {
        "enabled": True, "connected": True,
        "hits": 5, "misses": 3, "errors": 0,
        "total_requests": 8, "hit_rate": 0.625,
        "hit_rate_pct": 62.5, "ttl_seconds": 3600,
        "redis_url": "localhost:6379",
    }

    app.state.embedders = {PROVIDER: mock_embedder}
    app.state.embedding_cache = mock_cache
    app.state.settings = EmbeddingSettings()

    return TestClient(app), mock_embedder, mock_cache


class TestEmbedEndpointCache:
    def test_cache_miss_calls_provider(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.return_value = None
        r = client.post("/embed", json={"text": SAMPLE_TEXT, "provider": PROVIDER})
        assert r.status_code == 200
        embedder.embed.assert_called_once_with(SAMPLE_TEXT)

    def test_cache_miss_writes_to_cache(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.return_value = None
        client.post("/embed", json={"text": SAMPLE_TEXT, "provider": PROVIDER})
        cache.set.assert_called_once()

    def test_cache_hit_skips_provider(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.return_value = SAMPLE_VECTOR
        r = client.post("/embed", json={"text": SAMPLE_TEXT, "provider": PROVIDER})
        assert r.status_code == 200
        embedder.embed.assert_not_called()

    def test_cache_hit_field_true_on_hit(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.return_value = SAMPLE_VECTOR
        r = client.post("/embed", json={"text": SAMPLE_TEXT, "provider": PROVIDER})
        assert r.json()["cache_hit"] is True

    def test_cache_hit_field_false_on_miss(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.return_value = None
        r = client.post("/embed", json={"text": SAMPLE_TEXT, "provider": PROVIDER})
        assert r.json()["cache_hit"] is False

    def test_vector_from_cache_returned_correctly(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.return_value = [1.0, 2.0, 3.0]
        r = client.post("/embed", json={"text": SAMPLE_TEXT, "provider": PROVIDER})
        assert r.json()["vector"] == [1.0, 2.0, 3.0]


class TestEmbedBatchEndpointCache:
    def test_partial_cache_hit_only_misses_sent_to_provider(self, embed_client):
        client, embedder, cache = embed_client
        texts = ["cached text", "uncached text"]
        # First text is cached, second is not
        cache.get.side_effect = lambda t, p, m: SAMPLE_VECTOR if t == "cached text" else None
        embedder.embed_batch.return_value = [[0.9, 0.8, 0.7]]
        r = client.post("/embed/batch", json={"texts": texts, "provider": PROVIDER})
        assert r.status_code == 200
        # Only "uncached text" sent to provider
        embedder.embed_batch.assert_called_once_with(["uncached text"])

    def test_all_cached_skips_provider(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.return_value = SAMPLE_VECTOR
        r = client.post("/embed/batch", json={"texts": ["t1", "t2"], "provider": PROVIDER})
        assert r.status_code == 200
        embedder.embed_batch.assert_not_called()

    def test_cache_hits_misses_in_response(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.side_effect = lambda t, p, m: SAMPLE_VECTOR if t == "t1" else None
        embedder.embed_batch.return_value = [SAMPLE_VECTOR]
        r = client.post("/embed/batch", json={"texts": ["t1", "t2"], "provider": PROVIDER})
        body = r.json()
        assert body["cache_hits"] == 1
        assert body["cache_misses"] == 1

    def test_uncached_vectors_written_to_cache(self, embed_client):
        client, embedder, cache = embed_client
        cache.get.return_value = None
        embedder.embed_batch.return_value = [SAMPLE_VECTOR, SAMPLE_VECTOR]
        client.post("/embed/batch", json={"texts": ["t1", "t2"], "provider": PROVIDER})
        assert cache.set.call_count == 2


class TestCacheStatsEndpoint:
    def test_stats_returns_200(self, embed_client):
        client, _, _ = embed_client
        r = client.get("/embed/cache/stats")
        assert r.status_code == 200

    def test_stats_contains_hit_rate_pct(self, embed_client):
        client, _, _ = embed_client
        r = client.get("/embed/cache/stats")
        assert "hit_rate_pct" in r.json()

    def test_stats_no_cache_returns_defaults(self):
        from embedding.main import app
        app.state.embedding_cache = None
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.get("/embed/cache/stats")
        assert r.status_code == 200
        assert r.json()["enabled"] is False

    def test_flush_returns_deleted_count(self, embed_client):
        client, _, cache = embed_client
        cache.flush.return_value = 42
        r = client.delete("/embed/cache/flush")
        assert r.status_code == 200
        assert r.json()["deleted"] == 42
