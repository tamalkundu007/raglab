"""
EmbeddingCache — Redis-backed embedding cache for cost reduction.

Strategy:
    Cache key: SHA-256 hash of (chunk_text + embedding_model + provider).
    Cache value: JSON-serialised embedding vector.

    On embed request:
        1. Compute cache key.
        2. GET from Redis.
           HIT  → return cached vector; increment hit counter.
           MISS → call provider, write vector to Redis with TTL; increment miss counter.

    Hit-rate is logged per request and available via /embed/cache/stats.
    This is the ROI metric for cost conversations:
        "Re-ingesting a 10,000-chunk corpus where 80% of chunks are unchanged
         saves 80% of embedding API calls on every re-run."

Cache key design:
    SHA-256 of f"{provider}:{model}:{text}"
    - Provider-scoped: Azure OpenAI and OpenAI have different model spaces.
    - Model-scoped: same text, different model → different vector.
    - Text-scoped: deterministic, content-addressable.

Redis key format:
    raglab:embed:{sha256_hex}
    TTL: configurable (default 86400 s = 24 h).
    On TTL expiry, the next request re-computes and re-caches.

Module-level Redis import for test patchability:
    redis.Redis is imported at module level; tests patch via monkeypatch.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from raglab_common.logging import get_logger

log = get_logger(__name__)

# Module-level optional import — patchable in tests
try:
    import redis as _redis_module
    _REDIS_AVAILABLE = True
except ImportError:
    _redis_module = None  # type: ignore[assignment]
    _REDIS_AVAILABLE = False

_KEY_PREFIX = "raglab:embed:"


def _cache_key(text: str, provider: str, model: str) -> str:
    """Deterministic SHA-256 cache key scoped by provider + model + text."""
    raw = f"{provider}:{model}:{text}"
    return _KEY_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """
    Redis-backed embedding vector cache.

    Thread-safe for reads and writes (Redis handles atomicity).
    Gracefully degrades when Redis is unavailable — embedding still works,
    cache is simply bypassed (logged as warning, not error).

    Args:
        redis_url:    Redis connection URL (e.g. redis://localhost:6379/0).
        ttl_seconds:  Key expiry time. Default 86400 (24h).
        enabled:      Master toggle. If False, all operations are no-ops.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 86400,
        enabled: bool = True,
    ) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

        self._client: Any = None
        self._hits: int = 0
        self._misses: int = 0
        self._errors: int = 0

        if enabled:
            self._connect()

    def _connect(self) -> None:
        """Attempt Redis connection. Failure is non-fatal."""
        if not _REDIS_AVAILABLE:
            log.warning("embedding_cache.redis_unavailable",
                        reason="redis package not installed")
            self.enabled = False
            return
        try:
            self._client = _redis_module.Redis.from_url(
                self.redis_url,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # Ping to confirm connectivity
            self._client.ping()
            log.info("embedding_cache.connected", url=self.redis_url)
        except Exception as exc:
            log.warning("embedding_cache.connect_failed",
                        url=self.redis_url, error=str(exc))
            self._client = None
            self.enabled = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, text: str, provider: str, model: str) -> list[float] | None:
        """
        Look up a cached embedding vector.

        Returns:
            list[float] if cache hit, None on miss or error.
        """
        if not self.enabled or self._client is None:
            return None

        key = _cache_key(text, provider, model)
        try:
            raw = self._client.get(key)
            if raw is None:
                self._misses += 1
                log.debug("embedding_cache.miss", key=key[:16])
                return None
            vector = json.loads(raw)
            self._hits += 1
            log.debug("embedding_cache.hit", key=key[:16])
            return vector
        except Exception as exc:
            self._errors += 1
            log.warning("embedding_cache.get_error", error=str(exc))
            return None

    def set(self, text: str, provider: str, model: str, vector: list[float]) -> None:
        """
        Store an embedding vector in the cache.

        Silently skips on error — the embedding was already computed,
        a cache write failure is not worth propagating.
        """
        if not self.enabled or self._client is None:
            return

        key = _cache_key(text, provider, model)
        try:
            self._client.setex(key, self.ttl_seconds, json.dumps(vector))
            log.debug("embedding_cache.set", key=key[:16], ttl=self.ttl_seconds)
        except Exception as exc:
            self._errors += 1
            log.warning("embedding_cache.set_error", error=str(exc))

    def invalidate(self, text: str, provider: str, model: str) -> bool:
        """
        Delete a specific cache entry.

        Returns True if the key existed and was deleted.
        """
        if not self.enabled or self._client is None:
            return False
        key = _cache_key(text, provider, model)
        try:
            deleted = self._client.delete(key)
            return bool(deleted)
        except Exception:
            return False

    def flush(self) -> int:
        """
        Delete all raglab:embed:* keys. Returns count deleted.

        Use with care — only for dev/test; not exposed to the API.
        """
        if not self.enabled or self._client is None:
            return 0
        try:
            keys = self._client.keys(f"{_KEY_PREFIX}*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception:
            return 0

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction [0.0, 1.0]. 0.0 if no requests yet."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics dict for /embed/cache/stats endpoint."""
        total = self._hits + self._misses
        return {
            "enabled": self.enabled,
            "connected": self._client is not None,
            "hits": self._hits,
            "misses": self._misses,
            "errors": self._errors,
            "total_requests": total,
            "hit_rate": round(self.hit_rate, 4),
            "hit_rate_pct": round(self.hit_rate * 100, 2),
            "ttl_seconds": self.ttl_seconds,
            "redis_url": self.redis_url.split("@")[-1],  # strip credentials if present
        }

    def reset_stats(self) -> None:
        """Reset hit/miss/error counters (for testing)."""
        self._hits = 0
        self._misses = 0
        self._errors = 0
