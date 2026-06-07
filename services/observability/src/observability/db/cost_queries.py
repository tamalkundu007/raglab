"""
Cost dashboard queries — token usage, provider cost, embedding cache hit rate.

Reads from raglab_events:
  - llm-service spans: prompt_tokens, completion_tokens, total_cost attributes
  - embedding-service spans: cache_hits, cache_misses, hit_rate_pct
  - Per provider, per day, per model breakdowns

All functions: read-only, DB error → empty (no raise).
"""
from __future__ import annotations
from typing import Any
from raglab_common.logging import get_logger
log = get_logger(__name__)


async def get_token_summary(session: Any, hours: int = 24) -> dict:
    """Total tokens + estimated cost over last N hours."""
    from sqlalchemy import text
    sql = text("""
        SELECT
            SUM((attributes->>'prompt_tokens')::int)     AS total_prompt_tokens,
            SUM((attributes->>'completion_tokens')::int) AS total_completion_tokens,
            SUM((attributes->>'total_tokens')::int)      AS total_tokens,
            ROUND(SUM((attributes->>'estimated_cost_usd')::float)::numeric, 4) AS total_cost_usd,
            COUNT(DISTINCT trace_id)                     AS total_requests
        FROM raglab_events
        WHERE service_name = 'llm'
          AND attributes ? 'total_tokens'
          AND created_at > NOW() - INTERVAL ':hours hours'
    """)
    try:
        result = await session.execute(sql, {"hours": hours})
        row = result.mappings().one_or_none()
        return dict(row) if row else {}
    except Exception as exc:
        log.warning("cost_dashboard.token_summary_failed", error=str(exc))
        return {}


async def get_tokens_by_provider(session: Any, hours: int = 24) -> list[dict]:
    """Token usage and cost broken down by LLM provider."""
    from sqlalchemy import text
    sql = text("""
        SELECT
            attributes->>'provider'                                    AS provider,
            SUM((attributes->>'total_tokens')::int)                   AS total_tokens,
            ROUND(SUM((attributes->>'estimated_cost_usd')::float)::numeric, 4) AS cost_usd,
            COUNT(*)                                                   AS request_count,
            ROUND(AVG((attributes->>'duration_ms')::float)::numeric, 0) AS avg_latency_ms
        FROM raglab_events
        WHERE service_name = 'llm'
          AND attributes ? 'total_tokens'
          AND created_at > NOW() - INTERVAL ':hours hours'
        GROUP BY provider
        ORDER BY total_tokens DESC
    """)
    try:
        result = await session.execute(sql, {"hours": hours})
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        log.warning("cost_dashboard.by_provider_failed", error=str(exc))
        return []


async def get_daily_token_trend(session: Any, days: int = 7) -> list[dict]:
    """Daily token usage for the last N days (for D3 time-series)."""
    from sqlalchemy import text
    sql = text("""
        SELECT
            DATE_TRUNC('day', created_at)                             AS day,
            SUM((attributes->>'total_tokens')::int)                  AS total_tokens,
            ROUND(SUM((attributes->>'estimated_cost_usd')::float)::numeric, 4) AS cost_usd,
            COUNT(*)                                                  AS requests
        FROM raglab_events
        WHERE service_name = 'llm'
          AND attributes ? 'total_tokens'
          AND created_at > NOW() - INTERVAL ':days days'
        GROUP BY day
        ORDER BY day ASC
    """)
    try:
        result = await session.execute(sql, {"days": days})
        rows = result.mappings().all()
        return [{"day": str(r["day"])[:10], "total_tokens": r["total_tokens"],
                 "cost_usd": r["cost_usd"], "requests": r["requests"]} for r in rows]
    except Exception as exc:
        log.warning("cost_dashboard.daily_trend_failed", error=str(exc))
        return []


async def get_cache_stats_summary(session: Any, hours: int = 24) -> dict:
    """Embedding cache hit rate summary over last N hours."""
    from sqlalchemy import text
    sql = text("""
        SELECT
            SUM((attributes->>'cache_hits')::int)   AS total_hits,
            SUM((attributes->>'cache_misses')::int) AS total_misses,
            ROUND(AVG((attributes->>'hit_rate_pct')::float)::numeric, 1) AS avg_hit_rate_pct,
            COUNT(*)                                AS total_embed_requests
        FROM raglab_events
        WHERE service_name = 'embedding'
          AND attributes ? 'hit_rate_pct'
          AND created_at > NOW() - INTERVAL ':hours hours'
    """)
    try:
        result = await session.execute(sql, {"hours": hours})
        row = result.mappings().one_or_none()
        return dict(row) if row else {}
    except Exception as exc:
        log.warning("cost_dashboard.cache_stats_failed", error=str(exc))
        return {}
