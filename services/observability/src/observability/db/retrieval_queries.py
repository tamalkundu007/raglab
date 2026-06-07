"""
Retrieval scorer queries — read-only views over retrieval events.

Reads from raglab_events where service_name IN ('retrieval', 'pipeline')
and operation_name LIKE '%retrieval%' or attributes contain retrieval metadata.

Surfaces:
    - Per-query: strategy used, top-k results, scores
    - Healing delta: original vs final strategy, score improvement
    - Score distribution across recent queries
    - Top-performing vs worst-performing queries by score
"""

from __future__ import annotations
from typing import Any
from raglab_common.logging import get_logger

log = get_logger(__name__)


async def list_recent_queries(
    session: Any,
    limit: int = 50,
    collection: str | None = None,
) -> list[dict]:
    """
    Return recent retrieval queries from raglab_events.

    Looks for spans with operation_name containing 'retrieve' or 'retrieval',
    or attributes containing retriever_type / top_k metadata.
    """
    from sqlalchemy import text

    where = "service_name IN ('retrieval', 'pipeline') AND (operation_name LIKE '%retriev%' OR attributes ? 'retriever_type')"
    params: dict = {"limit": limit}

    if collection:
        where += " AND attributes->>'collection' = :collection"
        params["collection"] = collection

    sql = text(f"""
        SELECT
            trace_id,
            span_id,
            operation_name,
            start_time_ms,
            duration_ms,
            status,
            attributes->>'query'           AS query_text,
            attributes->>'retriever_type'  AS retriever_type,
            attributes->>'collection'      AS collection,
            (attributes->>'top_k')::int    AS top_k,
            (attributes->>'result_count')::int AS result_count,
            (attributes->>'top_score')::float  AS top_score,
            attributes->>'healed'          AS healed,
            attributes->>'original_strategy' AS original_strategy,
            attributes->>'final_strategy'    AS final_strategy
        FROM raglab_events
        WHERE {where}
        ORDER BY start_time_ms DESC
        LIMIT :limit
    """)

    try:
        result = await session.execute(sql, params)
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        log.warning("retrieval_scorer.list_queries_failed", error=str(exc))
        return []


async def get_query_detail(session: Any, trace_id: str) -> dict:
    """
    Return detailed retrieval span for a trace — candidates, scores, healing info.
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            trace_id, span_id, operation_name,
            start_time_ms, duration_ms, status,
            attributes, events
        FROM raglab_events
        WHERE trace_id = :trace_id
          AND service_name IN ('retrieval', 'pipeline')
          AND (operation_name LIKE '%retriev%' OR attributes ? 'retriever_type')
        ORDER BY start_time_ms ASC
        LIMIT 1
    """)

    try:
        result = await session.execute(sql, {"trace_id": trace_id})
        row = result.mappings().one_or_none()
        if not row:
            return {}
        import json
        d = dict(row)
        for field in ("attributes", "events"):
            if isinstance(d.get(field), str):
                try: d[field] = json.loads(d[field])
                except Exception: pass
        return d
    except Exception as exc:
        log.warning("retrieval_scorer.query_detail_failed", trace_id=trace_id, error=str(exc))
        return {}


async def get_score_distribution(
    session: Any,
    hours: int = 24,
) -> list[dict]:
    """
    Return score distribution buckets over the last N hours.
    Buckets: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            CASE
                WHEN (attributes->>'top_score')::float < 0.2 THEN '0.0-0.2'
                WHEN (attributes->>'top_score')::float < 0.4 THEN '0.2-0.4'
                WHEN (attributes->>'top_score')::float < 0.6 THEN '0.4-0.6'
                WHEN (attributes->>'top_score')::float < 0.8 THEN '0.6-0.8'
                ELSE '0.8-1.0'
            END AS bucket,
            COUNT(*) AS count
        FROM raglab_events
        WHERE service_name = 'retrieval'
          AND attributes ? 'top_score'
          AND created_at > NOW() - INTERVAL ':hours hours'
        GROUP BY bucket
        ORDER BY bucket ASC
    """)

    try:
        result = await session.execute(sql, {"hours": hours})
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        log.warning("retrieval_scorer.distribution_failed", error=str(exc))
        return []


async def get_healing_stats(session: Any, hours: int = 24) -> dict:
    """
    Return healing statistics: how many retrievals were healed, success rate.
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            COUNT(*)                                                    AS total_queries,
            SUM(CASE WHEN attributes->>'healed' = 'true' THEN 1 END)  AS healed_count,
            ROUND(AVG((attributes->>'top_score')::float)::numeric, 3) AS avg_top_score,
            COUNT(DISTINCT attributes->>'retriever_type')              AS strategy_count
        FROM raglab_events
        WHERE service_name = 'retrieval'
          AND attributes ? 'top_score'
          AND created_at > NOW() - INTERVAL ':hours hours'
    """)

    try:
        result = await session.execute(sql, {"hours": hours})
        row = result.mappings().one_or_none()
        return dict(row) if row else {}
    except Exception as exc:
        log.warning("retrieval_scorer.healing_stats_failed", error=str(exc))
        return {}
