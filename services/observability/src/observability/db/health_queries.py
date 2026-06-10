"""
Pipeline health queries — queue depth, DLQ, failed jobs.

Reads from raglab_events for pipeline/ingestion service spans.
"""
from __future__ import annotations
from typing import Any
from raglab_common.logging import get_logger
log = get_logger(__name__)


async def get_pipeline_health(session: Any, hours: int = 24) -> dict:
    """Return pipeline job success/failure counts and avg duration."""
    from sqlalchemy import text
    sql = text("""
        SELECT
            COUNT(*)                                            AS total_jobs,
            SUM(CASE WHEN status='ok'    THEN 1 ELSE 0 END)   AS successful,
            SUM(CASE WHEN status='error' THEN 1 ELSE 0 END)   AS failed,
            ROUND(AVG(duration_ms)::numeric, 0)               AS avg_duration_ms
        FROM raglab_events
        WHERE service_name = 'pipeline'
          AND parent_span_id IS NULL
          AND created_at > NOW() - INTERVAL ':hours hours'
    """)
    try:
        result = await session.execute(sql, {"hours": hours})
        row = result.mappings().one_or_none()
        return dict(row) if row else {}
    except Exception as exc:
        log.warning("pipeline_health.query_failed", error=str(exc))
        return {}


async def get_failed_jobs(session: Any, limit: int = 20) -> list[dict]:
    """Return recent failed pipeline jobs with error messages."""
    from sqlalchemy import text
    sql = text("""
        SELECT
            trace_id, start_time_ms, duration_ms,
            attributes->>'doc_id'       AS doc_id,
            attributes->>'filename'     AS filename,
            attributes->>'error'        AS error_message
        FROM raglab_events
        WHERE service_name = 'pipeline'
          AND status = 'error'
          AND parent_span_id IS NULL
        ORDER BY start_time_ms DESC
        LIMIT :limit
    """)
    try:
        result = await session.execute(sql, {"limit": limit})
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        log.warning("pipeline_health.failed_jobs_failed", error=str(exc))
        return []


async def get_heal_gate_summary(session: Any, hours: int = 24) -> list[dict]:
    """Return self-healing gate firing counts and scores."""
    from sqlalchemy import text
    sql = text("""
        SELECT
            attributes->>'gate_name'   AS gate_name,
            COUNT(*)                   AS fired_count,
            SUM(CASE WHEN (attributes->>'passed')::boolean THEN 1 ELSE 0 END) AS passed_count,
            ROUND(AVG((attributes->>'score')::float)::numeric, 3)  AS avg_score,
            attributes->>'action_taken' AS most_common_action
        FROM raglab_events
        WHERE service_name = 'pipeline'
          AND attributes ? 'gate_name'
          AND created_at > NOW() - INTERVAL ':hours hours'
        GROUP BY gate_name, attributes->>'action_taken'
        ORDER BY fired_count DESC
    """)
    try:
        result = await session.execute(sql, {"hours": hours})
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        log.warning("pipeline_health.gate_summary_failed", error=str(exc))
        return []
