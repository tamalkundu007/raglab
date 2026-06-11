"""
Observability event store — read-only queries against raglab_events.

This module NEVER writes to raglab_events. It only reads.
Writes happen exclusively via PostgresSpanExporter in raglab-common/tracing.py.

The raglab_events table schema (created by tracing.py):
    id              UUID
    trace_id        TEXT        -- 32-hex string
    span_id         TEXT        -- 16-hex string
    parent_span_id  TEXT|NULL
    service_name    TEXT
    operation_name  TEXT
    start_time_ms   BIGINT      -- Unix ms
    duration_ms     BIGINT
    status          TEXT        -- 'ok' | 'error' | 'unset'
    attributes      JSONB
    events          JSONB       -- list of {name, attributes}
    created_at      TIMESTAMPTZ

All query functions are async, accept an AsyncSession, return plain dicts.
Plain dicts avoid ORM overhead for read-heavy observability workloads.
"""

from __future__ import annotations

from typing import Any

from raglab_common.logging import get_logger

log = get_logger(__name__)


# ── Queries ───────────────────────────────────────────────────────────────────

async def list_recent_traces(
    session: Any,
    limit: int = 50,
    service_name: str | None = None,
    status: str | None = None,
    tenant_id: str | None = None,
) -> list[dict]:
    """
    Return the most recent traces (one row per unique trace_id).

    Groups spans by trace_id and returns summary: total spans,
    root service, first start, total duration.
    """
    from sqlalchemy import text

    where_clauses = ["1=1"]
    params: dict[str, Any] = {"limit": limit}

    if service_name:
        where_clauses.append("service_name = :service_name")
        params["service_name"] = service_name
    if status:
        where_clauses.append("status = :status")
        params["status"] = status
    if tenant_id:
        # R7: scope traces to tenant (via attributes->>'tenant_id' in span data)
        where_clauses.append("(attributes->>'tenant_id' = :tenant_id OR attributes IS NULL)")
        params["tenant_id"] = tenant_id

    where = " AND ".join(where_clauses)

    sql = text(f"""
        SELECT
            trace_id,
            COUNT(*)                         AS span_count,
            MIN(start_time_ms)               AS start_ms,
            MAX(start_time_ms + duration_ms) AS end_ms,
            MAX(start_time_ms + duration_ms)
                - MIN(start_time_ms)         AS total_duration_ms,
            array_agg(DISTINCT service_name) AS services,
            bool_or(status = 'error')        AS has_error
        FROM raglab_events
        WHERE {where}
        GROUP BY trace_id
        ORDER BY MIN(start_time_ms) DESC
        LIMIT :limit
    """)

    try:
        result = await session.execute(sql, params)
        rows = result.mappings().all()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning("observability.list_traces_failed", error=str(exc))
        return []


async def get_trace(session: Any, trace_id: str) -> list[dict]:
    """
    Return all spans for a single trace_id, ordered by start_time_ms.

    Used by the trace viewer to render the waterfall timeline.
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            trace_id,
            span_id,
            parent_span_id,
            service_name,
            operation_name,
            start_time_ms,
            duration_ms,
            status,
            attributes,
            events,
            created_at
        FROM raglab_events
        WHERE trace_id = :trace_id
        ORDER BY start_time_ms ASC
    """)

    try:
        result = await session.execute(sql, {"trace_id": trace_id})
        rows = result.mappings().all()
        return [dict(r) for r in rows]
    except Exception as exc:
        log.warning("observability.get_trace_failed", trace_id=trace_id, error=str(exc))
        return []


async def get_service_stats(
    session: Any,
    hours: int = 24,
) -> list[dict]:
    """
    Return per-service request counts, error rates, and avg duration
    over the last N hours.
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            service_name,
            COUNT(*)                                    AS total_spans,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count,
            ROUND(AVG(duration_ms)::numeric, 1)         AS avg_duration_ms,
            ROUND(MAX(duration_ms)::numeric, 1)         AS p100_duration_ms,
            COUNT(DISTINCT trace_id)                    AS unique_traces
        FROM raglab_events
        WHERE created_at > NOW() - INTERVAL ':hours hours'
          AND parent_span_id IS NULL
        GROUP BY service_name
        ORDER BY total_spans DESC
    """)

    try:
        result = await session.execute(sql, {"hours": hours})
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        log.warning("observability.service_stats_failed", error=str(exc))
        return []


async def get_trace_timeline(session: Any, trace_id: str) -> dict:
    """
    Build a timeline-ready structure for D3.js waterfall rendering.

    Returns:
        {
          trace_id: str,
          total_duration_ms: int,
          start_ms: int,
          spans: [
            {
              span_id, parent_span_id, service_name, operation_name,
              start_offset_ms,   # relative to trace start
              duration_ms,
              status,
              attributes,
              events,
              depth,             # tree depth for indentation
            }
          ]
        }
    """
    spans = await get_trace(session, trace_id)
    if not spans:
        return {"trace_id": trace_id, "total_duration_ms": 0, "start_ms": 0, "spans": []}

    trace_start = min(s["start_time_ms"] for s in spans)
    trace_end   = max(s["start_time_ms"] + s["duration_ms"] for s in spans)

    # Build parent→children map for depth calculation
    children: dict[str | None, list[str]] = {}
    span_map: dict[str, dict] = {}
    for s in spans:
        span_map[s["span_id"]] = s
        parent = s["parent_span_id"]
        children.setdefault(parent, []).append(s["span_id"])

    # BFS to assign depth
    depths: dict[str, int] = {}
    queue = [(sid, 0) for sid in children.get(None, [])]
    while queue:
        sid, depth = queue.pop(0)
        depths[sid] = depth
        for child_sid in children.get(sid, []):
            queue.append((child_sid, depth + 1))

    timeline_spans = []
    for s in spans:
        import json
        attrs = s["attributes"]
        evts  = s["events"]
        if isinstance(attrs, str):
            try: attrs = json.loads(attrs)
            except Exception: attrs = {}
        if isinstance(evts, str):
            try: evts = json.loads(evts)
            except Exception: evts = []

        timeline_spans.append({
            "span_id":        s["span_id"],
            "parent_span_id": s["parent_span_id"],
            "service_name":   s["service_name"],
            "operation_name": s["operation_name"],
            "start_offset_ms": s["start_time_ms"] - trace_start,
            "duration_ms":    s["duration_ms"],
            "status":         s["status"],
            "attributes":     attrs,
            "events":         evts,
            "depth":          depths.get(s["span_id"], 0),
        })

    return {
        "trace_id":          trace_id,
        "total_duration_ms": trace_end - trace_start,
        "start_ms":          trace_start,
        "spans":             timeline_spans,
    }
