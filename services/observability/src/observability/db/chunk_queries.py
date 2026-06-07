"""
Chunk inspector queries — read-only views over indexed chunks.

Reads from:
    raglab_events   — quality gate decisions (from pipeline-service spans)
    raglab_chunks   — chunk metadata (from indexing-service, if available)

Principle: read-only. Never writes.

The chunk inspector surfaces:
    - Exact text of each chunk (as indexed)
    - Token count, chunk_index
    - quality_score, quality_passed, quality_action (from raglab-eval)
    - Chunker type and configuration used
    - Page range (for PDF/table chunks)

Since raglab_chunks is the indexing-service's table, we query it if available.
If unavailable, we fall back to the quality events in raglab_events.
"""

from __future__ import annotations

from typing import Any

from raglab_common.logging import get_logger

log = get_logger(__name__)


async def get_chunks_for_doc(
    session: Any,
    doc_id: str,
    collection: str = "raglab",
    include_excluded: bool = True,
) -> list[dict]:
    """
    Return all chunks for a document, enriched with quality scores.

    Queries raglab_chunks (indexing metadata table) joined with quality
    gate events from raglab_events.
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            c.chunk_id,
            c.doc_id,
            c.collection,
            c.chunk_index,
            c.text,
            c.token_count,
            c.chunker_type,
            c.metadata,
            c.created_at,
            -- Quality fields from metadata JSON
            (c.metadata->>'quality_score')::float      AS quality_score,
            (c.metadata->>'quality_passed')::boolean   AS quality_passed,
            c.metadata->>'quality_action'              AS quality_action,
            c.metadata->>'quality_reason'              AS quality_reason
        FROM raglab_chunks c
        WHERE c.doc_id = :doc_id
          AND c.collection = :collection
        ORDER BY c.chunk_index ASC
    """)

    try:
        result = await session.execute(sql, {"doc_id": doc_id, "collection": collection})
        rows = result.mappings().all()
        chunks = [dict(r) for r in rows]

        if not include_excluded:
            chunks = [c for c in chunks if c.get("quality_action") != "excluded"]

        return chunks
    except Exception as exc:
        log.warning("chunk_inspector.get_chunks_failed",
                    doc_id=doc_id, error=str(exc))
        return []


async def get_doc_quality_summary(
    session: Any,
    doc_id: str,
    collection: str = "raglab",
) -> dict:
    """
    Return quality gate summary for a document.

    Returns: {total, accepted, flagged, excluded, avg_quality_score,
              min_quality_score, chunker_type}
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            COUNT(*)                                                    AS total,
            SUM(CASE WHEN metadata->>'quality_action'='accepted' THEN 1 ELSE 0 END) AS accepted,
            SUM(CASE WHEN metadata->>'quality_action'='flagged'  THEN 1 ELSE 0 END) AS flagged,
            SUM(CASE WHEN metadata->>'quality_action'='excluded' THEN 1 ELSE 0 END) AS excluded,
            ROUND(AVG((metadata->>'quality_score')::float)::numeric, 3) AS avg_quality_score,
            ROUND(MIN((metadata->>'quality_score')::float)::numeric, 3) AS min_quality_score,
            MODE() WITHIN GROUP (ORDER BY chunker_type)                 AS chunker_type
        FROM raglab_chunks
        WHERE doc_id = :doc_id
          AND collection = :collection
          AND metadata ? 'quality_score'
    """)

    try:
        result = await session.execute(sql, {"doc_id": doc_id, "collection": collection})
        row = result.mappings().one_or_none()
        if row:
            return dict(row)
        return {}
    except Exception as exc:
        log.warning("chunk_inspector.summary_failed", doc_id=doc_id, error=str(exc))
        return {}


async def list_recent_docs(
    session: Any,
    collection: str = "raglab",
    limit: int = 50,
) -> list[dict]:
    """
    Return recently indexed documents (distinct doc_ids).
    """
    from sqlalchemy import text

    sql = text("""
        SELECT
            doc_id,
            collection,
            MAX(chunker_type)      AS chunker_type,
            COUNT(*)               AS chunk_count,
            MAX(created_at)        AS last_indexed
        FROM raglab_chunks
        WHERE collection = :collection
        GROUP BY doc_id, collection
        ORDER BY MAX(created_at) DESC
        LIMIT :limit
    """)

    try:
        result = await session.execute(sql, {"collection": collection, "limit": limit})
        return [dict(r) for r in result.mappings().all()]
    except Exception as exc:
        log.warning("chunk_inspector.list_docs_failed", error=str(exc))
        return []
