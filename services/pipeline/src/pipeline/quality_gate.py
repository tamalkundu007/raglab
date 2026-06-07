"""
Pipeline quality gate — chunk quality remediation at ingestion time.

Sits between Step 2 (chunking) and Step 3 (embedding) in run_pipeline().
Scores each chunk via ChunkQualityScorer and applies the configured
quarantine strategy before any chunk reaches the vector store.

Three outcomes per chunk:
    accepted          — score >= threshold; proceeds to embedding + indexing
    flagged           — score < threshold, FLAG_ONLY strategy; keeps chunk
                        in the pipeline but injects quality metadata
    excluded          — score < threshold, EXCLUDE strategy; chunk dropped
    re_chunk_requested — score < threshold, RE_CHUNK strategy; chunk flagged
                         (actual re-chunking is a pipeline-service R6 feature;
                         in R5 we flag and document the intent)

Observable: every gate decision is logged with score + reason + action.
Toggleable: if chunk_quality_enabled=False, gate is a no-op pass-through.
"""

from __future__ import annotations

from typing import Any

from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel

log = get_logger(__name__)


def apply_quality_gate(
    chunks: list[ChunkModel],
    quality_config: dict[str, Any] | None,
    llm_caller=None,
) -> tuple[list[ChunkModel], dict[str, Any]]:
    """
    Apply chunk quality gate to a list of chunks.

    Args:
        chunks:         Chunks from ChunkerFactory.
        quality_config: Dict matching ChunkQualityConfig fields, or None to disable.
        llm_caller:     Optional injectable LLM caller for judge mode.

    Returns:
        (accepted_chunks, gate_summary)
        accepted_chunks — chunks that should proceed to embedding
        gate_summary    — dict with counts and per-chunk results for logging/tracing
    """
    if not quality_config or not quality_config.get("enabled", False):
        return chunks, {"enabled": False, "total": len(chunks), "accepted": len(chunks)}

    try:
        from raglab_eval import ChunkQualityConfig, ChunkQualityScorer, QuarantineStrategy
    except ImportError:
        log.warning("quality_gate.raglab_eval_not_installed")
        return chunks, {"enabled": False, "total": len(chunks), "accepted": len(chunks)}

    config = ChunkQualityConfig(**{
        k: v for k, v in quality_config.items()
        if k in ChunkQualityConfig.model_fields
    })
    scorer = ChunkQualityScorer(config)

    accepted: list[ChunkModel] = []
    flagged: list[ChunkModel] = []
    excluded: list[ChunkModel] = []
    results_summary: list[dict] = []

    for chunk in chunks:
        result = scorer.score(chunk, llm_caller=llm_caller)

        summary_entry = {
            "chunk_id": chunk.chunk_id,
            "score": result.score,
            "passed": result.passed,
            "action": result.action_taken,
            "reason": result.reason[:120],
        }
        results_summary.append(summary_entry)

        if result.action_taken == "excluded":
            excluded.append(chunk)
            log.info(
                "quality_gate.excluded",
                chunk_id=chunk.chunk_id,
                score=result.score,
                reason=result.reason[:80],
            )

        elif result.action_taken in ("flagged", "re_chunk_requested"):
            # Flag: keep in pipeline, inject quality metadata
            flagged_chunk = ChunkModel(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                metadata={
                    **chunk.metadata,
                    "quality_score": result.score,
                    "quality_passed": False,
                    "quality_action": result.action_taken,
                    "quality_reason": result.reason[:120],
                },
            )
            flagged.append(flagged_chunk)
            accepted.append(flagged_chunk)
            log.info(
                "quality_gate.flagged",
                chunk_id=chunk.chunk_id,
                score=result.score,
            )

        else:  # accepted
            accepted_chunk = ChunkModel(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                metadata={
                    **chunk.metadata,
                    "quality_score": result.score,
                    "quality_passed": True,
                    "quality_action": "accepted",
                },
            )
            accepted.append(accepted_chunk)

    gate_summary = {
        "enabled": True,
        "total": len(chunks),
        "accepted": len(accepted),
        "flagged": len(flagged),
        "excluded": len(excluded),
        "results": results_summary,
    }

    log.info(
        "quality_gate.complete",
        total=len(chunks),
        accepted=len(accepted),
        flagged=len(flagged),
        excluded=len(excluded),
    )

    return accepted, gate_summary
