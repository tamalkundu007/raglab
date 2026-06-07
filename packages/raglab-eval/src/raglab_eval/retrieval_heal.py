"""
RetrievalHealer — retrieval feedback loop with strategy escalation.

Detect → Score → Remediate pattern:

    Detect:  retrieval results are weak (too few, scores below floor, relevance gap)
    Score:   RetrievalHealResult with numeric score + reason
    Remediate: re-retrieve with next strategy in escalation_order; bounded retries

Three weakness signals:
    1. result_count < min_results — not enough chunks returned
    2. top_score < score_floor    — best match too weak for confident retrieval
    3. relevance_gap              — (future) top result much weaker than expected

Escalation order (configurable):
    Default: ["dense", "hybrid", "bm25"]
    On first failure: try dense → if still weak → try hybrid → if still weak → try bm25
    After max_healing_retries: return best available result + healed=True in metadata

Observable: every heal attempt logged with strategy + score + outcome.
Toggleable: config.enabled=False returns results as-is.

The healer is stateless — it holds no HTTP client.
It calls the passed `retriever_fn` callable:
    retriever_fn(query_text: str, strategy: str, top_k: int) → list[ChunkModel]

In production, pipeline-service passes a closure over the retrieval-service HTTP client.
In tests, retriever_fn is a pure Python mock.
"""

from __future__ import annotations

from typing import Any, Callable

from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel

from raglab_eval.models import RetrievalHealConfig, RetrievalHealResult

log = get_logger(__name__)


class RetrievalHealer:
    """
    Retrieval feedback loop — detects weak results and escalates strategy.

    Observable: every attempt logged.
    Toggleable: config.enabled=False is a pass-through.
    Stateless: retriever_fn injected.
    """

    def __init__(self, config: RetrievalHealConfig | None = None) -> None:
        self.config = config or RetrievalHealConfig()

    def heal(
        self,
        query_text: str,
        initial_results: list[ChunkModel],
        initial_strategy: str,
        retriever_fn: Callable[[str, str, int], list[ChunkModel]],
        top_k: int = 5,
    ) -> tuple[list[ChunkModel], RetrievalHealResult]:
        """
        Evaluate retrieval results and escalate if weak.

        Args:
            query_text:       The user query string.
            initial_results:  Results from the first retrieval attempt.
            initial_strategy: Strategy that produced initial_results (e.g. "dense").
            retriever_fn:     Callable(query, strategy, top_k) → list[ChunkModel].
            top_k:            How many chunks to request per attempt.

        Returns:
            (best_results, RetrievalHealResult)
        """
        cfg = self.config

        if not cfg.enabled:
            return initial_results, RetrievalHealResult(
                score=1.0, passed=True,
                reason="Retrieval healing disabled",
                action_taken="skipped",
                query_text=query_text,
                original_strategy=initial_strategy,
                final_strategy=initial_strategy,
                result_count=len(initial_results),
            )

        # Evaluate initial results
        weak, score, reason = self._is_weak(initial_results)

        if not weak:
            return initial_results, RetrievalHealResult(
                score=score, passed=True,
                reason=reason,
                action_taken="none",
                query_text=query_text,
                original_strategy=initial_strategy,
                final_strategy=initial_strategy,
                result_count=len(initial_results),
                top_score=self._top_score(initial_results),
            )

        log.info(
            "retrieval_healer.weak_detected",
            query=query_text[:60],
            strategy=initial_strategy,
            results=len(initial_results),
            score=score,
            reason=reason,
        )

        # Build escalation sequence — skip strategies already tried
        escalation = [s for s in cfg.escalation_order if s != initial_strategy]
        best_results = initial_results
        best_score = score
        final_strategy = initial_strategy
        retries = 0

        for strategy in escalation[:cfg.max_healing_retries]:
            retries += 1
            try:
                healed_results = retriever_fn(query_text, strategy, top_k)
            except Exception as exc:
                log.warning(
                    "retrieval_healer.attempt_failed",
                    strategy=strategy,
                    error=str(exc),
                )
                continue

            still_weak, new_score, new_reason = self._is_weak(healed_results)

            log.info(
                "retrieval_healer.attempt",
                strategy=strategy,
                results=len(healed_results),
                score=new_score,
                still_weak=still_weak,
            )

            if new_score > best_score:
                best_results = healed_results
                best_score = new_score
                final_strategy = strategy

            if not still_weak:
                break  # healed — stop escalating

        healed = final_strategy != initial_strategy
        passed = not self._is_weak(best_results)[0]

        # Inject heal metadata into results
        if healed:
            best_results = self._tag_healed(best_results, initial_strategy, final_strategy)

        action = "healed" if healed else "escalated_no_improvement"

        log.info(
            "retrieval_healer.complete",
            original=initial_strategy,
            final=final_strategy,
            retries=retries,
            result_count=len(best_results),
            healed=healed,
        )

        return best_results, RetrievalHealResult(
            score=best_score,
            passed=passed,
            reason=f"Escalated {initial_strategy}→{final_strategy} after {retries} attempt(s). {new_reason if retries else reason}",
            action_taken=action,
            query_text=query_text,
            original_strategy=initial_strategy,
            final_strategy=final_strategy,
            retries=retries,
            result_count=len(best_results),
            top_score=self._top_score(best_results),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_weak(
        self, results: list[ChunkModel]
    ) -> tuple[bool, float, str]:
        """
        Evaluate whether retrieval results are weak.

        Returns:
            (is_weak, score, reason)
            score: 0.0 = definitely weak, 1.0 = strong
        """
        cfg = self.config

        if len(results) < cfg.min_results:
            count_ratio = len(results) / max(cfg.min_results, 1)
            return True, count_ratio, f"Too few results: {len(results)} < {cfg.min_results}"

        top = self._top_score(results)
        if top is not None and top < cfg.score_floor:
            score = top / cfg.score_floor
            return True, score, f"Top score {top:.3f} below floor {cfg.score_floor}"

        return False, 1.0, f"Retrieval adequate: {len(results)} results, top_score={top}"

    @staticmethod
    def _top_score(results: list[ChunkModel]) -> float | None:
        """Extract the highest score from chunk metadata."""
        scores = []
        for chunk in results:
            for key in ("score", "rrf_score", "reranker_score"):
                s = chunk.metadata.get(key)
                if s is not None:
                    scores.append(float(s))
                    break
        return max(scores) if scores else None

    @staticmethod
    def _tag_healed(
        results: list[ChunkModel],
        original_strategy: str,
        final_strategy: str,
    ) -> list[ChunkModel]:
        """Inject heal metadata into result chunks."""
        tagged = []
        for chunk in results:
            tagged.append(ChunkModel(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                metadata={
                    **chunk.metadata,
                    "healed": True,
                    "original_strategy": original_strategy,
                    "final_strategy": final_strategy,
                },
            ))
        return tagged
