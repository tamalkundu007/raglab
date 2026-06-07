"""
raglab-eval — Self-Healing RAG evaluation package.

Version: 0.1.0

Modules:
    models        — shared types (EvalResult, ChunkQualityResult, etc.)
    chunk_quality — ChunkQualityScorer (heuristics + optional LLM judge)
    retrieval_heal — RetrievalHealer (Phase 4)
    groundedness  — GroundednessChecker (Phase 5)
"""

from raglab_eval.models import (
    EvalResult,
    ChunkQualityResult,
    ChunkQualityConfig,
    RetrievalHealResult,
    RetrievalHealConfig,
    GroundednessResult,
    GroundednessConfig,
    QuarantineStrategy,
    JudgeMode,
    GroundednessAction,
)
from raglab_eval.chunk_quality import ChunkQualityScorer
from raglab_eval.retrieval_heal import RetrievalHealer
from raglab_eval.groundedness import GroundednessChecker

__version__ = "0.1.0"

__all__ = [
    "EvalResult",
    "ChunkQualityResult", "ChunkQualityConfig",
    "RetrievalHealResult", "RetrievalHealConfig",
    "GroundednessResult", "GroundednessConfig",
    "QuarantineStrategy", "JudgeMode", "GroundednessAction",
    "ChunkQualityScorer",
    "RetrievalHealer",
    "GroundednessChecker",
]
