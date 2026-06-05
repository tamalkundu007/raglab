"""
raglab-retrievers — Retriever implementations for RAGLab.

Version: 0.2.0
Active in R1: DenseRetriever
Active in R3: BM25Retriever, HybridRetriever, MMRRetriever,
              ReRankerRetriever, CompressionRetriever

Public API:
    from raglab_retrievers import RetrieverFactory, DenseRetriever, BaseRetriever
"""

from raglab_retrievers.base import BaseRetriever
from raglab_retrievers.dense_retriever import DenseRetriever
from raglab_retrievers.factory import RetrieverFactory

__version__ = "0.2.0"

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "RetrieverFactory",
]
