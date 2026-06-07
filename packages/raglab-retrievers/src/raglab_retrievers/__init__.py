"""
raglab-retrievers — Retriever implementations for RAGLab.

Version: 0.3.0
Active in R1: DenseRetriever
Active in R3: BM25Retriever, HybridRetriever, MMRRetriever,
              ReRankerRetriever, CompressionRetriever
"""

from raglab_retrievers.base import BaseRetriever
from raglab_retrievers.dense_retriever import DenseRetriever
from raglab_retrievers.bm25_retriever import BM25Retriever, BM25Corpus
from raglab_retrievers.hybrid_retriever import HybridRetriever
from raglab_retrievers.mmr_retriever import MMRRetriever
from raglab_retrievers.reranker_retriever import ReRankerRetriever
from raglab_retrievers.compression_retriever import CompressionRetriever
from raglab_retrievers.graph_retriever import GraphRetriever
from raglab_retrievers.factory import RetrieverFactory

__version__ = "0.5.0"

__all__ = [
    "BaseRetriever", "RetrieverFactory",
    "DenseRetriever",
    "BM25Retriever", "BM25Corpus",
    "HybridRetriever",
    "MMRRetriever",
    "ReRankerRetriever",
    "CompressionRetriever", "GraphRetriever",
]
