"""layarank: High-performance local relevance filtering and reranking for RAG."""

from .errors import (
    ConfigurationError,
    ContextLimitError,
    LayaRankError,
    ModelExecutionError,
)
from .instructions import (
    PAIRWISE_INSTRUCTION,
    POINTWISE_RELEVANCE_INSTRUCTION,
    RELEVANCE_INSTRUCTION,
    RERANK_INSTRUCTION,
)
from .reranker import LayaRank, LayaRanker

__version__ = "0.1.0"
__all__ = [
    "LayaRank",
    "LayaRanker",
    "LayaRankError",
    "ConfigurationError",
    "ContextLimitError",
    "ModelExecutionError",
    "RERANK_INSTRUCTION",
    "RELEVANCE_INSTRUCTION",
    "POINTWISE_RELEVANCE_INSTRUCTION",
    "PAIRWISE_INSTRUCTION",
]
