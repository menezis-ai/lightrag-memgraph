"""
intelligence
============
ReAct agent with AFFINE features for IT Ops RAG intelligence.

L3 layer between LightRAG Retriever (L2) and Agentic Platform (L4).
"""

from .config import TwinRAGConfig
from .engine import TwinRAGEngine
from .models.schemas import (
    Citation,
    FeedbackEntry,
    IntentResult,
    IntentType,
    QueryResult,
    QueryTrace,
)

__all__ = [
    "TwinRAGConfig",
    "TwinRAGEngine",
    "Citation",
    "FeedbackEntry",
    "IntentResult",
    "IntentType",
    "QueryResult",
    "QueryTrace",
]
