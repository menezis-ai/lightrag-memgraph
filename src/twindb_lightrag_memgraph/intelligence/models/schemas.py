"""
twin_rag_intelligence/models/schemas.py
========================================
Pydantic models for all data structures in the package.
"""

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    GREETING = "GREETING"
    MALICIOUS = "MALICIOUS"
    ESCALATION = "ESCALATION"


class AnswerStatus(str, Enum):
    """Epistemic state of an answer returned by the intelligence pipeline."""

    GROUNDED = "grounded"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"
    QUERY_FAILED = "query_failed"
    NO_RETRIEVAL = "no_retrieval"


class IntentResult(BaseModel):
    intent: IntentType = IntentType.IN_SCOPE
    confidence: float = 0.0
    reason: str = ""


class Citation(BaseModel):
    passage_index: int
    text: str = Field(description="Chunk excerpt (500 chars max)")
    document_id: Optional[str] = None
    document_path: Optional[str] = None
    source_workspace: str = "unknown"
    score: float = 0.0


class QueryTrace(BaseModel):
    """Execution trace for observability (Dynatrace / structured logs)."""

    question: str
    workspace: str
    user_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    latency_ms: float = 0.0
    thought: str = ""
    resolved_query: str = ""
    expansion_terms: list[str] = Field(default_factory=list)
    raw_chunks_count: int = 0
    reranked_chunks_count: int = 0
    tokens_used: int = 0
    intent: Optional[IntentResult] = None
    early_exit: Optional[str] = None

    def start(self) -> None:
        self.start_time = time.time()

    def stop(self, early_exit: Optional[str] = None) -> None:
        self.end_time = time.time()
        self.latency_ms = (self.end_time - self.start_time) * 1000
        self.early_exit = early_exit


class QueryResult(BaseModel):
    """Final result returned to the Agentic Platform (L4)."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    answer_status: AnswerStatus = AnswerStatus.GROUNDED
    trace: Optional[QueryTrace] = None
    intent: Optional[IntentResult] = None


class FeedbackEntry(BaseModel):
    """User feedback entry."""

    query_trace_id: str
    question: str
    answer: str
    score: int = Field(description="1 (positive) or -1 (negative)")
    comment: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: float = 0.0
