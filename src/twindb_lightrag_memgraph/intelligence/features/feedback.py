"""
twin_rag_intelligence/features/feedback.py
============================================
Feedback / Evaluation loop.

Stores user feedback (thumbs up/down + optional comment) for quality monitoring.
This is a lightweight in-process store. In production, feedback is persisted
by the Agentic Platform (L4) or the Backend (L5).
"""

import logging
import time
from typing import Optional

from ..config import TwinRAGConfig
from ..models.schemas import FeedbackEntry

logger = logging.getLogger("twin_rag_intelligence.feedback")


class FeedbackStore:
    """
    In-memory feedback store for the RAG engine.

    In production, feedback is persisted externally (Supabase / Memgraph).
    This store provides a local buffer and metrics aggregation.
    """

    def __init__(self, config: TwinRAGConfig) -> None:
        self.config = config
        self._entries: list[FeedbackEntry] = []

    def record(
        self,
        query_trace_id: str,
        question: str,
        answer: str,
        score: int,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> FeedbackEntry:
        """
        Record a feedback entry.

        Args:
            query_trace_id: ID linking to the QueryTrace.
            question: Original question.
            answer: Generated answer.
            score: 1 (positive) or -1 (negative).
            comment: Optional user comment.
            user_id: Optional user identifier.

        Returns:
            The created FeedbackEntry.
        """
        entry = FeedbackEntry(
            query_trace_id=query_trace_id,
            question=question,
            answer=answer,
            score=score,
            comment=comment,
            user_id=user_id,
            timestamp=time.time(),
        )
        self._entries.append(entry)
        logger.info(
            "Feedback recorded: trace=%s score=%d comment=%s",
            query_trace_id,
            score,
            comment[:50] if comment else None,
        )
        return entry

    def get_stats(self) -> dict[str, int]:
        """Return aggregated feedback statistics."""
        positive = sum(1 for e in self._entries if e.score > 0)
        negative = sum(1 for e in self._entries if e.score < 0)
        return {
            "total": len(self._entries),
            "positive": positive,
            "negative": negative,
        }

    def get_entries(self, limit: int = 50) -> list[FeedbackEntry]:
        """Return the most recent feedback entries."""
        return self._entries[-limit:]

    def clear(self) -> None:
        """Clear the in-memory feedback store."""
        self._entries.clear()
