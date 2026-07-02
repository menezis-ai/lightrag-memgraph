"""Tests for Feedback store."""

import pytest

from twindb_lightrag_memgraph.intelligence.features.feedback import FeedbackStore


class TestFeedbackStore:
    """Feedback system tests."""

    @pytest.fixture
    def store(self, config):
        return FeedbackStore(config)

    def test_record_positive(self, store):
        entry = store.record(
            query_trace_id="trace-001",
            question="How to fix ORA-04030?",
            answer="Increase PGA.",
            score=1,
            user_id="user-42",
        )
        assert entry.score == 1
        assert entry.timestamp > 0

    def test_record_negative_with_comment(self, store):
        entry = store.record(
            query_trace_id="trace-002",
            question="VLAN config?",
            answer="Wrong answer.",
            score=-1,
            comment="The answer was about Oracle, not networking",
        )
        assert entry.score == -1
        assert entry.comment is not None

    def test_get_stats(self, store):
        store.record("t1", "q1", "a1", 1)
        store.record("t2", "q2", "a2", -1)
        store.record("t3", "q3", "a3", 1)

        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["positive"] == 2
        assert stats["negative"] == 1

    def test_get_entries_limit(self, store):
        for i in range(10):
            store.record(f"t{i}", f"q{i}", f"a{i}", 1)
        entries = store.get_entries(limit=5)
        assert len(entries) == 5

    def test_clear(self, store):
        store.record("t1", "q1", "a1", 1)
        store.clear()
        assert store.get_stats()["total"] == 0
