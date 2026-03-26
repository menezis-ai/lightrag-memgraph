"""Unit tests for TTL (Time-To-Live) support on KV storage nodes.

Memgraph Enterprise can auto-delete nodes with the :TTL label and a ``ttl``
integer property (Unix epoch expiry).  These tests verify that:
- TTL properties and labels are added when MEMGRAPH_TTL_SECONDS is set
- Only configured namespaces receive TTL labels
- The feature is completely inert when MEMGRAPH_TTL_SECONDS is unset

No Memgraph required -- all DB interactions are mocked.
"""

import time
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

import twindb_lightrag_memgraph._pool as pool

# ── Helpers ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def _noop_write_slot():
    yield


class QueryCapture:
    """Captures all queries and params passed to session.run()."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def make_session_factory(self):
        capture = self

        @asynccontextmanager
        async def mock_session():
            session = AsyncMock()

            async def tracking_run(query, **params):
                capture.calls.append((query, params))
                result = AsyncMock()
                result.consume = AsyncMock()
                result.single = AsyncMock(return_value=None)
                return result

            session.run = tracking_run
            yield session

        return mock_session


def _make_kv_store(monkeypatch, namespace="full_docs"):
    """Create a MemgraphKVStorage instance with mocked pool."""
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "test")
    from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage

    store = MemgraphKVStorage.__new__(MemgraphKVStorage)
    store.namespace = namespace
    store.workspace = "test"
    store.global_config = {}
    store.embedding_func = None
    return store


# ── _ttl module unit tests ───────────────────────────────────────────────


class TestGetTTLSeconds:
    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_TTL_SECONDS", raising=False)
        from twindb_lightrag_memgraph._ttl import get_ttl_seconds

        assert get_ttl_seconds() is None

    def test_valid_value(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_TTL_SECONDS", "86400")
        from twindb_lightrag_memgraph._ttl import get_ttl_seconds

        assert get_ttl_seconds() == 86400

    def test_invalid_value_returns_none(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_TTL_SECONDS", "garbage")
        from twindb_lightrag_memgraph._ttl import get_ttl_seconds

        assert get_ttl_seconds() is None

    def test_zero_returns_none(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_TTL_SECONDS", "0")
        from twindb_lightrag_memgraph._ttl import get_ttl_seconds

        assert get_ttl_seconds() is None

    def test_negative_returns_none(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_TTL_SECONDS", "-100")
        from twindb_lightrag_memgraph._ttl import get_ttl_seconds

        assert get_ttl_seconds() is None


class TestComputeTTLTimestamp:
    def test_returns_future_timestamp(self):
        from twindb_lightrag_memgraph._ttl import compute_ttl_timestamp

        now = time.time()
        ts = compute_ttl_timestamp(3600)
        assert ts is not None
        assert ts > now
        assert ts <= now + 3600 + 1  # +1 for rounding

    def test_returns_none_when_none(self):
        from twindb_lightrag_memgraph._ttl import compute_ttl_timestamp

        assert compute_ttl_timestamp(None) is None


class TestGetTTLNamespaces:
    def test_default_namespaces(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_TTL_LABELS", raising=False)
        from twindb_lightrag_memgraph._ttl import get_ttl_namespaces

        ns = get_ttl_namespaces()
        assert ns == {"full_docs", "text_chunks"}

    def test_custom_namespaces(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_TTL_LABELS", "full_docs")
        from twindb_lightrag_memgraph._ttl import get_ttl_namespaces

        ns = get_ttl_namespaces()
        assert ns == {"full_docs"}
        assert "text_chunks" not in ns


# ── KV upsert TTL integration tests ─────────────────────────────────────


class TestKVUpsertTTL:
    async def test_ttl_disabled_by_default(self, monkeypatch):
        """No MEMGRAPH_TTL_SECONDS env var -> upsert Cypher has no ttl."""
        monkeypatch.delenv("MEMGRAPH_TTL_SECONDS", raising=False)
        store = _make_kv_store(monkeypatch, namespace="full_docs")
        capture = QueryCapture()

        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", capture.make_session_factory()),
        ):
            await store.upsert({"doc1": {"text": "hello"}})

        assert len(capture.calls) == 1, "Expected exactly 1 query (no TTL label query)"
        query, params = capture.calls[0]
        assert "ttl" not in query.lower()
        assert "ttl_ts" not in params

    async def test_ttl_enabled_adds_property(self, monkeypatch):
        """MEMGRAPH_TTL_SECONDS=86400, namespace=full_docs -> ttl property set."""
        monkeypatch.setenv("MEMGRAPH_TTL_SECONDS", "86400")
        monkeypatch.delenv("MEMGRAPH_TTL_LABELS", raising=False)
        store = _make_kv_store(monkeypatch, namespace="full_docs")
        capture = QueryCapture()

        now = time.time()
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", capture.make_session_factory()),
        ):
            await store.upsert({"doc1": {"text": "hello"}})

        # First call: MERGE with ttl property
        assert len(capture.calls) == 2, "Expected 2 queries (MERGE + TTL label)"
        query, params = capture.calls[0]
        assert "n.ttl = $ttl_ts" in query
        assert "ttl_ts" in params
        assert params["ttl_ts"] > now

    async def test_ttl_respects_namespace_filter(self, monkeypatch):
        """TTL enabled but namespace not in the list -> no TTL property."""
        monkeypatch.setenv("MEMGRAPH_TTL_SECONDS", "3600")
        monkeypatch.delenv("MEMGRAPH_TTL_LABELS", raising=False)
        store = _make_kv_store(monkeypatch, namespace="llm_response_cache")
        capture = QueryCapture()

        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", capture.make_session_factory()),
        ):
            await store.upsert({"k1": {"v": 1}})

        assert len(capture.calls) == 1, "No TTL label query for excluded namespace"
        query, params = capture.calls[0]
        assert "ttl" not in query.lower()
        assert "ttl_ts" not in params

    async def test_ttl_custom_labels_env(self, monkeypatch):
        """MEMGRAPH_TTL_LABELS=full_docs -> full_docs gets TTL, text_chunks does not."""
        monkeypatch.setenv("MEMGRAPH_TTL_SECONDS", "7200")
        monkeypatch.setenv("MEMGRAPH_TTL_LABELS", "full_docs")

        # full_docs should get TTL
        store_fd = _make_kv_store(monkeypatch, namespace="full_docs")
        capture_fd = QueryCapture()
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", capture_fd.make_session_factory()),
        ):
            await store_fd.upsert({"d1": {"x": 1}})
        assert len(capture_fd.calls) == 2, "full_docs should have TTL queries"
        assert "n.ttl = $ttl_ts" in capture_fd.calls[0][0]

        # text_chunks should NOT get TTL
        store_tc = _make_kv_store(monkeypatch, namespace="text_chunks")
        capture_tc = QueryCapture()
        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", capture_tc.make_session_factory()),
        ):
            await store_tc.upsert({"c1": {"y": 2}})
        assert len(capture_tc.calls) == 1, "text_chunks should NOT have TTL queries"
        assert "ttl" not in capture_tc.calls[0][0].lower()

    async def test_ttl_label_query_executed(self, monkeypatch):
        """When TTL is enabled, a second query adds the :TTL label."""
        monkeypatch.setenv("MEMGRAPH_TTL_SECONDS", "3600")
        monkeypatch.delenv("MEMGRAPH_TTL_LABELS", raising=False)
        store = _make_kv_store(monkeypatch, namespace="full_docs")
        capture = QueryCapture()

        with (
            patch.object(pool, "acquire_write_slot", _noop_write_slot),
            patch.object(pool, "get_session", capture.make_session_factory()),
        ):
            await store.upsert({"doc1": {"text": "data"}})

        assert len(capture.calls) == 2
        label_query = capture.calls[1][0]
        assert "SET n:TTL" in label_query
        assert "n.ttl IS NOT NULL" in label_query
        assert "NOT n:TTL" in label_query
