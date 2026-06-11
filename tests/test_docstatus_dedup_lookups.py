"""
Offline unit tests for the LightRAG 1.5 dedup lookups on the DocStatus backend:
``get_doc_by_content_hash`` and ``get_doc_by_file_basename``.

These guard the Cypher query construction and the ``(doc_id, doc_data)`` return
contract without requiring a running Memgraph — all driver calls are mocked.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_mock_read_session(record):
    """Async session whose run().single() yields ``record`` (or None)."""
    mock_result = AsyncMock()
    mock_result.single = AsyncMock(return_value=record)
    mock_result.consume = AsyncMock()

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


@contextmanager
def _patch_read_pool(mock_session):
    with patch(
        "twindb_lightrag_memgraph._pool.get_read_session",
        return_value=mock_session,
    ):
        yield


def _make_docstatus_storage():
    from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

    return MemgraphDocStatusStorage(
        namespace="docstatus",
        global_config={},
        embedding_func=MagicMock(),
    )


class TestGetDocByContentHash:
    async def test_returns_doc_id_and_deserialized_data_when_found(self):
        record = {
            "id": "doc-42",
            "props": {"file_path": "r.pdf", "chunks_list": '["c1","c2"]'},
        }
        session = _make_mock_read_session(record)
        storage = _make_docstatus_storage()

        with _patch_read_pool(session):
            result = await storage.get_doc_by_content_hash("hash-abc")

        assert result is not None
        doc_id, doc_data = result
        assert doc_id == "doc-42"
        # JSON-encoded fields are deserialized back to Python objects.
        assert doc_data["chunks_list"] == ["c1", "c2"]
        # The content_hash is forwarded as a query parameter.
        assert session.run.call_args.kwargs["content_hash"] == "hash-abc"

    async def test_returns_none_when_not_found(self):
        session = _make_mock_read_session(None)
        storage = _make_docstatus_storage()

        with _patch_read_pool(session):
            result = await storage.get_doc_by_content_hash("missing")

        assert result is None

    async def test_empty_hash_short_circuits_without_query(self):
        session = _make_mock_read_session(None)
        storage = _make_docstatus_storage()

        with _patch_read_pool(session):
            result = await storage.get_doc_by_content_hash("")

        assert result is None
        session.run.assert_not_called()


class TestGetDocByFileBasename:
    async def test_matches_on_file_path_and_returns_tuple(self):
        record = {"id": "doc-7", "props": {"file_path": "report.pdf"}}
        session = _make_mock_read_session(record)
        storage = _make_docstatus_storage()

        with _patch_read_pool(session):
            result = await storage.get_doc_by_file_basename("report.pdf")

        assert result == ("doc-7", {"file_path": "report.pdf"})
        assert session.run.call_args.kwargs["basename"] == "report.pdf"

    async def test_returns_none_when_not_found(self):
        session = _make_mock_read_session(None)
        storage = _make_docstatus_storage()

        with _patch_read_pool(session):
            result = await storage.get_doc_by_file_basename("nope.pdf")

        assert result is None

    @pytest.mark.parametrize("basename", ["", "unknown_source"])
    async def test_sentinel_basenames_short_circuit_without_query(self, basename):
        session = _make_mock_read_session(None)
        storage = _make_docstatus_storage()

        with _patch_read_pool(session):
            result = await storage.get_doc_by_file_basename(basename)

        assert result is None
        session.run.assert_not_called()
