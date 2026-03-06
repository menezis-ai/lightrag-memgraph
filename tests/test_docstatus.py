"""
Integration tests for MemgraphDocStatusStorage.

Requires a running Memgraph instance (set MEMGRAPH_URI).
"""

import pytest
from lightrag.base import DocProcessingStatus, DocStatus

from twindb_lightrag_memgraph import register
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

register()


def _make_status(
    status=DocStatus.PENDING,
    summary="test doc",
    length=100,
    file_path="/test.txt",
    track_id=None,
) -> DocProcessingStatus:
    return DocProcessingStatus(
        content_summary=summary,
        content_length=length,
        file_path=file_path,
        status=status,
        created_at="2025-01-01T00:00:00",
        updated_at="2025-01-01T00:00:00",
        track_id=track_id,
    )


@pytest.fixture
async def doc_store():
    store = MemgraphDocStatusStorage(
        namespace="test_docstatus",
        global_config={},
        embedding_func=None,
    )
    await store.initialize()
    yield store
    await store.drop()


@pytest.mark.integration
class TestMemgraphDocStatusStorage:
    async def test_upsert_and_get(self, doc_store):
        status = _make_status()
        await doc_store.upsert({"doc1": status})
        result = await doc_store.get_by_id("doc1")
        assert result is not None
        assert result["status"] == "pending"

    async def test_get_status_counts(self, doc_store):
        await doc_store.upsert(
            {
                "d1": _make_status(DocStatus.PENDING),
                "d2": _make_status(DocStatus.PENDING),
                "d3": _make_status(DocStatus.PROCESSED),
            }
        )
        counts = await doc_store.get_status_counts()
        assert counts.get("pending") == 2
        assert counts.get("processed") == 1

    async def test_get_docs_by_status(self, doc_store):
        await doc_store.upsert(
            {
                "d1": _make_status(DocStatus.PENDING),
                "d2": _make_status(DocStatus.PROCESSED),
            }
        )
        pending = await doc_store.get_docs_by_status(DocStatus.PENDING)
        assert "d1" in pending
        assert "d2" not in pending
        assert isinstance(pending["d1"], DocProcessingStatus)

    async def test_get_docs_by_track_id(self, doc_store):
        await doc_store.upsert(
            {
                "d1": _make_status(track_id="batch-42"),
                "d2": _make_status(track_id="batch-99"),
            }
        )
        docs = await doc_store.get_docs_by_track_id("batch-42")
        assert "d1" in docs
        assert "d2" not in docs

    async def test_get_docs_paginated(self, doc_store):
        for i in range(5):
            await doc_store.upsert({f"d{i}": _make_status()})
        docs, total = await doc_store.get_docs_paginated(page=1, page_size=3)
        assert total == 5
        assert len(docs) == 3

    async def test_get_doc_by_file_path(self, doc_store):
        await doc_store.upsert({"d1": _make_status(file_path="/special.pdf")})
        result = await doc_store.get_doc_by_file_path("/special.pdf")
        assert result is not None
        assert result["file_path"] == "/special.pdf"

    async def test_filter_keys(self, doc_store):
        await doc_store.upsert({"existing": _make_status()})
        missing = await doc_store.filter_keys({"existing", "absent"})
        assert "absent" in missing
        assert "existing" not in missing

    async def test_is_empty(self, doc_store):
        assert await doc_store.is_empty() is True
        await doc_store.upsert({"d1": _make_status()})
        assert await doc_store.is_empty() is False

    async def test_delete(self, doc_store):
        await doc_store.upsert({"d1": _make_status()})
        await doc_store.delete(["d1"])
        assert await doc_store.get_by_id("d1") is None
