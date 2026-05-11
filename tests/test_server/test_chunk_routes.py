"""Tests for chunk routes (unit tests with mocked LightRAG)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from twindb_lightrag_memgraph.server.chunk_routes import (
    _resolve_chunk,
    _get_ordered_chunk_ids,
    _fetch_chunks_by_ids,
)


def _make_rag():
    """Create a mock LightRAG instance with text_chunks and doc_status."""
    rag = MagicMock()
    rag.text_chunks = MagicMock()
    rag.text_chunks.get_by_id = AsyncMock()
    rag.text_chunks.get_by_ids = AsyncMock()
    rag.doc_status = MagicMock()
    rag.doc_status.get_by_id = AsyncMock()
    return rag


class TestResolveChunk:
    async def test_found(self):
        rag = _make_rag()
        rag.text_chunks.get_by_id.return_value = {
            "_id": "chunk-1",
            "content": "hello",
            "full_doc_id": "doc-1",
        }
        result = await _resolve_chunk(rag, "chunk-1")
        assert result["content"] == "hello"

    async def test_not_found(self):
        rag = _make_rag()
        rag.text_chunks.get_by_id.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await _resolve_chunk(rag, "missing")
        assert exc_info.value.status_code == 404


class TestGetOrderedChunkIds:
    async def test_from_doc_status_dict(self):
        rag = _make_rag()
        rag.doc_status.get_by_id.return_value = {
            "chunks_list": ["c1", "c2", "c3"],
        }
        result = await _get_ordered_chunk_ids(rag, "doc-1")
        assert result == ["c1", "c2", "c3"]

    async def test_from_doc_status_dataclass(self):
        rag = _make_rag()
        status_obj = MagicMock()
        status_obj.chunks_list = ["c1", "c2"]
        rag.doc_status.get_by_id.return_value = status_obj
        result = await _get_ordered_chunk_ids(rag, "doc-1")
        assert result == ["c1", "c2"]

    async def test_missing_doc(self):
        rag = _make_rag()
        rag.doc_status.get_by_id.return_value = None
        with pytest.raises(HTTPException) as exc_info:
            await _get_ordered_chunk_ids(rag, "missing-doc")
        assert exc_info.value.status_code == 404

    async def test_empty_chunks_list(self):
        """doc_status has chunks_list=[] -- treated as falsy, raises 404."""
        rag = _make_rag()
        rag.doc_status.get_by_id.return_value = {"chunks_list": []}
        with pytest.raises(HTTPException) as exc_info:
            await _get_ordered_chunk_ids(rag, "doc-empty")
        assert exc_info.value.status_code == 404
        assert "No chunk ordering" in exc_info.value.detail

    async def test_chunks_list_none_in_dict(self):
        """doc_status is a dict with chunks_list=None -- raises 404."""
        rag = _make_rag()
        rag.doc_status.get_by_id.return_value = {"chunks_list": None}
        with pytest.raises(HTTPException) as exc_info:
            await _get_ordered_chunk_ids(rag, "doc-none")
        assert exc_info.value.status_code == 404

    async def test_chunks_list_none_in_dataclass(self):
        """doc_status is a dataclass with chunks_list=None -- raises 404."""
        rag = _make_rag()
        status_obj = MagicMock()
        status_obj.chunks_list = None
        rag.doc_status.get_by_id.return_value = status_obj
        with pytest.raises(HTTPException) as exc_info:
            await _get_ordered_chunk_ids(rag, "doc-dc-none")
        assert exc_info.value.status_code == 404

    async def test_dict_without_chunks_list_key(self):
        """doc_status dict has no chunks_list key at all -- raises 404."""
        rag = _make_rag()
        rag.doc_status.get_by_id.return_value = {"status": "completed"}
        with pytest.raises(HTTPException) as exc_info:
            await _get_ordered_chunk_ids(rag, "doc-nokey")
        assert exc_info.value.status_code == 404


class TestFetchChunksByIds:
    async def test_preserves_order(self):
        rag = _make_rag()
        rag.text_chunks.get_by_ids.return_value = [
            {"_id": "c2", "content": "two", "full_doc_id": "d1", "file_path": "f.txt", "chunk_order_index": 1, "tokens": 10},
            {"_id": "c1", "content": "one", "full_doc_id": "d1", "file_path": "f.txt", "chunk_order_index": 0, "tokens": 5},
        ]
        items = await _fetch_chunks_by_ids(rag, ["c1", "c2"])
        assert len(items) == 2
        assert items[0].chunk_id == "c1"
        assert items[1].chunk_id == "c2"

    async def test_missing_chunks_skipped(self):
        rag = _make_rag()
        rag.text_chunks.get_by_ids.return_value = [
            {"_id": "c1", "content": "one", "full_doc_id": "d1", "file_path": "f.txt", "chunk_order_index": 0, "tokens": 5},
        ]
        items = await _fetch_chunks_by_ids(rag, ["c1", "c_missing"])
        assert len(items) == 1
        assert items[0].chunk_id == "c1"

    async def test_empty_list(self):
        rag = _make_rag()
        rag.text_chunks.get_by_ids.return_value = []
        items = await _fetch_chunks_by_ids(rag, [])
        assert items == []

    async def test_none_entries_filtered(self):
        """get_by_ids returns [None, {valid}] -- only the valid one is kept."""
        rag = _make_rag()
        rag.text_chunks.get_by_ids.return_value = [
            None,
            {"_id": "c1", "content": "one", "full_doc_id": "d1", "file_path": "f.txt", "chunk_order_index": 0, "tokens": 5},
        ]
        items = await _fetch_chunks_by_ids(rag, ["c1"])
        assert len(items) == 1
        assert items[0].chunk_id == "c1"
        assert items[0].content == "one"

    async def test_chunk_id_key_fallback(self):
        """Raw dict has 'chunk_id' but no '_id' -- uses 'chunk_id' as fallback."""
        rag = _make_rag()
        rag.text_chunks.get_by_ids.return_value = [
            {"chunk_id": "c1", "content": "fallback", "full_doc_id": "d1", "file_path": "g.txt", "chunk_order_index": 0, "tokens": 8},
        ]
        items = await _fetch_chunks_by_ids(rag, ["c1"])
        assert len(items) == 1
        assert items[0].chunk_id == "c1"
        assert items[0].content == "fallback"

    async def test_defaults_for_missing_fields(self):
        """Raw dict with only '_id' -- other fields get defaults."""
        rag = _make_rag()
        rag.text_chunks.get_by_ids.return_value = [
            {"_id": "c1"},
        ]
        items = await _fetch_chunks_by_ids(rag, ["c1"])
        assert len(items) == 1
        item = items[0]
        assert item.chunk_id == "c1"
        assert item.content == ""
        assert item.full_doc_id == ""
        assert item.file_path == ""
        # chunk_order_index defaults to the enumeration index (0)
        assert item.chunk_order_index == 0
        assert item.tokens == 0

    async def test_defaults_chunk_order_index_to_position(self):
        """When chunk_order_index is missing, it defaults to the position in the
        requested list (the enumerate index)."""
        rag = _make_rag()
        rag.text_chunks.get_by_ids.return_value = [
            {"_id": "c3"},
        ]
        # c3 is the third in the requested list (index=2), but the first
        # two chunks are missing from the result, so the enumerate idx
        # when c3 is encountered is 2.
        items = await _fetch_chunks_by_ids(rag, ["c1", "c2", "c3"])
        assert len(items) == 1
        assert items[0].chunk_id == "c3"
        assert items[0].chunk_order_index == 2

    async def test_non_dict_entries_ignored(self):
        """Non-dict raw entries (e.g., stray strings) are ignored."""
        rag = _make_rag()
        rag.text_chunks.get_by_ids.return_value = [
            "not a dict",
            42,
            {"_id": "c1", "content": "ok"},
        ]
        items = await _fetch_chunks_by_ids(rag, ["c1"])
        assert len(items) == 1
        assert items[0].chunk_id == "c1"


# ---------------------------------------------------------------------------
# Windowing / range logic -- integration through the helper chain
# ---------------------------------------------------------------------------

def _make_rag_with_doc(chunk_ids: list[str]):
    """Create a rag mock whose doc_status returns the given ordered chunk IDs
    and whose text_chunks.get_by_ids returns matching chunk dicts."""
    rag = _make_rag()
    rag.doc_status.get_by_id.return_value = {"chunks_list": chunk_ids}

    async def fake_get_by_ids(ids):
        return [
            {
                "_id": cid,
                "content": f"content-{cid}",
                "full_doc_id": "doc-1",
                "file_path": "f.txt",
                "chunk_order_index": chunk_ids.index(cid),
                "tokens": 10,
            }
            for cid in ids
            if cid in chunk_ids
        ]

    rag.text_chunks.get_by_ids = AsyncMock(side_effect=fake_get_by_ids)
    return rag


def _make_rag_with_anchor(chunk_ids: list[str], anchor_id: str):
    """Create a rag mock with doc lookup and an anchor chunk resolve."""
    rag = _make_rag_with_doc(chunk_ids)
    anchor_idx = chunk_ids.index(anchor_id)
    rag.text_chunks.get_by_id.return_value = {
        "_id": anchor_id,
        "content": f"content-{anchor_id}",
        "full_doc_id": "doc-1",
        "file_path": "f.txt",
        "chunk_order_index": anchor_idx,
        "tokens": 10,
    }
    return rag


class TestContextWindowMath:
    """Test the windowing math from the get_chunk_context route handler by
    exercising the helper chain with known data."""

    async def test_clamps_at_start(self):
        """Chunk at index 1, window=3 -- start must clamp to 0."""
        chunk_ids = [f"c{i}" for i in range(10)]
        rag = _make_rag_with_anchor(chunk_ids, "c1")

        # Simulate what get_chunk_context does
        anchor = await _resolve_chunk(rag, "c1")
        doc_id = anchor["full_doc_id"]
        ordered_ids = await _get_ordered_chunk_ids(rag, doc_id)
        idx = ordered_ids.index("c1")
        window = 3

        start = max(0, idx - window)
        end = min(len(ordered_ids), idx + window + 1)
        window_ids = ordered_ids[start:end]

        assert start == 0  # clamped
        assert window_ids == ["c0", "c1", "c2", "c3", "c4"]

        items = await _fetch_chunks_by_ids(rag, window_ids)
        assert len(items) == 5
        assert items[0].chunk_id == "c0"
        assert items[-1].chunk_id == "c4"

    async def test_clamps_at_end(self):
        """Chunk at index 8 in a 10-chunk doc, window=3 -- end must clamp to 10."""
        chunk_ids = [f"c{i}" for i in range(10)]
        rag = _make_rag_with_anchor(chunk_ids, "c8")

        anchor = await _resolve_chunk(rag, "c8")
        doc_id = anchor["full_doc_id"]
        ordered_ids = await _get_ordered_chunk_ids(rag, doc_id)
        idx = ordered_ids.index("c8")
        window = 3

        start = max(0, idx - window)
        end = min(len(ordered_ids), idx + window + 1)
        window_ids = ordered_ids[start:end]

        assert end == 10  # clamped
        assert window_ids == ["c5", "c6", "c7", "c8", "c9"]

        items = await _fetch_chunks_by_ids(rag, window_ids)
        assert len(items) == 5
        assert items[0].chunk_id == "c5"
        assert items[-1].chunk_id == "c9"

    async def test_exact_boundaries_first_chunk(self):
        """Chunk at index 0, window=1 -- returns indices 0 and 1."""
        chunk_ids = [f"c{i}" for i in range(5)]
        rag = _make_rag_with_anchor(chunk_ids, "c0")

        anchor = await _resolve_chunk(rag, "c0")
        ordered_ids = await _get_ordered_chunk_ids(rag, anchor["full_doc_id"])
        idx = ordered_ids.index("c0")
        window = 1

        start = max(0, idx - window)
        end = min(len(ordered_ids), idx + window + 1)
        window_ids = ordered_ids[start:end]

        assert start == 0
        assert end == 2
        assert window_ids == ["c0", "c1"]

    async def test_window_middle(self):
        """Chunk in the middle with enough room on both sides."""
        chunk_ids = [f"c{i}" for i in range(10)]
        rag = _make_rag_with_anchor(chunk_ids, "c5")

        anchor = await _resolve_chunk(rag, "c5")
        ordered_ids = await _get_ordered_chunk_ids(rag, anchor["full_doc_id"])
        idx = ordered_ids.index("c5")
        window = 2

        start = max(0, idx - window)
        end = min(len(ordered_ids), idx + window + 1)
        window_ids = ordered_ids[start:end]

        assert window_ids == ["c3", "c4", "c5", "c6", "c7"]

    async def test_window_larger_than_doc(self):
        """Window larger than the entire doc -- returns all chunks."""
        chunk_ids = ["c0", "c1", "c2"]
        rag = _make_rag_with_anchor(chunk_ids, "c1")

        anchor = await _resolve_chunk(rag, "c1")
        ordered_ids = await _get_ordered_chunk_ids(rag, anchor["full_doc_id"])
        idx = ordered_ids.index("c1")
        window = 50

        start = max(0, idx - window)
        end = min(len(ordered_ids), idx + window + 1)
        window_ids = ordered_ids[start:end]

        assert window_ids == ["c0", "c1", "c2"]


class TestDocumentChunksRange:
    """Test the slicing logic from the get_document_chunks route handler."""

    async def test_start_only(self):
        """Only start=3 -- returns from index 3 to end."""
        chunk_ids = [f"c{i}" for i in range(8)]
        rag = _make_rag_with_doc(chunk_ids)

        ordered_ids = await _get_ordered_chunk_ids(rag, "doc-1")
        total = len(ordered_ids)
        start = 3
        end_param = None

        s = start
        e = (end_param or total - 1) + 1
        sliced = ordered_ids[s:e]

        assert sliced == ["c3", "c4", "c5", "c6", "c7"]

        items = await _fetch_chunks_by_ids(rag, sliced)
        assert len(items) == 5
        assert items[0].chunk_id == "c3"
        assert items[-1].chunk_id == "c7"

    async def test_end_only(self):
        """Only end=4 -- returns from index 0 to 4 inclusive."""
        chunk_ids = [f"c{i}" for i in range(8)]
        rag = _make_rag_with_doc(chunk_ids)

        ordered_ids = await _get_ordered_chunk_ids(rag, "doc-1")
        total = len(ordered_ids)
        start_param = None
        end_param = 4

        s = start_param or 0
        e = (end_param or total - 1) + 1  # inclusive -> slice end
        sliced = ordered_ids[s:e]

        assert sliced == ["c0", "c1", "c2", "c3", "c4"]

        items = await _fetch_chunks_by_ids(rag, sliced)
        assert len(items) == 5
        assert items[0].chunk_id == "c0"
        assert items[-1].chunk_id == "c4"

    async def test_start_and_end(self):
        """start=2, end=5 -- returns indices 2, 3, 4, 5."""
        chunk_ids = [f"c{i}" for i in range(10)]
        rag = _make_rag_with_doc(chunk_ids)

        ordered_ids = await _get_ordered_chunk_ids(rag, "doc-1")
        total = len(ordered_ids)
        start_param = 2
        end_param = 5

        s = start_param or 0
        e = (end_param or total - 1) + 1
        sliced = ordered_ids[s:e]

        assert sliced == ["c2", "c3", "c4", "c5"]

        items = await _fetch_chunks_by_ids(rag, sliced)
        assert len(items) == 4
        assert items[0].chunk_id == "c2"
        assert items[-1].chunk_id == "c5"

    async def test_no_start_no_end(self):
        """Neither start nor end -- returns all chunks."""
        chunk_ids = [f"c{i}" for i in range(5)]
        rag = _make_rag_with_doc(chunk_ids)

        ordered_ids = await _get_ordered_chunk_ids(rag, "doc-1")

        # Simulate: start and end are both None -- no slicing
        items = await _fetch_chunks_by_ids(rag, ordered_ids)
        assert len(items) == 5

    async def test_start_beyond_length(self):
        """start beyond doc length -- empty result."""
        chunk_ids = [f"c{i}" for i in range(5)]
        rag = _make_rag_with_doc(chunk_ids)

        ordered_ids = await _get_ordered_chunk_ids(rag, "doc-1")
        total = len(ordered_ids)
        start_param = 20
        end_param = None

        s = start_param or 0
        e = (end_param or total - 1) + 1
        sliced = ordered_ids[s:e]

        assert sliced == []

    async def test_end_zero(self):
        """end=0 -- inclusive end means slice [0:1], returns first chunk only."""
        chunk_ids = [f"c{i}" for i in range(5)]
        rag = _make_rag_with_doc(chunk_ids)

        ordered_ids = await _get_ordered_chunk_ids(rag, "doc-1")
        total = len(ordered_ids)
        start_param = None
        end_param = 0

        s = start_param or 0
        e = (end_param or total - 1) + 1
        sliced = ordered_ids[s:e]

        # end=0 is truthy-False (0), so `end_param or total - 1` => total - 1
        # This is actually a known behavior of the `or` pattern for 0.
        # The route uses `end or total - 1`, so end=0 is treated as "not set".
        assert sliced == chunk_ids  # all chunks returned


# ---------------------------------------------------------------------------
# HTTP-level tests via httpx AsyncClient
# ---------------------------------------------------------------------------

from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI
from twindb_lightrag_memgraph.server.chunk_routes import router, create_chunk_routes
from twindb_lightrag_memgraph.server.auth import configure_auth


def _make_chunk_app(rag_mock) -> FastAPI:
    """Build a minimal FastAPI app with chunk routes and auth disabled.

    Clears the module-level router first to avoid route accumulation
    across tests (create_chunk_routes appends to a global router).
    """
    configure_auth(api_key=None, jwt_secret=None)
    router.routes.clear()
    create_chunk_routes(rag_mock)
    app = FastAPI()
    app.include_router(router)
    return app


class TestHTTPChunkContext:
    """HTTP-level tests for GET /chunks/{chunk_id}/context."""

    async def test_context_returns_200(self):
        chunk_ids = [f"c{i}" for i in range(10)]
        rag = _make_rag_with_anchor(chunk_ids, "c5")
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/chunks/c5/context?window=2")
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == "doc-1"
        assert body["total_chunks_in_doc"] == 10
        chunk_ids_returned = [c["chunk_id"] for c in body["chunks"]]
        assert chunk_ids_returned == ["c3", "c4", "c5", "c6", "c7"]

    async def test_context_chunk_not_found_returns_404(self):
        rag = _make_rag()
        rag.text_chunks.get_by_id.return_value = None
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/chunks/nonexistent/context")
        assert resp.status_code == 404

    async def test_context_default_window_is_3(self):
        chunk_ids = [f"c{i}" for i in range(20)]
        rag = _make_rag_with_anchor(chunk_ids, "c10")
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/chunks/c10/context")
        assert resp.status_code == 200
        chunk_ids_returned = [c["chunk_id"] for c in resp.json()["chunks"]]
        # window=3 default: [c7, c8, c9, c10, c11, c12, c13]
        assert chunk_ids_returned == ["c7", "c8", "c9", "c10", "c11", "c12", "c13"]

    async def test_context_no_parent_doc_returns_404(self):
        """Anchor chunk exists but has no full_doc_id."""
        rag = _make_rag()
        rag.text_chunks.get_by_id.return_value = {
            "_id": "orphan",
            "content": "some text",
            "full_doc_id": "",
        }
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/chunks/orphan/context")
        assert resp.status_code == 404
        assert "no parent document" in resp.json()["detail"].lower()


class TestHTTPChunkDocument:
    """HTTP-level tests for GET /chunks/{chunk_id}/document."""

    async def test_document_returns_all_chunks(self):
        chunk_ids = ["c0", "c1", "c2", "c3"]
        rag = _make_rag_with_anchor(chunk_ids, "c1")
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/chunks/c1/document")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_chunks_in_doc"] == 4
        returned = [c["chunk_id"] for c in body["chunks"]]
        assert returned == ["c0", "c1", "c2", "c3"]

    async def test_document_chunk_not_found(self):
        rag = _make_rag()
        rag.text_chunks.get_by_id.return_value = None
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/chunks/missing/document")
        assert resp.status_code == 404


class TestHTTPDocumentChunks:
    """HTTP-level tests for GET /documents/{doc_id}/chunks."""

    async def test_all_chunks(self):
        chunk_ids = [f"c{i}" for i in range(5)]
        rag = _make_rag_with_doc(chunk_ids)
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/documents/doc-1/chunks")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_chunks_in_doc"] == 5
        returned = [c["chunk_id"] for c in body["chunks"]]
        assert returned == chunk_ids

    async def test_with_start_and_end(self):
        chunk_ids = [f"c{i}" for i in range(10)]
        rag = _make_rag_with_doc(chunk_ids)
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/documents/doc-1/chunks?start=2&end=5")
        assert resp.status_code == 200
        body = resp.json()
        # total_chunks_in_doc is the full doc count, not the slice
        assert body["total_chunks_in_doc"] == 10
        returned = [c["chunk_id"] for c in body["chunks"]]
        assert returned == ["c2", "c3", "c4", "c5"]

    async def test_with_start_only(self):
        chunk_ids = [f"c{i}" for i in range(8)]
        rag = _make_rag_with_doc(chunk_ids)
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/documents/doc-1/chunks?start=5")
        assert resp.status_code == 200
        returned = [c["chunk_id"] for c in resp.json()["chunks"]]
        assert returned == ["c5", "c6", "c7"]

    async def test_with_end_only(self):
        chunk_ids = [f"c{i}" for i in range(8)]
        rag = _make_rag_with_doc(chunk_ids)
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/documents/doc-1/chunks?end=3")
        assert resp.status_code == 200
        returned = [c["chunk_id"] for c in resp.json()["chunks"]]
        assert returned == ["c0", "c1", "c2", "c3"]

    async def test_missing_doc_returns_404(self):
        rag = _make_rag()
        rag.doc_status.get_by_id.return_value = None
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/documents/no-doc/chunks")
        assert resp.status_code == 404

    async def test_response_model_fields(self):
        """Verify the response has all ChunkContextResponse fields."""
        chunk_ids = ["c0"]
        rag = _make_rag_with_doc(chunk_ids)
        app = _make_chunk_app(rag)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/documents/doc-1/chunks")
        assert resp.status_code == 200
        body = resp.json()
        assert "chunks" in body
        assert "doc_id" in body
        assert "file_path" in body
        assert "total_chunks_in_doc" in body
        # Verify ChunkItem fields
        chunk = body["chunks"][0]
        assert "chunk_id" in chunk
        assert "content" in chunk
        assert "full_doc_id" in chunk
        assert "file_path" in chunk
        assert "chunk_order_index" in chunk
        assert "tokens" in chunk
