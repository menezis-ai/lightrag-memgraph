"""The chunk routes overlap two reads — pin what happens when one of them fails.

``_chunks_and_source_links`` runs the chunk KV read and the source-link read
concurrently. ``asyncio.gather`` was the obvious way to write that and the wrong
one: when the first coroutine raises, gather leaves the sibling running. The
chunk read is the request-critical one, so a failing KV would have left the
*optional* provenance read holding a connection on an already degraded read
pool — exactly when the pool is scarcest — while the request unwound to a 500.

These tests pin the corrected contract: same results on the happy path, and on
failure the optional read is cancelled and reaped before the error propagates.
"""

from __future__ import annotations

import asyncio

import pytest

from twindb_lightrag_memgraph.server import chunk_routes


class _Rag:
    def __init__(self, chunks: dict) -> None:
        self.text_chunks = _TextChunks(chunks)


class _TextChunks:
    def __init__(self, chunks: dict) -> None:
        self._chunks = chunks

    async def get_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        return [self._chunks[c] for c in chunk_ids]


def _chunk(chunk_id: str, index: int) -> dict:
    return {
        "_id": chunk_id,
        "content": f"content-{chunk_id}",
        "full_doc_id": "doc-1",
        "file_path": "document.pdf",
        "chunk_order_index": index,
        "tokens": 7,
    }


async def test_happy_path_returns_both_reads(monkeypatch):
    rag = _Rag({"c0": _chunk("c0", 0), "c1": _chunk("c1", 1)})

    async def _links(doc_id: str) -> list[dict]:
        return [{"id": "l1", "doc_id": doc_id, "url": "u", "label": "L"}]

    monkeypatch.setattr(chunk_routes, "_source_links_for_doc", _links)

    items, links = await chunk_routes._chunks_and_source_links(
        rag, ["c0", "c1"], "doc-1"
    )

    assert [i.chunk_id for i in items] == ["c0", "c1"]
    assert links == [{"id": "l1", "doc_id": "doc-1", "url": "u", "label": "L"}]


async def test_failed_chunk_read_cancels_the_optional_source_link_read(monkeypatch):
    """The regression this rewrite exists for."""
    started = asyncio.Event()
    cancelled = False
    completed = False

    async def _boom(_rag, _chunk_ids):
        # Let the sibling actually start before failing, so the test proves
        # cancellation rather than a coroutine that never ran.
        await started.wait()
        raise RuntimeError("text_chunks backend is down")

    async def _slow_links(_doc_id: str) -> list[dict]:
        nonlocal cancelled, completed
        started.set()
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled = True
            raise
        completed = True
        return []

    monkeypatch.setattr(chunk_routes, "_fetch_chunks_by_ids", _boom)
    monkeypatch.setattr(chunk_routes, "_source_links_for_doc", _slow_links)

    with pytest.raises(RuntimeError, match="text_chunks backend is down"):
        await chunk_routes._chunks_and_source_links(_Rag({}), ["c0"], "doc-1")

    assert cancelled, "the optional source-link read must be cancelled"
    assert not completed, "it must not keep running after the request failed"


async def test_failed_chunk_read_leaves_no_pending_task(monkeypatch):
    """The sibling is reaped, not merely cancelled and forgotten.

    An un-awaited cancelled task is how "Task exception was never retrieved"
    noise — and a connection released late — gets into production logs.
    """
    started = asyncio.Event()

    async def _boom(_rag, _chunk_ids):
        await started.wait()
        raise RuntimeError("down")

    async def _links(_doc_id: str) -> list[dict]:
        started.set()
        await asyncio.sleep(5)
        return []

    monkeypatch.setattr(chunk_routes, "_fetch_chunks_by_ids", _boom)
    monkeypatch.setattr(chunk_routes, "_source_links_for_doc", _links)

    before = {t for t in asyncio.all_tasks() if not t.done()}
    with pytest.raises(RuntimeError):
        await chunk_routes._chunks_and_source_links(_Rag({}), ["c0"], "doc-1")
    # Give the loop one turn to finalize any task that was left behind.
    await asyncio.sleep(0)
    leaked = {t for t in asyncio.all_tasks() if not t.done()} - before
    assert not leaked, f"pending task(s) survived the failure: {leaked}"


async def test_reads_actually_overlap(monkeypatch):
    """Guard the optimization itself: the two reads must not be serialized."""
    peak = 0
    live = 0

    async def _tracked_chunks(_rag, chunk_ids):
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return []

    async def _tracked_links(_doc_id):
        nonlocal peak, live
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.02)
        live -= 1
        return []

    monkeypatch.setattr(chunk_routes, "_fetch_chunks_by_ids", _tracked_chunks)
    monkeypatch.setattr(chunk_routes, "_source_links_for_doc", _tracked_links)

    await chunk_routes._chunks_and_source_links(_Rag({}), ["c0"], "doc-1")

    assert peak == 2, f"expected the two reads concurrent, observed peak={peak}"
