import asyncio

import pytest

from twindb_lightrag_memgraph.server.query.response_sources import (
    _build_sources_legacy_fallback,
    _filter_sources_by_advanced_filters,
)


@pytest.mark.asyncio
async def test_filter_sources_prefetches_unique_doc_tags_concurrently():
    in_flight = 0
    peak_in_flight = 0
    seen: list[str] = []

    async def fetch_doc_tags(doc_id: str, _folder: str) -> set[str]:
        nonlocal in_flight, peak_in_flight
        seen.append(doc_id)
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return {"keep"} if doc_id != "doc-drop" else {"drop"}

    sources = [
        {"doc_id": "doc-a", "name": "a.pdf"},
        {"doc_id": "doc-b", "name": "b.pdf"},
        {"doc_id": "doc-a", "name": "a-duplicate.pdf"},
        {"doc_id": "doc-drop", "name": "drop.pdf"},
    ]

    kept, incomplete = await _filter_sources_by_advanced_filters(
        sources,
        tag_filter={"all": ["keep"]},
        doc_filter=None,
        folder="default",
        fetch_doc_tags=fetch_doc_tags,
    )

    assert [source["name"] for source in kept] == [
        "a.pdf",
        "b.pdf",
        "a-duplicate.pdf",
    ]
    assert incomplete is False
    assert seen == ["doc-a", "doc-b", "doc-drop"]
    assert peak_in_flight > 1


@pytest.mark.asyncio
async def test_build_sources_legacy_fallback_resolves_doc_ids_concurrently(
    monkeypatch,
):
    in_flight = 0
    peak_in_flight = 0

    class _ChunksVdb:
        async def query(self, _query: str, *, top_k: int):
            return [
                {"id": f"chunk-{idx}", "file_path": f"doc-{idx}.pdf"}
                for idx in range(top_k)
            ]

    class _Rag:
        chunks_vdb = _ChunksVdb()

    async def fake_resolve(_rag, chunk_id: str) -> str:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return f"doc:{chunk_id}"

    monkeypatch.setattr(
        "twindb_lightrag_memgraph.server.query.response_sources._resolve_doc_for_chunk",
        fake_resolve,
    )

    sources = await _build_sources_legacy_fallback(_Rag(), "query", top_k=5)

    assert [source["doc_id"] for source in sources] == [
        "doc:chunk-0",
        "doc:chunk-1",
        "doc:chunk-2",
        "doc:chunk-3",
        "doc:chunk-4",
    ]
    assert peak_in_flight > 1
