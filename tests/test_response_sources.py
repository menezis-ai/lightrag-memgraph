import asyncio

import pytest

from twindb_lightrag_memgraph.server.query.response_sources import (
    _build_sources_legacy_fallback,
    _filter_sources_by_advanced_filters,
    _sort_sources_by_score,
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


def test_sort_sources_by_measured_score_keeps_citation_numbers():
    sources = [
        {"n": 1, "name": "unscored-a", "score": None},
        {"n": 2, "name": "medium", "score": 0.62},
        {"n": 3, "name": "best", "score": 0.94},
        {"n": 4, "name": "unscored-b"},
        {"n": 5, "name": "low", "score": 0.18},
        {"n": 6, "name": "invalid-score", "score": float("nan")},
    ]

    sorted_sources = _sort_sources_by_score(sources)

    assert [source["name"] for source in sorted_sources] == [
        "best",
        "medium",
        "low",
        "unscored-a",
        "unscored-b",
        "invalid-score",
    ]
    assert [source["n"] for source in sorted_sources] == [3, 2, 5, 1, 4, 6]


@pytest.mark.asyncio
async def test_filter_sources_prefers_one_batch_tag_lookup():
    single_calls: list[str] = []
    batch_calls: list[tuple[list[str], str]] = []

    async def fetch_doc_tags(doc_id: str, _folder: str) -> set[str]:
        single_calls.append(doc_id)
        return set()

    async def fetch_doc_tags_batch(
        doc_ids: list[str], folder: str
    ) -> dict[str, set[str]]:
        batch_calls.append((list(doc_ids), folder))
        return {
            doc_id: {"keep"} if doc_id != "doc-drop" else {"drop"} for doc_id in doc_ids
        }

    sources = [
        {"doc_id": "doc-a", "name": "a.pdf"},
        {"doc_id": "doc-b", "name": "b.pdf"},
        {"doc_id": "doc-a", "name": "a-copy.pdf"},
        {"doc_id": "doc-drop", "name": "drop.pdf"},
    ]

    kept, incomplete = await _filter_sources_by_advanced_filters(
        sources,
        tag_filter={"all": ["keep"]},
        doc_filter=None,
        folder="default",
        fetch_doc_tags=fetch_doc_tags,
        fetch_doc_tags_batch=fetch_doc_tags_batch,
    )

    assert [source["name"] for source in kept] == [
        "a.pdf",
        "b.pdf",
        "a-copy.pdf",
    ]
    assert incomplete is False
    assert batch_calls == [(["doc-a", "doc-b", "doc-drop"], "default")]
    assert single_calls == []


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
