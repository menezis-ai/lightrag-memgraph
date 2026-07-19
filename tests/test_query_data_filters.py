import asyncio

import pytest

from twindb_lightrag_memgraph.server.query import query_data_filters as qdf


class _Rag:
    doc_status = object()


class _BatchResolveTracker:
    def __init__(self) -> None:
        self.in_flight = 0
        self.peak_in_flight = 0

    async def track(self, value):
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        return value


def _install_batch_resolvers(monkeypatch) -> _BatchResolveTracker:
    tracker = _BatchResolveTracker()

    async def fake_chunk_batch(_rag, chunk_ids):
        return await tracker.track(
            {chunk_id: f"doc:{chunk_id}" for chunk_id in chunk_ids}
        )

    async def fake_file_batch(_rag, file_paths):
        return await tracker.track(
            {file_path: f"doc:{file_path}" for file_path in file_paths}
        )

    monkeypatch.setattr(qdf, "_resolve_doc_ids_for_chunk_ids", fake_chunk_batch)
    monkeypatch.setattr(qdf, "_resolve_doc_ids_for_file_paths", fake_file_batch)
    return tracker


async def _fake_tags(_doc_id, _folder):
    return {"keep"}


@pytest.mark.asyncio
async def test_doc_ids_for_query_data_row_resolves_independent_sources_concurrently(
    monkeypatch,
):
    in_flight = 0
    peak_in_flight = 0

    async def _track(value: str) -> str:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return value

    async def fake_chunk(_rag, chunk_id):
        return await _track(f"doc:{chunk_id}")

    async def fake_file(_rag, file_path):
        return await _track(f"doc:{file_path}")

    monkeypatch.setattr(qdf, "_resolve_doc_for_chunk", fake_chunk)
    monkeypatch.setattr(qdf, "_resolve_doc_for_file_path", fake_file)

    doc_ids = await qdf._doc_ids_for_query_data_row(
        _Rag(),
        {
            "doc_id": "direct-doc",
            "source_id": "chunk-a,chunk-b,chunk-c",
            "file_path": "file-a.pdf",
            "source": "file-b.pdf",
        },
    )

    assert doc_ids == {
        "direct-doc",
        "doc:chunk-a",
        "doc:chunk-b",
        "doc:chunk-c",
        "doc:file-a.pdf",
        "doc:file-b.pdf",
    }
    assert peak_in_flight > 1


@pytest.mark.asyncio
async def test_filter_rows_by_tags_resolves_chunk_and_file_batches_concurrently(
    monkeypatch,
):
    tracker = _install_batch_resolvers(monkeypatch)

    rows, ref_ids = await qdf._filter_rows_by_tags(
        _Rag(),
        [
            {
                "source_id": "chunk-a,chunk-b",
                "file_path": "file-a.pdf",
                "reference_id": "ref-a",
            }
        ],
        {"all": ["keep"]},
        "default",
        {},
        _fake_tags,
    )

    assert rows == [
        {
            "source_id": "chunk-a,chunk-b",
            "file_path": "file-a.pdf",
            "reference_id": "ref-a",
        }
    ]
    assert ref_ids == {"ref-a"}
    assert tracker.peak_in_flight > 1


@pytest.mark.asyncio
async def test_filter_rows_by_tags_preserves_serial_resolution_for_large_batches(
    monkeypatch,
):
    tracker = _install_batch_resolvers(monkeypatch)

    rows = [
        {
            "source_id": f"chunk-{idx}",
            "file_path": f"file-{idx}.pdf",
            "reference_id": f"ref-{idx}",
        }
        for idx in range(qdf._PARALLEL_RESOLVE_ID_BUDGET + 1)
    ]

    kept, ref_ids = await qdf._filter_rows_by_tags(
        _Rag(),
        rows,
        {"all": ["keep"]},
        "default",
        {},
        _fake_tags,
    )

    assert kept == rows
    assert len(ref_ids) == qdf._PARALLEL_RESOLVE_ID_BUDGET + 1
    assert tracker.peak_in_flight == 1
