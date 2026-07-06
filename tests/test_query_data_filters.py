import asyncio

import pytest

from twindb_lightrag_memgraph.server.query import query_data_filters as qdf


class _Rag:
    doc_status = object()


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
