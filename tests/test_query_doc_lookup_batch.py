from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from twindb_lightrag_memgraph import _constants, _pool
from twindb_lightrag_memgraph.server.query.doc_lookup import (
    _resolve_chunk_to_doc_id,
    _resolve_file_paths_to_doc_ids,
)
from twindb_lightrag_memgraph.server.query import router as query_router


class _BatchChunkStore:
    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.records = records
        self.calls: list[list[str]] = []

    async def get_by_ids(self, chunk_ids: list[str]) -> list[dict[str, Any] | None]:
        self.calls.append(list(chunk_ids))
        return [self.records.get(chunk_id) for chunk_id in chunk_ids]


class _LegacyDocStatus:
    def __init__(self, chunk_docs: dict[str, str]) -> None:
        self.chunk_docs = chunk_docs
        self.chunk_calls: list[list[str]] = []

    async def get_docs_by_chunks(self, chunk_ids: list[str]):
        self.chunk_calls.append(list(chunk_ids))
        return {
            self.chunk_docs[chunk_id]: object()
            for chunk_id in chunk_ids
            if chunk_id in self.chunk_docs
        }


async def test_chunk_doc_resolution_batches_records_then_falls_back_unresolved():
    text_chunks = _BatchChunkStore(
        {
            "chunk-a": {
                "chunk_id": "chunk-a",
                "full_doc_id": "doc-a",
            }
        }
    )
    doc_status = _LegacyDocStatus({"chunk-b": "doc-b"})
    rag = type(
        "Rag",
        (),
        {
            "text_chunks": text_chunks,
            "doc_status": doc_status,
        },
    )()

    resolved = await _resolve_chunk_to_doc_id(
        rag, ["chunk-a", "chunk-b", "chunk-a", "missing"]
    )

    assert resolved == {"chunk-a": "doc-a", "chunk-b": "doc-b"}
    assert text_chunks.calls == [["chunk-a", "chunk-b", "missing"]]
    assert doc_status.chunk_calls == [["chunk-b"], ["missing"]]


async def test_chunk_doc_resolution_uses_one_batch_when_records_are_complete():
    text_chunks = _BatchChunkStore(
        {
            f"chunk-{idx}": {
                "id": f"chunk-{idx}",
                "full_doc_id": f"doc-{idx}",
            }
            for idx in range(8)
        }
    )
    doc_status = _LegacyDocStatus({})
    rag = type(
        "Rag",
        (),
        {
            "text_chunks": text_chunks,
            "doc_status": doc_status,
        },
    )()

    resolved = await _resolve_chunk_to_doc_id(rag, [f"chunk-{idx}" for idx in range(8)])

    assert resolved == {f"chunk-{idx}": f"doc-{idx}" for idx in range(8)}
    assert len(text_chunks.calls) == 1
    assert doc_status.chunk_calls == []


class _BatchPathDocStatus:
    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []
        self.single_calls: list[str] = []

    async def get_docs_by_file_paths(self, file_paths: list[str]):
        self.batch_calls.append(list(file_paths))
        return {
            path: {"id": f"doc:{path}", "file_path": path}
            for path in file_paths
            if path != "missing.pdf"
        }

    async def get_doc_by_file_path(self, file_path: str):
        self.single_calls.append(file_path)
        return {"id": f"legacy:{file_path}"}


async def test_file_path_doc_resolution_prefers_one_batch_read():
    doc_status = _BatchPathDocStatus()
    rag = type("Rag", (), {"doc_status": doc_status})()

    resolved = await _resolve_file_paths_to_doc_ids(
        rag, ["a.pdf", "missing.pdf", "a.pdf", "b.pdf"]
    )

    assert resolved == {"a.pdf": "doc:a.pdf", "b.pdf": "doc:b.pdf"}
    assert doc_status.batch_calls == [["a.pdf", "missing.pdf", "b.pdf"]]
    assert doc_status.single_calls == []


async def test_file_path_doc_resolution_tolerates_rag_without_doc_status():
    resolved = await _resolve_file_paths_to_doc_ids(object(), ["a.pdf"])
    assert resolved == {}


async def test_file_path_doc_resolution_keeps_legacy_fallback_on_batch_failure():
    class _FailingBatchDocStatus(_BatchPathDocStatus):
        async def get_docs_by_file_paths(self, file_paths: list[str]):
            self.batch_calls.append(list(file_paths))
            raise RuntimeError("batch unavailable")

    doc_status = _FailingBatchDocStatus()
    rag = type("Rag", (), {"doc_status": doc_status})()

    resolved = await _resolve_file_paths_to_doc_ids(rag, ["a.pdf", "b.pdf"])

    assert resolved == {"a.pdf": "legacy:a.pdf", "b.pdf": "legacy:b.pdf"}
    assert doc_status.batch_calls == [["a.pdf", "b.pdf"]]
    assert doc_status.single_calls == ["a.pdf", "b.pdf"]


async def test_graph_tag_batch_uses_one_unwind_query(monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Result:
        def __aiter__(self):
            return self._rows().__aiter__()

        async def _rows(self):
            yield {"doc_id": "doc-a", "tags": ["RMan", "oracle"]}
            yield {"doc_id": "doc-b", "tags": []}

        async def consume(self):
            return None

    class _Session:
        async def run(self, query: str, **params):
            calls.append((query, params))
            return _Result()

    @asynccontextmanager
    async def get_read_session():
        yield _Session()

    monkeypatch.setattr(_pool, "get_read_session", get_read_session)
    monkeypatch.setattr(_constants, "resolve_workspace", lambda: "workspace")

    tags = await query_router._fetch_doc_graph_tags_batch(
        ["doc-a", "doc-b", "doc-a", "missing"], "default"
    )

    assert tags == {
        "doc-a": {"rman", "oracle"},
        "doc-b": set(),
        "missing": set(),
    }
    assert len(calls) == 1
    query, params = calls[0]
    assert "UNWIND $doc_ids AS doc_id" in query
    assert "OPTIONAL MATCH (d)-[:TAGGED_WITH]" in query
    assert params == {"doc_ids": ["doc-a", "doc-b", "missing"]}
