"""Native upload batch registration contract.

This pins the first ingestion boundary: every accepted ``POST /documents/upload``
request must schedule the indexing pipeline and become trackable.  The test
uses the real LightRAG HTTP route and multipart handling, but replaces the
expensive pipeline body with an in-memory ``PENDING`` registration so it can run
without Memgraph.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from lightrag.base import DocProcessingStatus, DocStatus


class FakeDocStatusStore:
    def __init__(self) -> None:
        self.docs: dict[str, DocProcessingStatus] = {}

    async def upsert(self, data: dict[str, DocProcessingStatus]) -> None:
        self.docs.update(data)

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, DocProcessingStatus] | None:
        for doc_id, doc in self.docs.items():
            if Path(doc.file_path).name == basename:
                return doc_id, doc
        return None

    async def get_doc_by_file_path(
        self, file_path: str
    ) -> tuple[str, DocProcessingStatus] | None:
        return await self.get_doc_by_file_basename(Path(file_path).name)

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        return {
            doc_id: doc for doc_id, doc in self.docs.items() if doc.track_id == track_id
        }

    async def get_by_id(self, doc_id: str) -> DocProcessingStatus | None:
        return self.docs.get(doc_id)


class FakeRag:
    workspace = "native_upload_batch_route"

    def __init__(self) -> None:
        self.doc_status = FakeDocStatusStore()

    async def aget_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:
        return await self.doc_status.get_docs_by_track_id(track_id)


@pytest.fixture
def native_upload_batch_app(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["pytest"])

    import lightrag.api.routers.document_routes as document_routes
    from lightrag.api.routers.document_routes import (
        DocumentManager,
        create_document_routes,
    )
    from tests.conftest import ensure_fresh_native_document_router

    indexed: list[tuple[str, str]] = []
    rag = FakeRag()

    async def reserve_enqueue_slot(_rag: Any) -> bool:
        return True

    async def release_enqueue_slot(_rag: Any) -> None:
        return None

    async def pipeline_index_file(rag: FakeRag, file_path: Path, track_id: str) -> None:
        indexed.append((file_path.name, track_id))
        now = datetime.now(timezone.utc).isoformat()
        await rag.doc_status.upsert(
            {
                f"doc-{track_id}": DocProcessingStatus(
                    content_summary=file_path.name,
                    content_length=file_path.stat().st_size,
                    file_path=file_path.name,
                    status=DocStatus.PENDING,
                    created_at=now,
                    updated_at=now,
                    track_id=track_id,
                )
            }
        )

    monkeypatch.setattr(
        document_routes, "_reserve_enqueue_slot", reserve_enqueue_slot, raising=False
    )
    monkeypatch.setattr(
        document_routes, "_release_enqueue_slot", release_enqueue_slot, raising=False
    )
    monkeypatch.setattr(
        document_routes,
        "check_pipeline_busy_or_raise",
        reserve_enqueue_slot,
        raising=False,
    )
    monkeypatch.setattr(document_routes, "pipeline_index_file", pipeline_index_file)

    doc_manager = DocumentManager(tmp_path / "input", workspace=rag.workspace)
    app = FastAPI()
    ensure_fresh_native_document_router()
    app.include_router(create_document_routes(rag, doc_manager, api_key=None))

    return rag, app, indexed


async def test_native_upload_route_registers_every_accepted_large_batch(
    native_upload_batch_app,
):
    rag, app, indexed = native_upload_batch_app
    upload_count = 39

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    "/documents/upload",
                    files={
                        "file": (
                            f"native-batch-{index:02d}.txt",
                            f"Native batch route document {index}".encode(),
                            "text/plain",
                        )
                    },
                )
                for index in range(upload_count)
            )
        )

        assert [response.status_code for response in responses] == [200] * upload_count

        track_ids = [response.json()["track_id"] for response in responses]
        assert len(set(track_ids)) == upload_count

        track_payloads = await asyncio.gather(
            *(
                client.get(f"/documents/track_status/{track_id}")
                for track_id in track_ids
            )
        )

    assert len(indexed) == upload_count
    assert {track_id for _, track_id in indexed} == set(track_ids)
    assert len(rag.doc_status.docs) == upload_count

    for response in track_payloads:
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["total_count"] == 1
        assert payload["documents"][0]["status"] == DocStatus.PENDING.value
