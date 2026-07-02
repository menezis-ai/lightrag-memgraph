"""Upload-to-list folder isolation contract.

This pins the full HTTP wiring around the native upload path:

* ``POST /documents/upload`` is an ingestion write and must capture the
  validated ``X-Twin-Folder`` request header.
* ``GET /documents`` is served by the Twin native shim and must pass that
  folder to storage pagination.

The test uses an in-memory fake RAG/doc_status store, but real FastAPI
middleware, multipart upload handling, and shim routing.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI, File, UploadFile
from httpx import ASGITransport, AsyncClient
from lightrag.base import DocProcessingStatus, DocStatus

from twindb_lightrag_memgraph import _install_storage_folder_capture
from twindb_lightrag_memgraph._constants import get_active_storage_folder
from twindb_lightrag_memgraph.server import native_shims
from twindb_lightrag_memgraph.server.native_shims import build_native_shims_router


class FakeDocStatusStore:
    def __init__(self) -> None:
        self.docs: dict[str, DocProcessingStatus] = {}

    async def upsert(self, data: dict[str, DocProcessingStatus]) -> None:
        self.docs.update(data)

    async def get_docs_paginated(self, **kwargs: Any):
        folder = kwargs.get("folder")
        rows = [
            (doc_id, doc)
            for doc_id, doc in self.docs.items()
            if (doc.metadata or {}).get("folder", "default") == folder
        ]
        return rows, len(rows)

    async def get_status_counts(self, folder: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for doc in self.docs.values():
            if (doc.metadata or {}).get("folder", "default") != folder:
                continue
            status = (
                doc.status.value if hasattr(doc.status, "value") else str(doc.status)
            )
            counts[status] = counts.get(status, 0) + 1
        return counts


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatusStore()


@pytest.fixture(autouse=True)
def _folder_env(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "default", "label": "Default folder", "kind": "primary"},
                {"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
            ]
        ),
    )


async def test_uploaded_sandbox_document_is_hidden_from_default(monkeypatch):
    async def no_tags(_docs, *, folder: str) -> None:
        return None

    monkeypatch.setattr(native_shims, "_attach_tags_via_graph", no_tags)

    rag = FakeRag()
    app = FastAPI()
    _install_storage_folder_capture(app)
    app.include_router(build_native_shims_router(lambda: rag))

    @app.post("/documents/upload")
    async def upload_probe(file: UploadFile = File(...)):
        folder = get_active_storage_folder()
        doc_id = f"doc-{file.filename}"
        await rag.doc_status.upsert(
            {
                doc_id: DocProcessingStatus(
                    content_summary="uploaded",
                    content_length=0,
                    file_path=file.filename or "upload.txt",
                    status=DocStatus.PROCESSED,
                    created_at="2026-06-20T00:00:00Z",
                    updated_at="2026-06-20T00:00:00Z",
                    chunks_count=0,
                    metadata={"folder": folder},
                )
            }
        )
        return {"status": "success", "message": "queued", "track_id": "upload-1"}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        uploaded = await client.post(
            "/documents/upload",
            headers={"X-Twin-Folder": "sandbox"},
            files={"file": ("sandbox.txt", b"sandbox payload", "text/plain")},
        )
        default_docs = await client.get("/documents")
        sandbox_docs = await client.get(
            "/documents",
            headers={"X-Twin-Folder": "sandbox"},
        )

    assert uploaded.status_code == 200
    assert default_docs.status_code == 200
    assert sandbox_docs.status_code == 200

    assert default_docs.json()["items"] == []
    assert default_docs.json()["total"] == 0

    sandbox_body = sandbox_docs.json()
    assert sandbox_body["total"] == 1
    assert [item["doc_id"] for item in sandbox_body["items"]] == ["doc-sandbox.txt"]
    assert sandbox_body["items"][0]["folder"] == "sandbox"
