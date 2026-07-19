"""Integration guard for Twin Folder scoping on the document list surface."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import native_shims
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.native_shims import build_native_shims_router


@dataclass
class FakeDocStatus:
    status: str
    file_path: str
    chunks_count: int
    chunks_list: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeDocStatusStore:
    def __init__(self) -> None:
        self.docs = {
            "doc-default-a": FakeDocStatus(
                status="completed",
                file_path="default-a.pdf",
                chunks_count=1,
                chunks_list=["c-default-a"],
                metadata={"folder": "default"},
            ),
            "doc-default-b": FakeDocStatus(
                status="completed",
                file_path="default-b.pdf",
                chunks_count=1,
                chunks_list=["c-default-b"],
                metadata={"folder": "default"},
            ),
            "doc-sandbox": FakeDocStatus(
                status="completed",
                file_path="sandbox.pdf",
                chunks_count=1,
                chunks_list=["c-sandbox"],
                metadata={"folder": "sandbox"},
            ),
        }

    async def get_docs_paginated(self, **kwargs: Any):
        folder = kwargs.get("folder")
        rows = [
            (doc_id, doc)
            for doc_id, doc in self.docs.items()
            if folder is None or (doc.metadata.get("folder") or "default") == folder
        ]
        return rows, len(rows)


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


@pytest.fixture()
async def client(monkeypatch):
    async def no_tags(_docs, *, folder: str) -> None:
        return None

    monkeypatch.setattr(native_shims, "_attach_tags_via_graph", no_tags)
    configure_auth(api_key="test-infra-root")
    rag = FakeRag()
    app = FastAPI()
    app.include_router(build_native_shims_router(lambda: rag))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-infra-root"},
    ) as c:
        yield c
    configure_auth()


async def _document_ids(client: AsyncClient, *, headers: dict[str, str] | None = None):
    r = await client.get("/documents", headers=headers)
    assert r.status_code == 200
    body = r.json()
    return body["total"], [item["doc_id"] for item in body["items"]]


class TestFolderScoping:
    async def test_default_folder_header_returns_only_default_docs(self, client):
        total, doc_ids = await _document_ids(
            client,
            headers={"X-Twin-Folder": "default"},
        )

        assert total == 2
        assert doc_ids == ["doc-default-a", "doc-default-b"]

    async def test_sandbox_folder_header_returns_only_sandbox_docs(self, client):
        total, doc_ids = await _document_ids(
            client,
            headers={"X-Twin-Folder": "sandbox"},
        )

        assert total == 1
        assert doc_ids == ["doc-sandbox"]

    async def test_missing_folder_header_uses_default_folder(self, client):
        total, doc_ids = await _document_ids(
            client,
        )

        assert total == 2
        assert doc_ids == ["doc-default-a", "doc-default-b"]
