"""Space scoping tests for the Twin native-route shims."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import native_shims
from twindb_lightrag_memgraph.server.native_shims import build_native_shims_router


@dataclass
class FakeDocStatus:
    status: str
    file_path: str
    chunks_count: int
    chunks_list: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_summary: str | None = None
    content_length: int | None = None
    created_at: str | None = None
    updated_at: str | None = None
    track_id: str | None = None
    error_msg: str | None = None


class FakeDocStatusStore:
    def __init__(self) -> None:
        self.docs = {
            "doc-default": FakeDocStatus(
                status="completed",
                file_path="default.pdf",
                chunks_count=1,
                chunks_list=["c-default"],
            ),
            "doc-sandbox": FakeDocStatus(
                status="completed",
                file_path="sandbox.pdf",
                chunks_count=1,
                chunks_list=["c-sandbox"],
                metadata={"space": "sandbox"},
            ),
        }

    async def get_docs_paginated(self, **_: Any):
        return list(self.docs.items()), len(self.docs)

    async def get_by_id(self, doc_id: str):
        return self.docs.get(doc_id)


class FakeTextChunks:
    async def get_by_ids(self, ids: list[str]):
        return [
            {"chunk_order_index": i, "content": f"text for {chunk_id}"}
            for i, chunk_id in enumerate(ids)
        ]


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatusStore()
        self.text_chunks = FakeTextChunks()
        self.deleted: list[str] = []

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        self.deleted.append(doc_id)


@pytest.fixture(autouse=True)
def _space_env(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_SPACE", "default")
    monkeypatch.setenv(
        "TWIN_SPACES_JSON",
        json.dumps(
            [
                {"id": "default", "label": "Default space", "kind": "primary"},
                {"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
            ]
        ),
    )


@pytest.fixture()
async def client(monkeypatch):
    async def no_tags(_docs, *, space: str) -> None:
        return None

    monkeypatch.setattr(native_shims, "_attach_tags_via_graph", no_tags)
    rag = FakeRag()
    app = FastAPI()
    app.include_router(build_native_shims_router(lambda: rag))
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c


class TestNativeShimSpaces:
    async def test_documents_default_space_sees_legacy_docs_only(self, client):
        r = await client.get("/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert [item["doc_id"] for item in body["items"]] == ["doc-default"]

    async def test_documents_sandbox_space_sees_sandbox_docs_only(self, client):
        r = await client.get("/documents", headers={"X-Twin-Space": "sandbox"})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert [item["doc_id"] for item in body["items"]] == ["doc-sandbox"]

    async def test_chunks_rejects_doc_from_another_space(self, client):
        r = await client.get("/documents/doc-sandbox/chunks")
        assert r.status_code == 404

    async def test_chunks_accepts_doc_in_requested_space(self, client):
        r = await client.get(
            "/documents/doc-sandbox/chunks",
            headers={"X-Twin-Space": "sandbox"},
        )
        assert r.status_code == 200
        assert r.json()[0]["chunk_id"] == "c-sandbox"
