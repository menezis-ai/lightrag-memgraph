"""Integration guard for Twin Space scoping on the document list surface."""

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


class FakeDocStatusStore:
    def __init__(self) -> None:
        self.docs = {
            "doc-default-a": FakeDocStatus(
                status="completed",
                file_path="default-a.pdf",
                chunks_count=1,
                chunks_list=["c-default-a"],
                metadata={"space": "default"},
            ),
            "doc-default-b": FakeDocStatus(
                status="completed",
                file_path="default-b.pdf",
                chunks_count=1,
                chunks_list=["c-default-b"],
                metadata={"space": "default"},
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


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatusStore()


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


async def _document_ids(client: AsyncClient, *, headers: dict[str, str] | None = None):
    r = await client.get("/documents", headers=headers)
    assert r.status_code == 200
    body = r.json()
    return body["total"], [item["doc_id"] for item in body["items"]]


class TestSpaceScoping:
    async def test_default_space_header_returns_only_default_docs(self, client):
        total, doc_ids = await _document_ids(
            client,
            headers={"X-Twin-Space": "default"},
        )

        assert total == 2
        assert doc_ids == ["doc-default-a", "doc-default-b"]

    async def test_sandbox_space_header_returns_only_sandbox_docs(self, client):
        total, doc_ids = await _document_ids(
            client,
            headers={"X-Twin-Space": "sandbox"},
        )

        assert total == 1
        assert doc_ids == ["doc-sandbox"]

    async def test_legacy_workspace_header_still_maps_to_space(self, client):
        total, doc_ids = await _document_ids(
            client,
            headers={"X-Twin-Workspace": "sandbox"},
        )

        assert total == 1
        assert doc_ids == ["doc-sandbox"]
