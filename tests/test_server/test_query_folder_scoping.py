"""The Twin query routes bind the active folder into ``storage_folder_context``.

Batch-2 cloisonnement is enforced *at the storage layer*
(``MemgraphVectorDBStorage.query``), which reads the folder from a ContextVar.
These tests prove the route wiring: every grounding call (``aquery_llm``,
``aquery``, ``aquery_data``, and the streaming generator) runs with the request's
``X-Twin-Folder`` bound into ``get_active_storage_folder()`` — and that absent a
folder header the default catalog folder is bound (never ``None`` here, since the
catalog always resolves a default). The storage-level *effect* of that binding is
covered live in ``tests/test_folder_query_scoping.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph._constants import get_active_storage_folder
from twindb_lightrag_memgraph.server.twin_query_routes import build_twin_query_router


def _envelope(answer: str = "grounded answer") -> dict[str, Any]:
    return {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [],
            "references": [{"reference_id": "1", "file_path": "/kb/a.pdf"}],
        },
        "metadata": {},
        "llm_response": {
            "content": answer,
            "response_iterator": None,
            "is_streaming": False,
        },
    }


class CapturingRag:
    """Records the folder visible (via the ContextVar) inside each grounding API."""

    def __init__(self) -> None:
        self.folder_in_llm: str | None = "<unset>"
        self.folder_in_query: str | None = "<unset>"
        self.folder_in_data: str | None = "<unset>"

    async def aquery_llm(self, query: str, *, param):
        self.folder_in_llm = get_active_storage_folder()
        return _envelope()

    async def aquery(self, query: str, *, param):
        self.folder_in_query = get_active_storage_folder()
        return "context body"

    async def aquery_data(self, query: str, *, param):
        self.folder_in_data = get_active_storage_folder()
        return {"status": "success", "message": "ok", "data": {}, "metadata": {}}


@pytest.fixture()
async def client(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "default", "label": "Default", "kind": "primary"},
                {"id": "sandbox", "label": "Sandbox", "kind": "sandbox"},
            ]
        ),
    )
    rag = CapturingRag()
    app = FastAPI()
    app.include_router(build_twin_query_router(lambda: rag))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        c._rag = rag
        yield c


class TestQueryBindsFolderContext:
    async def test_nominal_query_binds_header_folder_into_aquery_llm(self, client):
        r = await client.post(
            "/query",
            json={"query": "q", "mode": "mix"},
            headers={"X-Twin-Folder": "sandbox"},
        )
        assert r.status_code == 200
        assert client._rag.folder_in_llm == "sandbox"

    async def test_only_need_context_binds_folder_into_aquery(self, client):
        r = await client.post(
            "/query",
            json={"query": "q", "mode": "mix", "only_need_context": True},
            headers={"X-Twin-Folder": "sandbox"},
        )
        assert r.status_code == 200
        assert client._rag.folder_in_query == "sandbox"

    async def test_query_data_binds_folder_into_aquery_data(self, client):
        r = await client.post(
            "/query/data",
            json={"query": "q", "mode": "mix"},
            headers={"X-Twin-Folder": "sandbox"},
        )
        assert r.status_code == 200
        assert client._rag.folder_in_data == "sandbox"

    async def test_stream_binds_folder_inside_generator(self, client):
        # The generator runs after the handler returns — the ContextVar must be
        # bound inside generate(), not at the route boundary. Reading the full
        # streamed body drives the generator to completion.
        r = await client.post(
            "/query/stream",
            json={"query": "q", "mode": "mix"},
            headers={"X-Twin-Folder": "sandbox"},
        )
        assert r.status_code == 200
        assert client._rag.folder_in_llm == "sandbox"

    async def test_absent_header_binds_catalog_default(self, client):
        r = await client.post("/query", json={"query": "q", "mode": "mix"})
        assert r.status_code == 200
        # No header → catalog default folder bound (scoping still applies, to the
        # default folder — never an unscoped/None leak on the Twin route).
        assert client._rag.folder_in_llm == "default"
