"""Run the installed LightRAG HTTP app against a dedicated Memgraph database."""

import os
import re
import sys
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from lightrag_memgraph import _pool, register
from lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("LIGHTRAG_HTTP_INTEGRATION") != "1",
        reason="requires the LightRAG API extra and disposable Memgraph",
    ),
]


async def test_real_pagination_and_storage_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from lightrag.api import config

    monkeypatch.setattr(config, "_global_args", None)
    monkeypatch.setattr(config, "_initialized", False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lightrag-server",
            "--workspace",
            f"http_{uuid4().hex[:12]}",
            "--llm-binding",
            "openai",
            "--embedding-binding",
            "openai",
        ],
    )
    register()
    args = config.parse_args()
    args.kv_storage = "MemgraphKVStorage"
    args.vector_storage = "MemgraphVectorDBStorage"
    args.doc_status_storage = "MemgraphDocStatusStorage"
    args.graph_storage = "MemgraphStorage"
    args.llm_binding_host = args.embedding_binding_host = "http://127.0.0.1:1"
    args.llm_binding_api_key = "http-qualification"
    args.embedding_binding_api_key = "http-qualification"
    config.initialize_config(args, force=True)
    from lightrag.api import lightrag_server

    original_initialize = MemgraphDocStatusStorage.initialize
    storage = None

    async def capture_status_storage(self):
        nonlocal storage
        await original_initialize(self)
        storage = self

    monkeypatch.setattr(MemgraphDocStatusStorage, "initialize", capture_status_storage)
    app = lightrag_server.create_app(args)
    try:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://lightrag.test"
            ) as client,
        ):
            assert storage is not None
            timestamp = "2026-01-01T00:00:00+00:00"
            await storage.upsert(
                {
                    "doc-http": {
                        "content_summary": "HTTP qualification",
                        "content_length": 18,
                        "status": "processed",
                        "created_at": timestamp,
                        "updated_at": timestamp,
                        "file_path": "fixture.txt",
                    }
                }
            )
            payload = {"page": 1, "page_size": 10, "status_filter": "processed"}
            response = await client.post("/documents/paginated", json=payload)

            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("application/json")
            body = response.json()
            assert body["pagination"] == {
                "page": 1,
                "page_size": 10,
                "total_count": 1,
                "total_pages": 1,
                "has_next": False,
                "has_prev": False,
            }
            assert [doc["id"] for doc in body["documents"]] == ["doc-http"]
            assert body["documents"][0]["status"] == "processed"
            assert body["documents"][0]["file_path"] == "fixture.txt"
            assert body["status_counts"]["processed"] == 1

            failure = AsyncMock(side_effect=RuntimeError("Private storage diagnostic"))
            monkeypatch.setattr(storage, "get_docs_paginated", failure)
            response = await client.post("/documents/paginated", json=payload)

            failure.assert_awaited_once()
            assert response.status_code == 500
            assert response.headers["content-type"].startswith("application/json")
            body = response.json()
            assert set(body) == {"detail"}
            assert re.fullmatch(
                r"Internal server error \(error_id: [0-9a-f]{12}\)", body["detail"]
            )
    finally:
        await _pool.close_driver()
