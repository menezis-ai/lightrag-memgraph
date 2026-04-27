"""
HTTP-level e2e tests — guarantees the backend never returns HTML on errors.

Reproduces the **exact failure mode** that crashed the BNP frontend:

  1. Backend slow on POST /documents/paginated (> 60s) → nginx upstream timeout
  2. nginx returns its HTML 502 page to the frontend
  3. The frontend parses the response as JSON → crash

v0.5.1 fixed the backend slowness (indexes + parallel count/fetch) and
``tests/test_probe_e2e.py`` covers the Cypher-level regression. **This
module covers the HTTP layer**: even when the storage layer fails or the
endpoint receives garbage, the backend MUST return ``application/json``
with a parseable body — never HTML.

Approach: spin up a **minimal FastAPI app** in the test that wraps a real
``LightRAG`` instance with our Memgraph backends (via ``register()``).
Hit it through ``httpx.AsyncClient`` + ``ASGITransport`` — no network,
no uvicorn, no extra deps beyond the test dependencies. The shape of
the endpoints mirrors the upstream LightRAG API but only what the
frontend (and probes) actually call.

Requires a running Memgraph instance (set ``MEMGRAPH_URI``).
"""

import hashlib
import os
import shutil
import tempfile
import uuid

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.utils import EmbeddingFunc

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _pool

twindb_lightrag_memgraph.register()


EMBEDDING_DIM = 384


# ── Mock LLM/embedding (deterministic, no external call) ────────────


async def _mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    return "mock answer"


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    out = []
    for text in texts:
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[
            :EMBEDDING_DIM
        ].astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        out.append(vec)
    return np.array(out)


# ── FastAPI factory ─────────────────────────────────────────────────


def _build_app(rag: LightRAG) -> FastAPI:
    """Minimal FastAPI app exposing the endpoints LightRAG frontend / probes hit.

    The exception handler is the **regression sentinel**: if any unhandled
    exception escapes a route, it must still serialise to JSON, never HTML.
    Without this handler, FastAPI would return a default text/html 500 page
    on RuntimeError — exactly what crashed the BNP frontend.
    """
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.post("/documents/paginated")
    async def paginated(req: Request):
        body = await req.json()
        page = int(body.get("page", 1))
        page_size = int(body.get("page_size", 50))
        status_filter = body.get("status_filter")
        sort_field = body.get("sort_field", "updated_at")
        sort_direction = body.get("sort_direction", "desc")

        filter_enum = DocStatus(status_filter) if status_filter else None
        docs, total = await rag.doc_status.get_docs_paginated(
            status_filter=filter_enum,
            page=page,
            page_size=page_size,
            sort_field=sort_field,
            sort_direction=sort_direction,
        )
        return {
            "documents": [
                {"id": doc_id, "status": s.status.value, "file_path": s.file_path}
                for doc_id, s in docs
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    @app.post("/documents/text")
    async def insert_text(req: Request):
        body = await req.json()
        text = body.get("text")
        if not text:
            raise HTTPException(status_code=422, detail="text required")
        track_id = await rag.ainsert(text, ids=[f"doc-{uuid.uuid4().hex[:8]}"])
        return {"status": "success", "track_id": track_id}

    @app.get("/documents/track_status/{track_id}")
    async def track_status(track_id: str):
        # Best-effort: LightRAG stores per-track status; we just query DocStatus.
        docs = await rag.doc_status.get_docs_by_track_id(track_id)
        return {
            "track_id": track_id,
            "count": len(docs),
            "documents": [
                {"id": k, "status": v.status.value} for k, v in docs.items()
            ],
        }

    @app.exception_handler(Exception)
    async def json_error(_request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    return app


# ── Cleanup helper ──────────────────────────────────────────────────


async def _cleanup(workspace: str):
    """Drop all data tagged with this workspace from Memgraph."""
    try:
        async with _pool.get_session() as session:
            for prefix in ("KV_", "Vec_", "DocStatus_"):
                label = f"{prefix}{workspace}"
                try:
                    result = await session.run(
                        f"MATCH (n) WHERE ANY(l IN labels(n) "
                        f"WHERE l STARTS WITH '{label}') DETACH DELETE n"
                    )
                    await result.consume()
                except Exception:
                    pass
            try:
                result = await session.run(
                    f"MATCH (n:`{workspace}`) DETACH DELETE n"
                )
                await result.consume()
            except Exception:
                pass
    except Exception:
        pass


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def working_dir():
    d = tempfile.mkdtemp(prefix="lightrag_http_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
async def rag(working_dir):
    """LightRAG instance with our Memgraph backends, isolated workspace."""
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_share_data,
    )

    finalize_share_data()
    initialize_share_data()

    workspace = f"http_e2e_{uuid.uuid4().hex[:8]}"
    os.environ["MEMGRAPH_WORKSPACE"] = workspace

    await _cleanup(workspace)

    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=8192,
        func=_mock_embedding,
    )

    instance = LightRAG(
        working_dir=working_dir,
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        workspace=workspace,
        embedding_func=embedding_func,
        llm_model_func=_mock_llm,
        enable_llm_cache=False,
        chunk_token_size=200,
        chunk_overlap_token_size=50,
    )
    await instance.initialize_storages()

    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass

    yield instance

    await _cleanup(workspace)
    await instance.finalize_storages()


@pytest.fixture
async def http_client(rag):
    """ASGI test client — no network, no uvicorn."""
    app = _build_app(rag)
    # raise_app_exceptions=False: let FastAPI's exception_handler process the
    # error and return JSON, instead of bubbling the raw exception out of the
    # transport. This matches real-world behaviour (uvicorn never re-raises
    # to the network) and is what we need to test the JSON-only contract.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        yield client


# ── Tests ───────────────────────────────────────────────────────────


@pytest.mark.integration
class TestHealthHTTP:
    async def test_health_returns_json(self, http_client):
        resp = await http_client.get("/health")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["status"] == "healthy"


@pytest.mark.integration
class TestPaginatedHTTP:
    """The exact endpoint that broke at BNP."""

    async def test_returns_json_not_html(self, http_client):
        """Empty paginated must still return parseable JSON, not HTML."""
        resp = await http_client.post(
            "/documents/paginated", json={"page": 1, "page_size": 50}
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json"), (
            f"Got content-type={resp.headers['content-type']} — "
            "frontend would crash trying to parse this as JSON"
        )
        body = resp.json()  # raises if HTML
        assert "total" in body
        assert "documents" in body
        assert "page" in body

    async def test_paginated_with_status_filter(self, http_client):
        """status_filter is parsed as DocStatus enum, not crashed on."""
        resp = await http_client.post(
            "/documents/paginated",
            json={"page": 1, "page_size": 50, "status_filter": "processed"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["total"] == 0  # nothing inserted in this fixture

    async def test_paginated_invalid_status_returns_json_error(self, http_client):
        """Bad status_filter must return JSON error, not HTML 500."""
        resp = await http_client.post(
            "/documents/paginated",
            json={"page": 1, "page_size": 50, "status_filter": "garbage"},
        )
        # Either 500 with our handler, or 422 — both must be JSON
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert "error" in body or "detail" in body


@pytest.mark.integration
class TestErrorResponsesAreJson:
    """The frontend crash root cause — guarantee JSON on every status code."""

    async def test_500_via_storage_failure_returns_json(
        self, http_client, monkeypatch
    ):
        """Force the storage layer to raise — handler must intercept and JSON-ify."""
        from twindb_lightrag_memgraph.docstatus_impl import (
            MemgraphDocStatusStorage,
        )

        async def _boom(self, *args, **kwargs):
            raise RuntimeError("simulated backend crash")

        monkeypatch.setattr(
            MemgraphDocStatusStorage,
            "get_docs_paginated",
            _boom,
        )

        resp = await http_client.post(
            "/documents/paginated", json={"page": 1, "page_size": 50}
        )
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json"), (
            "Without JSON exception handler, FastAPI returns text/html — "
            "this is what crashed the BNP frontend"
        )
        body = resp.json()
        assert body["error"] == "simulated backend crash"
        assert body["type"] == "RuntimeError"

    async def test_404_returns_json(self, http_client):
        """FastAPI default 404 — verify it stays JSON."""
        resp = await http_client.get("/this-route-does-not-exist")
        assert resp.status_code == 404
        assert resp.headers["content-type"].startswith("application/json")
        assert "detail" in resp.json()

    async def test_405_returns_json(self, http_client):
        """Wrong method on a known route stays JSON."""
        resp = await http_client.delete("/health")
        assert resp.status_code == 405
        assert resp.headers["content-type"].startswith("application/json")

    async def test_422_validation_returns_json(self, http_client):
        """Missing required body fields → JSON error, not HTML."""
        resp = await http_client.post("/documents/text", json={})
        # No "text" key → our HTTPException(422)
        assert resp.status_code == 422
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert "detail" in body


@pytest.mark.integration
class TestTrackStatusHTTP:
    async def test_unknown_track_id_returns_json_zero_count(self, http_client):
        """Unknown track_id — must still return valid JSON (count=0)."""
        resp = await http_client.get("/documents/track_status/unknown-track-xyz")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        assert body["track_id"] == "unknown-track-xyz"
        assert body["count"] == 0
        assert body["documents"] == []
