"""
Native document route runtime coverage.

These tests exercise LightRAG's real /documents HTTP surface against the
Memgraph storage backends. They intentionally sit outside the WebUI shims so CI
guards the native upload and failed-reprocess paths, not only adapter imports or
route wiring.
"""

import asyncio
import hashlib
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _install_storage_folder_capture, _pool

twindb_lightrag_memgraph.register()


EMBEDDING_DIM = 384
WORKSPACE = "native_runtime"
DEFAULT_FOLDER = "default"

UPLOAD_DOC = (
    "Native upload runtime coverage document. It mentions Atlas, Boreal, and "
    "Cygnus as internal services so the parser, chunker, vector storage, graph "
    "extraction, and DocStatus membership all have concrete content to persist."
)

REPROCESS_DOC_ID = "doc-native-runtime-reprocess"
REPROCESS_DOC = (
    "Failed reprocess runtime coverage document. Atlas depends on Boreal, and "
    "Boreal reports incidents to Cygnus. This content must survive in full_docs "
    "while DocStatus is failed so /documents/reprocess_failed can resume it."
)


def _build_extraction_response() -> str:
    return "\n".join(
        [
            "entity<|#|>Atlas<|#|>service<|#|>Atlas is an internal service.",
            "entity<|#|>Boreal<|#|>service<|#|>Boreal is an internal service.",
            "entity<|#|>Cygnus<|#|>service<|#|>Cygnus receives incident reports.",
            "relation<|#|>Atlas<|#|>Boreal<|#|>depends on<|#|>Atlas depends on Boreal.",
            "relation<|#|>Boreal<|#|>Cygnus<|#|>reports to<|#|>Boreal reports incidents to Cygnus.",
            "<|COMPLETE|>",
        ]
    )


async def _mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    prompt_lower = prompt.lower() if isinstance(prompt, str) else ""
    if "entity_types" in prompt_lower or "extract" in prompt_lower:
        return _build_extraction_response()
    if "summary" in prompt_lower or "merge" in prompt_lower:
        return "Internal services and dependencies summary."
    return "Atlas depends on Boreal, and Boreal reports incidents to Cygnus."


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(digest * (EMBEDDING_DIM // 32 + 1), dtype=np.uint8)[
            :EMBEDDING_DIM
        ].astype(np.float32)
        norm = np.linalg.norm(vec)
        vectors.append(vec / norm if norm else vec)
    return np.array(vectors)


def _passthrough_priority_limit_async_func_call(*args, **kwargs):
    def decorator(func):
        async def wrapped(*call_args, **call_kwargs):
            return await func(*call_args, **call_kwargs)

        async def shutdown(*args, **kwargs):
            return None

        wrapped.shutdown = shutdown
        return wrapped

    return decorator


async def _shutdown_rag_workers(rag: LightRAG) -> None:
    async def _shutdown(shutdown):
        try:
            await shutdown(graceful=False)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            await shutdown()

    embedding_func = getattr(getattr(rag, "embedding_func", None), "func", None)
    shutdown = getattr(embedding_func, "shutdown", None)
    if callable(shutdown):
        await _shutdown(shutdown)

    role_funcs = getattr(rag, "role_llm_funcs", {})
    for wrapped in role_funcs.values():
        shutdown = getattr(wrapped, "shutdown", None)
        if callable(shutdown):
            await _shutdown(shutdown)


@pytest.fixture
def runtime_dirs():
    root = tempfile.mkdtemp(prefix="native_runtime_")
    yield {
        "working": os.path.join(root, "work"),
        "input": os.path.join(root, "input"),
    }
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
async def native_runtime(monkeypatch, runtime_dirs):
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", WORKSPACE)
    monkeypatch.setenv("INPUT_DIR", runtime_dirs["input"])
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", DEFAULT_FOLDER)
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        '[{"id":"default","label":"Default","kind":"primary"},'
        '{"id":"sandbox","label":"Sandbox","kind":"secondary"}]',
    )

    finalize_share_data()
    initialize_share_data()
    await _cleanup_workspace()

    import lightrag.lightrag as lightrag_module
    import lightrag.llm_roles as llm_roles_module

    monkeypatch.setattr(
        lightrag_module,
        "priority_limit_async_func_call",
        _passthrough_priority_limit_async_func_call,
    )
    monkeypatch.setattr(
        llm_roles_module,
        "priority_limit_async_func_call",
        _passthrough_priority_limit_async_func_call,
    )

    async def mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await _mock_llm(prompt, system_prompt, history_messages, **kwargs)

    async def mock_embedding(texts: list[str]) -> np.ndarray:
        return await _mock_embedding(texts)

    rag = LightRAG(
        working_dir=runtime_dirs["working"],
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        workspace=WORKSPACE,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=mock_embedding,
        ),
        llm_model_func=mock_llm,
        enable_llm_cache=False,
        chunk_token_size=120,
        chunk_overlap_token_size=20,
    )
    await rag.initialize_storages()

    try:
        from lightrag.kg.shared_storage import initialize_pipeline_status

        await initialize_pipeline_status()
    except Exception:
        pass

    from lightrag.api.routers.document_routes import (
        DocumentManager,
        create_document_routes,
    )

    doc_manager = DocumentManager(runtime_dirs["input"], workspace=WORKSPACE)
    app = FastAPI()
    _install_storage_folder_capture(app)
    app.include_router(create_document_routes(rag, doc_manager, api_key=None))

    try:
        yield rag, app
    finally:
        await _shutdown_rag_workers(rag)
        await _cleanup_workspace()
        await rag.finalize_storages()


async def _cleanup_workspace() -> None:
    try:
        async with _pool.get_session() as session:
            for prefix in ("KV_", "Vec_", "DocStatus_", "Folder_"):
                label = f"{prefix}{WORKSPACE}"
                result = await session.run(
                    "MATCH (n) WHERE ANY(l IN labels(n) WHERE l STARTS WITH $label) "
                    "DETACH DELETE n",
                    label=label,
                )
                await result.consume()
            result = await session.run(f"MATCH (n:`{WORKSPACE}`) DETACH DELETE n")
            await result.consume()
    except Exception:
        pass


async def _count_nodes(label_prefix: str) -> int:
    async with _pool.get_read_session() as session:
        result = await session.run(
            "MATCH (n) WHERE ANY(l IN labels(n) WHERE l STARTS WITH $prefix) "
            "RETURN count(n) AS count",
            prefix=label_prefix,
        )
        record = await result.single()
        await result.consume()
        return record["count"] if record else 0


async def _poll_track(client: httpx.AsyncClient, track_id: str) -> dict:
    for _ in range(50):
        response = await client.get(f"/documents/track_status/{track_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        documents = payload.get("documents", [])
        if documents and all(doc["status"] == DocStatus.PROCESSED for doc in documents):
            return payload
        await asyncio.sleep(0.1)
    return payload


def _doc_status_value(doc_status) -> str:
    status = doc_status["status"] if isinstance(doc_status, dict) else doc_status.status
    return status.value if isinstance(status, DocStatus) else str(status)


def _doc_field(doc_status, field: str):
    return doc_status[field] if isinstance(doc_status, dict) else getattr(doc_status, field)


@pytest.mark.integration
async def test_native_upload_ingests_file_through_runtime_pipeline(native_runtime):
    rag, app = native_runtime

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents/upload",
            files={
                "file": (
                    "native-runtime-upload.txt",
                    UPLOAD_DOC.encode(),
                    "text/plain",
                )
            },
            headers={"X-Twin-Folder": DEFAULT_FOLDER},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success"

        track_id = response.json()["track_id"]
        track_payload = await _poll_track(client, track_id)

    assert track_payload["total_count"] == 1
    uploaded = track_payload["documents"][0]
    assert uploaded["status"] == DocStatus.PROCESSED
    assert uploaded["file_path"] == "native-runtime-upload.txt"
    assert uploaded["chunks_count"] > 0

    doc_id = uploaded["id"]
    stored = await rag.doc_status.get_by_id(doc_id)
    assert stored is not None
    assert _doc_status_value(stored) == DocStatus.PROCESSED
    assert _doc_field(stored, "chunks_list")
    assert await rag.doc_status.get_folders_for_doc(doc_id) == [DEFAULT_FOLDER]
    assert await _count_nodes(f"Vec_{WORKSPACE}") > 0

    deletion = await rag.adelete_by_doc_id(doc_id)
    assert deletion.status == "success"
    assert await rag.doc_status.get_by_id(doc_id) is None


@pytest.mark.integration
async def test_reprocess_failed_resumes_failed_document_through_runtime_pipeline(
    native_runtime,
):
    rag, app = native_runtime
    track_id = "runtime-reprocess-track"

    await rag.apipeline_enqueue_documents(
        REPROCESS_DOC,
        ids=[REPROCESS_DOC_ID],
        file_paths=["native-runtime-reprocess.md"],
        track_id=track_id,
    )

    now = datetime.now(timezone.utc).isoformat()
    await rag.doc_status.upsert(
        {
            REPROCESS_DOC_ID: {
                "status": DocStatus.FAILED,
                "content_summary": REPROCESS_DOC[:100],
                "content_length": len(REPROCESS_DOC),
                "created_at": now,
                "updated_at": now,
                "file_path": "native-runtime-reprocess.md",
                "track_id": track_id,
                "chunks_count": 0,
                "chunks_list": [],
                "error_msg": "synthetic transient failure before reprocess",
                "metadata": {"folder": DEFAULT_FOLDER},
            }
        }
    )
    failed = await rag.doc_status.get_by_id(REPROCESS_DOC_ID)
    assert failed is not None
    assert _doc_status_value(failed) == DocStatus.FAILED

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        # Reprocess is workspace-global, not folder-scoped: a request issued
        # from sandbox must still resume a failed default-folder document.
        response = await client.post(
            "/documents/reprocess_failed",
            headers={"X-Twin-Folder": "sandbox"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "reprocessing_started"

        track_payload = await _poll_track(client, track_id)

    assert track_payload["total_count"] == 1
    reprocessed = track_payload["documents"][0]
    assert reprocessed["id"] == REPROCESS_DOC_ID
    assert reprocessed["status"] == DocStatus.PROCESSED
    assert reprocessed["chunks_count"] > 0

    stored = await rag.doc_status.get_by_id(REPROCESS_DOC_ID)
    assert stored is not None
    assert _doc_status_value(stored) == DocStatus.PROCESSED
    assert _doc_field(stored, "chunks_list")
    assert DEFAULT_FOLDER in await rag.doc_status.get_folders_for_doc(REPROCESS_DOC_ID)
    assert await _count_nodes(f"Vec_{WORKSPACE}") > 0
