"""PR-P2 acceptance: export(A) -> dry-run/apply(B) -> export(B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph._constants import storage_folder_context
from twindb_lightrag_memgraph.portability.exporter import export_kb
from twindb_lightrag_memgraph.portability.importer import apply_import
from twindb_lightrag_memgraph.portability.plan import create_dry_run, write_report
from twindb_lightrag_memgraph.portability.validate import validate_import
from twindb_lightrag_memgraph.server import folder_store

twindb_lightrag_memgraph.register()
pytestmark = pytest.mark.integration

SOURCE_WS = "portability_source"
TARGET_WS = "portability_target"
SOURCE_FOLDER = "staging"
TARGET_FOLDER = "production"
DIM = 4


async def _run(query: str, **params: Any) -> list[dict[str, Any]]:
    async with _pool.get_session() as session:
        result = await session.run(query, **params)
        rows = [dict(record) async for record in result]
        await result.consume()
    return rows


async def _wipe(workspace: str, folders: tuple[str, ...]) -> None:
    labels = [
        *(
            f"KV_{workspace}_{ns}"
            for ns in (
                "full_docs",
                "text_chunks",
                "full_entities",
                "full_relations",
                "entity_chunks",
                "relation_chunks",
                "llm_response_cache",
            )
        ),
        *(f"Vec_{workspace}_{ns}" for ns in ("chunks", "entities", "relationships")),
        f"DocStatus_{workspace}",
        f"Folder_{workspace}",
        f"GraphOverride_{workspace}",
        f"GraphRelOverride_{workspace}",
        f"WebuiSettings_{workspace}",
        f"TwinSourceLink_{workspace}",
        f"WebuiApiKey_{workspace}",
        workspace,
        *(f"WebuiTag_{folder}" for folder in folders),
        *(f"WebuiTagCategory_{folder}" for folder in folders),
        *(f"WebuiActivity_{folder}" for folder in folders),
        *(f"WebuiNotification_{folder}" for folder in folders),
    ]
    for label in labels:
        await _run(f"MATCH (n:`{label}`) DETACH DELETE n")
    for namespace in ("chunks", "entities", "relationships"):
        try:
            await _run(f"DROP VECTOR INDEX `vec_{workspace}_{namespace}`")
        except Exception:
            pass


def _embedding():
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str]):
        import numpy as np

        return np.asarray(
            [
                [float(len(text) % 11), float(index + 1), 0.25, -0.5]
                for index, text in enumerate(texts)
            ],
            dtype=np.float32,
        )

    return EmbeddingFunc(embedding_dim=DIM, max_token_size=8192, func=embed)


async def _seed_source(embedding_func) -> None:
    from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
    from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage
    from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

    full_docs = MemgraphKVStorage(
        namespace="full_docs",
        global_config={"workspace": SOURCE_WS},
        embedding_func=None,
    )
    chunks = MemgraphKVStorage(
        namespace="text_chunks",
        global_config={"workspace": SOURCE_WS},
        embedding_func=None,
    )
    await full_docs.initialize()
    await chunks.initialize()
    await full_docs.upsert({"doc-1": {"content": "portable document"}})
    await chunks.upsert(
        {
            "chunk-1": {
                "content": "portable chunk",
                "full_doc_id": "doc-1",
                "tokens": 2,
            }
        }
    )

    vectors = MemgraphVectorDBStorage(
        namespace="chunks",
        global_config={
            "workspace": SOURCE_WS,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.0},
        },
        embedding_func=embedding_func,
        meta_fields={"full_doc_id", "content", "file_path"},
    )
    await vectors.initialize()
    await vectors.upsert(
        {
            "chunk-1": {
                "content": "portable chunk",
                "full_doc_id": "doc-1",
                "file_path": "portable.txt",
                "chunk_order_index": 0,
                "tokens": 2,
            }
        }
    )

    docs = MemgraphDocStatusStorage(
        namespace="doc_status",
        global_config={"workspace": SOURCE_WS},
        embedding_func=None,
    )
    await docs.initialize()
    with storage_folder_context(SOURCE_FOLDER):
        await docs.upsert(
            {
                "doc-1": {
                    "status": "processed",
                    "file_path": "portable.txt",
                    "content_hash": "portable-hash",
                    "content_summary": "portable",
                    "content_length": 14,
                    "chunks_count": 1,
                    "chunks_list": ["chunk-1"],
                    "metadata": {"classification": {"class_id": "C1"}},
                    "folder": SOURCE_FOLDER,
                    "created_at": "2026-08-26T10:00:00+00:00",
                    "updated_at": "2026-08-26T10:00:01+00:00",
                }
            }
        )

    for entity in ("Alice", "Acme", "ManualA", "ManualB"):
        await _run(
            f"MERGE (n:`{SOURCE_WS}` {{entity_id: $entity}}) "
            "SET n.entity_type = 'Concept', n.description = $description, "
            "n.source_id = 'chunk-1', n.file_path = 'portable.txt' "
            "SET n:`Concept`",
            entity=entity,
            description=f"{entity} portable",
        )
    await _run(
        f"MATCH (a:`{SOURCE_WS}` {{entity_id: 'Alice'}}), "
        f"(b:`{SOURCE_WS}` {{entity_id: 'Acme'}}) "
        "MERGE (a)-[r:DIRECTED]->(b) "
        "SET r.description = 'uses', r.source_id = 'chunk-1'"
    )
    await _run(
        f"MERGE (f:`Folder_{SOURCE_WS}` {{id: $folder}}) "
        f"WITH f MATCH (a:`{SOURCE_WS}` {{entity_id: 'ManualA'}}), "
        f"(b:`{SOURCE_WS}` {{entity_id: 'ManualB'}}) "
        "MERGE (a)-[:GRAPH_MEMBER_OF]->(f) "
        "MERGE (b)-[:GRAPH_MEMBER_OF]->(f) "
        "MERGE (a)-[r:DIRECTED]->(b) "
        "SET r.keywords = 'manual link', r.twin_folder_json = $folders",
        folder=SOURCE_FOLDER,
        folders=json.dumps([SOURCE_FOLDER]),
    )


@pytest.fixture
async def clean_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_PORTABILITY_DIR", str(tmp_path / "portability"))
    monkeypatch.setenv("TWIN_FOLDERS_RUNTIME_FILE", str(tmp_path / "folders.json"))
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", str(tmp_path / "procedures.json"))
    monkeypatch.setenv("TWIN_MIP_MAX_CLASSIFICATION", "C2")
    folder_store.reset_runtime_store()
    await _wipe(SOURCE_WS, (SOURCE_FOLDER, TARGET_FOLDER))
    await _wipe(TARGET_WS, (SOURCE_FOLDER, TARGET_FOLDER))
    yield tmp_path
    folder_store.reset_runtime_store()
    await _wipe(SOURCE_WS, (SOURCE_FOLDER, TARGET_FOLDER))
    await _wipe(TARGET_WS, (SOURCE_FOLDER, TARGET_FOLDER))


async def test_operator_roundtrip_with_folder_mapping(clean_roundtrip, monkeypatch):
    tmp_path: Path = clean_roundtrip
    embedding_func = _embedding()
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", SOURCE_FOLDER)
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps([{"id": SOURCE_FOLDER, "label": "Staging", "kind": "primary"}]),
    )
    await _seed_source(embedding_func)
    source_bundle = tmp_path / "source-bundle"
    source = await export_kb(
        source_bundle,
        workspace=SOURCE_WS,
        embedding_func=embedding_func,
        actor="integration-test",
    )

    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", TARGET_FOLDER)
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps([{"id": TARGET_FOLDER, "label": "Production", "kind": "primary"}]),
    )
    mapping = {SOURCE_FOLDER: TARGET_FOLDER}
    report = await create_dry_run(
        source_bundle,
        workspace=TARGET_WS,
        folder_map=mapping,
        embedding_func=embedding_func,
    )
    assert report["blocking"] == [], report["blocking"]
    report_path = write_report(tmp_path / "report.json", report)
    checkpoint = tmp_path / "checkpoint.json"
    applied = await apply_import(
        source_bundle,
        report_path=report_path,
        checkpoint_path=checkpoint,
        embedding_func=embedding_func,
    )
    assert applied["ok"] is True and applied["resumed"] is False

    validation = await validate_import(
        source_bundle,
        workspace=TARGET_WS,
        folder_map=mapping,
        embedding_func=embedding_func,
    )
    assert validation["ok"] is True, validation["problems"]
    assert validation["expected_state_hash"] == validation["actual_state_hash"]
    assert validation["expected_state_hash"] != source.state_hash

    from twindb_lightrag_memgraph.server import folder, graph_reader

    (manual,) = await _run(
        f"MATCH (a:`{TARGET_WS}` {{entity_id: 'ManualA'}})-[r:DIRECTED]->"
        f"(b:`{TARGET_WS}` {{entity_id: 'ManualB'}}) "
        "RETURN r.twin_folder_json AS folders, r.source_id AS source_id"
    )
    assert json.loads(manual["folders"]) == [TARGET_FOLDER]
    assert manual["source_id"] is None
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": TARGET_FOLDER, "label": "Production", "kind": "primary"},
                {"id": SOURCE_FOLDER, "label": "Staging", "kind": "team"},
            ]
        ),
    )
    with folder.scoped_folder(TARGET_FOLDER):
        visible = await graph_reader._read_one_relation(
            TARGET_WS, "ManualA", "ManualB", {}, set()
        )
    with folder.scoped_folder(SOURCE_FOLDER):
        hidden = await graph_reader._read_one_relation(
            TARGET_WS, "ManualA", "ManualB", {}, set()
        )
    assert visible is not None
    assert hidden is None

    replay = await apply_import(
        source_bundle,
        report_path=report_path,
        checkpoint_path=checkpoint,
        embedding_func=embedding_func,
    )
    assert replay["ok"] is True and replay["resumed"] is True
