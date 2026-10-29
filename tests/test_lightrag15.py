"""Real storage regressions for the LightRAG 1.5 control-plane contracts."""

from uuid import uuid4

import lightrag.base as base
import pytest
from lightrag.base import DocStatus

from lightrag_memgraph import _pool
from lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from lightrag_memgraph.kv_impl import MemgraphKVStorage


def row(**changes):
    return {
        "status": "pending",
        "created_at": "2026-01-01",
        "updated_at": "2026-01-01",
        "content_summary": "test",
        "content_length": 4,
        "file_path": "report.pdf",
        "chunks_list": ["chunk-a"],
        "metadata": {},
        **changes,
    }


@pytest.fixture
async def store(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "lr15_" + uuid4().hex)
    instance = MemgraphDocStatusStorage("status", {}, None)
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.drop()
        async with _pool.get_session() as session:
            for prop in (
                "id",
                "status",
                "file_path",
                "track_id",
                "updated_at",
                "created_at",
                "content_hash",
            ):
                await (
                    await session.run(f"DROP INDEX ON :`{instance._label()}`({prop})")
                ).consume()


@pytest.mark.integration
async def test_content_dedup_excludes_self_and_pointers_and_orders_candidates(store):
    await store.upsert(
        {
            "self": row(content_hash="hash"),
            "pointer": row(
                content_hash="hash",
                created_at="2025",
                metadata={"is_duplicate": True, "original_doc_id": "self"},
            ),
            "later": row(content_hash="hash", created_at="2028"),
            "earlier": row(content_hash="hash", created_at="2027"),
        }
    )
    doc_id, props = await store.get_doc_by_content_hash("hash", exclude_doc_id="self")
    assert doc_id == "earlier"
    assert props["chunks_list"] == ["chunk-a"]
    assert await store.get_doc_by_content_hash("absent") is None
    assert (await store.get_doc_by_file_basename("report.pdf"))[0] == "self"


@pytest.mark.integration
async def test_status_filters_and_immutable_creation_on_upsert(store):
    await store.upsert(
        {"a": row(), "b": row(status="processed"), "c": row(status="failed")}
    )
    await store.upsert({"a": row(created_at="2099", status="processing")})
    assert (await store.get_by_id("a"))["created_at"] == "2026-01-01"
    docs, total = await store.get_docs_paginated(
        status_filter=DocStatus.FAILED,
        status_filters=[DocStatus.PROCESSING, DocStatus.PROCESSED],
    )
    assert total == 2
    assert {doc_id for doc_id, _ in docs} == {"a", "b"}


modern = pytest.mark.skipif(
    not hasattr(base, "DocStatusPage"),
    reason="LightRAG release predates scheduling API",
)


@pytest.mark.integration
async def test_dedup_continues_past_a_full_window_of_duplicate_markers(store):
    await store.upsert(
        {
            f"pointer-{i:03d}": row(
                created_at="2025",
                content_hash="same",
                metadata={"is_duplicate": True, "original_doc_id": "self"},
            )
            for i in range(130)
        }
    )
    await store.upsert(
        {
            "self": row(content_hash="same"),
            "other": row(created_at="2027", content_hash="same"),
        }
    )
    assert (await store.get_doc_by_content_hash("same", exclude_doc_id="self"))[
        0
    ] == "other"
    assert (await store.get_doc_by_file_basename("report.pdf"))[0] == "self"


@pytest.mark.integration
@modern
async def test_keyset_page_survives_status_transition_and_hydrates_full_rows(store):
    await store.upsert(
        {
            key: row(metadata={"custom_chunk_patch": {"chunks": [key]}})
            for key in ("c", "a", "b")
        }
    )
    first = await store.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=1, strict=True
    )
    assert list(first.docs) == ["a"]
    assert first.docs["a"].has_custom_chunk_journal
    assert not hasattr(first.docs["a"], "chunks_list")
    await store.update_doc_status_fields(
        "a", {"status": DocStatus.PROCESSED, "metadata": {}}
    )
    second = await store.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, position=first.next_position, strict=True
    )
    assert list(second.docs) == ["b", "c"]
    last = await store.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, position=second.next_position, strict=True
    )
    assert last.next_position is base.CURSOR_END
    assert last.docs == {}
    docs = await store.get_docs_by_ids(["a", "missing"], strict=True)
    assert set(docs) == {"a"}
    assert not docs["a"].has_custom_chunk_journal
    full = await store.get_full_docs_by_ids(["a", "missing"], strict=True)
    assert full["a"].chunks_list == ["chunk-a"]
    assert full["a"].content_summary == "test"
    assert await store.count_docs_by_statuses([DocStatus.PENDING]) == 2
    # Stage statuses, content hash and nested metadata survive full hydration.
    await store.update_doc_status_fields(
        "b", {"status": DocStatus.PARSING, "content_hash": "hash-b"}
    )
    hydrated = (await store.get_full_docs_by_ids(["b"], strict=True))["b"]
    assert hydrated.status is DocStatus.PARSING
    assert hydrated.content_hash == "hash-b"
    assert hydrated.metadata == {"custom_chunk_patch": {"chunks": ["b"]}}


@pytest.mark.integration
@modern
async def test_strict_kv_read_does_not_report_corruption_as_absence(store):
    kv = MemgraphKVStorage("full_docs", {}, None)
    await kv.initialize()
    try:
        assert await kv.get_by_id_strict("absent") is None
        await kv.upsert({"doc": {"content": "stored"}})
        assert await kv.get_by_id_strict("doc") == {"content": "stored"}
        async with _pool.get_session() as session:
            await (
                await session.run(
                    f"MATCH (n:`{kv._label()}` {{id: 'doc'}}) SET n.data = 'null'"
                )
            ).consume()
        with pytest.raises(ValueError):
            await kv.get_by_id_strict("doc")
    finally:
        await kv.drop()
        async with _pool.get_session() as session:
            await (await session.run(f"DROP INDEX ON :`{kv._label()}`(id)")).consume()


@pytest.mark.integration
@modern
async def test_filtered_legacy_page_advances_and_strict_read_raises(store):
    await store.upsert({"bad": row(created_at="2025"), "good": row()})
    async with _pool.get_session() as session:
        await (
            await session.run(
                f"MATCH (n:`{store._label()}` {{id: 'bad'}}) REMOVE n.created_at"
            )
        ).consume()
    with pytest.raises((ValueError, TypeError)):
        await store.get_docs_by_statuses_page([DocStatus.PENDING], limit=1, strict=True)
    first = await store.get_docs_by_statuses_page([DocStatus.PENDING], limit=1)
    assert first.docs == {}
    assert first.next_position is not base.CURSOR_END
    second = await store.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=1, position=first.next_position, strict=True
    )
    assert list(second.docs) == ["good"]
    with pytest.raises(ValueError, match="immutable"):
        await store.update_doc_status_fields("good", {"created_at": "later"})
    from lightrag.exceptions import StorageRecordNotFoundError

    with pytest.raises(StorageRecordNotFoundError):
        await store.update_doc_status_fields("absent", {"status": "processed"})
    await store.update_doc_status_fields(
        "absent", {"status": "processed"}, missing_ok=True
    )


@pytest.mark.integration
@modern
async def test_source_resolution_conflicts_and_duplicate_only_absence(store):
    await store.upsert({"marker": row(metadata={"is_duplicate": True})})
    assert isinstance(
        await store.resolve_doc_source_strict("report.pdf"), base.SourceAbsent
    )
    await store.upsert({"a": row()})
    unique = await store.resolve_doc_source_strict("report.pdf")
    assert isinstance(unique, base.SourceUnique)
    assert unique.doc_id == "a"
    await store.upsert({"b": row()})
    conflict = await store.resolve_doc_source_strict("report.pdf")
    assert isinstance(conflict, base.SourceConflict)
    assert set(conflict.sample_doc_ids) == {"a", "b"}


@pytest.mark.integration
@modern
async def test_control_plane_propagates_transport_errors_and_invalid_json(
    store, monkeypatch
):
    await store.upsert({"doc": row()})
    async with _pool.get_session() as session:
        await (
            await session.run(
                f"MATCH (n:`{store._label()}` {{id: 'doc'}}) SET n.metadata = 'invalid'"
            )
        ).consume()
    with pytest.raises(ValueError):
        await store.get_by_id_strict("doc")
    with pytest.raises(ValueError):
        await store.get_full_docs_by_ids(["doc"], strict=True)

    kv = MemgraphKVStorage("full_docs", {}, None)
    await kv.initialize()

    def broken():
        raise OSError("transport unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(_pool, "get_read_session", broken)
        calls = [
            store.get_doc_by_content_hash("hash"),
            store.resolve_doc_source_strict("report.pdf"),
            store.get_docs_by_statuses_page([DocStatus.PENDING], limit=1, strict=True),
            store.get_docs_by_ids(["doc"], strict=True),
            store.get_full_docs_by_ids(["doc"], strict=True),
            store.count_docs_by_statuses([DocStatus.PENDING]),
            kv.get_by_id_strict("doc"),
        ]
        for call in calls:
            with pytest.raises(OSError, match="transport unavailable"):
                await call


@pytest.mark.integration
@modern
async def test_graph_first_ingestion_and_delete_without_embedding_provider(
    store, tmp_path
):
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
    from tests.test_e2e import SAMPLE_DOC, _mock_llm

    finalize_share_data()
    initialize_share_data()
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=store.workspace,
        kv_storage="MemgraphKVStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        vector_storage="NoopVectorDBStorage",
        embedding_func=None,
        llm_model_func=_mock_llm,
        enable_llm_cache=False,
    )
    await rag.initialize_storages()
    try:
        await rag.ainsert(SAMPLE_DOC, ids=["graph-first"], file_paths=["report.txt"])
        processed = await rag.doc_status.get_docs_by_statuses(
            [DocStatus.PROCESSED], strict=True
        )
        assert set(processed) == {"graph-first"}
        assert await rag.chunk_entity_relation_graph.has_node("Paris")
        result = await rag.adelete_by_doc_id("graph-first")
        assert result.status == "success"
        assert await rag.doc_status.get_by_id_strict("graph-first") is None
        assert not await rag.chunk_entity_relation_graph.has_node("Paris")
    finally:
        for storage in rag._index_storages():
            await storage.drop()
        await rag.finalize_storages()
