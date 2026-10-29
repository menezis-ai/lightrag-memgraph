"""Real-server ownership, migration refusal and cross-tenant deletion regressions."""

import asyncio
import uuid

import pytest
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.kg.shared_storage import initialize_share_data

from lightrag_memgraph import LegacyStorageError, StorageIdentityError, _pool
from lightrag_memgraph._identity import (
    REGISTRY_LABEL,
    StorageIdentity,
    _index_label,
    _rows,
)
from lightrag_memgraph.migration import inspect_storage_identities
from tests.test_storage_identity import label, stores

pytestmark = pytest.mark.integration


@pytest.fixture
async def scope():
    initialize_share_data(workers=1)
    stem = "identity_" + uuid.uuid4().hex[:12]
    allocated, extra = [], set()

    def make(kind, workspace=None, namespace="chunks"):
        store = stores(workspace or stem, namespace)[
            ["kv", "vector", "status", "graph"].index(kind)
        ]
        allocated.append((kind, store))
        return store

    yield stem, make, extra
    # Remove only exact labels/indexes allocated by this fixture, even after a
    # deliberately refused initialize(). Never drop arbitrary application data.
    labels = extra | {label(s) for _, s in allocated}
    names = {
        name
        for kind, s in allocated
        if kind == "vector"
        for name in (
            s._index_name(),
            s._text_index_name(),
            f"vec_{s.workspace}_{s.namespace}",
            f"text_{s.workspace}_{s.namespace}",
        )
    }
    async with _pool.get_session() as session:
        for row in await _rows(session, "SHOW VECTOR INDEX INFO"):
            if _index_label(row) in labels or row["index_name"] in names:
                await _rows(session, f"DROP VECTOR INDEX `{row['index_name']}`")
        for name in names:
            indexes = await _rows(session, "SHOW INDEX INFO")
            if any(f"(name: {name})" in row.get("index type", "") for row in indexes):
                await _rows(session, f"DROP TEXT INDEX `{name}`")
        for physical in labels:
            await _rows(session, f"MATCH (n:`{physical}`) DETACH DELETE n")
        for row in await _rows(session, "SHOW INDEX INFO"):
            physical = _index_label(row)
            if physical in labels and row.get("index type") in (
                "label",
                "label+property",
            ):
                prop = row.get("property")
                if isinstance(prop, (list, tuple)):
                    prop = prop[0] if len(prop) == 1 else None
                suffix = f"(`{prop}`)" if prop else ""
                await _rows(session, f"DROP INDEX ON :`{physical}`{suffix}")
        await _rows(
            session,
            f"MATCH (r:`{REGISTRY_LABEL}`) WHERE r.id IN $ids DELETE r",
            ids=[label(s) for _, s in allocated],
        )
    for kind, store in allocated:
        if kind == "graph" and store._driver is not None:
            await store.finalize()
    await _pool.close_driver()


async def put(kind, store, owner):
    if kind == "graph":
        await store.upsert_node(
            "same", {"entity_id": "same", "entity_type": "TEST", "description": owner}
        )
    elif kind == "status":
        await store.upsert(
            {
                "same": DocProcessingStatus(
                    content_summary=owner,
                    content_length=4,
                    file_path="/test",
                    status=DocStatus.PROCESSED,
                    created_at="2026-01-01",
                    updated_at="2026-01-01",
                )
            }
        )
    elif kind == "vector":
        store.meta_fields = {"content"}
        await store.upsert(
            {"same": {"embedding": [1.0, 0.0, 0.0, 0.0], "content": owner}}
        )
    else:
        await store.upsert({"same": {"content": owner}})


async def read(kind, store):
    value = await (
        store.get_node("same") if kind == "graph" else store.get_by_id("same")
    )
    return (
        value
        and value[
            {"graph": "description", "status": "content_summary"}.get(kind, "content")
        ]
    )


async def delete(kind, store):
    if kind == "graph":
        await store.remove_nodes(["same"])
    else:
        await store.delete(["same"])


@pytest.mark.parametrize("kind", ["kv", "vector", "status", "graph"])
@pytest.mark.parametrize("boundary", ["underscore", "namespace"])
async def test_same_ids_remain_isolated_through_reads_writes_deletes_and_drop(
    scope, kind, boundary
):
    stem, make, _ = scope
    a = make(kind, stem + "_a", "b_c" if boundary == "underscore" else "one")
    b = make(
        kind,
        stem + "_a_b" if boundary == "underscore" else stem + "_a",
        "c" if boundary == "underscore" else "two",
    )
    await a.initialize()
    await b.initialize()
    await put(kind, a, "alpha")
    assert await read(kind, b) is None
    await put(kind, b, "beta")
    assert await read(kind, a) == "alpha"
    await delete(kind, a)
    assert await read(kind, b) == "beta"
    await put(kind, a, "alpha_again")
    assert (await a.drop())["status"] == "success"
    assert await read(kind, a) is None
    assert await read(kind, b) == "beta"
    if kind == "vector":
        assert [
            hit["id"]
            for hit in await b.query("", 5, query_embedding=[1.0, 0.0, 0.0, 0.0])
        ] == ["same"]


@pytest.mark.parametrize("kind", ["kv", "vector", "status"])
@pytest.mark.parametrize("alias", ["legacy", "physical"])
async def test_graph_workspace_matching_storage_label_cannot_read_or_drop_storage(
    scope, kind, alias
):
    stem, make, _ = scope
    data = make(kind)
    await data.initialize()
    await put(kind, data, "preserve")
    identity = StorageIdentity(kind, stem, "chunks")
    graph = make(
        "graph", identity.legacy_label if alias == "legacy" else label(data), "graph"
    )
    await graph.initialize()
    assert await read("graph", graph) is None
    await put("graph", graph, "graph")
    await graph.drop()
    assert await read(kind, data) == "preserve"


async def test_registry_name_and_legacy_entity_type_aliases_are_valid_logical_workspaces(
    scope,
):
    stem, make, _ = scope
    first = make("graph", namespace="graph")
    await first.initialize()
    await put("graph", first, "first")
    # LightRAG 1.4 adds an entity-type label; emulate that on newer upstreams.
    alias = stem + "_type"
    async with _pool.get_session() as session:
        await _rows(session, f"MATCH (n:`{label(first)}`) SET n:`{alias}`")
    for workspace in (alias, REGISTRY_LABEL):
        second = make("graph", workspace, stem)
        await second.initialize()
        await put("graph", second, "second")
        await second.drop()
        assert await read("graph", first) == "first"


@pytest.mark.parametrize("kind", ["kv", "vector", "status", "graph"])
@pytest.mark.parametrize("empty_index", [False, True])
async def test_legacy_data_or_even_empty_indexes_refuse_startup_without_mutation(
    scope, kind, empty_index
):
    stem, make, extra = scope
    store = make(kind)
    identity = StorageIdentity(kind, stem, "chunks")
    old = identity.legacy_label
    extra.add(old)
    async with _pool.get_session() as session:
        await _rows(session, f"CREATE INDEX ON :`{old}`(id)")
        if not empty_index:
            await _rows(
                session,
                f'CREATE (n:`{old}` {{id: "same", entity_id: "same", data: "original", embedding: [1.0, 0.0, 0.0, 0.0]}})',
            )
        if kind == "vector":
            await _rows(
                session,
                f'CREATE VECTOR INDEX `{identity.legacy_indexes[0]}` ON :`{old}`(embedding) WITH CONFIG {{dimension: 4, capacity: 100, metric: "cos"}}',
            )
        before = await _rows(
            session, f"MATCH (n:`{old}`) RETURN properties(n) AS props"
        )
        schema_before = await _rows(session, "SHOW INDEX INFO")
    with pytest.raises(LegacyStorageError, match="ambiguous ownership"):
        await store.initialize()
    with pytest.raises(StorageIdentityError, match="not initialized"):
        await read(kind, store)
    async with _pool.get_session() as session:
        assert (
            await _rows(session, f"MATCH (n:`{old}`) RETURN properties(n) AS props")
            == before
        )
        assert await _rows(session, "SHOW INDEX INFO") == schema_before
        assert (
            await _rows(
                session,
                f"MATCH (r:`{REGISTRY_LABEL}` {{id: $id}}) RETURN r",
                id=label(store),
            )
            == []
        )
        if kind == "vector" and not empty_index:
            hits = await _rows(
                session,
                "CALL vector_search.search($name, 5, [1.0,0.0,0.0,0.0]) YIELD node RETURN node.id AS id",
                name=identity.legacy_indexes[0],
            )
            assert hits == [{"id": "same"}]
    # Explicit source re-ingestion in a fresh workspace is supported.
    fresh = make(kind, stem + "_fresh")
    await fresh.initialize()
    await put(kind, fresh, "reingested")
    assert await read(kind, fresh) == "reingested"


@pytest.mark.parametrize("occupied", ["node", "index"])
async def test_unregistered_v2_target_cannot_be_adopted(scope, occupied):
    _, make, _ = scope
    store = make("kv")
    async with _pool.get_session() as session:
        await _rows(
            session,
            f'CREATE (n:`{label(store)}` {{id: "foreign"}})'
            if occupied == "node"
            else f"CREATE INDEX ON :`{label(store)}`(id)",
        )
    with pytest.raises(StorageIdentityError, match="Unclaimed data or indexes"):
        await store.initialize()


async def test_concurrent_startup_reopen_and_drop_preserve_one_owner(scope):
    _, make, _ = scope
    instances = [make("kv") for _ in range(24)]
    await asyncio.gather(*(s.initialize() for s in instances))
    await put("kv", instances[0], "shared_owner")
    assert await read("kv", instances[-1]) == "shared_owner"
    await instances[0].drop()
    await instances[-1].initialize()
    async with _pool.get_session() as session:
        owners = await _rows(
            session,
            f"MATCH (r:`{REGISTRY_LABEL}` {{id: $id}}) RETURN properties(r) AS props",
            id=label(instances[0]),
        )
    assert owners == [
        {
            "props": StorageIdentity(
                "kv", instances[0].workspace, instances[0].namespace
            ).record
        }
    ]


async def test_inventory_is_read_only_and_reports_unknown_registry_records(scope):
    stem, make, _ = scope
    store = make("kv")
    await store.initialize()
    await put("kv", store, "preserve")
    async with _pool.get_session() as session:
        before = await _rows(session, "MATCH (n) RETURN count(n) AS count")
        indexes = await _rows(session, "SHOW INDEX INFO")
    report = await inspect_storage_identities()
    assert report["read_only"] is True
    assert StorageIdentity("kv", stem, "chunks").record in report["registered"]
    async with _pool.get_session() as session:
        assert await _rows(session, "MATCH (n) RETURN count(n) AS count") == before
        assert await _rows(session, "SHOW INDEX INFO") == indexes
        # An old graph can occupy the registry label. Preserve and reject it.
        await _rows(
            session,
            f"CREATE (r:`{REGISTRY_LABEL}` {{id: $id, foreign: true}})",
            id=label(store) + "_foreign",
        )
    try:
        assert (await inspect_storage_identities())["invalid_registry_entries"] == 1
        other = make("kv", stem + "_other")
        with pytest.raises(StorageIdentityError, match="Unrecognized data"):
            await other.initialize()
    finally:
        async with _pool.get_session() as session:
            await _rows(
                session,
                f"MATCH (r:`{REGISTRY_LABEL}` {{id: $id}}) DELETE r",
                id=label(store) + "_foreign",
            )


async def test_empty_legacy_text_index_is_preserved_and_blocks_initialization(scope):
    from lightrag_memgraph.capabilities import inspect_memgraph

    stem, make, extra = scope
    store = make("vector")
    identity = StorageIdentity("vector", stem, "chunks")
    extra.add(identity.legacy_label)
    async with _pool.get_session() as session:
        if not (await inspect_memgraph(session)).text_search:
            pytest.skip("Engine has no text index support")
        await _rows(
            session,
            f"CREATE TEXT INDEX `{identity.legacy_indexes[1]}` ON :`{identity.legacy_label}`(content)",
        )
        before = await _rows(session, "SHOW INDEX INFO")
    with pytest.raises(LegacyStorageError, match="ambiguous ownership"):
        await store.initialize()
    async with _pool.get_session() as session:
        assert await _rows(session, "SHOW INDEX INFO") == before


@pytest.mark.parametrize("kind", ["kv", "vector", "status", "graph"])
async def test_finalize_and_pool_reset_require_reverification_before_data_access(
    scope, kind
):
    _, make, _ = scope
    store = make(kind)
    await store.initialize()
    await put(kind, store, "preserve")
    await store.finalize()
    with pytest.raises(StorageIdentityError, match="not initialized"):
        await read(kind, store)
    await store.initialize()
    assert await read(kind, store) == "preserve"
    await _pool.close_driver()
    with pytest.raises(StorageIdentityError, match="not initialized"):
        await read(kind, store)
    await store.initialize()
    assert await read(kind, store) == "preserve"
