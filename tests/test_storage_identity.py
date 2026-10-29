"""Storage identity must remain injective across kinds and underscore boundaries."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from lightrag.utils import EmbeddingFunc

from lightrag_memgraph import register
from lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from lightrag_memgraph.kv_impl import MemgraphKVStorage
from lightrag_memgraph.vector_impl import MemgraphVectorDBStorage

register()
from lightrag.kg.memgraph_impl import MemgraphStorage  # noqa: E402


def stores(workspace, namespace):
    return [
        MemgraphKVStorage(namespace, {}, None, workspace=workspace),
        MemgraphVectorDBStorage(
            namespace,
            {},
            EmbeddingFunc(embedding_dim=4, func=AsyncMock()),
            workspace=workspace,
        ),
        MemgraphDocStatusStorage(namespace, {}, None, workspace=workspace),
        MemgraphStorage(namespace, {}, None, workspace=workspace),
    ]


def label(store):
    return (
        store._get_workspace_label()
        if isinstance(store, MemgraphStorage)
        else store._label()
    )


def test_underscore_boundaries_and_namespaces_cannot_alias():
    first = stores("a", "b_c")
    second = stores("a_b", "c")
    for a, b in zip(first, second):
        assert label(a) != label(b)
    assert first[1]._index_name() != second[1]._index_name()
    assert first[1]._text_index_name() != second[1]._text_index_name()
    for a, b in zip(stores("a", "one"), stores("a", "two")):
        assert label(a) != label(b)


def test_all_kinds_and_reserved_looking_logical_names_are_distinct():
    values = (
        "a",
        "a_b",
        "b_c",
        "_",
        "__",
        "KV_a_text_chunks",
        "Vec_a_b",
        "DocStatus_a",
        "LRMG2_REGISTRY",
        "w1_a_n1_b",
    )
    identities = {}
    for workspace in values:
        for namespace in values:
            for store in stores(workspace, namespace):
                key = type(store).__name__, workspace, namespace
                assert label(store) not in identities, (
                    key,
                    identities.get(label(store)),
                )
                identities[label(store)] = key


def test_graph_cannot_use_a_storage_label_as_its_physical_workspace():
    kv = MemgraphKVStorage("text_chunks", {}, None, workspace="a")
    graph = MemgraphStorage("graph", {}, None, workspace=kv._label())
    assert graph._get_workspace_label() != kv._label()


@pytest.mark.parametrize(
    "entity_type", ["LRMG2_REGISTRY", "lrmg2_kv_w1_a_n1_b", " LRMG2_VECTOR_w1_a_n1_b "]
)
async def test_entity_type_cannot_inject_reserved_storage_labels(entity_type):
    graph = MemgraphStorage("graph", {}, None, workspace="types")
    graph._driver = MagicMock()
    with pytest.raises(ValueError, match="reserved storage prefix"):
        await graph.upsert_node(
            "ENTITY", {"entity_id": "ENTITY", "entity_type": entity_type}
        )
    graph._driver.session.assert_not_called()


@pytest.mark.parametrize("kind", ["kv", "vector", "status", "graph"])
async def test_cold_storage_cannot_access_data_or_drop(kind):
    from lightrag_memgraph import StorageIdentityError

    store = stores("cold", "chunks")[["kv", "vector", "status", "graph"].index(kind)]
    calls = (
        [store.get_node("same"), store.drop()]
        if kind == "graph"
        else [store.get_by_id("same"), store.delete(["same"]), store.drop()]
    )
    for call in calls:
        with pytest.raises(StorageIdentityError, match="not initialized"):
            await call


@pytest.mark.parametrize("kind", ["kv", "vector", "status", "graph"])
@pytest.mark.parametrize(
    "change", ["namespace", "workspace", "connection", "pool_reset"]
)
async def test_verified_identity_cannot_be_reused_after_configuration_changes(
    monkeypatch, kind, change
):
    from tests.conftest import initialized_mock
    from lightrag_memgraph import StorageIdentityError, _pool
    from lightrag_memgraph._identity import require_identity

    store = initialized_mock(
        stores("verified", "chunks")[["kv", "vector", "status", "graph"].index(kind)]
    )
    require_identity(store, kind)
    if change in ("namespace", "workspace"):
        setattr(store, change, "other")
    elif change == "connection":
        monkeypatch.setenv("MEMGRAPH_DATABASE", "other")
    else:
        monkeypatch.setattr(_pool, "_identity_epoch", _pool._identity_epoch + 1)
    with pytest.raises(StorageIdentityError, match="not initialized"):
        require_identity(store, kind)


async def test_pool_refuses_to_verify_new_database_through_cached_old_driver(
    monkeypatch,
):
    from lightrag_memgraph import _pool
    from lightrag_memgraph._constants import connection_fingerprint

    driver = MagicMock()
    monkeypatch.setattr(_pool, "_driver", driver)
    monkeypatch.setattr(_pool, "_driver_scope", connection_fingerprint())
    monkeypatch.setenv("MEMGRAPH_DATABASE", "other")
    with pytest.raises(RuntimeError, match="connection settings changed"):
        await _pool.get_driver()
    driver.session.assert_not_called()
