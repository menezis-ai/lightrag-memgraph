"""Storage constructors prefer LightRAG's explicit workspace over process state."""

import os

import numpy as np
from lightrag.kg.memgraph_impl import MemgraphStorage
from lightrag.utils import EmbeddingFunc

from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage
from twindb_lightrag_memgraph.patches.registry import (
    _explicit_workspace_memgraph_init,
)
from twindb_lightrag_memgraph.vector_impl import MemgraphVectorDBStorage


async def _embed(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 4), dtype=np.float32)


_EMBEDDING = EmbeddingFunc(embedding_dim=4, max_token_size=32, func=_embed)


def test_storage_constructors_use_explicit_workspace(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "process_global")
    config = {"workspace": "request_local"}

    kv = MemgraphKVStorage("kv", config, None)
    docs = MemgraphDocStatusStorage("doc_status", config, None)
    vectors = MemgraphVectorDBStorage("chunks", config, _EMBEDDING)
    native_init = MemgraphStorage.__init__
    graph_init = _explicit_workspace_memgraph_init(native_init)
    graph = MemgraphStorage.__new__(MemgraphStorage)
    graph_init(graph, "chunk_entity_relation", config, None, workspace=None)

    assert kv.workspace == "request_local"
    assert docs.workspace == "request_local"
    assert vectors.workspace == "request_local"
    assert graph.workspace == "request_local"


def test_storage_constructors_keep_legacy_environment_fallback(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "legacy_workspace")

    kv = MemgraphKVStorage("kv", {}, None)
    docs = MemgraphDocStatusStorage("doc_status", {}, None)
    vectors = MemgraphVectorDBStorage("chunks", {}, _EMBEDDING)
    native_init = MemgraphStorage.__init__
    graph_init = _explicit_workspace_memgraph_init(native_init)
    graph = MemgraphStorage.__new__(MemgraphStorage)
    graph_init(graph, "chunk_entity_relation", {}, None, workspace="")

    assert kv.workspace == "legacy_workspace"
    assert docs.workspace == "legacy_workspace"
    assert vectors.workspace == "legacy_workspace"
    assert graph.workspace == "legacy_workspace"


def test_graph_wrapper_delegates_native_initialization(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", "process_global")
    calls = []

    def native_init(self, namespace, global_config, embedding_func, workspace=None):
        calls.append((namespace, global_config, embedding_func, workspace))
        self.workspace = os.environ.get("MEMGRAPH_WORKSPACE") or workspace or "base"
        self.native_state = "preserved"

    graph_init = _explicit_workspace_memgraph_init(native_init)
    graph = object.__new__(_ReviewedGraphStorage)
    graph_init(
        graph,
        "chunk_entity_relation",
        {"workspace": "request_local"},
        None,
        workspace="request_local",
    )

    assert calls == [
        (
            "chunk_entity_relation",
            {"workspace": "request_local"},
            None,
            "request_local",
        )
    ]
    assert graph.native_state == "preserved"
    assert graph.workspace == "request_local"
    assert graph_init._twindb_explicit_workspace_patch is True


class _ReviewedGraphStorage:
    pass
