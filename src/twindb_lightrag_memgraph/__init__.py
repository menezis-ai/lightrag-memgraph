"""
twindb-lightrag-memgraph
========================
Memgraph storage backends (KV, Vector, DocStatus) for LightRAG.

Patches LightRAG's storage registry at runtime so a single Memgraph
database can host an entire LightRAG instance. LightRAG already ships
its own ``MemgraphStorage`` for the **graph** layer; this package fills
the three remaining slots (KV, Vector, DocStatus) without touching
LightRAG's source code.

Usage::

    from twindb_lightrag_memgraph import register
    register()  # MUST be called before instantiating LightRAG

    rag = LightRAG(
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",  # already built into LightRAG
        ...,
    )
"""

from __future__ import annotations

import logging
from importlib.metadata import version as _pkg_version

from ._hooks import clear_post_index_hooks, register_post_index_hook

logger = logging.getLogger("twindb_lightrag_memgraph")

try:
    __version__ = _pkg_version("twindb-lightrag-memgraph")
except Exception:
    __version__ = "dev"

_registered = False

__all__ = [
    "register",
    "register_post_index_hook",
    "clear_post_index_hooks",
    "__version__",
]


def register() -> None:
    """Monkey-patch LightRAG's storage registries to add Memgraph backends.

    Idempotent — safe to call multiple times. Patches three dicts in
    ``lightrag.kg``: ``STORAGE_IMPLEMENTATIONS``, ``STORAGE_ENV_REQUIREMENTS``,
    and ``STORAGES``. Must be called BEFORE any ``LightRAG(...)`` that names a
    Memgraph storage class — LightRAG resolves storage classes at construction
    time.

    Module paths registered in ``STORAGES`` are absolute
    (``twindb_lightrag_memgraph.kv_impl`` and siblings) because LightRAG's
    ``lazy_external_import`` resolves with ``package="lightrag"``, so a
    relative path would point inside the LightRAG package and fail.
    """
    global _registered
    if _registered:
        return

    import lightrag.kg as kg_registry

    kg_registry.STORAGE_IMPLEMENTATIONS["KV_STORAGE"]["implementations"].append(
        "MemgraphKVStorage"
    )
    kg_registry.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"][
        "implementations"
    ].append("MemgraphVectorDBStorage")
    kg_registry.STORAGE_IMPLEMENTATIONS["DOC_STATUS_STORAGE"][
        "implementations"
    ].append("MemgraphDocStatusStorage")

    kg_registry.STORAGE_ENV_REQUIREMENTS["MemgraphKVStorage"] = ["MEMGRAPH_URI"]
    kg_registry.STORAGE_ENV_REQUIREMENTS["MemgraphVectorDBStorage"] = [
        "MEMGRAPH_URI"
    ]
    kg_registry.STORAGE_ENV_REQUIREMENTS["MemgraphDocStatusStorage"] = [
        "MEMGRAPH_URI"
    ]

    kg_registry.STORAGES["MemgraphKVStorage"] = "twindb_lightrag_memgraph.kv_impl"
    kg_registry.STORAGES["MemgraphVectorDBStorage"] = (
        "twindb_lightrag_memgraph.vector_impl"
    )
    kg_registry.STORAGES["MemgraphDocStatusStorage"] = (
        "twindb_lightrag_memgraph.docstatus_impl"
    )

    try:
        import lightrag

        suffix = f"+memgraph-{__version__}"
        if suffix not in lightrag.__version__:
            lightrag.__version__ = f"{lightrag.__version__}{suffix}"
    except Exception:
        pass

    _registered = True
    logger.info(
        "Registered Memgraph storages (KV / Vector / DocStatus) with LightRAG"
    )
