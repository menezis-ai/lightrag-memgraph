"""
lightrag-memgraph
========================
Extension package that registers 3 Memgraph storage backends
into LightRAG's registry WITHOUT modifying LightRAG source code.

Usage:
    from lightrag_memgraph import register
    register()  # Call ONCE before instantiating LightRAG

    rag = LightRAG(
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",          # Already built-in
        ...
    )
"""

import inspect
import logging
from functools import wraps
from importlib.metadata import version as _pkg_version
from threading import Lock

from .capabilities import MemgraphCapabilities as MemgraphCapabilities
from .capabilities import inspect_memgraph as inspect_memgraph

from ._hooks import clear_post_index_hooks as clear_post_index_hooks
from ._hooks import register_post_index_hook as register_post_index_hook
from ._identity import LegacyStorageError as LegacyStorageError
from ._identity import StorageIdentityError as StorageIdentityError

logger = logging.getLogger("lightrag_memgraph")

try:
    __version__ = _pkg_version("lightrag-memgraph")
except Exception:
    __version__ = "dev"

_registered = False
_registration_lock = Lock()

__all__ = [
    "register",
    "register_post_index_hook",
    "clear_post_index_hooks",
    "MemgraphCapabilities",
    "inspect_memgraph",
    "__version__",
    "LegacyStorageError",
    "StorageIdentityError",
]


def register() -> None:
    """Install process-wide Memgraph integration before creating LightRAG instances."""
    with _registration_lock:
        if not _registered:
            _register()


def _register() -> None:
    global _registered
    import lightrag
    import lightrag.kg as kg_registry

    upstream_version = getattr(lightrag, "__version__", None)
    new_impls = {
        "KV_STORAGE": ("MemgraphKVStorage", "lightrag_memgraph.kv_impl"),
        "VECTOR_STORAGE": ("MemgraphVectorDBStorage", "lightrag_memgraph.vector_impl"),
        "DOC_STATUS_STORAGE": (
            "MemgraphDocStatusStorage",
            "lightrag_memgraph.docstatus_impl",
        ),
    }
    try:
        if not isinstance(upstream_version, str):
            raise TypeError("LightRAG version must be a string")
        implementations = [
            (kg_registry.STORAGE_IMPLEMENTATIONS[key]["implementations"], name)
            for key, (name, _) in new_impls.items()
        ]
        if not all(isinstance(impls, list) for impls, _ in implementations):
            raise TypeError("storage implementations must be lists")
        if not all(
            isinstance(registry, dict)
            for registry in (kg_registry.STORAGE_ENV_REQUIREMENTS, kg_registry.STORAGES)
        ):
            raise TypeError("storage registries must be dictionaries")
        graph_type, graph_methods = _prepare_builtin_memgraph_storage()
        patches = (
            (graph_type, graph_methods),
            _prepare_operate_hot_paths(graph_type),
            _prepare_insert_done(),
        )
    except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Unsupported LightRAG registration contract ({upstream_version}): {exc}"
        ) from exc

    for impls, name in implementations:
        if name not in impls:
            impls.append(name)
    for name, module in new_impls.values():
        kg_registry.STORAGE_ENV_REQUIREMENTS[name] = ["MEMGRAPH_URI"]
        kg_registry.STORAGES[name] = module
    for target, methods in patches:
        for name, method in methods.items():
            setattr(target, name, method)

    # Upstream merge writes stay inside LightRAG's entity/relationship locks.
    suffix = f"+memgraph-{__version__}"
    if suffix not in upstream_version:
        lightrag.__version__ = upstream_version + suffix

    _registered = True
    msg = (
        f"lightrag-memgraph v{__version__} — "
        "PATCH APPLIED SUCCESSFULLY\n"
        "  Graph DB ........ Memgraph (MemgraphStorage, patched for TLS + multi-db)\n"
        "  Vector DB ....... Memgraph native vector_search (MemgraphVectorDBStorage)\n"
        "  KV Storage ...... Memgraph (MemgraphKVStorage)\n"
        "  DocStatus ....... Memgraph (MemgraphDocStatusStorage)"
    )
    print(msg)
    logger.info(msg)


def _validate_signature(name, method, replacement, *, allow_shorter=False):
    original = inspect.signature(method)
    expected = inspect.signature(replacement)
    # Remove annotations so upstream type hints can evolve independently.
    original_params = tuple(p.replace(annotation=p.empty) for p in original.parameters.values())
    expected_params = tuple(p.replace(annotation=p.empty) for p in expected.parameters.values())
    compatible = original_params == expected_params or (
        allow_shorter and original_params == expected_params[:-1]
    )
    if not compatible or inspect.iscoroutinefunction(method) != inspect.iscoroutinefunction(replacement):
        raise TypeError(f"{name}{original} does not match {expected}")
    return len(original_params) == len(expected_params)


def _prepare_builtin_memgraph_storage():
    """Replace MemgraphStorage.initialize to support MEMGRAPH_ENCRYPTED
    and wrap the driver so that database routing works correctly for both
    direct (``bolt://``) and routing (``neo4j+s://``) protocols.

    * ``neo4j://`` / ``neo4j+s://`` — ``database=`` is forwarded to
      ``session()`` so the driver can route to the correct cluster member.
    * ``bolt://`` / ``bolt+s://`` — ``database=`` is stripped and
      ``USE DATABASE`` is issued inside the session (Memgraph Community
      workaround for GQL 50N42).

    The wrapper covers *all* built-in methods (has_node, upsert_node, …)
    without having to monkey-patch each one individually.
    """
    from contextlib import asynccontextmanager

    from lightrag.base import BaseGraphStorage
    from lightrag.kg.memgraph_impl import MemgraphStorage
    from lightrag.kg.shared_storage import get_data_init_lock
    from neo4j import AsyncGraphDatabase

    from ._constants import resolve_workspace, validate_entity_type, validate_identifier
    from ._pool import _read_connection_config, _uses_routing_protocol
    from . import _pool
    from ._identity import (
        guard_graph_method, initialization_context, initialize_identity,
        invalidate_identity, mark_identity_ready, require_identity, storage_label,
    )

    inspect.signature(BaseGraphStorage.__init__).bind(
        None, namespace=None, workspace=None, global_config=None, embedding_func=None
    )
    if not callable(get_data_init_lock):
        raise TypeError("get_data_init_lock is not callable")
    inspect.signature(get_data_init_lock).bind()
    methods = {}

    _original_logger = None
    try:
        from lightrag.utils import logger as _original_logger
    except ImportError:
        pass

    def _patched_init(self, namespace, global_config, embedding_func, workspace=None):
        validate_identifier(namespace, "namespace")
        # Upstream constructors in supported LightRAG releases give the global
        # MEMGRAPH_WORKSPACE variable priority. All four stores must instead
        # honor the same per-instance workspace, without temporarily editing env.
        BaseGraphStorage.__init__(
            self,
            namespace=namespace,
            workspace=resolve_workspace(workspace, global_config),
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver = None
        invalidate_identity(self)

    methods["__init__"] = _patched_init

    def _physical_graph_label(self):
        return storage_label("graph", self.workspace, self.namespace)

    methods["_get_workspace_label"] = _physical_graph_label

    # Older LightRAG writers interpolate entity_type as a label. Keep their
    # immediate transaction/retry path and reject unsafe delimiters before I/O.
    if not getattr(MemgraphStorage.upsert_node, "_lightrag_memgraph_validates_entity_type", False):
        original_upsert_node = MemgraphStorage.upsert_node
        inspect.signature(original_upsert_node).bind(None, None, None)

        async def _safe_upsert_node(self, node_id, node_data):
            if "entity_type" in node_data:
                validate_entity_type(node_data["entity_type"])
            require_identity(self, "graph")
            return await original_upsert_node(self, node_id, node_data)

        _safe_upsert_node._lightrag_memgraph_validates_entity_type = True
        _safe_upsert_node._lightrag_memgraph_identity_guarded = True
        methods["upsert_node"] = _safe_upsert_node

    batch_upsert = getattr(MemgraphStorage, "upsert_nodes_batch", None)
    if batch_upsert is not None and not getattr(
        batch_upsert, "_lightrag_memgraph_validates_entity_type", False
    ):
        original_upsert_nodes_batch = batch_upsert
        inspect.signature(batch_upsert).bind(None, None)

        async def _safe_upsert_nodes_batch(self, nodes):
            for _, node_data in nodes:
                if "entity_type" in node_data:
                    validate_entity_type(node_data["entity_type"])
            require_identity(self, "graph")
            return await original_upsert_nodes_batch(self, nodes)

        _safe_upsert_nodes_batch._lightrag_memgraph_validates_entity_type = True
        _safe_upsert_nodes_batch._lightrag_memgraph_identity_guarded = True
        methods["upsert_nodes_batch"] = _safe_upsert_nodes_batch

    class _SafeDriverWrapper:
        """Thin proxy around an AsyncDriver that intercepts session().

        When *use_routing* is True (``neo4j://`` / ``neo4j+s://``), the
        ``database=`` parameter is forwarded natively so the driver can
        route queries to the correct cluster member.

        When *use_routing* is False (``bolt://`` / ``bolt+s://``), the
        ``database=`` kwarg is stripped and ``USE DATABASE`` is issued
        inside the session instead.  On Memgraph Community (no Enterprise
        license), ``USE DATABASE`` fails — we detect this once and skip
        it for all subsequent sessions.
        """

        def __init__(self, real_driver, database, use_routing):
            self._real = real_driver
            self._database = database
            self._use_routing = use_routing
            self._enterprise_supported: bool | None = None

        def session(self, **kwargs):
            kwargs.pop("database", None)
            if self._use_routing and self._database:
                kwargs["database"] = self._database
            return self._safe_session(**kwargs)

        @asynccontextmanager
        async def _safe_session(self, **kwargs):
            from neo4j.exceptions import ClientError as _ClientError

            async with self._real.session(**kwargs) as session:
                if (
                    not self._use_routing
                    and self._database
                    and self._database != "memgraph"
                ):
                    if self._enterprise_supported is False:
                        if self._database != "memgraph":
                            raise RuntimeError(
                                f"Database {self._database!r} requires Memgraph Enterprise"
                            )
                        pass  # Community — skip
                    else:
                        try:
                            validate_identifier(self._database, "database")
                            result = await session.run(f"USE DATABASE {self._database}")
                            await result.consume()
                            if self._enterprise_supported is None:
                                self._enterprise_supported = True
                        except _ClientError as exc:
                            if "enterprise feature" in str(exc).lower():
                                self._enterprise_supported = False
                                if self._database != "memgraph":
                                    raise RuntimeError(
                                        f"Database {self._database!r} requires Memgraph Enterprise"
                                    ) from exc
                                logger.info(
                                    "Memgraph Community detected (graph pool)"
                                    " — USE DATABASE not available"
                                )
                            else:
                                raise
                yield session

        async def close(self):
            await self._real.close()

        def __getattr__(self, name):
            return getattr(self._real, name)

    async def _patched_initialize(self):
        async with get_data_init_lock():
            invalidate_identity(self)
            context = initialization_context(self, "graph")
            previous_driver = self._driver
            self._driver = None
            if previous_driver is not None:
                await previous_driver.close()
            uri, database, driver_kwargs = _read_connection_config()
            database = database or "memgraph"
            validate_identifier(database, "database")

            raw_driver = AsyncGraphDatabase.driver(uri, **driver_kwargs)
            self._driver = _SafeDriverWrapper(
                raw_driver, database, _uses_routing_protocol()
            )
            self._DATABASE = database

            try:
                async with self._driver.session() as session:
                    async with _pool.acquire_write_slot():
                        await initialize_identity(self, "graph", session)
                    try:
                        workspace_label = self._get_workspace_label()
                        result = await session.run(
                            f"CREATE INDEX ON :`{workspace_label}`(entity_id)"
                        )
                        await result.consume()
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            pass
                        elif _original_logger:
                            _original_logger.warning(
                                "[MemgraphGraph:%s] Index creation failed: %s",
                                self.workspace,
                                e,
                            )

                    result = await session.run("RETURN 1")
                    await result.consume()
                    mark_identity_ready(self, "graph", context)
                    if _original_logger:
                        _original_logger.info(
                            f"[MemgraphGraph:{self.workspace}] GRAPH storage "
                            f"connected to Memgraph "
                            f"(db={database}, patched for TLS + multi-db)"
                        )
            except Exception as e:
                failed_driver = self._driver
                self._driver = None
                invalidate_identity(self)
                try:
                    await failed_driver.close()
                except Exception:
                    pass
                if _original_logger:
                    _original_logger.error(
                        f"[{self.workspace}] Failed to connect to Memgraph: {type(e).__name__}"
                    )
                raise

    methods["initialize"] = _patched_initialize

    # -- Batch overrides: single-UNWIND queries instead of N round-trips --

    async def _patched_get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        ws = self._get_workspace_label()
        query = f"UNWIND $ids AS eid MATCH (n:`{ws}` {{entity_id: eid}}) RETURN eid, n"
        result = {}
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            records = await session.run(query, ids=node_ids)
            async for record in records:
                node_dict = dict(record["n"])
                if "labels" in node_dict:
                    node_dict["labels"] = [
                        lbl for lbl in node_dict["labels"] if lbl != ws
                    ]
                result[record["eid"]] = node_dict
            await records.consume()
        return result

    async def _patched_node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        if not node_ids:
            return {}
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        ws = self._get_workspace_label()
        query = (
            f"UNWIND $ids AS eid "
            f"MATCH (n:`{ws}` {{entity_id: eid}}) "
            f"OPTIONAL MATCH (n)-[r]-() "
            f"RETURN eid, count(r) AS degree"
        )
        result = {}
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            records = await session.run(query, ids=node_ids)
            async for record in records:
                result[record["eid"]] = record["degree"]
            await records.consume()
        # Missing nodes get degree 0 (matches original node_degree behavior)
        for nid in node_ids:
            if nid not in result:
                result[nid] = 0
        return result

    async def _patched_get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        if not pairs:
            return {}
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        ws = self._get_workspace_label()
        query = (
            f"UNWIND $pairs AS pair "
            f"MATCH (s:`{ws}` {{entity_id: pair.src}})"
            f"-[r]-"
            f"(t:`{ws}` {{entity_id: pair.tgt}}) "
            f"WITH pair, collect(properties(r))[0] AS props "
            f"RETURN pair.src AS src, pair.tgt AS tgt, props"
        )
        _defaults = {
            "weight": 1.0,
            "source_id": None,
            "description": None,
            "keywords": None,
        }
        result = {}
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            records = await session.run(
                query, pairs=[{"src": p["src"], "tgt": p["tgt"]} for p in pairs]
            )
            async for record in records:
                edge_props = dict(record["props"]) if record["props"] else {}
                for key, default_value in _defaults.items():
                    if key not in edge_props:
                        edge_props[key] = default_value
                result[(record["src"], record["tgt"])] = edge_props
            await records.consume()
        return result

    async def _patched_edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        if not edge_pairs:
            return {}
        # Collect unique node IDs, batch-fetch degrees, then sum per pair
        unique_ids = list({nid for pair in edge_pairs for nid in pair})
        degrees = await self.node_degrees_batch(unique_ids)
        return {
            (src, tgt): degrees.get(src, 0) + degrees.get(tgt, 0)
            for src, tgt in edge_pairs
        }

    async def _patched_get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        if not node_ids:
            return {}
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        ws = self._get_workspace_label()
        query = (
            f"UNWIND $ids AS eid "
            f"MATCH (n:`{ws}` {{entity_id: eid}}) "
            f"OPTIONAL MATCH (n)-[r]-(connected:`{ws}`) "
            f"WHERE connected.entity_id IS NOT NULL "
            f"RETURN eid, "
            f"collect([n.entity_id, connected.entity_id]) AS edges"
        )
        result = {}
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            records = await session.run(query, ids=node_ids)
            async for record in records:
                raw_edges = record["edges"]
                edges = [
                    (pair[0], pair[1])
                    for pair in raw_edges
                    if pair[0] is not None and pair[1] is not None
                ]
                result[record["eid"]] = edges
            await records.consume()
        # Missing nodes get empty list
        for nid in node_ids:
            if nid not in result:
                result[nid] = []
        return result

    # -- Fused queries: merge two gather() calls into one round-trip --

    async def _patched_get_nodes_with_degrees_batch(
        self, node_ids: list[str]
    ) -> tuple[dict[str, dict], dict[str, int]]:
        """Fused get_nodes_batch + node_degrees_batch in a single query."""
        if not node_ids:
            return {}, {}
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        ws = self._get_workspace_label()
        query = (
            f"UNWIND $ids AS eid "
            f"MATCH (n:`{ws}` {{entity_id: eid}}) "
            f"OPTIONAL MATCH (n)-[r]-() "
            f"RETURN eid, n, count(r) AS degree"
        )
        nodes = {}
        degrees = {}
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            records = await session.run(query, ids=node_ids)
            async for record in records:
                node_dict = dict(record["n"])
                if "labels" in node_dict:
                    node_dict["labels"] = [
                        lbl for lbl in node_dict["labels"] if lbl != ws
                    ]
                nodes[record["eid"]] = node_dict
                degrees[record["eid"]] = record["degree"]
            await records.consume()
        for nid in node_ids:
            if nid not in degrees:
                degrees[nid] = 0
        return nodes, degrees

    async def _patched_get_edges_with_degrees_batch(
        self, pairs: list[dict[str, str]]
    ) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], int]]:
        """Fused get_edges_batch + edge_degrees_batch in a single session.

        Pipelines two queries (edge props + node degrees) in one session
        instead of two separate sessions via asyncio.gather().
        """
        if not pairs:
            return {}, {}
        if self._driver is None:
            raise RuntimeError(
                "Memgraph driver is not initialized. Call 'await initialize()' first."
            )
        ws = self._get_workspace_label()

        edge_query = (
            f"UNWIND $pairs AS pair "
            f"MATCH (s:`{ws}` {{entity_id: pair.src}})"
            f"-[r]-"
            f"(t:`{ws}` {{entity_id: pair.tgt}}) "
            f"WITH pair, collect(properties(r))[0] AS props "
            f"RETURN pair.src AS src, pair.tgt AS tgt, props"
        )
        # Collect unique node IDs for degree computation
        unique_ids = list({nid for p in pairs for nid in (p["src"], p["tgt"])})
        degree_query = (
            f"UNWIND $ids AS eid "
            f"MATCH (n:`{ws}` {{entity_id: eid}}) "
            f"OPTIONAL MATCH (n)-[r]-() "
            f"RETURN eid, count(r) AS degree"
        )

        _defaults = {
            "weight": 1.0,
            "source_id": None,
            "description": None,
            "keywords": None,
        }
        edge_data = {}
        node_degrees = {}

        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            # Pipeline both queries in the same session
            pair_params = [{"src": p["src"], "tgt": p["tgt"]} for p in pairs]
            edge_records = await session.run(edge_query, pairs=pair_params)
            async for record in edge_records:
                key = (record["src"], record["tgt"])
                edge_props = dict(record["props"]) if record["props"] else {}
                for k, default_value in _defaults.items():
                    if k not in edge_props:
                        edge_props[k] = default_value
                edge_data[key] = edge_props
            await edge_records.consume()

            deg_records = await session.run(degree_query, ids=unique_ids)
            async for record in deg_records:
                node_degrees[record["eid"]] = record["degree"]
            await deg_records.consume()

        # Sum src + tgt degrees per edge pair
        edge_degrees = {}
        for p in pairs:
            key = (p["src"], p["tgt"])
            edge_degrees[key] = node_degrees.get(p["src"], 0) + node_degrees.get(
                p["tgt"], 0
            )
        return edge_data, edge_degrees

    methods["get_nodes_batch"] = _patched_get_nodes_batch
    methods["node_degrees_batch"] = _patched_node_degrees_batch
    methods["get_edges_batch"] = _patched_get_edges_batch
    methods["edge_degrees_batch"] = _patched_edge_degrees_batch
    methods["get_nodes_edges_batch"] = _patched_get_nodes_edges_batch
    methods["get_nodes_with_degrees_batch"] = _patched_get_nodes_with_degrees_batch
    methods["get_edges_with_degrees_batch"] = _patched_get_edges_with_degrees_batch

    if not getattr(MemgraphStorage.finalize, "_lightrag_memgraph_identity_finalizer", False):
        original_finalize = MemgraphStorage.finalize
        inspect.signature(original_finalize).bind(None)

        @wraps(original_finalize)
        async def _identity_finalize(self):
            invalidate_identity(self)
            return await original_finalize(self)

        _identity_finalize._lightrag_memgraph_identity_finalizer = True
        methods["finalize"] = _identity_finalize

    for name, replacement in methods.items():
        original = getattr(MemgraphStorage, name, None)
        if original is None and name in {
            "get_nodes_with_degrees_batch", "get_edges_with_degrees_batch"
        }:
            continue
        _validate_signature(f"MemgraphStorage.{name}", original, replacement)

    for name in sorted(set(dir(MemgraphStorage)) | methods.keys()):
        if name.startswith("_") or name in {"initialize", "finalize", "index_done_callback"}:
            continue
        method = methods.get(name) or getattr(MemgraphStorage, name)
        if inspect.iscoroutinefunction(method) and not getattr(method, "_lightrag_memgraph_identity_guarded", False):
            methods[name] = guard_graph_method(method)
    return MemgraphStorage, methods


def _prepare_operate_hot_paths(graph_type):
    """Use fused queries for Memgraph and delegate other graphs to LightRAG."""
    import lightrag.operate as operate
    from lightrag.utils import logger as _lr_logger

    _original_get_node_data = inspect.unwrap(
        operate._get_node_data, stop=lambda method: not getattr(method, "_memgraph_fused", False)
    )
    _original_find_edges = inspect.unwrap(
        operate._find_most_related_edges_from_entities,
        stop=lambda method: not getattr(method, "_memgraph_fused", False),
    )

    async def _fused_get_node_data(
        query, knowledge_graph_inst, entities_vdb, query_param, query_embedding=None
    ):
        query_kwargs = {"query_embedding": query_embedding} if accepts_embedding else {}
        if not isinstance(knowledge_graph_inst, graph_type):
            return await _original_get_node_data(
                query, knowledge_graph_inst, entities_vdb, query_param, **query_kwargs
            )
        _lr_logger.info(
            f"Query nodes: {query} (top_k:{query_param.top_k}, "
            f"cosine:{entities_vdb.cosine_better_than_threshold})"
        )
        query_kwargs = {"top_k": query_param.top_k}
        if query_embedding is not None:
            query_kwargs["query_embedding"] = query_embedding
        results = await entities_vdb.query(query, **query_kwargs)
        if not len(results):
            return [], []

        node_ids = [r["entity_name"] for r in results]

        nodes_dict, degrees_dict = await knowledge_graph_inst.get_nodes_with_degrees_batch(
            node_ids
        )

        node_datas = [nodes_dict.get(nid) for nid in node_ids]
        node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

        if not all([n is not None for n in node_datas]):
            _lr_logger.warning("Some nodes are missing, maybe the storage is damaged")

        node_datas = [
            {
                **n,
                "entity_name": k["entity_name"],
                "rank": d,
                "created_at": k.get("created_at"),
            }
            for k, n, d in zip(results, node_datas, node_degrees)
            if n is not None
        ]

        use_relations = await operate._find_most_related_edges_from_entities(
            node_datas,
            query_param,
            knowledge_graph_inst,
        )

        _lr_logger.info(
            f"Local query: {len(node_datas)} entites, {len(use_relations)} relations"
        )
        return node_datas, use_relations

    async def _fused_find_edges(node_datas, query_param, knowledge_graph_inst):
        if not isinstance(knowledge_graph_inst, graph_type):
            return await _original_find_edges(node_datas, query_param, knowledge_graph_inst)
        node_names = [dp["entity_name"] for dp in node_datas]
        batch_edges_dict = await knowledge_graph_inst.get_nodes_edges_batch(node_names)

        all_edges = []
        seen = set()
        for node_name in node_names:
            this_edges = batch_edges_dict.get(node_name, [])
            for e in this_edges:
                sorted_edge = tuple(sorted(e))
                if sorted_edge not in seen:
                    seen.add(sorted_edge)
                    all_edges.append(sorted_edge)

        edge_pairs_dicts = [{"src": e[0], "tgt": e[1]} for e in all_edges]

        edge_data_dict, edge_degrees_dict = (
            await knowledge_graph_inst.get_edges_with_degrees_batch(edge_pairs_dicts)
        )

        all_edges_data = []
        for pair in all_edges:
            edge_props = edge_data_dict.get(pair)
            if edge_props is not None:
                if "weight" not in edge_props:
                    _lr_logger.warning(
                        f"Edge {pair} missing 'weight' attribute, using default value 1.0"
                    )
                    edge_props["weight"] = 1.0
                combined = {
                    "src_tgt": pair,
                    "rank": edge_degrees_dict.get(pair, 0),
                    **edge_props,
                }
                all_edges_data.append(combined)

        all_edges_data = sorted(
            all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        return all_edges_data

    accepts_embedding = _validate_signature(
        "operate._get_node_data", _original_get_node_data, _fused_get_node_data,
        allow_shorter=True,
    )
    _validate_signature(
        "operate._find_most_related_edges_from_entities", _original_find_edges, _fused_find_edges
    )
    node_query = wraps(_original_get_node_data)(_fused_get_node_data)
    edge_query = wraps(_original_find_edges)(_fused_find_edges)
    node_query._memgraph_fused = edge_query._memgraph_fused = True
    return operate, {
        "_get_node_data": node_query,
        "_find_most_related_edges_from_entities": edge_query,
    }


def _prepare_insert_done():
    """Build the process-wide post-index hook wrapper."""
    from lightrag.lightrag import LightRAG

    from ._hooks import _run_post_index_hooks

    original = LightRAG._insert_done
    if getattr(original, "_memgraph_post_index_hook", False):
        return LightRAG, {}

    async def hooked_insert_done(self, pipeline_status=None, pipeline_status_lock=None):
        await original(self, pipeline_status, pipeline_status_lock)
        await _run_post_index_hooks(self)

    _validate_signature("LightRAG._insert_done", original, hooked_insert_done)
    wrapper = wraps(original)(hooked_insert_done)
    wrapper._memgraph_post_index_hook = True
    return LightRAG, {"_insert_done": wrapper}
