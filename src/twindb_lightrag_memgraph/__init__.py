"""
twindb-lightrag-memgraph
========================
Extension package that registers 3 Memgraph storage backends
into LightRAG's registry WITHOUT modifying LightRAG source code.

Usage:
    from twindb_lightrag_memgraph import register
    register()  # Call ONCE before instantiating LightRAG

    rag = LightRAG(
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",          # Already built-in
        ...
    )
"""

import logging
from importlib.metadata import version as _pkg_version

from ._hooks import clear_post_index_hooks, register_post_index_hook

logger = logging.getLogger("twindb_lightrag_memgraph")

_NOT_INITIALIZED_MSG = (
    "Memgraph driver is not initialized. Call 'await initialize()' first."
)

try:
    __version__ = _pkg_version("twindb-lightrag-memgraph")
except Exception:
    __version__ = "dev"

_registered = False


def register() -> None:
    """Monkey-patch LightRAG's storage registries to add Memgraph backends.

    Safe to call multiple times (idempotent).
    Patches 3 dicts in lightrag.kg: STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS, and STORAGES.
    """
    global _registered
    if _registered:
        return

    import lightrag.kg as kg_registry

    # 1. STORAGE_IMPLEMENTATIONS - declare our classes as valid implementations
    _new_impls = {
        "KV_STORAGE": "MemgraphKVStorage",
        "VECTOR_STORAGE": "MemgraphVectorDBStorage",
        "DOC_STATUS_STORAGE": "MemgraphDocStatusStorage",
    }
    for storage_type, class_name in _new_impls.items():
        impls = kg_registry.STORAGE_IMPLEMENTATIONS[storage_type]["implementations"]
        if class_name not in impls:
            impls.append(class_name)

    # 2. STORAGE_ENV_REQUIREMENTS - env vars required for each backend
    kg_registry.STORAGE_ENV_REQUIREMENTS.update(
        {
            "MemgraphKVStorage": ["MEMGRAPH_URI"],
            "MemgraphVectorDBStorage": ["MEMGRAPH_URI"],
            "MemgraphDocStatusStorage": ["MEMGRAPH_URI"],
        }
    )

    # 3. STORAGES - absolute module paths (importlib ignores package= for these)
    kg_registry.STORAGES.update(
        {
            "MemgraphKVStorage": "twindb_lightrag_memgraph.kv_impl",
            "MemgraphVectorDBStorage": "twindb_lightrag_memgraph.vector_impl",
            "MemgraphDocStatusStorage": "twindb_lightrag_memgraph.docstatus_impl",
        }
    )

    # 4. Monkey-patch built-in MemgraphStorage to use our TLS config
    #    and avoid session(database=...) which breaks on Community/Coordinator
    _patch_builtin_memgraph_storage()

    # 5. Buffer merge_nodes_and_edges writes (130+ RTT → 2 UNWIND queries)
    _patch_merge_write_path()

    # 6. Post-indexation hook on LightRAG._insert_done
    _patch_insert_done()

    _registered = True
    msg = (
        f"twindb-lightrag-memgraph v{__version__} — "
        "PATCH APPLIED SUCCESSFULLY\n"
        "  Graph DB ........ Memgraph (MemgraphStorage, patched for TLS + multi-db)\n"
        "  Vector DB ....... Memgraph native vector_search (MemgraphVectorDBStorage)\n"
        "  KV Storage ...... Memgraph (MemgraphKVStorage)\n"
        "  DocStatus ....... Memgraph (MemgraphDocStatusStorage)"
    )
    print(msg)
    logger.info(msg)


def _patch_builtin_memgraph_storage():
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

    from lightrag.kg.memgraph_impl import MemgraphStorage
    from lightrag.kg.shared_storage import get_data_init_lock
    from neo4j import AsyncGraphDatabase

    from ._constants import validate_identifier
    from ._pool import _read_connection_config, _uses_routing_protocol

    _original_logger = None
    try:
        from lightrag.utils import logger as _original_logger
    except ImportError:
        pass

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
                        pass  # Community — skip
                    else:
                        try:
                            _use_result = await session.run(
                                f"USE DATABASE {self._database}"
                            )
                            await _use_result.consume()
                            if self._enterprise_supported is None:
                                self._enterprise_supported = True
                        except _ClientError as exc:
                            if (
                                "enterprise" in str(exc).lower()
                                or "license" in str(exc).lower()
                            ):
                                self._enterprise_supported = False
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
                    try:
                        workspace_label = self._get_workspace_label()
                        _idx_result = await session.run(
                            f"CREATE INDEX ON :{workspace_label}(entity_id)"
                        )
                        await _idx_result.consume()
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            pass  # Expected on repeated initialize(); index is already created
                        elif _original_logger:
                            _original_logger.warning(
                                "[MemgraphGraph:%s] Index creation failed: %s",
                                self.workspace,
                                e,
                            )

                    _ping = await session.run("RETURN 1")
                    await _ping.consume()
                    if _original_logger:
                        _original_logger.info(
                            f"[MemgraphGraph:{self.workspace}] GRAPH storage "
                            f"connected to Memgraph "
                            f"(db={database}, patched for TLS + multi-db)"
                        )
            except Exception as e:
                if _original_logger:
                    _original_logger.error(
                        f"[{self.workspace}] Failed to connect to Memgraph: {type(e).__name__}"
                    )
                raise

    MemgraphStorage.initialize = _patched_initialize

    # -- Batch overrides: single-UNWIND queries instead of N round-trips --

    async def _patched_get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        if not node_ids:
            return {}
        if self._driver is None:
            raise RuntimeError(_NOT_INITIALIZED_MSG)
        ws = self._get_workspace_label()
        query = (
            f"UNWIND $ids AS eid "
            f"MATCH (n:`{ws}` {{entity_id: eid}}) "
            f"RETURN eid, n"
        )
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
            raise RuntimeError(_NOT_INITIALIZED_MSG)
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
            raise RuntimeError(_NOT_INITIALIZED_MSG)
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
            raise RuntimeError(_NOT_INITIALIZED_MSG)
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
            raise RuntimeError(_NOT_INITIALIZED_MSG)
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
            raise RuntimeError(_NOT_INITIALIZED_MSG)
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

    MemgraphStorage.get_nodes_batch = _patched_get_nodes_batch
    MemgraphStorage.node_degrees_batch = _patched_node_degrees_batch
    MemgraphStorage.get_edges_batch = _patched_get_edges_batch
    MemgraphStorage.edge_degrees_batch = _patched_edge_degrees_batch
    MemgraphStorage.get_nodes_edges_batch = _patched_get_nodes_edges_batch
    MemgraphStorage.get_nodes_with_degrees_batch = _patched_get_nodes_with_degrees_batch
    MemgraphStorage.get_edges_with_degrees_batch = _patched_get_edges_with_degrees_batch

    # -- Monkey-patch operate.py hot paths to use fused queries --
    _patch_operate_hot_paths()


def _patch_operate_hot_paths():
    """Replace two operate.py functions to use fused single-query methods.

    Falls back to the original asyncio.gather() pattern when the graph
    storage backend does not expose fused methods (non-Memgraph).
    """
    import asyncio

    import lightrag.operate as operate
    from lightrag.utils import logger as _lr_logger

    _original_get_node_data = operate._get_node_data
    _original_find_edges = operate._find_most_related_edges_from_entities

    async def _fused_get_node_data(
        query, knowledge_graph_inst, entities_vdb, query_param
    ):
        _lr_logger.info(
            f"Query nodes: {query} (top_k:{query_param.top_k}, "
            f"cosine:{entities_vdb.cosine_better_than_threshold})"
        )
        results = await entities_vdb.query(query, top_k=query_param.top_k)
        if not len(results):
            return [], []

        node_ids = [r["entity_name"] for r in results]

        if hasattr(knowledge_graph_inst, "get_nodes_with_degrees_batch"):
            nodes_dict, degrees_dict = (
                await knowledge_graph_inst.get_nodes_with_degrees_batch(node_ids)
            )
        else:
            nodes_dict, degrees_dict = await asyncio.gather(
                knowledge_graph_inst.get_nodes_batch(node_ids),
                knowledge_graph_inst.node_degrees_batch(node_ids),
            )

        node_datas = [nodes_dict.get(nid) for nid in node_ids]
        node_degrees = [degrees_dict.get(nid, 0) for nid in node_ids]

        if not all(n is not None for n in node_datas):
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

        if hasattr(knowledge_graph_inst, "get_edges_with_degrees_batch"):
            edge_data_dict, edge_degrees_dict = (
                await knowledge_graph_inst.get_edges_with_degrees_batch(
                    edge_pairs_dicts
                )
            )
        else:
            edge_pairs_tuples = list(all_edges)
            edge_data_dict, edge_degrees_dict = await asyncio.gather(
                knowledge_graph_inst.get_edges_batch(edge_pairs_dicts),
                knowledge_graph_inst.edge_degrees_batch(edge_pairs_tuples),
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

    operate._get_node_data = _fused_get_node_data
    operate._find_most_related_edges_from_entities = _fused_find_edges


def _patch_merge_write_path():
    """Replace merge_nodes_and_edges with a buffered version.

    Instead of 130+ individual upsert_node/upsert_edge Bolt round-trips
    per document, the buffered version accumulates all upserts in memory
    and flushes them as 2-3 UNWIND queries at the end.

    Double-patch required: operate.merge_nodes_and_edges is imported by
    lightrag.lightrag via ``from ... import``, creating a local copy.
    """
    from lightrag import lightrag as _lr_mod
    from lightrag import operate
    from lightrag.kg.memgraph_impl import MemgraphStorage

    from ._buffered_graph import _BufferedGraphProxy

    _original_merge = operate.merge_nodes_and_edges

    async def _buffered_merge_nodes_and_edges(*args, **kwargs):
        # Extract knowledge_graph_inst from args or kwargs.
        # Signature evolved across lightrag versions:
        #   old: (entity_map, edge_map, knowledge_graph_inst, global_config)
        #   new: (chunk_results, knowledge_graph_inst, entity_vdb, ...)
        # We support both by checking kwargs first, then positional args.
        graph_inst = kwargs.get("knowledge_graph_inst")
        if graph_inst is None:
            # Positional: index 2 (old) or index 1 (new).
            # Detect by type: MemgraphStorage is always the graph instance.
            for i, arg in enumerate(args):
                if isinstance(arg, MemgraphStorage):
                    graph_inst = arg
                    break

        if not isinstance(graph_inst, MemgraphStorage):
            return await _original_merge(*args, **kwargs)

        proxy = _BufferedGraphProxy(graph_inst)
        # Replace the graph instance in args/kwargs
        if "knowledge_graph_inst" in kwargs:
            kwargs["knowledge_graph_inst"] = proxy
        else:
            args = list(args)
            for i, arg in enumerate(args):
                if arg is graph_inst:
                    args[i] = proxy
                    break
            args = tuple(args)
        await _original_merge(*args, **kwargs)
        await proxy.flush()

    _buffered_merge_nodes_and_edges.__name__ = "buffered_merge_nodes_and_edges"

    # Double-patch: operate module + lightrag.lightrag module
    operate.merge_nodes_and_edges = _buffered_merge_nodes_and_edges
    _lr_mod.merge_nodes_and_edges = _buffered_merge_nodes_and_edges
    logger.info("Patched merge_nodes_and_edges with buffered UNWIND writer")


def _patch_insert_done():
    """Wrap ``LightRAG._insert_done`` to fire post-indexation hooks.

    After the original method completes (all storage ``index_done_callback()``
    have run), every callback registered via :func:`register_post_index_hook`
    is invoked with the ``LightRAG`` instance.

    Single patch — ``_insert_done`` is called via ``self.``, so no
    double-patch is needed.
    """
    from lightrag.lightrag import LightRAG

    from ._hooks import _run_post_index_hooks

    _original = LightRAG._insert_done

    async def _hooked_insert_done(
        self, pipeline_status=None, pipeline_status_lock=None
    ):
        await _original(self, pipeline_status, pipeline_status_lock)
        await _run_post_index_hooks(self)

    _hooked_insert_done.__name__ = "hooked_insert_done"
    LightRAG._insert_done = _hooked_insert_done
    logger.info("Patched LightRAG._insert_done with post-indexation hooks")
