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

# Module-level state captured during LightRAG bootstrap so that other
# patches can reach the host's LightRAG instance without re-instantiating
# it. Populated by ``_patch_capture_rag`` when ``shim_native_routes=True``.
_twindb_state: dict[str, object] = {}


def _env_flag(name: str) -> bool:
    import os

    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def register(
    replace_ui: bool | None = None,
    mount_server: bool | None = None,
    shim_native_routes: bool | None = None,
    security_baseline: bool = True,
    classify: bool | None = None,
    classification_label_map_path: str | None = None,
    classification_ceiling: str | None = None,
    webui_dist: str | None = None,
    twin_api_prefix: str = "/twin/api",
    webui_stores: str = "memgraph",
    webui_categories_config: str | None = None,
) -> None:
    """Monkey-patch LightRAG's storage registries to add Memgraph backends.

    Safe to call multiple times (idempotent).
    Patches 3 dicts in lightrag.kg: STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS, and STORAGES.

    Args:
        replace_ui: If True, also wrap ``lightrag.api.lightrag_server.create_app``
            to swap the native ``/webui`` Mount with our WebUI fork (TS build).
            Requires ``webui_dist`` to point to a built ``dist/`` (containing
            ``index.html``) — or relies on an embedded ``webui_dist/`` shipped
            inside this package.
        mount_server: If True, mount our Twin sub-app on ``twin_api_prefix``
            (FastAPI: ``twindb_lightrag_memgraph.server.app:create_app``).
            Lifespans are chained so the sub-app initializes during parent startup.
        shim_native_routes: If True, prepend a Twin-shaped APIRouter at the
            HEAD of ``app.router.routes`` to shadow LightRAG's native
            ``GET /documents``, ``GET /health``, ``GET /pipeline_status`` (+
            add the missing ``GET /documents/{id}/chunks``, per-doc scan,
            REST-style delete, curated ``/openapi``). Requires the host
            LightRAG bootstrap to use the standard ``create_document_routes``
            factory so the ``rag`` instance can be captured.
        webui_stores: Which backend to use for the Twin overlay's
            tags / activity / notifications stores when ``mount_server=True``.

              - ``"memgraph"`` (default): Memgraph-backed stores
                (storage-scoped via
                env var ``WORKSPACE``). Fresh install boots empty; mutations
                persist. Requires ``MEMGRAPH_URI`` and runs the
                async store factories inside a lifespan wrapper around the
                LightRAG host's lifespan.
              - ``"seed"``: in-memory fixtures from ``webui_seed``. Useful
                only for demo/dev; user-generated surfaces are pre-populated
                and mutations are lost on restart.
        webui_categories_config: Optional path to a JSON file that
            *mirrors* the tag-category taxonomy on every boot. Doctrine
            "Config as Code" — the tenant admin owns the taxonomy in Git
            (or a Kubernetes ConfigMap), and Twin enforces it.

              - Schema: ``[{"id": "...", "label": "...", "color": "#RRGGBB"}, …]``.
              - Semantics: **replace**, not merge. Every reboot mirrors
                the file's content. Removing a category from the file
                makes it disappear on next boot (tags pointing at it
                become orphan; logged as warning, not auto-deleted).
              - Only honored when ``webui_stores="memgraph"``. With
                ``webui_stores="seed"`` the in-memory fixtures win.
              - When ``None`` (default), the internal seed
                ``webui_seed.TAG_CATEGORIES`` is bootstrapped on a
                fresh folder-backed store via
                :meth:`MemgraphTagStore.bootstrap_categories_if_empty`.
        security_baseline: If True (default), neutralize runtime supply-chain
            hazards before any LightRAG module gets a chance to mis-behave:
            blocks ``pipmaster`` install entrypoints and the
            ``check_and_install_dependencies`` call in ``lightrag_server``.
            Cf. audit Prisme G §1 — required for production deployment.
            Disable only in dev environments where you accept runtime pip calls.
        classify: Enable the pre-ingestion MIP classification hook. ``None``
            means auto-enable only when ``TWIN_MIP_LABEL_MAP`` is configured.
            ``True`` forces the hook on; ``False`` disables it.
        classification_label_map_path: Optional JSON label map path passed to
            the classifier. Defaults to ``TWIN_MIP_LABEL_MAP``.
        classification_ceiling: Optional max class accepted at ingest.
            Defaults to ``TWIN_MIP_MAX_CLASSIFICATION`` then ``"C2"``.
        webui_dist: Optional explicit path to the WebUI fork's ``dist/``.
            When ``None`` and ``replace_ui=True``, falls back to
            ``<package>/webui_dist`` (set up by ``scripts/build_webui.sh``).
        twin_api_prefix: URL prefix where the Twin sub-app is mounted
            (default ``/twin/api``).

    Storage-only behavior (default ``replace_ui=False, mount_server=False``) is
    identical to v1.0.x — instances already in production are unaffected.
    """
    global _registered
    if _registered:
        return

    # Runtime-overlay flags are env-drivable so deployments whose boot
    # path already calls a bare ``register()`` (the patch historically in
    # production) can activate the UI/server/shims with environment
    # variables only — no code change on the host side. Explicit booleans
    # passed by the caller always win; ``None`` defers to the env.
    if replace_ui is None:
        replace_ui = _env_flag("TWIN_REPLACE_UI")
    if mount_server is None:
        mount_server = _env_flag("TWIN_MOUNT_SERVER")
    if shim_native_routes is None:
        shim_native_routes = _env_flag("TWIN_SHIM_NATIVE_ROUTES")

    # 0. Security baseline FIRST — must run before any lightrag.api.* or
    #    lightrag.llm.* import that would trigger pipmaster auto-install.
    #    Idempotent via sentinels on the target modules.
    if security_baseline:
        _patch_security_baseline()

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

    # 6b. Optional MIP pre-ingestion classification gate.
    import os

    classify_enabled = (
        bool(os.environ.get("TWIN_MIP_LABEL_MAP")) if classify is None else classify
    )
    if classify_enabled:
        from ._classification_hook import install_lightrag_ingestion_hook

        install_lightrag_ingestion_hook(
            label_map_path=classification_label_map_path,
            ceiling=classification_ceiling,
        )

    # 7. Append our version to lightrag.__version__ so the WebUI displays it
    #    next to the LightRAG version string in the top-right corner.
    _patch_version_string()

    # 8. Optionally extend the FastAPI app: swap WebUI + mount Twin sub-app
    #    + shim native routes for the agent-readable contract.
    #    Opt-in via flags — default off keeps prod instances unaffected.
    if shim_native_routes:
        # Must wrap create_document_routes BEFORE create_app runs so that
        # when the host's create_app calls it, we capture the rag instance.
        _patch_capture_rag()

    if replace_ui or mount_server or shim_native_routes:
        _patch_lightrag_server_create_app(
            webui_dist=_resolve_webui_dist(webui_dist) if replace_ui else None,
            twin_api_prefix=twin_api_prefix if mount_server else None,
            shim_native_routes=shim_native_routes,
            webui_stores=webui_stores if mount_server else "seed",
            webui_categories_config=(
                webui_categories_config if mount_server else None
            ),
        )

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
        query, knowledge_graph_inst, entities_vdb, query_param,
        query_embedding=None,
    ):
        _lr_logger.info(
            f"Query nodes: {query} (top_k:{query_param.top_k}, "
            f"cosine:{entities_vdb.cosine_better_than_threshold})"
        )
        results = await entities_vdb.query(
            query, top_k=query_param.top_k, query_embedding=query_embedding,
        )
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


def _patch_security_baseline() -> None:
    """Defense-in-depth security baseline patches for production.

    Implements the supply-chain controls identified in audit Prisme G §1 and
    required by DORA art. 9 (ICT supply-chain integrity). When this baseline
    is active, the process cannot pull new dependencies at runtime — the wheel
    must already contain everything it needs.

    Currently blocks:
        1. ``pipmaster`` runtime install entrypoints (sync + async + all manager
           classes: ``PackageManager``, ``AsyncPackageManager``,
           ``UvPackageManager``, ``CondaPackageManager``).
        2. ``lightrag.api.lightrag_server.check_and_install_dependencies()``
           which would auto-install uvicorn/tiktoken/fastapi at boot.

    Idempotent. Sentinels are set on each target module so repeated
    ``register()`` calls don't stack patches.
    """
    _disable_pipmaster_runtime_install()
    _disable_lightrag_dependency_autoinstall()


_RUNTIME_INSTALL_REFUSED_MSG = (
    "Runtime pip install blocked by TwinRAG security baseline (pipmaster "
    "neutralized). All dependencies must be pinned in pyproject.toml and "
    "resolved at build time. Attempted: {package!r}. "
    "See audit Prisme G §1 (supply-chain integrity, DORA art. 9). "
    "To disable in dev environments only: register(security_baseline=False)."
)


def _disable_pipmaster_runtime_install() -> None:
    """Replace every pipmaster install entrypoint with a hard refusal.

    Coverage: module-level convenience helpers (sync + async) + every
    ``install*`` / ``ensure*`` method on every manager class.
    """
    try:
        import pipmaster as pm
    except ImportError:
        return  # pipmaster not installed — nothing to do

    if getattr(pm, "_twindb_install_blocked", False):
        return

    def _refuse(*args, **kwargs):
        pkg = kwargs.get("package", kwargs.get("package_name", "<unknown>"))
        if pkg == "<unknown>":
            for a in args:
                if isinstance(a, str):
                    pkg = a
                    break
        raise RuntimeError(_RUNTIME_INSTALL_REFUSED_MSG.format(package=pkg))

    async def _refuse_async(*args, **kwargs):
        return _refuse(*args, **kwargs)

    # Module-level convenience functions
    _sync_targets = (
        "install", "install_edit", "install_if_missing", "install_multiple",
        "install_multiple_if_not_installed", "install_or_update",
        "install_or_update_multiple", "install_requirements", "install_version",
        "ensure_packages", "ensure_requirements",
    )
    for name in _sync_targets:
        if hasattr(pm, name):
            setattr(pm, name, _refuse)

    _async_targets = (
        "async_install", "async_install_if_missing", "async_install_multiple",
        "async_ensure_packages", "async_ensure_requirements",
    )
    for name in _async_targets:
        if hasattr(pm, name):
            setattr(pm, name, _refuse_async)

    # Class-level methods on every manager
    for cls_name in (
        "PackageManager", "AsyncPackageManager",
        "UvPackageManager", "CondaPackageManager",
    ):
        cls = getattr(pm, cls_name, None)
        if cls is None:
            continue
        is_async_cls = cls_name == "AsyncPackageManager"
        for method_name in list(cls.__dict__):
            if not (method_name.startswith("install") or method_name.startswith("ensure")):
                continue
            replacement = _refuse_async if is_async_cls else _refuse
            setattr(cls, method_name, replacement)

    pm._twindb_install_blocked = True
    logger.info("twindb: pipmaster runtime install blocked (security baseline)")


def _disable_lightrag_dependency_autoinstall() -> None:
    """Neutralize ``lightrag.api.lightrag_server.check_and_install_dependencies``
    iff the module is already imported. Otherwise skip silently — the module
    parses ``sys.argv`` at import time and would raise ``SystemExit`` under
    pytest (which has its own argv). The hook in
    ``_patch_lightrag_server_create_app`` re-calls this once the module is
    safely loaded by an explicit ``replace_ui=`` / ``mount_server=`` path.

    Replaced with a no-op that logs a warning if ever called. Required
    packages (uvicorn, tiktoken, fastapi) are expected to be pinned in
    pyproject.toml and installed at build time.

    Pipmaster itself is already neutralized by
    ``_disable_pipmaster_runtime_install``, so even if this no-op never
    fires, an actual install attempt would still ``RuntimeError`` — the
    no-op only converts an attempted install into a logged warning instead
    of a boot crash.
    """
    import sys
    srv = sys.modules.get("lightrag.api.lightrag_server")
    if srv is None:
        return  # not yet imported — will be patched lazily via the create_app hook

    if getattr(srv, "_twindb_autoinstall_blocked", False):
        return

    def _noop():
        logger.warning(
            "twindb: lightrag.api.lightrag_server.check_and_install_dependencies "
            "was called but is a no-op under TwinRAG security baseline. "
            "Verify uvicorn/tiktoken/fastapi are pinned in pyproject.toml."
        )

    if hasattr(srv, "check_and_install_dependencies"):
        srv.check_and_install_dependencies = _noop
        srv._twindb_autoinstall_blocked = True
        logger.info("twindb: lightrag check_and_install_dependencies neutralized")


def _resolve_webui_dist(explicit: str | None) -> str:
    """Resolve the WebUI fork ``dist/`` path with this priority:
       1. explicit argument (raise if missing index.html);
       2. embedded ``<package>/webui_dist`` (set up by build script);
       3. dev fallback: ``lightrag_webui_twin/dist`` sibling of the repo root.

    Raises ``FileNotFoundError`` if none works.
    """
    from pathlib import Path
    import twindb_lightrag_memgraph as _pkg

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser().resolve())

    pkg_dir = Path(_pkg.__file__).parent
    candidates.append((pkg_dir / "webui_dist").resolve())

    # Dev fallback: when running from a checkout (editable install)
    # walk up to find sibling lightrag_webui_twin/dist
    repo_root = pkg_dir.parent.parent  # src/twindb_lightrag_memgraph → src → repo
    candidates.append((repo_root / "lightrag_webui_twin" / "dist").resolve())

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "index.html").is_file():
            return str(candidate)

    raise FileNotFoundError(
        "register(replace_ui=True): no WebUI dist found. Tried:\n  - "
        + "\n  - ".join(str(c) for c in candidates)
        + "\nBuild it with: cd lightrag_webui_twin && bun install --frozen-lockfile && bun run build"
    )


def _patch_capture_rag() -> None:
    """Wrap ``create_document_routes(rag, ...)`` so we capture ``rag``.

    LightRAG instantiates the RAG inside ``create_app`` and immediately
    feeds it to ``create_document_routes`` (and the query/graph routes)
    as the first positional arg. We wrap that factory to grab a reference
    before delegating to the original.

    Idempotent — sentinel attribute on the target module prevents
    stacking wrappers across repeated ``register()`` calls in tests.
    """
    import lightrag.api.routers.document_routes as dr

    if getattr(dr, "_twindb_capture_rag_patched", False):
        return

    orig_factory = dr.create_document_routes

    def wrapped_factory(rag, *args, **kwargs):
        _twindb_state["rag"] = rag
        logger.info("twindb: captured LightRAG instance for shim routes")
        return orig_factory(rag, *args, **kwargs)

    wrapped_factory.__wrapped__ = orig_factory
    wrapped_factory.__name__ = "wrapped_create_document_routes"
    dr.create_document_routes = wrapped_factory

    # The lightrag_server module imports create_document_routes by name
    # at module-load time, so we must also rebind there if the module is
    # already imported.
    import sys

    if "lightrag.api.lightrag_server" in sys.modules:
        srv_mod = sys.modules["lightrag.api.lightrag_server"]
        if hasattr(srv_mod, "create_document_routes"):
            srv_mod.create_document_routes = wrapped_factory

    dr._twindb_capture_rag_patched = True


def _patch_lightrag_server_create_app(
    webui_dist: str | None = None,
    twin_api_prefix: str | None = None,
    shim_native_routes: bool = False,
    webui_stores: str = "memgraph",
    webui_categories_config: str | None = None,
) -> None:
    """Wrap ``lightrag.api.lightrag_server.create_app`` to optionally:
       - replace the native ``/webui`` Mount with our WebUI fork (``webui_dist``);
       - mount our Twin sub-app on ``twin_api_prefix``;
       - prepend native-route shims (``shim_native_routes``) for the
         agent-readable surface the React port expects.

    Idempotent: a sentinel attribute on the module prevents stacking wrappers
    when ``register()`` is called more than once (e.g. in test loops).

    All three features default to off. Calling with all None/False is a no-op
    (still installs the wrapper as a passthrough so subsequent re-registrations
    are coherent).

    Order: this must run AFTER ``_patch_version_string()`` because the source
    string of ``lightrag_server.core_version`` is bound at module import.
    """
    import lightrag.api.lightrag_server as srv

    if getattr(srv, "_twindb_create_app_patched", False):
        logger.debug("_patch_lightrag_server_create_app: already patched — skip")
        return

    # Now that lightrag_server is safely imported (explicit user opt-in via
    # replace_ui/mount_server), revisit the security baseline's autoinstall
    # neutralization which had to skip earlier if the module was not yet loaded.
    _disable_lightrag_dependency_autoinstall()

    orig_create_app = srv.create_app

    def wrapped_create_app(args):
        app = orig_create_app(args)
        if shim_native_routes:
            _inject_native_shims(app)
        if webui_dist is not None:
            _replace_webui_mount(app, webui_dist)
        if twin_api_prefix is not None:
            _mount_twin_subapp(
                app,
                twin_api_prefix,
                webui_stores=webui_stores,
                webui_categories_config=webui_categories_config,
                auth_args=args,
            )
        return app

    wrapped_create_app.__wrapped__ = orig_create_app
    wrapped_create_app.__name__ = "wrapped_create_app"
    srv.create_app = wrapped_create_app
    srv._twindb_create_app_patched = True
    logger.info(
        "twindb: lightrag.api.lightrag_server.create_app wrapped "
        "(replace_ui=%s, mount_server=%s, shim_native_routes=%s)",
        webui_dist is not None,
        twin_api_prefix is not None,
        shim_native_routes,
    )


def _inject_native_shims(app) -> None:
    """Prepend the Twin native shim routes at the HEAD of ``app.router.routes``.

    FastAPI matches routes in registration order — the first hit wins.
    To shadow LightRAG's natives (already registered by the time
    ``create_app`` returns) we insert at index 0.

    The shims call back into the host's ``LightRAG`` instance captured
    by ``_patch_capture_rag``. If capture failed (host bootstrap took a
    non-standard path), routes raise 500 with a clear error message —
    we never silently fall back to a different RAG.
    """
    from .server.auth import require_auth
    from .server.native_shims import build_health_shim, build_native_shims_router

    def _get_rag():
        rag = _twindb_state.get("rag")
        if rag is None:
            raise RuntimeError(
                "twindb shim: host LightRAG instance not captured. "
                "register(shim_native_routes=True) requires create_document_routes "
                "to be called by the host (the standard lightrag-server entrypoint)."
            )
        return rag

    # Shim routes other than /auth-status, /login, /logout, /health must
    # require auth — they expose document listing, deletion, and pipeline
    # state. Audit 2026-06-10 finding C1.
    shim_router = build_native_shims_router(_get_rag, auth_dependency=require_auth)
    health_router = build_health_shim(_get_rag)

    # Prepend each shim route to app.router.routes so they beat LightRAG's
    # natives in the first-match-wins game.
    insert_at = 0
    for r in list(shim_router.routes) + list(health_router.routes):
        app.router.routes.insert(insert_at, r)
        insert_at += 1

    logger.info(
        "twindb: prepended %d native shim route(s) at app.router HEAD",
        insert_at,
    )


def _build_runtime_config() -> dict[str, object]:
    """Build the TwinRuntimeConfig dict that gets substituted into index.html.

    Shape mirrors ``lightrag_webui_twin/src/types/auth.ts:TwinRuntimeConfig``.

    ``debugUser`` is the dev escape hatch that auto-authenticates the
    React port. It is omitted whenever a server-side auth backend is
    configured: IdP (``TWIN_IDP_JWKS_URL``), local JWT
    (``LIGHTRAG_JWT_SECRET`` / ``TOKEN_SECRET`` / ``AUTH_ACCOUNTS``), or
    static API key (``LIGHTRAG_API_KEY``). Only a fully open dev/demo
    instance gets ``debugUser``.

    Override per-deploy via env vars (read late so a single ``register()``
    call can re-render against a different identity without re-importing).
    """
    import os

    from ._folders import build_runtime_folder_config, load_folder_catalog
    from .server.idp_jwt import IdpConfig as _IdpConfig

    _auth_backend_active = bool(
        _IdpConfig.from_env() is not None
        or os.environ.get("LIGHTRAG_JWT_SECRET")
        or os.environ.get("TOKEN_SECRET")
        or os.environ.get("AUTH_ACCOUNTS")
        or os.environ.get("LIGHTRAG_API_KEY")
    )

    api_base = os.environ.get("TWIN_API_BASE_URL", "/twin/api")
    lightrag_base = os.environ.get("TWIN_LIGHTRAG_BASE_URL", "")
    idp_logout = os.environ.get(
        "TWIN_IDP_LOGOUT_URL",
        "https://idp.twin.local/realms/twin/protocol/openid-connect/logout",
    )
    folder_catalog = load_folder_catalog()
    runtime_folder_config = build_runtime_folder_config()
    debug_user = {
        "sso_subject": os.environ.get("TWIN_DEBUG_USER_EMAIL", "operator@twin.local"),
        "email": os.environ.get("TWIN_DEBUG_USER_EMAIL", "operator@twin.local"),
        # Neutral anonymous-operator label — must never look like a real
        # colleague (activity events carry this name in open-access mode).
        "name": os.environ.get("TWIN_DEBUG_USER_NAME", "operator@twin.local"),
        "palier": {
            "level": 3,
            "label": "Steward",
            "scopes": ["twin:read", "twin:write", "twin:approve"],
        },
        "folders": [folder.id for folder in folder_catalog.folders],
        "idp": "local-debug",
        "idp_realm": "twin-local",
        "sub": "local-debug-sub",
        "session_expires": "2099-12-31T23:59:00Z",
        "gateway_scopes": [
            "read:documents",
            "write:documents",
            "read:query",
            "read:activity",
            "admin:tags",
            "admin:folders",
        ],
    }
    config: dict[str, object] = {
        "apiBaseUrl": api_base,
        "lightragBaseUrl": lightrag_base,
        "idpLogoutUrl": idp_logout,
        **runtime_folder_config,
    }
    # debugUser bypasses the LoginScreen, so expose it only for fully
    # open dev/demo instances.
    if not _auth_backend_active:
        config["debugUser"] = debug_user
    return config


def _replace_webui_mount(app, webui_dist: str) -> None:
    """Substitute the ``.app`` of the route named ``"webui"`` in ``app.router.routes``.

    Strategy chosen (Prisme A §6 Option A): mutate the existing ``Mount`` rather
    than appending a second one, so the route order — and thus precedence —
    stays identical to the native ``create_app`` output. This makes the swap
    invisible to downstream middlewares, the ``/`` redirect, and the
    ``/docs`` / ``/auth-status`` / ``/login`` companions.

    The substituted Mount serves a :class:`_TemplatedStaticFiles` that
    rewrites ``__TWIN_CONFIG_JSON__`` inside ``index.html`` on the fly,
    so the SPA receives the runtime config from the server (cf. plan §3.4)
    without any build-time injection or extra route.

    Side mount: the Twin React port is built with ``base: '/'`` (so the
    same dist also serves a standalone demo at root). When mounted
    under ``/webui/`` the inline references like ``/assets/index-XYZ.js``
    and ``/favicon.svg`` would 404. We mount a sibling ``/assets`` +
    favicon static handler so absolute-root references resolve. This is
    invisible to LightRAG natives because ``/assets`` is not part of
    LightRAG's surface — the Twin sub-paths are additive.
    """
    import json
    from pathlib import Path

    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse
    from starlette.routing import Mount

    legacy_hash_guard = (
        "<script>"
        "(function(){"
        "if(window.location.hash==='#/login'){"
        "window.history.replaceState(null,'',window.location.pathname+window.location.search);"
        "}"
        "}());"
        "</script>"
    )

    class _TemplatedStaticFiles(StaticFiles):
        """StaticFiles subclass that substitutes ``__TWIN_CONFIG_JSON__`` in index.html.

        Intercepts the lookup of ``index.html`` (both via the empty-path
        directory-default and explicit ``GET /webui/index.html``) and
        returns an :class:`HTMLResponse` with the placeholder replaced by
        the runtime config JSON. All other paths fall through to
        :meth:`StaticFiles.get_response` unchanged.

        The template is read once and cached on the instance; the config
        is also computed once at instance creation time. To pick up a new
        config value, call ``register()`` again (the wrapper rebinds and
        re-runs ``_replace_webui_mount``).
        """

        def __init__(
            self,
            *args,
            runtime_config_json: str,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self._runtime_config_json = runtime_config_json
            self._template_cache: str | None = None
            self._template_path = Path(self.directory) / "index.html"

        async def get_response(self, path: str, scope):
            # Starlette normalizes the mount-relative path via os.path.normpath,
            # so GET /webui/ arrives as path == "." (NOT "" or "/"). Explicit
            # GET /webui/index.html arrives as path == "index.html". Both
            # resolve to the same file → both are substitution targets.
            if path in (".", "index.html"):
                if self._template_cache is None:
                    try:
                        self._template_cache = self._template_path.read_text(
                            encoding="utf-8"
                        )
                    except FileNotFoundError:
                        logger.error(
                            "twindb webui template not found at %s",
                            self._template_path,
                        )
                        return await super().get_response(path, scope)
                html = self._template_cache.replace(
                    "__TWIN_CONFIG_JSON__",
                    self._runtime_config_json,
                )
                if legacy_hash_guard not in html:
                    html = html.replace("</head>", f"{legacy_hash_guard}</head>", 1)
                return HTMLResponse(
                    html,
                    headers={
                        # The Vite bundle filenames are content-hashed. If a
                        # browser reuses an old index.html after a deploy, it
                        # asks for deleted /assets/*.js files and the SPA boots
                        # blank. Revalidate the HTML entrypoint every time;
                        # immutable caching remains safe for hashed assets.
                        "Cache-Control": "no-store, max-age=0",
                    },
                )
            return await super().get_response(path, scope)

    webui_route = None
    for route in app.router.routes:
        if isinstance(route, Mount) and getattr(route, "name", None) == "webui":
            webui_route = route
            break

    if webui_route is None:
        logger.warning(
            "_replace_webui_mount: no Mount(name='webui') found on app — "
            "LightRAG likely started without WebUI assets (check_frontend_build "
            "returned False). UI replacement skipped."
        )
        return

    runtime_config_json = json.dumps(_build_runtime_config())

    webui_route.app = _TemplatedStaticFiles(
        directory=webui_dist,
        html=True,
        check_dir=True,
        runtime_config_json=runtime_config_json,
    )

    # Side-mount /assets at root so the dist's absolute references resolve.
    dist_path = Path(webui_dist)
    assets_dir = dist_path / "assets"
    if assets_dir.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="twin-assets-root",
        )

    # favicon + small static files referenced from index.html at root
    for fname in ("favicon.svg", "favicon.ico", "favicon.png", "icons.svg"):
        fpath = dist_path / fname
        if fpath.is_file():
            captured = str(fpath)

            async def _serve(path=captured):
                return FileResponse(path)

            app.get(f"/{fname}", include_in_schema=False)(_serve)

    # Bundle may also reference /mockServiceWorker.js (used by VITE_FORCE_MSW
    # standalone build). Harmless to expose even in hardened mode — it only
    # activates if VITE_FORCE_MSW was true at build-time.
    msw_path = dist_path / "mockServiceWorker.js"
    if msw_path.is_file():
        captured_msw = str(msw_path)

        async def _serve_msw():
            return FileResponse(captured_msw, media_type="application/javascript")

        app.get("/mockServiceWorker.js", include_in_schema=False)(_serve_msw)

    logger.info(
        "twindb: WebUI mount at /webui swapped → %s (with __TWIN_CONFIG_JSON__ substitution)",
        webui_dist,
    )


def _mount_twin_subapp(
    app,
    prefix: str,
    webui_stores: str = "memgraph",
    webui_categories_config: str | None = None,
    auth_args=None,
) -> None:
    """Mount the Twin overlay as an ``APIRouter`` directly on the host app.

    Doctrine: one app, one LightRAG, one lifespan. Earlier revisions
    instantiated a full FastAPI sub-app with a chained lifespan that
    booted a second LightRAG — fine for the standalone factory in
    ``server/app.py`` (still usable for unit tests), doubled the
    resource footprint in production. The current implementation
    includes the existing ``webui_router`` directly so:

      - one LightRAG instance for the whole process (the host's),
      - the same ``/twin/api/*`` surface from ``webui_router`` serves
        every endpoint the React port expects (folders, notifications,
        tags + CRUD, thesaurus, activity, graph).

    Storage backend selection (``webui_stores``):

      - ``"memgraph"``: the WebUI store is built **inside a chained
        lifespan** because the Memgraph store factories are async. On
        startup, after LightRAG's own lifespan has run
        ``rag.initialize_storages()``, we instantiate Memgraph-backed
        backends scoped to ``$WORKSPACE`` and swap them into a
        :class:`WebuiStore`. Fresh installs boot empty for documents,
        tags, activity, notifications, and graph.
      - ``"seed"``: the WebUI store is built sync from
        :func:`WebuiStore.from_seed`, so fixtures (tags, activity,
        notifications) are visible immediately. No lifespan wrapping.
        This mode is demo/dev only.
    """
    import os

    from fastapi import Depends

    from .server.auth import configure_auth, require_auth
    from .server.idp_jwt import IdpConfig as _IdpConfig, configure_idp

    def _arg_value(*names: str):
        if auth_args is None:
            return None
        for name in names:
            value = getattr(auth_args, name, None)
            if value:
                return value
        return None

    _resolved_api_key = _arg_value("api_key", "lightrag_api_key") or os.environ.get(
        "LIGHTRAG_API_KEY"
    )
    _resolved_jwt_secret = (
        _arg_value("jwt_secret", "lightrag_jwt_secret")
        or os.environ.get("LIGHTRAG_JWT_SECRET")
        or os.environ.get("TOKEN_SECRET")
    )
    _idp_cfg = _IdpConfig.from_env()

    configure_auth(
        api_key=_resolved_api_key,
        jwt_secret=_resolved_jwt_secret,
        jwt_algorithm=_arg_value("jwt_algorithm", "lightrag_jwt_algorithm")
        or os.environ.get("LIGHTRAG_JWT_ALGORITHM", "HS256"),
        jwt_expiration_hours=int(
            _arg_value("jwt_expiration_hours", "lightrag_jwt_expiration_hours")
            or os.environ.get("LIGHTRAG_JWT_EXPIRATION_HOURS")
            or os.environ.get("TOKEN_EXPIRE_HOURS", "4")
        ),
        jwt_username=_arg_value("jwt_username", "lightrag_jwt_username")
        or os.environ.get("LIGHTRAG_JWT_USERNAME", "admin"),
        jwt_password=_arg_value("jwt_password", "lightrag_jwt_password")
        or os.environ.get("LIGHTRAG_JWT_PASSWORD", "changeme"),
        auth_accounts=os.environ.get("AUTH_ACCOUNTS"),
    )

    # Activate the IdP JWT middleware if TWIN_IDP_JWKS_URL is set in
    # the env. Idempotent: dormant when no URL is configured.
    configure_idp(_idp_cfg)

    # Mock-kill safeguard: if the operator activates an IdP (a strong
    # signal that this is a real deployment, not a standalone demo),
    # warn loudly when ``webui_stores`` is still the demo "seed"
    # backend. The visible Twin overlay (tags / activity /
    # notifications / documents) would otherwise be in-memory fixtures
    # that look like real production data until the first restart
    # erases them.
    if _idp_cfg is not None and webui_stores == "seed":
        logger.warning(
            "twindb: DEMO STORES IN PROD — webui_stores='seed' with "
            "active IdP (%s). Tags, activity, notifications, documents, "
            "and graph entities are in-memory fixtures and WILL NOT "
            "survive a restart. Pass webui_stores='memgraph' on the "
            "deployment runbook before going live.",
            _idp_cfg.idp_name,
        )

    try:
        from .server.webui_router import (
            WebuiStore,
            router as webui_router,
            set_store,
        )
    except ImportError as exc:
        raise ImportError(
            "mount_server=True requires the 'server' extra: "
            "pip install 'twindb-lightrag-memgraph[server]'"
        ) from exc

    app.include_router(
        webui_router,
        prefix=prefix,
        dependencies=[Depends(require_auth)],
    )

    # Twin overlay query route — wraps LightRAG `aquery` and returns
    # structured `{response, sources}` so the React port can render
    # clickable citations. Mounted under the same prefix so the
    # frontend just calls `${TWIN}/query` instead of LightRAG's
    # native `/query`.
    try:
        from .server.twin_query_routes import build_twin_query_router

        def _get_rag_for_twin_query():
            rag = _twindb_state.get("rag")
            if rag is None:
                raise RuntimeError(
                    "twindb twin_query: host LightRAG instance not captured. "
                    "register(mount_server=True) requires shim_native_routes=True "
                    "so the rag instance is available."
                )
            return rag

        twin_query_router = build_twin_query_router(_get_rag_for_twin_query)
        app.include_router(
            twin_query_router,
            prefix=prefix,
            dependencies=[Depends(require_auth)],
        )
    except ImportError:
        # twin_query_routes is part of the server extra; if the
        # extra wasn't installed we silently skip — the legacy
        # LightRAG native /query still works for the React port.
        pass

    if webui_stores == "seed":
        # Sync setup; the seed includes pre-populated tags / activity /
        # notifications visible from the very first request.
        set_store(WebuiStore.from_seed())
        logger.info(
            "twindb: Twin overlay router included at %s "
            "(in-memory seed; %d routes)",
            prefix,
            len(webui_router.routes),
        )
        return

    if webui_stores != "memgraph":
        raise ValueError(
            f"webui_stores={webui_stores!r} is not supported. "
            "Use 'seed' or 'memgraph'."
        )

    # Memgraph branch — async store factories require a lifespan hook.
    from contextlib import asynccontextmanager

    parent_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def chained_lifespan(parent_app):
        async with parent_lifespan(parent_app):
            # LightRAG has finished initialize_storages() — Memgraph
            # connection pool is up, indexes are created. Safe to talk
            # to the WebUI store Memgraph backends.
            #
            # We **bypass** the ``make_memgraph_*_store()`` factories
            # because they call ``bootstrap_if_empty()``, which seeds
            # the folder-backed store with the demo fixtures on first init.
            # That makes a "fresh" folder look pre-populated (not
            # what an operator on a clean install expects). We
            # instantiate the classes directly + ``initialize()`` only.
            try:
                from .server.folder import load_folder_catalog
                from .server.webui_activitystore import MemgraphActivityStore
                from .server.webui_notificationstore import (
                    MemgraphNotificationStore,
                )
                from .server.webui_tagstore import MemgraphTagStore

                catalog = load_folder_catalog()
                for folder in catalog.folders:
                    tag_store = MemgraphTagStore(workspace=folder.id)
                    await tag_store.initialize()
                    # Categories — governance taxonomy, NOT user-generated.
                    # Two modes:
                    #   1. webui_categories_config set → mirror an external
                    #      JSON file on every boot (Config-as-Code doctrine,
                    #      Option 3). The file is source of truth, edits
                    #      propagate to Memgraph at next reboot.
                    #   2. No config path → bootstrap once from the internal
                    #      seed (Oracle / Infra / Network / Payment /
                    #      Lifecycle / Governance). Useful for demo + early
                    #      dev when the admin hasn't shipped a config yet.
                    if webui_categories_config:
                        n = await tag_store.replace_categories_from_config(
                            webui_categories_config
                        )
                        logger.info(
                            "twindb: categories sourced from %s (%d entries, space=%s)",
                            webui_categories_config,
                            n,
                            folder.id,
                        )
                    else:
                        await tag_store.bootstrap_categories_if_empty()
                    activity_store = MemgraphActivityStore(workspace=folder.id)
                    await activity_store.initialize()
                    notif_store = MemgraphNotificationStore(workspace=folder.id)
                    await notif_store.initialize()

                    store = WebuiStore.for_folder(folder.id, mode="memgraph")
                    store._tag_backend = tag_store
                    store._activity_backend = activity_store
                    store._notification_backend = notif_store
                    set_store(store, folder=folder.id)
                logger.info(
                    "twindb: Twin overlay stores switched to Memgraph "
                    "(folders=%s) — fresh folders boot empty.",
                    ",".join(folder.id for folder in catalog.folders),
                )
            except Exception:
                logger.exception(
                    "twindb: FAILED to switch stores to Memgraph; "
                    "keeping in-memory seed.",
                )
                raise
            yield

    app.router.lifespan_context = chained_lifespan
    logger.info(
        "twindb: Twin overlay router included at %s "
        "(memgraph stores pending lifespan startup; %d routes)",
        prefix,
        len(webui_router.routes),
    )


def _patch_version_string():
    """Append our package version to ``lightrag.__version__`` so the WebUI
    displays it next to the LightRAG version in the top-right corner.

    The LightRAG WebUI reads ``core_version`` from auth/health endpoints,
    which is bound at import time as ``from lightrag import __version__``.
    Patching the source attribute *before* ``lightrag_server`` imports it
    propagates the change to the entire server (and thus the WebUI).

    Idempotent: if already patched, do nothing. The marker prevents
    appending twice on repeated ``register()`` calls in test or dev loops.
    """
    import lightrag

    marker = f"+memgraph-{__version__}"
    current = getattr(lightrag, "__version__", "")
    if marker in current:
        return  # already patched

    lightrag.__version__ = f"{current}{marker}"
    logger.info(
        "Patched lightrag.__version__ → %s (visible in WebUI top-right)",
        lightrag.__version__,
    )
