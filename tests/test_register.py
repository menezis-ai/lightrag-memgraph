"""
Tests for the register() function.

These tests are OFFLINE - they only verify the monkey-patch logic
on LightRAG's in-memory registries, no Memgraph connection needed.
"""

import importlib

import lightrag.kg as kg_registry

import twindb_lightrag_memgraph


def _reset_registration():
    """Force re-registration by clearing the flag."""
    twindb_lightrag_memgraph._registered = False


class TestRegister:
    def test_register_idempotent(self):
        """Calling register() twice should not duplicate entries."""
        _reset_registration()
        twindb_lightrag_memgraph.register()
        twindb_lightrag_memgraph.register()

        count = kg_registry.STORAGE_IMPLEMENTATIONS["KV_STORAGE"][
            "implementations"
        ].count("MemgraphKVStorage")
        assert count == 1

    def test_kv_in_implementations(self):
        """MemgraphKVStorage must appear in KV_STORAGE implementations."""
        _reset_registration()
        twindb_lightrag_memgraph.register()

        impls = kg_registry.STORAGE_IMPLEMENTATIONS["KV_STORAGE"]["implementations"]
        assert "MemgraphKVStorage" in impls

    def test_vector_in_implementations(self):
        """MemgraphVectorDBStorage must appear in VECTOR_STORAGE implementations."""
        _reset_registration()
        twindb_lightrag_memgraph.register()

        impls = kg_registry.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
        assert "MemgraphVectorDBStorage" in impls

    def test_docstatus_in_implementations(self):
        """MemgraphDocStatusStorage must appear in DOC_STATUS_STORAGE implementations."""
        _reset_registration()
        twindb_lightrag_memgraph.register()

        impls = kg_registry.STORAGE_IMPLEMENTATIONS["DOC_STATUS_STORAGE"][
            "implementations"
        ]
        assert "MemgraphDocStatusStorage" in impls

    def test_env_requirements(self):
        """All 3 backends must declare MEMGRAPH_URI as required."""
        _reset_registration()
        twindb_lightrag_memgraph.register()

        for name in (
            "MemgraphKVStorage",
            "MemgraphVectorDBStorage",
            "MemgraphDocStatusStorage",
        ):
            assert name in kg_registry.STORAGE_ENV_REQUIREMENTS
            assert "MEMGRAPH_URI" in kg_registry.STORAGE_ENV_REQUIREMENTS[name]

    def test_storages_module_paths(self):
        """STORAGES dict must contain absolute paths for our 3 backends."""
        _reset_registration()
        twindb_lightrag_memgraph.register()

        expected = {
            "MemgraphKVStorage": "twindb_lightrag_memgraph.kv_impl",
            "MemgraphVectorDBStorage": "twindb_lightrag_memgraph.vector_impl",
            "MemgraphDocStatusStorage": "twindb_lightrag_memgraph.docstatus_impl",
        }
        for class_name, module_path in expected.items():
            assert kg_registry.STORAGES[class_name] == module_path

    def test_absolute_import_resolution(self):
        """Absolute module paths must resolve even when package='lightrag'."""
        paths = {
            "MemgraphKVStorage": "twindb_lightrag_memgraph.kv_impl",
            "MemgraphVectorDBStorage": "twindb_lightrag_memgraph.vector_impl",
            "MemgraphDocStatusStorage": "twindb_lightrag_memgraph.docstatus_impl",
        }
        for class_name, module_path in paths.items():
            # Simulate what lazy_external_import does with package="lightrag"
            mod = importlib.import_module(module_path, package="lightrag")
            cls = getattr(mod, class_name)
            assert cls is not None
            assert cls.__name__ == class_name

    def test_verify_storage_implementation_passes(self):
        """LightRAG's verify_storage_implementation must accept our backends."""
        _reset_registration()
        twindb_lightrag_memgraph.register()

        from lightrag.kg import verify_storage_implementation

        # These must NOT raise
        verify_storage_implementation("KV_STORAGE", "MemgraphKVStorage")
        verify_storage_implementation("VECTOR_STORAGE", "MemgraphVectorDBStorage")
        verify_storage_implementation("DOC_STATUS_STORAGE", "MemgraphDocStatusStorage")

    async def test_mount_twin_subapp_requires_configured_bearer(self, monkeypatch):
        """The production /twin/api mount must inherit the Twin auth dependency."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from twindb_lightrag_memgraph.server.auth import configure_auth

        monkeypatch.setenv("LIGHTRAG_API_KEY", "mount-secret")
        app = FastAPI()
        twindb_lightrag_memgraph._mount_twin_subapp(
            app,
            "/twin/api",
            webui_stores="seed",
        )

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                denied = await client.get("/twin/api/tags")
                assert denied.status_code == 401

                allowed = await client.get(
                    "/twin/api/tags",
                    headers={"Authorization": "Bearer mount-secret"},
                )
                assert allowed.status_code == 200
        finally:
            configure_auth(api_key=None, jwt_secret=None)

    def test_overlay_installs_procedure_seam_before_admin_activation(self, monkeypatch):
        """A cold-started disabled profile must still react to a later toggle."""
        from twindb_lightrag_memgraph.patches import registry

        calls = []
        app = object()
        monkeypatch.setattr(registry, "_patch_native_entity_taxonomy", lambda: None)
        monkeypatch.setattr(registry, "_patch_upload_duplicate_lookup", lambda: None)
        monkeypatch.setattr(
            registry,
            "_patch_pipeline_enqueue_conversion",
            lambda: calls.append("procedure-seam"),
        )
        monkeypatch.setattr(registry._conversion, "is_enabled", lambda: False)
        monkeypatch.setattr(registry._vision, "is_enabled", lambda: False)

        result = registry._wrapped_create_app_impl(
            object(),
            orig_create_app=lambda _args: app,
            webui_dist=None,
            twin_api_prefix=None,
            shim_native_routes=False,
            webui_stores="seed",
            webui_categories_config=None,
        )

        assert result is app
        assert calls == ["procedure-seam"]


class TestEnvDrivenFlags:
    """register() overlay flags resolve from env when not passed explicitly.

    Deployments whose boot path calls a bare register() (the patch
    historically in production) must be able to activate the UI/server/
    shims with environment variables only.
    """

    def test_env_flags_activate_overlay(self, monkeypatch):
        _reset_registration()
        monkeypatch.setenv("TWIN_REPLACE_UI", "true")
        monkeypatch.setenv("TWIN_MOUNT_SERVER", "1")
        monkeypatch.setenv("TWIN_SHIM_NATIVE_ROUTES", "yes")
        calls = {}

        def fake_patch(**kwargs):
            calls.update(kwargs)

        monkeypatch.setattr(
            twindb_lightrag_memgraph,
            "_patch_lightrag_server_create_app",
            fake_patch,
        )
        monkeypatch.setattr(
            twindb_lightrag_memgraph,
            "_resolve_webui_dist",
            lambda explicit: "/fake/dist",
        )
        monkeypatch.setattr(
            twindb_lightrag_memgraph, "_patch_capture_rag", lambda: None
        )
        twindb_lightrag_memgraph.register()

        assert calls["webui_dist"] == "/fake/dist"
        assert calls["twin_api_prefix"] == "/twin/api"
        assert calls["shim_native_routes"] is True

    def test_env_unset_keeps_storage_only_default(self, monkeypatch):
        _reset_registration()
        for var in ("TWIN_REPLACE_UI", "TWIN_MOUNT_SERVER", "TWIN_SHIM_NATIVE_ROUTES"):
            monkeypatch.delenv(var, raising=False)
        called = []
        monkeypatch.setattr(
            twindb_lightrag_memgraph,
            "_patch_lightrag_server_create_app",
            lambda **kw: called.append(kw),
        )
        twindb_lightrag_memgraph.register()
        assert called == []

    def test_explicit_false_overrides_env(self, monkeypatch):
        _reset_registration()
        monkeypatch.setenv("TWIN_REPLACE_UI", "true")
        called = []
        monkeypatch.setattr(
            twindb_lightrag_memgraph,
            "_patch_lightrag_server_create_app",
            lambda **kw: called.append(kw),
        )
        twindb_lightrag_memgraph.register(
            replace_ui=False, mount_server=False, shim_native_routes=False
        )
        assert called == []

    def test_missing_webui_dist_degrades_replace_ui_only(self, monkeypatch, caplog):
        """When replace_ui=True but the embedded dist is absent, register()
        must keep mount_server / shim_native_routes alive instead of raising
        and losing everything (BNP 2026-06-12 silent-failure path).
        """
        _reset_registration()
        for var in ("TWIN_REPLACE_UI", "TWIN_MOUNT_SERVER", "TWIN_SHIM_NATIVE_ROUTES"):
            monkeypatch.delenv(var, raising=False)
        calls = {}

        def fake_patch(**kwargs):
            calls.update(kwargs)

        def raise_missing(explicit):
            raise FileNotFoundError("no WebUI dist found.")

        monkeypatch.setattr(
            twindb_lightrag_memgraph,
            "_patch_lightrag_server_create_app",
            fake_patch,
        )
        monkeypatch.setattr(
            twindb_lightrag_memgraph, "_resolve_webui_dist", raise_missing
        )
        monkeypatch.setattr(
            twindb_lightrag_memgraph, "_patch_capture_rag", lambda: None
        )

        with caplog.at_level("ERROR", logger="twindb_lightrag_memgraph"):
            twindb_lightrag_memgraph.register(
                replace_ui=True, mount_server=True, shim_native_routes=True
            )

        assert calls, "expected _patch_lightrag_server_create_app to still run"
        assert calls["webui_dist"] is None
        assert calls["twin_api_prefix"] == "/twin/api"
        assert calls["shim_native_routes"] is True
        assert any(
            "no WebUI dist found" in rec.message for rec in caplog.records
        ), "expected a loud ERROR log about the missing dist"
