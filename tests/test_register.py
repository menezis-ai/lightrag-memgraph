"""
Tests for the register() function.

These tests are OFFLINE - they only verify the monkey-patch logic
on LightRAG's in-memory registries, no Memgraph connection needed.
"""

import importlib

import lightrag.kg as kg_registry

import lightrag_memgraph


def _reset_registration():
    """Force re-registration by clearing the flag."""
    lightrag_memgraph._registered = False


class TestRegister:
    def test_register_preserves_merge_entry_points(self, monkeypatch):
        """Both upstream import sites must keep their lock-protected write path."""
        from lightrag import lightrag as lr_module
        from lightrag import operate

        original = object()
        monkeypatch.setattr(operate, "merge_nodes_and_edges", original)
        # LightRAG 1.5.0 imports merge inside pipeline helpers; later releases
        # and 1.4.x also expose a module-level import site.
        module_entry_point = hasattr(lr_module, "merge_nodes_and_edges")
        if module_entry_point:
            monkeypatch.setattr(lr_module, "merge_nodes_and_edges", original)
        monkeypatch.setattr(lightrag_memgraph, "_registered", False)
        lightrag_memgraph.register()

        assert operate.merge_nodes_and_edges is original
        if module_entry_point:
            assert lr_module.merge_nodes_and_edges is original

    def test_register_exposes_package_version_once(self, monkeypatch):
        import lightrag

        monkeypatch.setattr(lightrag_memgraph, "__version__", "0.6.0")
        monkeypatch.setattr(lightrag_memgraph, "_registered", False)
        monkeypatch.setattr(lightrag, "__version__", "1.4.16")
        lightrag_memgraph.register()
        lightrag_memgraph.register()

        assert lightrag.__version__ == "1.4.16+memgraph-0.6.0"

    def test_register_idempotent(self):
        """Calling register() twice should not duplicate entries."""
        _reset_registration()
        lightrag_memgraph.register()
        lightrag_memgraph.register()

        count = kg_registry.STORAGE_IMPLEMENTATIONS["KV_STORAGE"][
            "implementations"
        ].count("MemgraphKVStorage")
        assert count == 1

    def test_kv_in_implementations(self):
        """MemgraphKVStorage must appear in KV_STORAGE implementations."""
        _reset_registration()
        lightrag_memgraph.register()

        impls = kg_registry.STORAGE_IMPLEMENTATIONS["KV_STORAGE"]["implementations"]
        assert "MemgraphKVStorage" in impls

    def test_vector_in_implementations(self):
        """MemgraphVectorDBStorage must appear in VECTOR_STORAGE implementations."""
        _reset_registration()
        lightrag_memgraph.register()

        impls = kg_registry.STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
        assert "MemgraphVectorDBStorage" in impls

    def test_docstatus_in_implementations(self):
        """MemgraphDocStatusStorage must appear in DOC_STATUS_STORAGE implementations."""
        _reset_registration()
        lightrag_memgraph.register()

        impls = kg_registry.STORAGE_IMPLEMENTATIONS["DOC_STATUS_STORAGE"][
            "implementations"
        ]
        assert "MemgraphDocStatusStorage" in impls

    def test_env_requirements(self):
        """All 3 backends must declare MEMGRAPH_URI as required."""
        _reset_registration()
        lightrag_memgraph.register()

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
        lightrag_memgraph.register()

        expected = {
            "MemgraphKVStorage": "lightrag_memgraph.kv_impl",
            "MemgraphVectorDBStorage": "lightrag_memgraph.vector_impl",
            "MemgraphDocStatusStorage": "lightrag_memgraph.docstatus_impl",
        }
        for class_name, module_path in expected.items():
            assert kg_registry.STORAGES[class_name] == module_path

    def test_absolute_import_resolution(self):
        """Absolute module paths must resolve even when package='lightrag'."""
        paths = {
            "MemgraphKVStorage": "lightrag_memgraph.kv_impl",
            "MemgraphVectorDBStorage": "lightrag_memgraph.vector_impl",
            "MemgraphDocStatusStorage": "lightrag_memgraph.docstatus_impl",
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
        lightrag_memgraph.register()

        from lightrag.kg import verify_storage_implementation

        # These must NOT raise
        verify_storage_implementation("KV_STORAGE", "MemgraphKVStorage")
        verify_storage_implementation("VECTOR_STORAGE", "MemgraphVectorDBStorage")
        verify_storage_implementation("DOC_STATUS_STORAGE", "MemgraphDocStatusStorage")
