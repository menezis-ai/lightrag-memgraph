"""Unmocked import-resolution tests for the standalone server factory.

Audit 2026-07-02 COMPAT-1/COMPAT-2: ``server/app.py`` imported
``openai_embedding`` (the upstream name is ``openai_embed`` on every supported
LightRAG) and awaited ``_rag.initialize()`` (the real API is
``initialize_storages()``) — the lifespan had NEVER run against a real
LightRAG because every test mocked those seams (one literally
``patch(..., create=True)``-ed the nonexistent symbol into existence).

These tests import the exact symbols the factory needs from the INSTALLED
lightrag, with zero mocks, so the CI matrix itself proves resolution on
1.4.9.11 / 1.4.11 / 1.4.12. The integration test at the bottom runs the full
unmocked lifespan against a real Memgraph.
"""

from __future__ import annotations

import inspect
import uuid
from pathlib import Path

import pytest

import twindb_lightrag_memgraph.server.app as app_module
from twindb_lightrag_memgraph.server.app import (
    _build_embedding_func,
    _build_llm_func,
    create_app,
)
from twindb_lightrag_memgraph.server.settings import LightRAGServerSettings


def _make_settings(**overrides) -> LightRAGServerSettings:
    defaults = dict(
        working_dir="/tmp/lightrag_factory_resolution_test",
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        graph_storage="MemgraphStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        enable_langsmith_tracing=False,
        graph_relation_id_backfill_on_startup=False,
        webui_tag_backend="memory",
        webui_activity_backend="memory",
        webui_notifications_backend="memory",
        api_key="factory-resolution-key",
    )
    defaults.update(overrides)
    return LightRAGServerSettings(**defaults)


class TestFactorySymbolResolution:
    """The exact upstream imports the factory performs — unmocked."""

    def test_openai_embed_resolves_from_installed_lightrag(self):
        from lightrag.llm.openai import openai_embed

        # Depending on the version, openai_embed is a bare coroutine function
        # or an EmbeddingFunc instance (decorated upstream). Both are
        # async-callable, which is all the factory relies on.
        assert callable(openai_embed)
        assert inspect.iscoroutinefunction(openai_embed) or (
            inspect.iscoroutinefunction(type(openai_embed).__call__)
        )

    def test_openai_complete_resolves_from_installed_lightrag(self):
        from lightrag.llm.openai import openai_complete

        assert inspect.iscoroutinefunction(openai_complete)

    def test_embedding_func_dataclass_resolves_from_installed_lightrag(self):
        from lightrag.utils import EmbeddingFunc

        fields = getattr(EmbeddingFunc, "__dataclass_fields__", {})
        for name in ("embedding_dim", "func", "max_token_size"):
            assert name in fields

    def test_lightrag_exposes_initialize_storages(self):
        from lightrag import LightRAG

        assert inspect.iscoroutinefunction(LightRAG.initialize_storages)

    def test_build_embedding_func_resolves_unmocked(self):
        """Calling the builder executes its real import — no patching."""
        from lightrag.utils import EmbeddingFunc

        func = _build_embedding_func(_make_settings(), api_key="unused")

        assert isinstance(func, EmbeddingFunc)
        assert inspect.iscoroutinefunction(func.func)

    def test_build_llm_func_resolves_unmocked(self):
        func = _build_llm_func(_make_settings(), api_key="unused")

        assert inspect.iscoroutinefunction(func)


class TestLifespanSourceContract:
    """Source-level pins (read from the file: immune to runtime patching)."""

    def test_lifespan_awaits_initialize_storages_not_initialize(self):
        source = Path(app_module.__file__).read_text(encoding="utf-8")
        assert "await _rag.initialize_storages()" in source
        assert "await _rag.initialize()" not in source

    def test_factory_never_imports_or_calls_openai_embedding(self):
        """Pin usage, not prose: the docstring may name the dead symbol, but
        no import or call of it may reappear."""
        source = Path(app_module.__file__).read_text(encoding="utf-8")
        assert "import openai_embedding" not in source
        assert "openai_embedding(" not in source


@pytest.mark.integration
async def test_standalone_lifespan_boots_unmocked_against_memgraph(
    monkeypatch, tmp_path
):
    """Full unmocked lifespan boot: real register(), real LightRAG from the
    installed package, real Memgraph storages. Stub LLM credentials only —
    no model call happens at boot."""
    from twindb_lightrag_memgraph.server.auth import configure_auth

    for var in ("TWIN_REPLACE_UI", "TWIN_MOUNT_SERVER", "TWIN_SHIM_NATIVE_ROUTES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-stub-boot-only-never-called")

    workspace = f"fboot_{uuid.uuid4().hex[:10]}"
    settings = _make_settings(
        working_dir=str(tmp_path),
        workspace=workspace,
    )
    app = create_app(settings)

    try:
        async with app.router.lifespan_context(app):
            rag = app_module._rag
            assert rag is not None
            assert type(rag).__name__ == "LightRAG"
            # initialize_storages() really ran (upstream enum flips):
            assert "INITIALIZED" in str(getattr(rag, "_storages_status", ""))
            # …and a real Bolt round-trip works through the Memgraph KV slot:
            assert await rag.full_docs.get_by_id("doc-that-does-not-exist") is None
        assert app_module._rag is None
    finally:
        configure_auth(api_key=None, jwt_secret=None)
