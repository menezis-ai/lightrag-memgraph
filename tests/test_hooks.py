"""Unit tests for post-indexation hooks (_insert_done patch).

No Memgraph required — LightRAG instance is mocked.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph._hooks import (
    _post_index_hooks,
    _run_post_index_hooks,
    clear_post_index_hooks,
    register_post_index_hook,
)


@pytest.fixture(autouse=True)
def _clean_hooks():
    """Ensure hooks list is empty before and after each test."""
    clear_post_index_hooks()
    yield
    clear_post_index_hooks()


def _mock_lightrag(workspace="test_ws"):
    """Create a mock LightRAG instance with storage backends."""
    rag = MagicMock()
    rag.workspace = workspace

    # Mock the 12 storage instances that _insert_done iterates over
    for attr in (
        "full_docs",
        "doc_status",
        "text_chunks",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
        "entities_vdb",
        "relationships_vdb",
        "chunks_vdb",
        "chunk_entity_relation_graph",
    ):
        mock_storage = AsyncMock()
        mock_storage.index_done_callback = AsyncMock()
        setattr(rag, attr, mock_storage)

    return rag


# ── Hook execution ───────────────────────────────────────────────────


class TestHookExecution:
    async def test_hook_called_after_insert_done(self):
        """Registered hook should be called with the LightRAG instance."""
        twindb_lightrag_memgraph.register()

        rag = _mock_lightrag()
        received = []

        async def capture(inst):
            received.append(inst)

        register_post_index_hook(capture)

        await _run_post_index_hooks(rag)

        assert len(received) == 1
        assert received[0] is rag

    async def test_multiple_hooks_run_in_order(self):
        """Multiple hooks should run in registration order."""
        order = []

        async def hook_a(inst):
            order.append("a")

        async def hook_b(inst):
            order.append("b")

        register_post_index_hook(hook_a)
        register_post_index_hook(hook_b)

        await _run_post_index_hooks(MagicMock())

        assert order == ["a", "b"]

    async def test_failing_hook_does_not_break_pipeline(self):
        """A failing hook should not prevent subsequent hooks from running."""
        second_hook = AsyncMock()

        async def bad_hook(inst):
            raise RuntimeError("boom")

        register_post_index_hook(bad_hook)
        register_post_index_hook(second_hook)

        await _run_post_index_hooks(MagicMock())

        second_hook.assert_awaited_once()

    async def test_no_hooks_is_noop(self):
        """Empty hook list should not raise."""
        await _run_post_index_hooks(MagicMock())

    async def test_hook_receives_instance_with_workspace(self):
        """Hook should receive the LightRAG instance with workspace access."""
        received = {}

        async def capture_hook(inst):
            received["workspace"] = inst.workspace

        register_post_index_hook(capture_hook)
        rag = _mock_lightrag(workspace="prod")

        await _run_post_index_hooks(rag)

        assert received["workspace"] == "prod"


# ── Registry management ──────────────────────────────────────────────


class TestRegistry:
    def test_clear_hooks(self):
        """clear_post_index_hooks should empty the list."""
        register_post_index_hook(AsyncMock())
        assert len(_post_index_hooks) == 1

        clear_post_index_hooks()
        assert len(_post_index_hooks) == 0

    def test_register_post_index_hook_exported(self):
        """register_post_index_hook should be importable from top-level package."""
        from twindb_lightrag_memgraph import register_post_index_hook as fn

        assert callable(fn)


# ── Patch registration ───────────────────────────────────────────────


class TestPatchRegistration:
    def test_insert_done_patched(self):
        """_insert_done should be patched after register()."""
        twindb_lightrag_memgraph.register()
        from lightrag.lightrag import LightRAG

        assert "hooked" in LightRAG._insert_done.__name__
