"""Unit tests for post-indexation hooks (_insert_done patch).

No Memgraph required — LightRAG instance is mocked.
"""

import asyncio
import functools
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

import lightrag_memgraph
from lightrag_memgraph._hooks import (
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
        lightrag_memgraph.register()

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

    @pytest.mark.parametrize(
        ("callback_kind", "expected_name"),
        [("function", "bad_hook"), ("partial", "partial"), ("instance", "FailingHook")],
    )
    async def test_callback_failure_logs_original_cause_and_continues(
        self, callback_kind, expected_name, caplog
    ):
        """Functions and nameless callbacks retain their error and do not stop later hooks."""
        failure = RuntimeError("hook failure")
        cause = ValueError("original cause")

        async def bad_hook(inst, marker="audit"):
            raise failure from cause

        class FailingHook:
            async def __call__(self, inst):
                raise failure from cause

        callbacks = {
            "function": bad_hook,
            "partial": functools.partial(bad_hook, marker="audit"),
            "instance": FailingHook(),
        }
        healthy_hook = AsyncMock()
        rag = MagicMock()
        register_post_index_hook(callbacks[callback_kind])
        register_post_index_hook(healthy_hook)

        with caplog.at_level(logging.ERROR, logger="lightrag_memgraph"):
            await _run_post_index_hooks(rag)

        healthy_hook.assert_awaited_once_with(rag)
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.getMessage() == f"Post-index hook {expected_name} failed"
        assert record.exc_info[1] is failure
        assert record.exc_info[1].__cause__ is cause
        assert "ValueError: original cause" in caplog.text

    @pytest.mark.parametrize(
        "metadata", ["raises", "non_string", "empty", "str_subclass"]
    )
    async def test_unusable_callback_name_does_not_replace_hook_error(
        self, metadata, caplog
    ):
        """Broken metadata and repr cannot escape the callback failure handler."""
        failure = RuntimeError("hook failure")

        class UnprintableName(str):
            def __str__(self):
                raise RuntimeError("callback name must not be stringified")

        class BrokenNameHook:
            @property
            def __name__(self):
                if metadata == "raises":
                    raise LookupError("unreadable name")
                if metadata == "str_subclass":
                    return UnprintableName("untrusted callback name")
                return self if metadata == "non_string" else ""

            def __repr__(self):
                raise AssertionError("callback repr must not be used")

            async def __call__(self, inst):
                raise failure

        healthy_hook = AsyncMock()
        register_post_index_hook(BrokenNameHook())
        register_post_index_hook(healthy_hook)

        with caplog.at_level(logging.ERROR, logger="lightrag_memgraph"):
            await _run_post_index_hooks(MagicMock())

        healthy_hook.assert_awaited_once()
        assert len(caplog.records) == 1
        record = caplog.records[0]
        expected_name = "<unknown>" if metadata == "raises" else "BrokenNameHook"
        assert record.getMessage() == f"Post-index hook {expected_name} failed"
        assert record.exc_info[1] is failure

    @pytest.mark.parametrize("error_type", [asyncio.CancelledError, KeyboardInterrupt])
    async def test_base_exception_propagates_without_running_later_hooks(
        self, error_type, caplog
    ):
        """Cancellation and process interruption remain outside failure isolation."""
        failure = error_type("stop hooks")

        async def interrupted_hook(inst):
            raise failure

        healthy_hook = AsyncMock()
        register_post_index_hook(interrupted_hook)
        register_post_index_hook(healthy_hook)

        with pytest.raises(error_type) as raised:
            await _run_post_index_hooks(MagicMock())

        assert raised.value is failure
        healthy_hook.assert_not_awaited()
        assert caplog.records == []

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
        from lightrag_memgraph import register_post_index_hook as fn

        assert callable(fn)


# ── Patch registration ───────────────────────────────────────────────


class TestPatchRegistration:
    def test_insert_done_patched(self):
        """_insert_done should be patched after register()."""
        lightrag_memgraph.register()
        from lightrag.lightrag import LightRAG

        assert "hooked" in LightRAG._insert_done.__name__
