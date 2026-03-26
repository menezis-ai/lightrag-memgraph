"""
Tests for the pre-insert memory budget enforcement patch.

Unit tests (no Memgraph needed) for:
- _patch_enqueue_budget_check() -- the monkey-patch on apipeline_enqueue_documents
- MEMGRAPH_BUDGET_ENFORCE env var modes: off, warn, reject
- Graceful handling of budget check failures
"""

from unittest.mock import AsyncMock, patch

import pytest

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph._memory import BudgetCheck, MemoryBudgetExceeded


@pytest.fixture(autouse=True)
def _ensure_registered():
    """Ensure register() has been called so the patch is active."""
    twindb_lightrag_memgraph.register()


def _make_budget_check(fits: bool) -> BudgetCheck:
    """Create a BudgetCheck with the given fits value."""
    return BudgetCheck(
        fits=fits,
        used_bytes=50 * 1024**3,
        limit_bytes=75 * 1024**3,
        estimated_cost_bytes=10 * 1024**3,
        remaining_bytes=15 * 1024**3 if fits else 0,
        headroom_ratio=0.20 if fits else 0.0,
    )


class TestBudgetEnforceOff:
    async def test_enforce_off_by_default(self, monkeypatch):
        """With MEMGRAPH_BUDGET_ENFORCE unset, no budget check occurs."""
        monkeypatch.delenv("MEMGRAPH_BUDGET_ENFORCE", raising=False)

        from lightrag.lightrag import LightRAG

        mock_original = AsyncMock(return_value=None)

        with patch(
            "twindb_lightrag_memgraph._memory.check_memory_budget",
            new_callable=AsyncMock,
        ) as mock_check:
            # Call the patched method via the class — we need a mock self
            mock_self = AsyncMock(spec=LightRAG)
            # Bypass the patch chain by calling the patched function directly
            # The current patched method is LightRAG.apipeline_enqueue_documents
            patched_fn = LightRAG.apipeline_enqueue_documents

            # We need to mock the _original captured in the closure.
            # Simplest approach: just verify check_memory_budget is never called.
            # Use a fresh patch on the inner _original via the module-level approach.
            with patch.object(
                LightRAG,
                "apipeline_enqueue_documents",
                wraps=patched_fn,
            ):
                # Re-patch to inject our mock original
                pass

            # Simpler approach: just call the patched function and verify
            # check_memory_budget was NOT called
            try:
                await patched_fn(mock_self, ["some text"])
            except Exception:
                pass  # Original may fail — that's fine, we just check the gate

            mock_check.assert_not_called()


class TestBudgetEnforceWarn:
    async def test_enforce_warn_logs_but_proceeds(self, monkeypatch):
        """With mode=warn and fits=False, log warning but call original."""
        monkeypatch.setenv("MEMGRAPH_BUDGET_ENFORCE", "warn")

        from lightrag.lightrag import LightRAG

        budget = _make_budget_check(fits=False)
        patched_fn = LightRAG.apipeline_enqueue_documents

        with (
            patch(
                "twindb_lightrag_memgraph._memory.check_memory_budget",
                new_callable=AsyncMock,
                return_value=budget,
            ) as mock_check,
            patch("twindb_lightrag_memgraph.logger") as mock_logger,
        ):
            mock_self = AsyncMock(spec=LightRAG)
            try:
                await patched_fn(mock_self, ["some text"])
            except Exception:
                pass  # Original may fail

            mock_check.assert_awaited()
            mock_logger.warning.assert_called()
            # Verify warning message contains budget info
            warn_args = mock_logger.warning.call_args
            assert "[Budget]" in warn_args[0][0]


class TestBudgetEnforceReject:
    async def test_enforce_reject_raises(self, monkeypatch):
        """With mode=reject and fits=False, raise MemoryBudgetExceeded."""
        monkeypatch.setenv("MEMGRAPH_BUDGET_ENFORCE", "reject")

        from lightrag.lightrag import LightRAG

        budget = _make_budget_check(fits=False)
        patched_fn = LightRAG.apipeline_enqueue_documents

        with patch(
            "twindb_lightrag_memgraph._memory.check_memory_budget",
            new_callable=AsyncMock,
            return_value=budget,
        ):
            mock_self = AsyncMock(spec=LightRAG)
            with pytest.raises(MemoryBudgetExceeded) as exc_info:
                await patched_fn(mock_self, ["some text"])

            assert exc_info.value.budget_check is budget
            assert "WOULD EXCEED BUDGET" in str(exc_info.value)

    async def test_enforce_reject_allows_within_budget(self, monkeypatch):
        """With mode=reject and fits=True, proceed normally."""
        monkeypatch.setenv("MEMGRAPH_BUDGET_ENFORCE", "reject")

        from lightrag.lightrag import LightRAG

        budget = _make_budget_check(fits=True)
        patched_fn = LightRAG.apipeline_enqueue_documents

        with patch(
            "twindb_lightrag_memgraph._memory.check_memory_budget",
            new_callable=AsyncMock,
            return_value=budget,
        ):
            mock_self = AsyncMock(spec=LightRAG)
            # Should NOT raise — budget fits
            try:
                await patched_fn(mock_self, ["some text"])
            except MemoryBudgetExceeded:
                pytest.fail("MemoryBudgetExceeded raised despite fits=True")
            except Exception:
                pass  # Original may fail — that's fine, gate passed


class TestBudgetEnforceEdgeCases:
    async def test_enforce_handles_single_string_input(self, monkeypatch):
        """When input is a string (not list), it should be wrapped in a list."""
        monkeypatch.setenv("MEMGRAPH_BUDGET_ENFORCE", "warn")

        from lightrag.lightrag import LightRAG

        budget = _make_budget_check(fits=True)
        patched_fn = LightRAG.apipeline_enqueue_documents

        with patch(
            "twindb_lightrag_memgraph._memory.check_memory_budget",
            new_callable=AsyncMock,
            return_value=budget,
        ) as mock_check:
            mock_self = AsyncMock(spec=LightRAG)
            try:
                await patched_fn(mock_self, "a single string")
            except Exception:
                pass

            mock_check.assert_awaited()
            # Verify texts= was passed as a list wrapping the string
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs["texts"] == ["a single string"]

    async def test_enforce_budget_check_failure_proceeds(self, monkeypatch):
        """If check_memory_budget raises, log warning and proceed."""
        monkeypatch.setenv("MEMGRAPH_BUDGET_ENFORCE", "reject")

        from lightrag.lightrag import LightRAG

        patched_fn = LightRAG.apipeline_enqueue_documents

        with (
            patch(
                "twindb_lightrag_memgraph._memory.check_memory_budget",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Memgraph unreachable"),
            ),
            patch("twindb_lightrag_memgraph.logger") as mock_logger,
        ):
            mock_self = AsyncMock(spec=LightRAG)
            # Should NOT raise MemoryBudgetExceeded — the check itself failed
            try:
                await patched_fn(mock_self, ["some text"])
            except MemoryBudgetExceeded:
                pytest.fail(
                    "MemoryBudgetExceeded raised despite check_memory_budget failing"
                )
            except Exception:
                pass  # Original may fail

            # Verify warning was logged about the failure
            mock_logger.warning.assert_called()
            warn_args = mock_logger.warning.call_args
            assert "Budget check failed" in warn_args[0][0]


class TestMemoryBudgetExceededException:
    def test_exception_attributes(self):
        budget = _make_budget_check(fits=False)
        exc = MemoryBudgetExceeded(budget)
        assert exc.budget_check is budget
        assert "WOULD EXCEED BUDGET" in str(exc)

    def test_exception_is_exception(self):
        budget = _make_budget_check(fits=False)
        exc = MemoryBudgetExceeded(budget)
        assert isinstance(exc, Exception)

    def test_exception_importable_from_top_level(self):
        from twindb_lightrag_memgraph import MemoryBudgetExceeded as Exc

        assert Exc is MemoryBudgetExceeded


class TestPatchRegistration:
    def test_enqueue_patched(self):
        """apipeline_enqueue_documents should be patched after register()."""
        from lightrag.lightrag import LightRAG

        assert "budget_gated" in LightRAG.apipeline_enqueue_documents.__name__
