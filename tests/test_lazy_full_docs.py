"""Tests for the lazy full_docs proxy: purge + reconstruction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from twindb_lightrag_memgraph._lazy_full_docs import (
    is_enabled,
    patch_full_docs_with_lazy_reconstruction,
    purge_processed_full_docs,
    reset_reconstruction_state,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset reconstruction tracking between tests."""
    reset_reconstruction_state()
    yield
    reset_reconstruction_state()


class TestIsEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_PURGE_FULL_DOCS", raising=False)
        assert is_enabled() is False

    def test_enabled_when_on(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_PURGE_FULL_DOCS", "on")
        assert is_enabled() is True

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_PURGE_FULL_DOCS", "ON")
        assert is_enabled() is True


class TestPurgeProcessedFullDocs:
    async def test_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("MEMGRAPH_PURGE_FULL_DOCS", raising=False)
        mock_rag = MagicMock()
        await purge_processed_full_docs(mock_rag)
        # doc_status should never be accessed
        mock_rag.doc_status.get_docs_by_status.assert_not_called()

    async def test_purges_processed_docs(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_PURGE_FULL_DOCS", "on")
        mock_rag = MagicMock()
        mock_rag.doc_status.get_docs_by_status = AsyncMock(
            return_value={"doc1": MagicMock(), "doc2": MagicMock()}
        )
        mock_rag.full_docs.get_by_id = AsyncMock(
            side_effect=lambda id: {"content": "text"} if id == "doc1" else None
        )
        mock_rag.full_docs.delete = AsyncMock()
        mock_rag.text_chunks = MagicMock()

        await purge_processed_full_docs(mock_rag)

        # Only doc1 had content — only doc1 should be purged
        mock_rag.full_docs.delete.assert_awaited_once_with(["doc1"])

    async def test_no_processed_docs(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_PURGE_FULL_DOCS", "on")
        mock_rag = MagicMock()
        mock_rag.doc_status.get_docs_by_status = AsyncMock(return_value={})
        mock_rag.full_docs = MagicMock()
        mock_rag.text_chunks = MagicMock()

        await purge_processed_full_docs(mock_rag)
        mock_rag.full_docs.delete.assert_not_called()

    async def test_exception_does_not_propagate(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_PURGE_FULL_DOCS", "on")
        mock_rag = MagicMock()
        mock_rag.doc_status.get_docs_by_status = AsyncMock(
            side_effect=RuntimeError("DB error")
        )
        mock_rag.full_docs = MagicMock()
        mock_rag.text_chunks = MagicMock()

        # Should not raise
        await purge_processed_full_docs(mock_rag)


class TestLazyReconstruction:
    async def test_returns_original_when_present(self):
        mock_rag = MagicMock()
        mock_rag.full_docs.get_by_id = AsyncMock(
            return_value={"content": "original", "file_path": "/a.pdf"}
        )
        mock_rag.doc_status = MagicMock()
        mock_rag.text_chunks = MagicMock()

        patch_full_docs_with_lazy_reconstruction(mock_rag)

        result = await mock_rag.full_docs.get_by_id("doc1")
        assert result == {"content": "original", "file_path": "/a.pdf"}

    async def test_reconstructs_from_chunks(self):
        mock_rag = MagicMock()
        mock_rag.full_docs.get_by_id = AsyncMock(return_value=None)
        mock_rag.doc_status.get_by_id = AsyncMock(
            return_value={
                "chunks_list": ["c1", "c2", "c3"],
                "file_path": "/doc.pdf",
            }
        )
        mock_rag.text_chunks.get_by_ids = AsyncMock(
            return_value=[
                {"content": "Part 1.", "chunk_order_index": 0},
                {"content": "Part 2.", "chunk_order_index": 1},
                {"content": "Part 3.", "chunk_order_index": 2},
            ]
        )

        patch_full_docs_with_lazy_reconstruction(mock_rag)

        result = await mock_rag.full_docs.get_by_id("doc1")
        assert result is not None
        assert result["file_path"] == "/doc.pdf"
        assert "Part 1." in result["content"]
        assert "Part 2." in result["content"]
        assert "Part 3." in result["content"]

    async def test_returns_none_when_no_doc_status(self):
        mock_rag = MagicMock()
        mock_rag.full_docs.get_by_id = AsyncMock(return_value=None)
        mock_rag.doc_status.get_by_id = AsyncMock(return_value=None)
        mock_rag.text_chunks = MagicMock()

        patch_full_docs_with_lazy_reconstruction(mock_rag)

        result = await mock_rag.full_docs.get_by_id("missing")
        assert result is None

    async def test_returns_none_when_no_chunks(self):
        mock_rag = MagicMock()
        mock_rag.full_docs.get_by_id = AsyncMock(return_value=None)
        mock_rag.doc_status.get_by_id = AsyncMock(
            return_value={"chunks_list": [], "file_path": "/doc.pdf"}
        )
        mock_rag.text_chunks = MagicMock()

        patch_full_docs_with_lazy_reconstruction(mock_rag)

        result = await mock_rag.full_docs.get_by_id("empty")
        assert result is None

    async def test_reconstruction_exception_returns_none(self):
        mock_rag = MagicMock()
        mock_rag.full_docs.get_by_id = AsyncMock(return_value=None)
        mock_rag.doc_status.get_by_id = AsyncMock(side_effect=RuntimeError("DB error"))
        mock_rag.text_chunks = MagicMock()

        patch_full_docs_with_lazy_reconstruction(mock_rag)

        result = await mock_rag.full_docs.get_by_id("doc1")
        assert result is None


class TestReconstructionPatchTracking:
    async def test_patches_only_once_per_instance(self, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_PURGE_FULL_DOCS", "on")
        mock_rag = MagicMock()
        mock_rag.doc_status.get_docs_by_status = AsyncMock(return_value={})
        mock_rag.full_docs = MagicMock()
        mock_rag.full_docs.get_by_id = AsyncMock(return_value=None)
        mock_rag.text_chunks = MagicMock()

        with patch(
            "twindb_lightrag_memgraph._lazy_full_docs.patch_full_docs_with_lazy_reconstruction"
        ) as mock_patch:
            await purge_processed_full_docs(mock_rag)
            await purge_processed_full_docs(mock_rag)
            # Should be called only once for the same instance
            mock_patch.assert_called_once()
