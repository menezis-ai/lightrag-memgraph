"""FAILED-doc LLM extraction-cache purge (audit 2026-07-02 addendum, finding B).

LightRAG persists the entity-extraction LLM cache even when the document
itself fails: the extraction cache is gated by
``enable_llm_cache_for_entity_extract`` (default true, independent of
``enable_llm_cache``) and the FAILED handlers in ``process_document`` flush
the cache *before* writing the FAILED status row (verified on the 1.4.9.11
wheel). The Memgraph DocStatus backend therefore purges the extraction-cache
rows tied to a FAILED doc's chunks, so a re-ingestion re-calls the LLM
instead of replaying the cached (possibly truncated/imparsable) responses.

Compat contract (docs/test-doctrine-lightrag-compat.md):
``TWIN_PURGE_LLM_CACHE_ON_FAILED=0`` restores LightRAG-native behavior —
cache rows survive the failure, and no purge query is ever issued.
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from twindb_lightrag_memgraph.kv_impl import MemgraphKVStorage

WS = "cache_purge_it"


def _doc_store(workspace: str) -> MemgraphDocStatusStorage:
    return MemgraphDocStatusStorage(
        namespace="doc_status",
        global_config={},
        embedding_func=None,
    )


@pytest.fixture
async def stores(monkeypatch):
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", WS)
    doc = _doc_store(WS)
    chunks = MemgraphKVStorage(
        namespace="text_chunks", global_config={}, embedding_func=None
    )
    cache = MemgraphKVStorage(
        namespace="llm_response_cache", global_config={}, embedding_func=None
    )
    for s in (doc, chunks, cache):
        await s.initialize()
    yield doc, chunks, cache
    for s in (doc, chunks, cache):
        await s.drop()


async def _seed(chunks: MemgraphKVStorage, cache: MemgraphKVStorage) -> None:
    # Shapes mirror LightRAG 1.4.9.11: text_chunks rows embed full_doc_id,
    # extraction-cache rows embed chunk_id, query-cache rows carry no chunk.
    await chunks.upsert(
        {
            "chunk-fail-1": {"content": "part one", "full_doc_id": "doc-fail"},
            "chunk-fail-2": {"content": "part two", "full_doc_id": "doc-fail"},
            "chunk-ok-1": {"content": "healthy", "full_doc_id": "doc-ok"},
        }
    )
    await cache.upsert(
        {
            "default:extract:hfail1": {
                "return": "{broken json",
                "cache_type": "extract",
                "chunk_id": "chunk-fail-1",
            },
            "default:extract:hfail2": {
                "return": "{still broken",
                "cache_type": "extract",
                "chunk_id": "chunk-fail-2",
            },
            "default:extract:hok": {
                "return": "{}",
                "cache_type": "extract",
                "chunk_id": "chunk-ok-1",
            },
            "default:query:hq": {
                "return": "an answer",
                "cache_type": "query",
                "chunk_id": None,
            },
        }
    )


@pytest.mark.integration
class TestFailedDocCachePurge:
    async def test_failed_upsert_purges_extraction_cache_of_failed_doc_only(
        self, stores
    ):
        doc, chunks, cache = stores
        await _seed(chunks, cache)

        await doc.upsert(
            {
                "doc-fail": {
                    "status": "failed",
                    "file_path": "f.pdf",
                    "error_msg": "LLM output unparsable",
                }
            }
        )

        # The failed doc's extraction-cache rows are gone …
        assert await cache.get_by_id("default:extract:hfail1") is None
        assert await cache.get_by_id("default:extract:hfail2") is None
        # … other docs' extraction cache and query cache are untouched …
        assert await cache.get_by_id("default:extract:hok") is not None
        assert await cache.get_by_id("default:query:hq") is not None
        # … and the chunk rows themselves are not the purge's business.
        assert await chunks.get_by_id("chunk-fail-1") is not None
        # The FAILED status row itself persisted normally.
        failed = await doc.get_by_id("doc-fail")
        assert failed is not None and failed["status"] == "failed"

    async def test_flag_off_keeps_lightrag_native_behavior(self, stores, monkeypatch):
        """Compat: with the flag off, cache rows survive a FAILED doc exactly
        as they do on native LightRAG storage backends."""
        doc, chunks, cache = stores
        await _seed(chunks, cache)
        monkeypatch.setenv("TWIN_PURGE_LLM_CACHE_ON_FAILED", "0")

        await doc.upsert({"doc-fail": {"status": "failed", "file_path": "f.pdf"}})

        assert await cache.get_by_id("default:extract:hfail1") is not None
        assert await cache.get_by_id("default:extract:hfail2") is not None

    async def test_processed_upsert_does_not_purge(self, stores):
        doc, chunks, cache = stores
        await _seed(chunks, cache)

        await doc.upsert({"doc-fail": {"status": "processed", "file_path": "f.pdf"}})

        assert await cache.get_by_id("default:extract:hfail1") is not None


class TestPurgeGracefulDegradation:
    """No Memgraph needed — the purge must never break ingestion."""

    async def test_purge_swallows_storage_errors(self, monkeypatch):
        from twindb_lightrag_memgraph import _pool

        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "cache_purge_unit")
        store = _doc_store("cache_purge_unit")

        def boom():
            raise RuntimeError("memgraph down")

        monkeypatch.setattr(_pool, "get_read_session", boom)
        # Must not raise into the (simulated) ingestion pipeline.
        await store._purge_failed_doc_llm_cache(["doc-x"])

    async def test_flag_off_short_circuits_before_any_session(self, monkeypatch):
        from twindb_lightrag_memgraph import _pool

        monkeypatch.setenv("MEMGRAPH_WORKSPACE", "cache_purge_unit")
        monkeypatch.setenv("TWIN_PURGE_LLM_CACHE_ON_FAILED", "false")
        store = _doc_store("cache_purge_unit")
        calls: list[int] = []
        monkeypatch.setattr(_pool, "get_read_session", lambda: calls.append(1))

        await store._purge_failed_doc_llm_cache(["doc-x"])

        assert calls == []  # flag off → zero storage traffic
