"""Tests for the S4c tag-governance stores.

- ``TestInMemoryTagStore`` is pure unit (no Memgraph dependency).
- ``TestMemgraphTagStore`` is integration (marker auto-skips when
  MEMGRAPH_URI is unset, per tests/conftest.py).
"""

from __future__ import annotations

import pytest

from twindb_lightrag_memgraph.server import webui_seed
from twindb_lightrag_memgraph.server.webui_tagstore import (
    InMemoryTagStore,
    MemgraphTagStore,
    make_memgraph_store,
)


class TestInMemoryTagStore:
    def test_seeded_from_module(self):
        store = InMemoryTagStore()
        assert len(store.list_tags()) == len(webui_seed.TAGS)
        assert len(store.list_categories()) == len(webui_seed.TAG_CATEGORIES)

    def test_list_returns_a_deep_copy(self):
        store = InMemoryTagStore()
        first = store.list_tags()
        first[0]["tag"] = "MUTATED"
        # Subsequent read is unaffected.
        again = store.list_tags()
        assert again[0]["tag"] != "MUTATED"

    def test_custom_data_overrides_seed(self):
        custom_tags = [
            {
                "tag": "alpha",
                "tier": 1,
                "category": "infra",
                "status": "active",
                "def": "...",
                "aliases": [],
                "deprecates": [],
                "sources_count": 0,
                "chunks_count": 0,
                "query_freq_30d": 0,
                "created": {"by": "x", "at": "2026-01-01"},
                "last_edit": {"by": "x", "at": "2026-01-01"},
                "related": [],
                "examples": [],
            },
        ]
        custom_cats = [{"id": "infra", "label": "Infra", "color": "#000"}]
        store = InMemoryTagStore(tags=custom_tags, categories=custom_cats)
        assert [t["tag"] for t in store.list_tags()] == ["alpha"]
        assert [c["id"] for c in store.list_categories()] == ["infra"]


# ---------------------------------------------------------------------------
# Integration — Memgraph backend
# ---------------------------------------------------------------------------


@pytest.fixture
def _ws():
    """Unique workspace per test run so the KV doesn't bleed across cases."""
    import secrets

    return f"twstore_{secrets.token_hex(4)}"


async def _cleanup(workspace: str) -> None:
    """Drop tag + membership labels for the test workspace."""
    from twindb_lightrag_memgraph import _pool

    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            for label in (
                f"WebuiTag_{workspace}",
                f"WebuiTagCategory_{workspace}",
                f"DocStatus_{workspace}",
                f"Folder_{workspace}",
            ):
                result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                await result.consume()


@pytest.mark.integration
class TestMemgraphTagStore:
    async def test_bootstrap_writes_seed_then_subsequent_calls_skip(self, _ws):
        try:
            store = MemgraphTagStore(workspace=_ws)
            await store.initialize()
            first = await store.bootstrap_if_empty()
            second = await store.bootstrap_if_empty()
            assert first is True
            assert second is False
        finally:
            await _cleanup(_ws)

    async def test_list_tags_returns_seed_data(self, _ws):
        try:
            store = MemgraphTagStore(workspace=_ws)
            await store.initialize()
            await store.bootstrap_if_empty()
            tags = await store.list_tags()
            cats = await store.list_categories()
            assert len(tags) == len(webui_seed.TAGS)
            assert len(cats) == len(webui_seed.TAG_CATEGORIES)
            tag_names = {t["tag"] for t in tags}
            assert "rman" in tag_names
            assert "argocd" in tag_names  # requested tier survives the round-trip
            # Requested-tier fields survive
            argocd = next(t for t in tags if t["tag"] == "argocd")
            assert argocd["tier"] == "requested"
            assert argocd["status"] == "pending-review"
        finally:
            await _cleanup(_ws)

    async def test_make_memgraph_store_bootstraps_atomic(self, _ws):
        try:
            store = await make_memgraph_store(workspace=_ws)
            tags = await store.list_tags()
            assert len(tags) == len(webui_seed.TAGS)
        finally:
            await _cleanup(_ws)

    async def test_usage_counts_only_member_docs(self, _ws, monkeypatch):
        monkeypatch.setenv("MEMGRAPH_WORKSPACE", _ws)
        try:
            from twindb_lightrag_memgraph import _pool

            store = MemgraphTagStore(workspace=_ws)
            await store.initialize()
            await store.upsert_tag(
                {
                    "tag": "scoped",
                    "tier": 3,
                    "category": "infra",
                    "status": "active",
                    "def": "folder-scoped tag",
                    "aliases": [],
                    "deprecates": [],
                    "sources_count": 0,
                    "chunks_count": 0,
                    "query_freq_30d": 0,
                    "created": {"by": "test", "at": "2026-01-01"},
                    "last_edit": {"by": "test", "at": "2026-01-01"},
                    "related": [],
                    "examples": [],
                }
            )
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    CREATE (member:`DocStatus_{_ws}` {{id: 'member', chunks_count: 2}})
                    CREATE (other:`DocStatus_{_ws}` {{id: 'other', chunks_count: 5}})
                    CREATE (folder:`Folder_{_ws}` {{id: $folder}})
                    WITH member, other, folder
                    MATCH (tag:`WebuiTag_{_ws}` {{id: 'scoped'}})
                    MERGE (member)-[:MEMBER_OF]->(folder)
                    MERGE (member)-[:TAGGED_WITH]->(tag)
                    MERGE (other)-[:TAGGED_WITH]->(tag)
                    """,
                    folder=_ws,
                )
                await result.consume()

            scoped = next(t for t in await store.list_tags() if t["tag"] == "scoped")
            assert scoped["sources_count"] == 1
            assert scoped["chunks_count"] == 2
        finally:
            await _cleanup(_ws)

    async def test_two_workspaces_are_isolated(self, _ws):
        ws2 = _ws + "_b"
        try:
            await make_memgraph_store(workspace=_ws)
            await make_memgraph_store(workspace=ws2)
            store_a = MemgraphTagStore(workspace=_ws)
            store_b = MemgraphTagStore(workspace=ws2)
            tags_a = await store_a.list_tags()
            tags_b = await store_b.list_tags()
            assert len(tags_a) == len(tags_b) == len(webui_seed.TAGS)
            # Drop one workspace and confirm the other survives.
            await _cleanup(_ws)
            tags_a_after = await store_a.list_tags()
            tags_b_after = await store_b.list_tags()
            assert tags_a_after == []
            assert len(tags_b_after) == len(webui_seed.TAGS)
        finally:
            await _cleanup(_ws)
            await _cleanup(ws2)


# ---------------------------------------------------------------------------
# Integration — router reads through the Memgraph backend
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestRouterWithMemgraphBackend:
    async def test_get_tags_hits_memgraph_when_backend_set(self, _ws):
        """End-to-end: WebuiStore + MemgraphTagStore + /tags handler."""
        from twindb_lightrag_memgraph.server import webui_router

        original = webui_router.get_store()
        try:
            backend = await make_memgraph_store(workspace=_ws)
            store = webui_router.WebuiStore.from_seed()
            store._tag_backend = backend  # noqa: SLF001 — test wiring
            webui_router.set_store(store)

            tags = await store.list_tags()
            cats = await store.list_tag_categories()
            assert len(tags) == len(webui_seed.TAGS)
            assert len(cats) == len(webui_seed.TAG_CATEGORIES)
        finally:
            webui_router.set_store(original)
            await _cleanup(_ws)
