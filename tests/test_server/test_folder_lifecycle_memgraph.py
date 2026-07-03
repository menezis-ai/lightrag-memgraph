"""Folder lifecycle on a Memgraph deployment (MG-3 / MG-4, audit 2026-07-02).

MG-3: folders created *after* boot must inherit the deployment's store
construction mode — Memgraph-backed tag/activity/notification stores on a
memgraph deployment (so the folder's audit trail survives a restart and is
shared across workers), seed stores only in seed mode (LightRAG-compat
doctrine: behavior identical when the feature is off).

MG-4: the folder-delete residual-data guard must probe the database
(``MEMBER_OF`` memberships, ``GRAPH_MEMBER_OF`` entities, ``WebuiTag_*``),
not only the in-process store, and a permitted delete must not strand the
``Folder_{ws}`` node or the folder's store labels.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server import folder_store, webui_router
from twindb_lightrag_memgraph.server.app import create_app
from twindb_lightrag_memgraph.server.webui import routes_folders
from twindb_lightrag_memgraph.server.webui import store as store_module
from twindb_lightrag_memgraph.server.webui.store import (
    WebuiStore,
    deployment_store_mode,
    ensure_folder_store,
    get_store,
    set_store,
)
from twindb_lightrag_memgraph.server.webui_activitystore import (
    InMemoryActivityStore,
    MemgraphActivityStore,
)
from twindb_lightrag_memgraph.server.webui_notificationstore import (
    InMemoryNotificationStore,
    MemgraphNotificationStore,
)
from twindb_lightrag_memgraph.server.webui_tagstore import (
    InMemoryTagStore,
    MemgraphTagStore,
)


def _register_memgraph_template(default_id: str) -> WebuiStore:
    """Register the store the boot wiring would have built for *default_id*.

    Mirrors ``patches/registry.py:_init_overlay_memgraph_stores`` /
    ``server/app.py:_init_webui_backends`` minus the awaited ``initialize()``
    calls (constructors are DB-free).
    """
    store = WebuiStore.for_folder(default_id, mode="memgraph")
    store._tag_backend = MemgraphTagStore(workspace=default_id)  # noqa: SLF001
    store._activity_backend = MemgraphActivityStore(  # noqa: SLF001
        workspace=default_id
    )
    store._notification_backend = MemgraphNotificationStore(  # noqa: SLF001
        workspace=default_id
    )
    set_store(store, folder=default_id)
    return store


def _make_activity_event(summary: str) -> dict[str, Any]:
    return {
        "id": f"evt_{secrets.token_hex(6)}",
        "ts": "2026-07-02T10:00:00Z",
        "rel": "now",
        "day": "Today",
        "kind": "settings",
        "sev": "info",
        "actor": {"user": "operator", "role": "operator"},
        "target": {"type": "folder", "label": "lifecycle"},
        "summary": summary,
        "meta": {},
    }


@pytest.fixture()
def catalog_env(monkeypatch):
    """Single env-seeded default folder, no runtime file, clean stores."""
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {
                    "id": "default",
                    "label": "Default folder",
                    "kind": "primary",
                    "description": "SRE seed",
                }
            ]
        ),
    )
    monkeypatch.setenv("TWIN_MAX_FOLDERS", "5")
    monkeypatch.delenv("TWIN_FOLDERS_RUNTIME_FILE", raising=False)
    folder_store.reset_runtime_store()
    webui_router.reset_store()
    yield
    folder_store.reset_runtime_store()
    webui_router.reset_store()


@pytest.fixture()
def quiet_backend_init(monkeypatch):
    """Replace the (Memgraph-touching) backend init with a recording no-op."""
    calls: list[WebuiStore] = []

    async def _fake_init(store: WebuiStore) -> None:
        calls.append(store)

    monkeypatch.setattr(store_module, "initialize_store_backends", _fake_init)
    return calls


async def _drain_pending_inits() -> None:
    pending = list(store_module._pending_backend_inits)  # noqa: SLF001
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


# ---------------------------------------------------------------------------
# MG-3 — lazy store construction inherits the deployment mode (unit)
# ---------------------------------------------------------------------------


class TestLazyStoreModeInheritance:
    async def test_runtime_folder_inherits_memgraph_backends(
        self, catalog_env, quiet_backend_init
    ):
        """The MG-3 defect: a folder created after boot must NOT get in-RAM
        stores when the deployment booted in memgraph mode."""
        _register_memgraph_template("default")

        store = get_store("runtimef")
        await _drain_pending_inits()

        assert store.mode == "memgraph"
        assert isinstance(store.tags, MemgraphTagStore)
        assert isinstance(store.activity, MemgraphActivityStore)
        assert isinstance(store.notifications, MemgraphNotificationStore)
        # Backends must be scoped to the NEW folder, not the template's.
        assert store.tags._workspace == "runtimef"  # noqa: SLF001
        assert store.activity._workspace == "runtimef"  # noqa: SLF001
        assert store.notifications._workspace == "runtimef"  # noqa: SLF001
        # Init was scheduled from the sync path.
        assert store in quiet_backend_init

    async def test_partial_backend_wiring_is_mirrored(
        self, catalog_env, quiet_backend_init
    ):
        """`server/app.py` wires per-setting subsets — mirror exactly those."""
        template = WebuiStore.for_folder("default", mode="memgraph")
        template._tag_backend = MemgraphTagStore(workspace="default")  # noqa: SLF001
        set_store(template, folder="default")

        store = get_store("runtimef")
        await _drain_pending_inits()

        assert store.mode == "memgraph"
        assert isinstance(store.tags, MemgraphTagStore)
        assert isinstance(store.activity, InMemoryActivityStore)
        assert isinstance(store.notifications, InMemoryNotificationStore)

    async def test_seed_mode_still_yields_seed_stores(self, catalog_env):
        """Compat doctrine: with the feature off (seed deployment), lazy
        construction is byte-identical to the historical behavior."""
        assert deployment_store_mode() == "seed"

        store = get_store("runtimef")

        assert store.mode == "seed"
        assert isinstance(store.tags, InMemoryTagStore)
        assert isinstance(store.activity, InMemoryActivityStore)
        assert isinstance(store.notifications, InMemoryNotificationStore)
        assert not store_module._pending_backend_inits  # noqa: SLF001

    async def test_deployment_store_mode_helper(self, catalog_env):
        assert deployment_store_mode() == "seed"
        _register_memgraph_template("default")
        assert deployment_store_mode() == "memgraph"

    async def test_ensure_folder_store_awaits_init(
        self, catalog_env, quiet_backend_init
    ):
        _register_memgraph_template("default")

        store = await ensure_folder_store("runtimef")

        assert store.mode == "memgraph"
        assert quiet_backend_init == [store]
        # Registered: subsequent get_store returns the same object.
        assert get_store("runtimef") is store

    async def test_ensure_folder_store_seed_mode_no_init(
        self, catalog_env, quiet_backend_init
    ):
        store = await ensure_folder_store("runtimef")
        assert store.mode == "seed"
        assert quiet_backend_init == []


# ---------------------------------------------------------------------------
# MG-4 — delete guard consults the database (unit, probe monkeypatched)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def client(catalog_env, quiet_backend_init):
    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


class TestDeleteGuardMemgraphMode:
    async def _provision(self, client, folder_id: str = "sandbox") -> None:
        r = await client.post("/folders", json={"id": folder_id, "label": "S"})
        assert r.status_code == 201

    async def test_delete_409_when_db_has_memberships(self, client, monkeypatch):
        _register_memgraph_template("default")
        await self._provision(client)

        async def _probe(folder_id: str) -> dict[str, int]:
            return {"documents": 2, "graph entities": 0, "tags": 0}

        async def _cleanup(folder_id: str) -> None:
            raise AssertionError("cleanup must not run when the guard refuses")

        monkeypatch.setattr(routes_folders, "_memgraph_residual_data", _probe)
        monkeypatch.setattr(
            routes_folders, "_cleanup_memgraph_folder_residue", _cleanup
        )

        r = await client.delete("/folders/sandbox")

        assert r.status_code == 409
        assert "2 documents" in r.json()["detail"]
        # Folder must survive the refused delete.
        ids = [f["id"] for f in (await client.get("/folders")).json()]
        assert "sandbox" in ids

    async def test_delete_503_when_probe_fails(self, client, monkeypatch):
        """Fail-closed: an unverifiable folder is never deleted."""
        _register_memgraph_template("default")
        await self._provision(client)

        async def _probe(folder_id: str) -> dict[str, int]:
            raise RuntimeError("memgraph down")

        monkeypatch.setattr(routes_folders, "_memgraph_residual_data", _probe)

        r = await client.delete("/folders/sandbox")

        assert r.status_code == 503
        ids = [f["id"] for f in (await client.get("/folders")).json()]
        assert "sandbox" in ids

    async def test_delete_204_runs_residue_cleanup(self, client, monkeypatch):
        _register_memgraph_template("default")
        await self._provision(client)

        cleaned: list[str] = []

        async def _probe(folder_id: str) -> dict[str, int]:
            return {"documents": 0, "graph entities": 0, "tags": 0}

        async def _cleanup(folder_id: str) -> None:
            cleaned.append(folder_id)

        monkeypatch.setattr(routes_folders, "_memgraph_residual_data", _probe)
        monkeypatch.setattr(
            routes_folders, "_cleanup_memgraph_folder_residue", _cleanup
        )

        r = await client.delete("/folders/sandbox")

        assert r.status_code == 204
        assert cleaned == ["sandbox"]
        ids = [f["id"] for f in (await client.get("/folders")).json()]
        assert "sandbox" not in ids

    async def test_delete_seed_mode_never_probes_db(self, client, monkeypatch):
        """Compat doctrine: seed deployments keep the pure in-process guard."""

        async def _probe(folder_id: str) -> dict[str, int]:
            raise AssertionError("DB probe must not run in seed mode")

        monkeypatch.setattr(routes_folders, "_memgraph_residual_data", _probe)
        await self._provision(client)

        r = await client.delete("/folders/sandbox")

        assert r.status_code == 204


# ---------------------------------------------------------------------------
# Integration — real Memgraph
# ---------------------------------------------------------------------------


async def _run_write(query: str, **params: Any) -> None:
    from twindb_lightrag_memgraph import _pool

    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            result = await session.run(query, **params)
            await result.consume()


async def _run_count(query: str, **params: Any) -> int:
    from twindb_lightrag_memgraph import _pool

    async with _pool.get_read_session() as session:
        result = await session.run(query, **params)
        record = await result.single()
        await result.consume()
        return int(record["c"]) if record else 0


async def _drop_folder_labels(folder_id: str) -> None:
    for label in (
        f"WebuiTag_{folder_id}",
        f"WebuiTagCategory_{folder_id}",
        f"WebuiActivity_{folder_id}",
        f"WebuiNotification_{folder_id}",
    ):
        await _run_write(f"MATCH (n:`{label}`) DETACH DELETE n")


@pytest.fixture()
def mg_env(monkeypatch):
    """Isolated workspace + uuid'd env-seeded default folder (siblings share
    the Memgraph instance)."""
    ws = f"mg34ws_{secrets.token_hex(4)}"
    default_id = f"dflt_{secrets.token_hex(4)}"
    monkeypatch.setenv("MEMGRAPH_WORKSPACE", ws)
    monkeypatch.setenv("WORKSPACE", ws)
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", default_id)
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps([{"id": default_id, "label": "Default", "kind": "primary"}]),
    )
    monkeypatch.setenv("TWIN_MAX_FOLDERS", "5")
    monkeypatch.delenv("TWIN_FOLDERS_RUNTIME_FILE", raising=False)
    folder_store.reset_runtime_store()
    webui_router.reset_store()
    yield ws, default_id
    folder_store.reset_runtime_store()
    webui_router.reset_store()


@pytest.mark.integration
class TestRuntimeFolderStorePersistence:
    """MG-3 proof: an event written through a runtime-created folder's store
    is readable through a FRESH store instance on the same Memgraph."""

    async def test_activity_event_survives_store_instance(self, mg_env):
        ws, default_id = mg_env
        _register_memgraph_template(default_id)
        folder_id = f"rtf_{secrets.token_hex(4)}"
        try:
            store = await ensure_folder_store(folder_id)
            assert store.mode == "memgraph"

            event = _make_activity_event("runtime folder lifecycle event")
            await store.record_activity(event)

            # Fresh backend — a restart / another worker sees the event.
            fresh = MemgraphActivityStore(workspace=folder_id)
            items, total, _ = await fresh.list()
            assert total == 1
            assert items[0]["id"] == event["id"]
            assert items[0]["summary"] == "runtime folder lifecycle event"
        finally:
            await _drop_folder_labels(folder_id)
            await _drop_folder_labels(default_id)

    async def test_notifications_and_categories_provisioned(self, mg_env):
        ws, default_id = mg_env
        _register_memgraph_template(default_id)
        folder_id = f"rtf_{secrets.token_hex(4)}"
        try:
            store = await ensure_folder_store(folder_id)

            await store.push_notification(
                {
                    "id": f"ntf_{secrets.token_hex(4)}",
                    "kind": "info",
                    "title": "hello",
                    "detail": "persisted",
                }
            )
            fresh_notifs = MemgraphNotificationStore(workspace=folder_id)
            items = await fresh_notifs.list()
            assert any(n.get("title") == "hello" for n in items)

            # ensure_folder_store bootstrapped the governance taxonomy.
            fresh_tags = MemgraphTagStore(workspace=folder_id)
            categories = await fresh_tags.list_categories()
            assert len(categories) > 0
        finally:
            await _drop_folder_labels(folder_id)
            await _drop_folder_labels(default_id)


@pytest.mark.integration
class TestFolderDeleteLifecycleIntegration:
    """MG-4 end-to-end: guard sees real MEMBER_OF data; permitted delete
    strands nothing."""

    @pytest.fixture()
    async def mg_client(self, mg_env):
        ws, default_id = mg_env
        app = create_app()
        _register_memgraph_template(default_id)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c, ws, default_id
        await _drop_folder_labels(default_id)

    async def test_membership_blocks_delete_then_cleanup(self, mg_client):
        client, ws, _default_id = mg_client
        folder_id = f"rt_{secrets.token_hex(4)}"
        doc_id = f"doc_{secrets.token_hex(4)}"
        try:
            r = await client.post("/folders", json={"id": folder_id, "label": "RT"})
            assert r.status_code == 201

            # Real membership, the surface the old guard never saw.
            await _run_write(
                f"MERGE (f:`Folder_{ws}` {{id: $folder}}) "
                f"MERGE (d:`DocStatus_{ws}` {{id: $doc}}) "
                "MERGE (d)-[:MEMBER_OF]->(f)",
                folder=folder_id,
                doc=doc_id,
            )

            r = await client.delete(f"/folders/{folder_id}")
            assert r.status_code == 409
            assert "1 documents" in r.json()["detail"]

            # Remove the membership → delete is permitted.
            await _run_write(
                f"MATCH (d:`DocStatus_{ws}` {{id: $doc}}) DETACH DELETE d",
                doc=doc_id,
            )
            r = await client.delete(f"/folders/{folder_id}")
            assert r.status_code == 204

            # Nothing stranded: store labels + Folder node are gone.
            for label in (
                f"WebuiTag_{folder_id}",
                f"WebuiTagCategory_{folder_id}",
                f"WebuiActivity_{folder_id}",
                f"WebuiNotification_{folder_id}",
            ):
                assert (
                    await _run_count(f"MATCH (n:`{label}`) RETURN count(n) AS c") == 0
                ), f"stranded label {label}"
            assert (
                await _run_count(
                    f"MATCH (f:`Folder_{ws}` {{id: $folder}}) " "RETURN count(f) AS c",
                    folder=folder_id,
                )
                == 0
            )
        finally:
            await _run_write(
                f"MATCH (d:`DocStatus_{ws}` {{id: $doc}}) DETACH DELETE d",
                doc=doc_id,
            )
            await _run_write(
                f"MATCH (f:`Folder_{ws}` {{id: $folder}}) DETACH DELETE f",
                folder=folder_id,
            )
            await _drop_folder_labels(folder_id)

    async def test_tags_block_delete(self, mg_client):
        client, ws, _default_id = mg_client
        folder_id = f"rt_{secrets.token_hex(4)}"
        try:
            r = await client.post("/folders", json={"id": folder_id, "label": "RT"})
            assert r.status_code == 201

            await _run_write(
                f"MERGE (t:`WebuiTag_{folder_id}` {{id: $id}}) " "SET t.data = $data",
                id="scoped",
                data=json.dumps({"tag": "scoped"}),
            )

            r = await client.delete(f"/folders/{folder_id}")
            assert r.status_code == 409
            assert "tags" in r.json()["detail"]

            await _run_write(f"MATCH (t:`WebuiTag_{folder_id}`) DETACH DELETE t")
            r = await client.delete(f"/folders/{folder_id}")
            assert r.status_code == 204
        finally:
            await _run_write(
                f"MATCH (f:`Folder_{ws}` {{id: $folder}}) DETACH DELETE f",
                folder=folder_id,
            )
            await _drop_folder_labels(folder_id)
