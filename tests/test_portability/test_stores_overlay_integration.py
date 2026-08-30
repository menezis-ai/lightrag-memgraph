"""T1.3 overlay round-trips against a real Memgraph."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph.portability.stores import PortabilityError, Scope
from twindb_lightrag_memgraph.portability.stores_overlay import (
    ActivityStore,
    SettingsStore,
    SourceLinkStore,
    TagCategoryStore,
    TagStore,
)
from twindb_lightrag_memgraph.server.source_links_store import (
    MemgraphSourceLinkStore,
)
from twindb_lightrag_memgraph.server.vision_settings_store import update_settings
from twindb_lightrag_memgraph.server.webui_activitystore import MemgraphActivityStore
from twindb_lightrag_memgraph.server.webui_tagstore import MemgraphTagStore

twindb_lightrag_memgraph.register()
pytestmark = pytest.mark.integration

WS_A, WS_B = "overlay_a", "overlay_b"
SOURCE_FOLDERS = ("of1", "of2")
TARGET_FOLDERS = ("og1", "og2")
FOLDER_MAP = dict(zip(SOURCE_FOLDERS, TARGET_FOLDERS, strict=True))
BUNDLE_ID = "47ffb8d7-8946-4f7a-9d32-f856414c46e7"


async def _run(query: str, **params: Any) -> list[dict[str, Any]]:
    async with _pool.get_session() as session:
        result = await session.run(query, **params)
        rows = [dict(record) async for record in result]
        await result.consume()
    return rows


async def _collect(iterator: AsyncIterator[dict]) -> list[dict]:
    return [record async for record in iterator]


async def _aiter(records: list[dict]) -> AsyncIterator[dict]:
    for record in records:
        yield record


async def _wipe() -> None:
    for label in (
        *(f"WebuiTag_{folder}" for folder in SOURCE_FOLDERS + TARGET_FOLDERS),
        *(f"WebuiTagCategory_{folder}" for folder in SOURCE_FOLDERS + TARGET_FOLDERS),
        *(f"WebuiActivity_{folder}" for folder in SOURCE_FOLDERS + TARGET_FOLDERS),
        f"WebuiSettings_{WS_A}",
        f"WebuiSettings_{WS_B}",
        f"TwinSourceLink_{WS_A}",
        f"TwinSourceLink_{WS_B}",
    ):
        await _run(f"MATCH (n:`{label}`) DETACH DELETE n")


@pytest.fixture
async def seeded_overlay():
    await _wipe()
    for folder, label in zip(SOURCE_FOLDERS, ("One", "Two"), strict=True):
        tags = MemgraphTagStore(workspace=folder)
        await tags.initialize()
        await tags.upsert_tag(
            {
                "tag": "shared",
                "label": f"Shared {label}",
                "status": "approved",
                "category": "ops",
            }
        )
        await tags._write_many(
            tags._cat_label,
            "id",
            [{"id": "ops", "label": f"Operations {label}", "color": "#123456"}],
        )
        activity = MemgraphActivityStore(workspace=folder)
        await activity.initialize()
        await activity.append(
            {
                "id": f"event-{folder}",
                "kind": "document.updated",
                "sev": "info",
                "actor": {"user": "tester"},
                "target": {"id": f"doc-{folder}", "label": label},
                "summary": f"Updated {label}",
                "ts": "2026-08-25T10:00:00Z",
            }
        )

    await update_settings(
        WS_A,
        min_ocr_chars=33,
        drop_classes=["logo", "signature"],
        procedure_enabled=True,
        updated_by="admin",
    )
    links = MemgraphSourceLinkStore(WS_A)
    await links.initialize()
    await links.create(
        {
            "id": "link-1",
            "doc_id": "doc-of1",
            "url": "https://docs.example.invalid/runbook",
            "label": "Runbook",
            "created_by": "admin",
            "created_at": "2026-08-25T10:00:00Z",
            "updated_by": "admin",
            "updated_at": "2026-08-25T10:00:00Z",
            "version": 1,
            "deleted": False,
            "deleted_by": None,
            "deleted_at": None,
        }
    )

    yield Scope(
        workspace=WS_A,
        folder_ids=SOURCE_FOLDERS,
        bundle_id=BUNDLE_ID,
    ), Scope(
        workspace=WS_B,
        folder_ids=TARGET_FOLDERS,
        folder_map=FOLDER_MAP,
        bundle_id=BUNDLE_ID,
    )
    await _wipe()


async def test_folder_scoped_tags_categories_and_activity_round_trip(seeded_overlay):
    source, target = seeded_overlay

    tags = TagStore()
    tag_records = await _collect(tags.export_records(source))
    assert [(r["folder_id"], r["id"]) for r in tag_records] == [
        ("of1", "shared"),
        ("of2", "shared"),
    ]
    assert await tags.import_records(_aiter(tag_records), target) == 2
    restored_tags = await _collect(tags.export_records(target))
    assert [r["folder_id"] for r in restored_tags] == ["og1", "og2"]
    assert [r["value"]["label"] for r in restored_tags] == [
        "Shared One",
        "Shared Two",
    ]

    categories = TagCategoryStore()
    category_records = await _collect(categories.export_records(source))
    assert await categories.import_records(_aiter(category_records), target) == 2
    restored_categories = await _collect(categories.export_records(target))
    assert [r["value"]["label"] for r in restored_categories] == [
        "Operations One",
        "Operations Two",
    ]

    activity = ActivityStore()
    activity_records = await _collect(activity.export_records(source))
    assert all(record["origin"] == {"workspace": WS_A} for record in activity_records)
    assert await activity.import_records(_aiter(activity_records), target) == 2
    restored_activity = await _collect(activity.export_records(target))
    assert [r["folder_id"] for r in restored_activity] == ["og1", "og2"]
    assert [r["origin"] for r in restored_activity] == [
        {"workspace": WS_A},
        {"workspace": WS_A},
    ]


async def test_settings_and_source_links_round_trip(seeded_overlay):
    source, target = seeded_overlay

    settings = SettingsStore()
    records = await _collect(settings.export_records(source))
    assert len(records) == 1 and records[0]["id"] == "vision"
    assert await settings.import_records(_aiter(records), target) == 1
    assert await _collect(settings.export_records(target)) == records

    links = SourceLinkStore()
    records = await _collect(links.export_records(source))
    assert records[0]["deleted_by"] is None and records[0]["deleted_at"] is None
    assert await links.import_records(_aiter(records), target) == 1
    assert await _collect(links.export_records(target)) == records
    # Same bundle replay is idempotent.
    assert await links.import_records(_aiter(records), target) == 1
    assert await links.count(target) == 1


async def test_settings_nested_allow_list_refuses_endpoint(seeded_overlay):
    source, _ = seeded_overlay
    await _run(
        f"MATCH (n:`WebuiSettings_{WS_A}` {{id: 'vision'}}) " "SET n.data = $data",
        data='{"min_ocr_chars": 1, "endpoint": "https://secret.invalid"}',
    )
    with pytest.raises(PortabilityError, match="endpoint"):
        await _collect(SettingsStore().export_records(source))
