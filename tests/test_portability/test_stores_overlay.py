"""T1.3 unit contracts for file-backed overlay stores."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import pytest

from twindb_lightrag_memgraph import _procedure_store
from twindb_lightrag_memgraph.portability.bundle import BundleWriter
from twindb_lightrag_memgraph.portability.stores import Scope
from twindb_lightrag_memgraph.portability.stores_overlay import (
    ActivityStore,
    ProcedureStore,
    RuntimeFolderStore,
)
from twindb_lightrag_memgraph.server import folder_store


async def _collect(iterator: AsyncIterator[dict]) -> list[dict]:
    return [record async for record in iterator]


async def _aiter(records: list[dict]) -> AsyncIterator[dict]:
    for record in records:
        yield record


@pytest.fixture(autouse=True)
def _reset_runtime_folders(monkeypatch):
    monkeypatch.delenv("TWIN_FOLDERS_RUNTIME_FILE", raising=False)
    folder_store.reset_runtime_store()
    yield
    folder_store.reset_runtime_store()


async def test_runtime_folders_round_trip_with_folder_map():
    folder_store.add_runtime_folder(
        folder_id="rf1",
        label="Runtime one",
        kind="team",
        description="A mutable folder",
        sources=7,
    )
    store = RuntimeFolderStore()
    source = Scope(workspace="base", folder_ids=("rf1",))
    records = await _collect(store.export_records(source))
    assert records == [
        {
            "id": "rf1",
            "label": "Runtime one",
            "kind": "team",
            "description": "A mutable folder",
            "sources": 7,
        }
    ]

    folder_store.reset_runtime_store()
    target = Scope(workspace="restored", folder_ids=("rf9",), folder_map={"rf1": "rf9"})
    assert await store.import_records(_aiter(records), target) == 1
    restored = folder_store.get_runtime_folder("rf9")
    assert restored is not None
    assert restored.as_runtime_config() == {**records[0], "id": "rf9"}
    # Exact replay is idempotent.
    assert await store.import_records(_aiter(records), target) == 1


async def test_procedure_round_trip_preserves_id_state_and_png(monkeypatch, tmp_path):
    source_file = tmp_path / "source-procedures.json"
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", str(source_file))
    png = b"\x89PNG\r\n\x1a\nportable"
    bundle_id = _procedure_store.create_bundle(
        file_name="runbook.pdf",
        original_path="/input/runbook.pdf",
        track_id="track-1",
        state="pending",
        reason="review",
        source="forced",
        folder="f1",
        content_hash="abc",
        full_text="procedure text",
        schematics=[
            {
                "page": 2,
                "png_base64": base64.b64encode(png).decode("ascii"),
                "blind": {"tasks": []},
            }
        ],
        classification={"class_id": "C2"},
        schematics_total=1,
        operator_classification="C2",
    )
    source_record = _procedure_store.get_bundle(bundle_id)
    assert source_record is not None

    writer = BundleWriter(tmp_path / "bundle")
    portable = ProcedureStore(bundle_writer=writer)
    records = await _collect(portable.export_records(Scope(workspace="base")))
    assert len(records) == 1 and records[0]["id"] == bundle_id
    file_ref = records[0]["schematics"][0]["file"]
    assert "png_base64" not in records[0]["schematics"][0]
    assert (writer.root / file_ref).read_bytes() == png

    target_file = tmp_path / "target-procedures.json"
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", str(target_file))
    importer = ProcedureStore(bundle_root=writer.root)
    assert (
        await importer.import_records(_aiter(records), Scope(workspace="target")) == 1
    )
    assert _procedure_store.get_bundle(bundle_id) == source_record
    # Exact replay does not duplicate or rewrite the record.
    assert (
        await importer.import_records(_aiter(records), Scope(workspace="target")) == 1
    )
    assert len(_procedure_store.list_bundles()) == 1


async def test_procedure_degraded_store_refuses_export(monkeypatch, tmp_path):
    store_file = tmp_path / "procedures.json"
    store_file.write_text("not-json", encoding="utf-8")
    monkeypatch.setenv("TWIN_PROCEDURE_STORE_FILE", str(store_file))
    store = ProcedureStore(bundle_writer=BundleWriter(tmp_path / "bundle"))
    with pytest.raises(_procedure_store.StoreDegradedError):
        await _collect(store.export_records(Scope(workspace="base")))


async def test_activity_semantic_records_ignore_transport_bundle_id(monkeypatch):
    from twindb_lightrag_memgraph.portability import stores_overlay

    event = {
        "id": "event-1",
        "kind": "document.updated",
        "summary": "Updated",
    }

    async def rows(_query: str, **params):
        if params.get("after") is not None:
            return []
        return [{"p": {"id": "event-1", "data": json.dumps(event)}}]

    monkeypatch.setattr(stores_overlay, "read_rows", rows)
    store = ActivityStore()
    first = await _collect(
        store.export_records(
            Scope(workspace="base", folder_ids=("f1",), bundle_id="bundle-one")
        )
    )
    second = await _collect(
        store.export_records(
            Scope(workspace="base", folder_ids=("f1",), bundle_id="bundle-two")
        )
    )

    assert first == second
    assert first[0]["origin"] == {"workspace": "base"}
