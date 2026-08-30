"""T2.4 — count/index/folder/state-hash validation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import threading

import pytest

from twindb_lightrag_memgraph.portability import validate
from twindb_lightrag_memgraph.portability.bundle import BundleReader

from .test_bundle import _build


class FakeCountStore:
    def __init__(self, name: str, counts: dict[str, int]):
        self.name = name
        self.counts = counts

    async def count(self, _scope):
        return self.counts[self.name]


def test_normalized_state_hash_maps_folder_scoped_records(tmp_path):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    with BundleReader(tmp_path / "bundle") as reader:
        assert reader.inspect().ok
        identity = validate.normalized_source_state_hash(reader, {"f1": "f1"})
        mapped = validate.normalized_source_state_hash(reader, {"f1": "g1"})
    assert identity == manifest.state_hash
    assert mapped != manifest.state_hash


def test_procedure_normalization_maps_primary_and_duplicate_folders():
    record = {
        "id": "p1",
        "folder": "f1",
        "duplicate_requests": [
            {"path": "/a", "folder": "f1"},
            {"path": "/b", "folder": None},
        ],
    }
    mapped = validate._normalized_record("procedures", record, {"f1": "g1"})
    assert mapped["folder"] == "g1"
    assert mapped["duplicate_requests"][0]["folder"] == "g1"
    assert mapped["duplicate_requests"][1]["folder"] is None


def test_graph_edge_normalization_maps_manual_relation_folder_stamp():
    record = {
        "src": "a",
        "tgt": "b",
        "props": {"twin_folder_json": json.dumps(["staging"])},
    }
    mapped = validate._normalized_record(
        "graph.edges", record, {"staging": "production"}
    )
    assert json.loads(mapped["props"]["twin_folder_json"]) == ["production"]


async def test_validation_cancellation_waits_for_extraction_then_closes(
    monkeypatch, tmp_path
):
    extraction_started = threading.Event()
    release_extraction = threading.Event()
    reader_closed = threading.Event()

    class SlowReader:
        def __init__(self, _source):
            pass

        def __enter__(self):
            extraction_started.set()
            release_extraction.wait(timeout=2)
            return self

        def close(self):
            reader_closed.set()

    monkeypatch.setattr(validate, "BundleReader", SlowReader)
    task = asyncio.create_task(
        validate.validate_import(tmp_path / "bundle.tar.gz", workspace="target")
    )
    assert await asyncio.to_thread(extraction_started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_extraction.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reader_closed.is_set()


async def test_validation_nominal_and_count_regression(monkeypatch, tmp_path):
    manifest, _ = _build(tmp_path / "bundle", with_png=True)
    expected = {
        entry.store: entry.records
        for entry in manifest.files
        if entry.path.endswith(".jsonl")
    }
    counts = dict(expected)

    monkeypatch.setattr(
        validate,
        "portable_store",
        lambda spec: FakeCountStore(spec.name, counts),
    )

    index_rows = [
        {
            "index_name": f"vec_target_{namespace}",
            "label": f":Vec_target_{namespace}",
            "property": "embedding",
            "metric": "cos",
            "dimension": manifest.embedding.dim,
            "capacity": 100_000,
            "size": expected[f"vec.{namespace}"],
        }
        for namespace in ("chunks", "entities", "relationships")
    ]

    async def rows(query, **_params):
        assert query == "SHOW VECTOR INDEX INFO"
        return index_rows

    async def scalar(_query, **_params):
        return 0

    current_reexport = manifest

    async def reexport(*_args, **_kwargs):
        return current_reexport

    monkeypatch.setattr(validate, "read_rows", rows)
    monkeypatch.setattr(validate, "read_scalar", scalar)
    monkeypatch.setattr(validate, "export_kb", reexport)

    nominal = await validate.validate_import(tmp_path / "bundle", workspace="target")
    assert nominal["ok"] is True, nominal["problems"]
    assert nominal["procedure_files"][0]["ok"] is True

    procedure = next(
        entry for entry in manifest.files if entry.path.startswith("files/procedures/")
    )
    current_reexport = replace(
        manifest,
        files=[
            replace(entry, sha256="f" * 64) if entry.path == procedure.path else entry
            for entry in manifest.files
        ],
    )
    changed_file = await validate.validate_import(
        tmp_path / "bundle", workspace="target"
    )
    assert changed_file["ok"] is False
    assert any("procedure file" in problem for problem in changed_file["problems"])

    current_reexport = manifest
    counts["docstatus"] = 1
    regressed = await validate.validate_import(tmp_path / "bundle", workspace="target")
    assert regressed["ok"] is False
    assert any("docstatus" in problem for problem in regressed["problems"])

    counts["docstatus"] = expected["docstatus"]
    index_rows[0]["metric"] = "l2"
    wrong_index = await validate.validate_import(
        tmp_path / "bundle", workspace="target"
    )
    assert wrong_index["ok"] is False
    assert any("metric" in problem for problem in wrong_index["problems"])
