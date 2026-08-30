"""T2.1/T2.2 — target compatibility and deterministic dry-run reports."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import replace

import pytest

from twindb_lightrag_memgraph.portability import plan
from twindb_lightrag_memgraph.portability.bundle import BundleReader
from twindb_lightrag_memgraph.portability.compat import TargetFacts, check
from twindb_lightrag_memgraph.portability.canonical import jcs_dumps, jcs_sha256
from twindb_lightrag_memgraph.portability.manifest import state_hash_of

from .test_bundle import _build


def _facts(manifest, **overrides):
    base = TargetFacts(
        workspace="target",
        database="memgraph",
        memgraph_version="3.12.0",
        lightrag_version="1.5.6",
        embedding_model="target-model",
        embedding_dim=manifest.embedding.dim,
        embedding_probe=manifest.embedding.probe.vectors,
        vector_capacity=100_000,
        vector_indexes=[],
        classification_ceiling="C2",
        allow_unverified=False,
        pipeline_idle=True,
        store_counts={},
        env_folders=[{"id": "f1", "label": "Target", "kind": "primary"}],
        runtime_folders=[],
        max_folders=5,
    )
    return replace(base, **overrides)


def test_compatibility_matrix_accepts_nominal_and_rejects_probe(tmp_path):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    nominal = check(manifest, _facts(manifest))
    assert all(item["ok"] for item in nominal), nominal

    divergent = [list(vector) for vector in manifest.embedding.probe.vectors]
    divergent[1] = [1.0, -1.0, 1.0, -1.0]
    verdicts = check(manifest, _facts(manifest, embedding_probe=divergent))
    embedding = next(item for item in verdicts if item["dimension"] == "embedding")
    assert embedding["ok"] is False
    assert embedding["target"]["min_cosine"] < 0.999


async def test_dry_run_is_hash_stable_and_requires_explicit_env_mapping(
    monkeypatch, tmp_path
):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)

    async def facts(*_args, **_kwargs):
        return _facts(manifest)

    monkeypatch.setattr(plan, "collect_target_facts", facts)
    blocked = await plan.create_dry_run(tmp_path / "bundle", workspace="target")
    assert any(
        item["code"] == "folder_mapping_required" for item in blocked["blocking"]
    )

    first = await plan.create_dry_run(
        tmp_path / "bundle",
        workspace="target",
        folder_map={"f1": "f1"},
    )
    second = await plan.create_dry_run(
        tmp_path / "bundle",
        workspace="target",
        folder_map={"f1": "f1"},
    )
    assert first["blocking"] == []
    assert first["report_hash"] == second["report_hash"]
    assert first["created_at"] != ""


async def test_bundle_inspection_does_not_block_event_loop(monkeypatch, tmp_path):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    inspection_started = threading.Event()
    release_inspection = threading.Event()
    original_inspect = plan.BundleReader.inspect

    def slow_inspect(reader):
        inspection_started.set()
        release_inspection.wait(timeout=2)
        return original_inspect(reader)

    async def facts(*_args, **_kwargs):
        return _facts(manifest)

    monkeypatch.setattr(plan.BundleReader, "inspect", slow_inspect)
    monkeypatch.setattr(plan, "collect_target_facts", facts)
    task = asyncio.create_task(
        plan.create_dry_run(
            tmp_path / "bundle",
            workspace="target",
            folder_map={"f1": "f1"},
        )
    )
    started_at = asyncio.get_running_loop().time()
    assert await asyncio.to_thread(inspection_started.wait, 2)
    assert asyncio.get_running_loop().time() - started_at < 0.75
    release_inspection.set()
    report = await task
    assert report["blocking"] == []


async def test_dry_run_cancellation_drains_inspection_before_reader_close(
    monkeypatch, tmp_path
):
    _build(tmp_path / "bundle", with_png=False)
    inspection_started = threading.Event()
    release_inspection = threading.Event()
    inspection_finished = threading.Event()
    reader_closed = threading.Event()
    original_inspect = plan.BundleReader.inspect
    original_close = plan.BundleReader.close

    def slow_inspect(reader):
        inspection_started.set()
        release_inspection.wait(timeout=2)
        try:
            return original_inspect(reader)
        finally:
            inspection_finished.set()

    def tracked_close(reader):
        reader_closed.set()
        return original_close(reader)

    monkeypatch.setattr(plan.BundleReader, "inspect", slow_inspect)
    monkeypatch.setattr(plan.BundleReader, "close", tracked_close)
    task = asyncio.create_task(
        plan.create_dry_run(tmp_path / "bundle", workspace="target")
    )
    assert await asyncio.to_thread(inspection_started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not inspection_finished.is_set()
    assert not reader_closed.is_set()

    release_inspection.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert inspection_finished.is_set()
    assert reader_closed.is_set()


async def test_dry_run_cancellation_waits_for_extraction_then_closes(
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

    monkeypatch.setattr(plan, "BundleReader", SlowReader)
    task = asyncio.create_task(
        plan.create_dry_run(tmp_path / "bundle.tar.gz", workspace="target")
    )
    assert await asyncio.to_thread(extraction_started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_extraction.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reader_closed.is_set()


async def test_report_hash_ignores_harmless_probe_float_variation(
    monkeypatch, tmp_path
):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    exact = _facts(manifest)
    varied_vectors = [list(vector) for vector in exact.embedding_probe]
    varied_vectors[0][0] += 0.001
    varied = replace(exact, embedding_probe=varied_vectors)
    calls = iter((exact, varied))

    async def facts(*_args, **_kwargs):
        return next(calls)

    monkeypatch.setattr(plan, "collect_target_facts", facts)
    first = await plan.create_dry_run(
        tmp_path / "bundle",
        workspace="target",
        folder_map={"f1": "f1"},
    )
    second = await plan.create_dry_run(
        tmp_path / "bundle",
        workspace="target",
        folder_map={"f1": "f1"},
    )
    first_probe = next(
        item for item in first["compat"] if item["dimension"] == "embedding"
    )
    second_probe = next(
        item for item in second["compat"] if item["dimension"] == "embedding"
    )
    assert first_probe["ok"] is second_probe["ok"] is True
    assert first_probe["target"]["min_cosine"] != second_probe["target"]["min_cosine"]
    assert "embedding_probe" not in first["target"]
    assert first["report_hash"] == second["report_hash"]


async def test_dry_run_blocks_non_empty_capacity_classification_and_busy_target(
    monkeypatch, tmp_path
):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    root = tmp_path / "bundle"
    manifest_path = root / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    vector_entry = next(
        item for item in manifest_data["files"] if item["store"] == "vec.chunks"
    )
    payload = (jcs_dumps({}) + "\n").encode() * 11
    (root / vector_entry["path"]).write_bytes(payload)
    vector_entry.update(
        records=11, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    manifest_data["state_hash"] = state_hash_of(
        {item["path"]: item["sha256"] for item in manifest_data["files"]}
    )
    manifest_data["manifest_hash"] = jcs_sha256(
        {key: value for key, value in manifest_data.items() if key != "manifest_hash"}
    )
    manifest_path.write_text(jcs_dumps(manifest_data) + "\n")

    async def facts(*_args, **_kwargs):
        return _facts(
            manifest,
            vector_capacity=100_000,
            vector_indexes=[
                {
                    "index_name": "vec_target_chunks",
                    "label": ":Vec_wrong_chunks",
                    "property": "wrong_embedding",
                    "metric": "l2",
                    "dimension": manifest.embedding.dim + 1,
                    "capacity": 5,
                    "size": 0,
                }
            ],
            pipeline_idle=False,
            store_counts={"docstatus": 1},
            classification_ceiling="C1",
        )

    monkeypatch.setattr(plan, "collect_target_facts", facts)
    report = await plan.create_dry_run(
        tmp_path / "bundle",
        workspace="target",
        folder_map={"f1": "f1"},
    )
    codes = {item["code"] for item in report["blocking"]}
    assert {
        "compat_classification",
        "target_not_empty",
        "target_pipeline_busy",
        "vector_capacity",
        "vector_index_label",
        "vector_index_property",
        "vector_index_metric",
        "vector_index_dimension",
    } <= codes
    chunks = next(item for item in report["capacity"] if item["store"] == "vec.chunks")
    assert chunks["capacity"] == 5
    assert chunks["configured_capacity"] == 100_000


def test_report_round_trip_and_tamper_detection(tmp_path):
    report = plan.seal_report(
        {
            "format": plan.REPORT_FORMAT,
            "format_version": plan.REPORT_VERSION,
            "created_at": "2026-08-26T00:00:00Z",
            "bundle": {},
            "target": {},
            "folders": {},
            "compat": [],
            "classification": {},
            "capacity": [],
            "stats": {},
            "options": {},
            "blocking": [],
            "report_hash": "",
        }
    )
    path = plan.write_report(tmp_path / "report.json", report)
    assert plan.load_report(path) == report
    path.write_text(path.read_text().replace('"blocking":[]', '"blocking":[1]'))
    try:
        plan.load_report(path)
    except plan.ReportError as exc:
        assert "report_hash" in str(exc)
    else:  # pragma: no cover - a tampered approval must never parse
        raise AssertionError("tampered report was accepted")


def test_source_runtime_folder_ids_are_read_from_canonical_member(tmp_path):
    _build(tmp_path / "bundle", with_png=False)
    with BundleReader(tmp_path / "bundle") as reader:
        assert reader.inspect().ok
        assert plan._runtime_source_ids(reader) == set()
