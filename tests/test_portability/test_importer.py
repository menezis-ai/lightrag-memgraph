"""T2.3 — approved apply, stale-report gate, checkpoints and resume."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncIterator

import pytest

from twindb_lightrag_memgraph.portability import importer, plan
from twindb_lightrag_memgraph.portability.canonical import jcs_dumps, jcs_sha256
from twindb_lightrag_memgraph.portability.stores import store_by_name

from .test_bundle import _build


class FakeImportStore:
    def __init__(self, name: str, calls: list[str], fail_on: str | None = None):
        self.spec = store_by_name(name)
        self.calls = calls
        self.fail_on = fail_on

    async def import_records(self, records: AsyncIterator[dict], scope) -> int:
        self.calls.append(self.spec.name)
        count = 0
        async for _record in records:
            count += 1
        if self.spec.name == self.fail_on:
            raise RuntimeError("injected interruption")
        return count


def _approved(manifest):
    return plan.seal_report(
        {
            "format": plan.REPORT_FORMAT,
            "format_version": plan.REPORT_VERSION,
            "created_at": "2026-08-26T10:00:00Z",
            "bundle": {
                "bundle_id": manifest.bundle_id,
                "manifest_hash": manifest.manifest_hash,
                "state_hash": manifest.state_hash,
                "source_workspace": manifest.source["workspace"],
            },
            "target": {"workspace": "target"},
            "folders": {
                "requested_mapping": {"f1": "f1"},
                "effective_mapping": {"f1": "f1"},
            },
            "compat": [],
            "classification": {},
            "capacity": [],
            "stats": {},
            "options": {"allow_unverified": False},
            "blocking": [],
            "report_hash": "",
        }
    )


async def test_apply_uses_ordered_stores_and_replay_resumes_without_writes(
    monkeypatch, tmp_path
):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    report = _approved(manifest)
    report_path = plan.write_report(tmp_path / "report.json", report)
    checkpoint = tmp_path / "checkpoint.json"
    calls: list[str] = []

    async def current(*_args, **_kwargs):
        return report

    async def sweep(_workspace):
        calls.append("__sweep__")
        return {}

    monkeypatch.setattr(importer, "create_dry_run", current)
    monkeypatch.setattr(
        importer,
        "portable_store",
        lambda spec, **_kwargs: FakeImportStore(spec.name, calls),
    )
    from twindb_lightrag_memgraph.server import graph_reader

    monkeypatch.setattr(graph_reader, "sweep_stale_source_refs", sweep)

    first = await importer.apply_import(
        tmp_path / "bundle",
        report_path=report_path,
        checkpoint_path=checkpoint,
    )
    assert first["ok"] is True and first["resumed"] is False
    expected = [name for name in importer.IMPORT_ORDER if name in first["imported"]]
    assert calls[:-1] == expected
    assert calls[-1] == "__sweep__"

    calls.clear()
    replay = await importer.apply_import(
        tmp_path / "bundle",
        report_path=report_path,
        checkpoint_path=checkpoint,
    )
    assert replay["resumed"] is True
    assert calls == []


async def test_stale_report_refuses_before_first_store(monkeypatch, tmp_path):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    report = _approved(manifest)
    report_path = plan.write_report(tmp_path / "report.json", report)
    calls: list[str] = []

    async def changed(*_args, **_kwargs):
        return {**report, "report_hash": "f" * 64}

    monkeypatch.setattr(importer, "create_dry_run", changed)
    monkeypatch.setattr(
        importer,
        "portable_store",
        lambda spec, **_kwargs: FakeImportStore(spec.name, calls),
    )
    with pytest.raises(importer.StaleReportError, match="target changed"):
        await importer.apply_import(
            tmp_path / "bundle",
            report_path=report_path,
            checkpoint_path=tmp_path / "checkpoint.json",
        )
    assert calls == []


async def test_apply_opens_only_the_persisted_approved_report_hash(tmp_path):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    report = _approved(manifest)
    report_path = plan.write_report(tmp_path / "report.json", report)

    with pytest.raises(plan.ReportError, match="persisted approval"):
        await importer.apply_import(
            tmp_path / "bundle",
            report_path=report_path,
            checkpoint_path=tmp_path / "checkpoint.json",
            approved_report_hash="f" * 64,
        )


async def test_apply_cancellation_waits_for_extraction_then_closes(
    monkeypatch, tmp_path
):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    report = _approved(manifest)
    report_path = plan.write_report(tmp_path / "report.json", report)
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

    monkeypatch.setattr(importer, "BundleReader", SlowReader)
    task = asyncio.create_task(
        importer.apply_import(
            tmp_path / "bundle.tar.gz",
            report_path=report_path,
            checkpoint_path=tmp_path / "checkpoint.json",
        )
    )
    assert await asyncio.to_thread(extraction_started.wait, 1)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_extraction.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reader_closed.is_set()


async def test_interrupted_store_is_replayed_from_last_checkpoint(
    monkeypatch, tmp_path
):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    report = _approved(manifest)
    report_path = plan.write_report(tmp_path / "report.json", report)
    checkpoint = tmp_path / "checkpoint.json"
    calls: list[str] = []

    async def current(*_args, **_kwargs):
        return report

    monkeypatch.setattr(importer, "create_dry_run", current)
    monkeypatch.setattr(
        importer,
        "portable_store",
        lambda spec, **_kwargs: FakeImportStore(spec.name, calls, fail_on="docstatus"),
    )
    with pytest.raises(RuntimeError, match="injected"):
        await importer.apply_import(
            tmp_path / "bundle",
            report_path=report_path,
            checkpoint_path=checkpoint,
        )
    assert calls[:3] == ["runtime_folders", "folders", "docstatus"]

    calls.clear()
    monkeypatch.setattr(
        importer,
        "portable_store",
        lambda spec, **_kwargs: FakeImportStore(spec.name, calls),
    )
    from twindb_lightrag_memgraph.server import graph_reader

    async def sweep(_workspace):
        return {}

    monkeypatch.setattr(graph_reader, "sweep_stale_source_refs", sweep)
    result = await importer.apply_import(
        tmp_path / "bundle",
        report_path=report_path,
        checkpoint_path=checkpoint,
    )
    assert result["resumed"] is True
    assert calls[0] == "docstatus"
    assert "runtime_folders" not in calls and "folders" not in calls


async def test_apply_refuses_report_with_blockers(tmp_path):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    report = _approved(manifest)
    report["blocking"] = [{"code": "target_not_empty", "message": "no"}]
    report = plan.seal_report(report)
    report_path = plan.write_report(tmp_path / "report.json", report)
    with pytest.raises(importer.ImportRefused, match="blocking"):
        await importer.apply_import(
            tmp_path / "bundle",
            report_path=report_path,
            checkpoint_path=tmp_path / "checkpoint.json",
        )


async def test_apply_validates_workspace_before_choosing_checkpoint(tmp_path):
    manifest, _ = _build(tmp_path / "bundle", with_png=False)
    report = _approved(manifest)
    report["target"]["workspace"] = "../../escape"
    report = plan.seal_report(report)
    report_path = plan.write_report(tmp_path / "report.json", report)

    with pytest.raises(ValueError, match="workspace"):
        await importer.apply_import(tmp_path / "bundle", report_path=report_path)
    assert not (tmp_path / "escape.json").exists()


async def test_resume_refuses_changed_procedure_file_and_manifest(
    monkeypatch, tmp_path
):
    manifest, _ = _build(tmp_path / "bundle", with_png=True)
    report = _approved(manifest)
    report_path = plan.write_report(tmp_path / "report.json", report)
    checkpoint = tmp_path / "checkpoint.json"
    calls: list[str] = []

    async def current(*_args, **_kwargs):
        return report

    async def sweep(_workspace):
        return {}

    monkeypatch.setattr(importer, "create_dry_run", current)
    monkeypatch.setattr(
        importer,
        "portable_store",
        lambda spec, **_kwargs: FakeImportStore(spec.name, calls),
    )
    from twindb_lightrag_memgraph.server import graph_reader

    monkeypatch.setattr(graph_reader, "sweep_stale_source_refs", sweep)
    await importer.apply_import(
        tmp_path / "bundle",
        report_path=report_path,
        checkpoint_path=checkpoint,
    )

    image = tmp_path / "bundle" / "files/procedures/x/1.png"
    changed = image.read_bytes()[:-1] + b"1"
    image.write_bytes(changed)
    manifest_path = tmp_path / "bundle" / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_entry = next(
        entry for entry in data["files"] if entry["path"] == "files/procedures/x/1.png"
    )
    image_entry["sha256"] = hashlib.sha256(changed).hexdigest()
    data["manifest_hash"] = jcs_sha256(
        {key: value for key, value in data.items() if key != "manifest_hash"}
    )
    manifest_path.write_text(jcs_dumps(data) + "\n", encoding="utf-8")
    assert data["state_hash"] == manifest.state_hash

    with pytest.raises(importer.ImportRefused, match="different bundle"):
        await importer.apply_import(
            tmp_path / "bundle",
            report_path=report_path,
            checkpoint_path=checkpoint,
        )
