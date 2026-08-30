"""Approved, checkpointed KB bundle apply (KB-PORTABILITY-PLAN T2.3).

The first attempt replays the dry-run and compares its deterministic hash
before writing.  Once the first store is committed the target is necessarily
non-empty, so a resume is authorised only by a private checkpoint bound to the
bundle, target workspace and approved report hash.  Every store adapter is
idempotent; a crash after a store commit but before the checkpoint write can
therefore safely replay that store.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from .._constants import (
    resolve_portability_batch_size,
    resolve_portability_dir,
    validate_identifier,
)
from .bundle import (
    BundleError,
    BundleReader,
    close_bundle_reader,
    open_bundle_reader,
    run_reader_io,
)
from .canonical import jcs_dumps, jcs_loads, jcs_sha256
from .jsonl import iter_jsonl
from .plan import ReportError, create_dry_run, load_report
from .stores import Scope, portable_store, store_by_name

IMPORT_ORDER: tuple[str, ...] = (
    "runtime_folders",
    "folders",
    "docstatus",
    "member_of",
    "kv.full_docs",
    "kv.text_chunks",
    "kv.full_entities",
    "kv.full_relations",
    "kv.entity_chunks",
    "kv.relation_chunks",
    "vec.chunks",
    "vec.entities",
    "vec.relationships",
    "graph.nodes",
    "graph.edges",
    "graph.member_of",
    "graph.overrides",
    "tag_categories",
    "tags",
    "tagged_with",
    "settings",
    "source_links",
    "activity",
    "procedures",
)


class ImportRefused(ReportError):
    """An approved apply cannot safely start or resume."""


class StaleReportError(ImportRefused):
    """The target facts changed since the approved dry-run."""


def _checkpoint_hash(data: dict[str, Any]) -> str:
    return jcs_sha256({k: v for k, v in data.items() if k != "checkpoint_hash"})


def _write_checkpoint(path: Path, data: dict[str, Any]) -> None:
    sealed = dict(data)
    sealed["checkpoint_hash"] = _checkpoint_hash(sealed)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".part")
    try:
        temporary.write_text(jcs_dumps(sealed) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = jcs_loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ImportRefused(f"cannot read checkpoint: {exc}") from exc
    expected = {
        "format",
        "bundle_id",
        "manifest_hash",
        "state_hash",
        "report_hash",
        "workspace",
        "folder_map",
        "completed",
        "imported",
        "status",
        "checkpoint_hash",
    }
    if not isinstance(data, dict) or set(data) != expected:
        raise ImportRefused("checkpoint has an unsupported shape")
    if data.get("format") != "twin-kb-import-checkpoint/1":
        raise ImportRefused("unsupported checkpoint format")
    if data.get("checkpoint_hash") != _checkpoint_hash(data):
        raise ImportRefused("checkpoint_hash does not match checkpoint content")
    if not isinstance(data.get("completed"), list) or not isinstance(
        data.get("imported"), dict
    ):
        raise ImportRefused("checkpoint completed/imported fields are invalid")
    completed = data["completed"]
    imported = data["imported"]
    allowed_completed = {*IMPORT_ORDER, "__sweep__"}
    if (
        any(
            not isinstance(item, str) or item not in allowed_completed
            for item in completed
        )
        or len(completed) != len(set(completed))
        or any(
            not isinstance(name, str)
            or name not in IMPORT_ORDER
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for name, count in imported.items()
        )
        or data.get("status") not in {"applying", "applied"}
    ):
        raise ImportRefused("checkpoint progress fields are invalid")
    return data


def default_checkpoint_path(bundle_id: str, workspace: str) -> Path:
    return (
        Path(resolve_portability_dir())
        / "checkpoints"
        / f"{bundle_id}-{workspace}.json"
    )


async def _records(
    reader: BundleReader, path: str, digest: str
) -> AsyncIterator[dict[str, Any]]:
    for record in iter_jsonl(reader.path_of(path), digest):
        yield record


def _assert_binding(
    *,
    report: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    bundle_id: str,
    manifest_hash: str,
    state_hash: str,
    workspace: str,
) -> None:
    bundle = report.get("bundle")
    target = report.get("target")
    if not isinstance(bundle, dict) or not isinstance(target, dict):
        raise ImportRefused("report bundle/target binding is invalid")
    if (
        bundle.get("bundle_id") != bundle_id
        or bundle.get("manifest_hash") != manifest_hash
        or bundle.get("state_hash") != state_hash
    ):
        raise ImportRefused("report was approved for a different bundle")
    if target.get("workspace") != workspace:
        raise ImportRefused("report was approved for a different target workspace")
    if report.get("blocking"):
        raise ImportRefused("approved report still contains blocking findings")
    if checkpoint is None:
        return
    expected = {
        "bundle_id": bundle_id,
        "manifest_hash": manifest_hash,
        "state_hash": state_hash,
        "report_hash": report["report_hash"],
        "workspace": workspace,
        "folder_map": report["folders"]["effective_mapping"],
    }
    differences = {
        key: {"checkpoint": checkpoint.get(key), "expected": value}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if differences:
        raise ImportRefused(f"checkpoint binding mismatch: {differences}")


async def apply_import(
    source: str | Path,
    *,
    report_path: str | Path,
    checkpoint_path: str | Path | None = None,
    embedding_func: Any | None = None,
    batch: int | None = None,
    approved_report_hash: str | None = None,
) -> dict[str, Any]:
    """Apply one approved report, resuming only through its bound checkpoint."""
    report = await asyncio.to_thread(
        load_report,
        report_path,
        expected_report_hash=approved_report_hash,
    )
    target = report.get("target")
    folders = report.get("folders")
    options = report.get("options")
    if not isinstance(target, dict) or not isinstance(folders, dict):
        raise ImportRefused("report target/folders fields are invalid")
    if not isinstance(options, dict):
        raise ImportRefused("report options field is invalid")
    workspace = validate_identifier(str(target.get("workspace") or ""), "workspace")
    requested_raw = folders.get("requested_mapping")
    effective_raw = folders.get("effective_mapping")
    if not isinstance(requested_raw, dict) or not isinstance(effective_raw, dict):
        raise ImportRefused("report folder mappings are invalid")
    requested_map = {
        validate_identifier(str(source), "source folder"): validate_identifier(
            str(destination), "target folder"
        )
        for source, destination in requested_raw.items()
    }
    effective_map = {
        validate_identifier(str(source), "source folder"): validate_identifier(
            str(destination), "target folder"
        )
        for source, destination in effective_raw.items()
    }
    allow_unverified = bool(report.get("options", {}).get("allow_unverified"))
    batch_size = batch if batch is not None else resolve_portability_batch_size()
    if batch_size < 1:
        raise ImportRefused("batch must be >= 1")

    reader = BundleReader(Path(source))
    try:
        await open_bundle_reader(reader)
        inspection = await run_reader_io(reader.inspect)
        if not inspection.ok:
            raise BundleError(
                "bundle integrity failed: " + "; ".join(inspection.problems)
            )
        assert reader.manifest is not None and reader.root is not None
        manifest = reader.manifest
        checkpoint_target = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else default_checkpoint_path(manifest.bundle_id, workspace)
        )
        checkpoint = _load_checkpoint(checkpoint_target)
        _assert_binding(
            report=report,
            checkpoint=checkpoint,
            bundle_id=manifest.bundle_id,
            manifest_hash=manifest.manifest_hash,
            state_hash=manifest.state_hash,
            workspace=workspace,
        )

        resumed = checkpoint is not None
        if checkpoint is None:
            current = await create_dry_run(
                reader.root,
                workspace=workspace,
                folder_map=requested_map,
                embedding_func=embedding_func,
                allow_unverified=allow_unverified,
            )
            if current["report_hash"] != report["report_hash"]:
                raise StaleReportError(
                    "target changed since dry-run; generate and approve a new report"
                )
            checkpoint = {
                "format": "twin-kb-import-checkpoint/1",
                "bundle_id": manifest.bundle_id,
                "manifest_hash": manifest.manifest_hash,
                "state_hash": manifest.state_hash,
                "report_hash": report["report_hash"],
                "workspace": workspace,
                "folder_map": effective_map,
                "completed": [],
                "imported": {},
                "status": "applying",
                "checkpoint_hash": "",
            }
            _write_checkpoint(checkpoint_target, checkpoint)

        scope = Scope(
            workspace=workspace,
            folder_ids=tuple(sorted(set(effective_map.values()))),
            folder_map=effective_map,
            batch_size=batch_size,
            bundle_id=manifest.bundle_id,
            embedding_dim=manifest.embedding.dim,
        )
        entries = {entry.store: entry for entry in manifest.files}
        completed = {str(name) for name in checkpoint["completed"]}
        imported = {str(k): int(v) for k, v in checkpoint["imported"].items()}

        for name in IMPORT_ORDER:
            entry = entries.get(name)
            if entry is None or name in completed:
                continue
            spec = store_by_name(name)
            store = portable_store(
                spec,
                bundle_root=reader.root if name == "procedures" else None,
            )
            count = await store.import_records(
                _records(reader, entry.path, entry.sha256), scope
            )
            if count != entry.records:
                raise ImportRefused(
                    f"{name}: imported {count} records, manifest declares {entry.records}"
                )
            imported[name] = count
            completed.add(name)
            checkpoint["completed"] = [
                item for item in IMPORT_ORDER if item in completed
            ]
            checkpoint["imported"] = dict(sorted(imported.items()))
            _write_checkpoint(checkpoint_target, checkpoint)

        warnings: list[str] = []
        if "__sweep__" not in completed:
            try:
                from ..server.graph_reader import sweep_stale_source_refs

                await sweep_stale_source_refs(workspace)
            except Exception as exc:  # documented best-effort hygiene step
                warnings.append(f"stale source reference sweep failed: {exc}")
            completed.add("__sweep__")
            checkpoint["completed"] = [
                *[item for item in IMPORT_ORDER if item in completed],
                "__sweep__",
            ]

        checkpoint["status"] = "applied"
        checkpoint["imported"] = dict(sorted(imported.items()))
        _write_checkpoint(checkpoint_target, checkpoint)
        return {
            "ok": True,
            "status": "applied",
            "bundle_id": manifest.bundle_id,
            "state_hash": manifest.state_hash,
            "workspace": workspace,
            "resumed": resumed,
            "checkpoint": str(checkpoint_target),
            "imported": dict(sorted(imported.items())),
            "warnings": warnings,
        }
    finally:
        await close_bundle_reader(reader)


__all__ = [
    "IMPORT_ORDER",
    "ImportRefused",
    "StaleReportError",
    "apply_import",
    "default_checkpoint_path",
]
