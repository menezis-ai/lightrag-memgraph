"""Import dry-run and approved report contract (KB-PORTABILITY-PLAN T2.2).

The report is the operator approval boundary. Its hash excludes the display
timestamp, itself and the diagnostic cosine value; the fixed probe id,
dimension and pass/fail verdict remain bound. ``apply`` can therefore replay
the target facts without rejecting harmless endpoint float noise, while still
refusing a semantic change before the first write. A report with any
``blocking`` entry is useful diagnostically but can never be applied.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._constants import validate_identifier
from .bundle import (
    BundleError,
    BundleReader,
    close_bundle_reader,
    open_bundle_reader,
    run_reader_io,
)
from .canonical import jcs_dumps, jcs_loads, jcs_sha256
from .compat import (
    TargetFacts,
    check,
    check_vector_index_contract,
    collect_target_facts,
)
from .jsonl import iter_jsonl
from .stores import PortabilityError

REPORT_FORMAT = "twin-kb-import-report"
REPORT_VERSION = "1.0"


class DryRunRefused(PortabilityError):
    """The bundle or requested target mapping cannot produce a safe plan."""


class ReportError(PortabilityError):
    """An approved report is malformed or its hash does not match."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def report_hash(report: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in report.items()
        if key not in {"created_at", "report_hash"}
    }
    # The embedding decision is the fixed probe id + dimension + cosine
    # verdict. The exact cosine is useful diagnostics but may vary harmlessly
    # across GPU replicas while staying above the contractual threshold.
    raw_compatibility = payload.get("compat")
    if not isinstance(raw_compatibility, list):
        return jcs_sha256(payload)
    compatibility = []
    for item in raw_compatibility:
        if not isinstance(item, dict):
            compatibility.append(item)
            continue
        normalized = dict(item)
        if normalized.get("dimension") == "embedding" and isinstance(
            normalized.get("target"), dict
        ):
            target = dict(normalized["target"])
            target.pop("min_cosine", None)
            normalized["target"] = target
        compatibility.append(normalized)
    payload["compat"] = compatibility
    return jcs_sha256(payload)


def seal_report(report: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(report)
    sealed["report_hash"] = report_hash(sealed)
    return sealed


def _block(code: str, message: str, **details: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if details:
        item["details"] = details
    return item


def _runtime_source_ids(reader: BundleReader) -> set[str]:
    assert reader.manifest is not None
    entry = next(
        item for item in reader.manifest.files if item.store == "runtime_folders"
    )
    return {
        str(record["id"])
        for record in iter_jsonl(reader.path_of(entry.path), entry.sha256)
    }


def _folder_plan(
    reader: BundleReader,
    facts: TargetFacts,
    requested: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assert reader.manifest is not None
    manifest = reader.manifest
    source = {str(folder["id"]): dict(folder) for folder in manifest.folders}
    unknown = sorted(set(requested) - set(source))
    if unknown:
        raise DryRunRefused(f"folder mapping names unknown source folders: {unknown}")
    for src, dst in requested.items():
        validate_identifier(src, "source folder")
        validate_identifier(dst, "target folder")

    effective = {src: requested.get(src, src) for src in source}
    destinations = list(effective.values())
    if len(set(destinations)) != len(destinations):
        raise DryRunRefused(
            "folder mapping must be one-to-one; merging folders is outside v1"
        )

    source_runtime = _runtime_source_ids(reader)
    env_ids = {str(folder["id"]) for folder in facts.env_folders}
    runtime_ids = {str(folder["id"]) for folder in facts.runtime_folders}
    blockers: list[dict[str, Any]] = []
    collisions: list[dict[str, str]] = []
    creations: list[dict[str, Any]] = []

    for src, dst in effective.items():
        explicitly_mapped = src in requested
        if dst in env_ids:
            collisions.append({"source": src, "target": dst, "kind": "env"})
            if not explicitly_mapped:
                blockers.append(
                    _block(
                        "folder_mapping_required",
                        f"source folder {src!r} collides with target env folder {dst!r}; approve it with --map-folder {src}={dst}",
                    )
                )
            if src in source_runtime:
                blockers.append(
                    _block(
                        "runtime_folder_to_env",
                        f"runtime source folder {src!r} cannot map to env-seeded target folder {dst!r} without changing semantic state",
                    )
                )
            continue
        if dst in runtime_ids:
            collisions.append({"source": src, "target": dst, "kind": "runtime"})
            blockers.append(
                _block(
                    "runtime_folder_collision",
                    f"target runtime folder {dst!r} already exists; v1 requires an empty target",
                )
            )
            continue
        if src not in source_runtime:
            blockers.append(
                _block(
                    "target_env_folder_missing",
                    f"env-seeded source folder {src!r} must map to an env-seeded target folder",
                )
            )
            continue
        folder = source[src]
        creations.append(
            {
                "source": src,
                "id": dst,
                "label": folder.get("label", dst),
                "kind": folder.get("kind", "custom"),
            }
        )

    final_ids = env_ids | runtime_ids | {item["id"] for item in creations}
    if len(final_ids) > facts.max_folders:
        blockers.append(
            _block(
                "folder_limit",
                f"target would have {len(final_ids)} folders but TWIN_MAX_FOLDERS is {facts.max_folders}",
            )
        )

    return (
        {
            "requested_mapping": dict(sorted(requested.items())),
            "effective_mapping": dict(sorted(effective.items())),
            "source": [source[key] for key in sorted(source)],
            "source_runtime_ids": sorted(source_runtime),
            "target_env": facts.env_folders,
            "target_runtime": facts.runtime_folders,
            "collisions": collisions,
            "creations": creations,
            "max_folders": facts.max_folders,
        },
        blockers,
    )


def _capacity_plan(
    reader: BundleReader, facts: TargetFacts
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert reader.manifest is not None
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    indexes = {str(item.get("index_name") or ""): item for item in facts.vector_indexes}
    for entry in sorted(reader.manifest.files, key=lambda item: item.store):
        if not entry.store.startswith("vec."):
            continue
        namespace = entry.store.removeprefix("vec.")
        index_name = f"vec_{facts.workspace}_{namespace}"
        existing = indexes.get(index_name)
        effective_capacity = facts.vector_capacity
        contract = None
        if existing is not None:
            contract = check_vector_index_contract(
                existing,
                workspace=facts.workspace,
                namespace=namespace,
                embedding_dim=reader.manifest.embedding.dim,
                minimum_capacity=max(1, entry.records),
                expected_size=0,
            )
            actual_capacity = contract["actual"]["capacity"]
            effective_capacity = actual_capacity if actual_capacity is not None else 0
        row = {
            "store": entry.store,
            "records": entry.records,
            "capacity": effective_capacity,
            "configured_capacity": facts.vector_capacity,
            "index": existing,
            "contract": contract,
        }
        rows.append(row)
        if existing is None and entry.records > facts.vector_capacity:
            blockers.append(
                _block(
                    "vector_capacity",
                    f"{entry.store} has {entry.records} vectors but target capacity is {facts.vector_capacity}",
                )
            )
        if contract is not None:
            for problem in contract["problems"]:
                code = (
                    "vector_capacity"
                    if problem == "capacity"
                    else f"vector_index_{problem}"
                )
                blockers.append(
                    _block(
                        code,
                        f"existing index {index_name!r} violates its canonical {problem} contract",
                        expected=contract["expected"],
                        actual=contract["actual"],
                    )
                )
    return rows, blockers


async def create_dry_run(
    source: str | Path,
    *,
    workspace: str,
    folder_map: dict[str, str] | None = None,
    embedding_func: Any | None = None,
    allow_unverified: bool | None = None,
) -> dict[str, Any]:
    """Inspect *source*, probe *workspace*, and return a sealed report."""
    workspace = validate_identifier(workspace, "workspace")
    requested = dict(folder_map or {})
    reader = BundleReader(Path(source))
    try:
        await open_bundle_reader(reader)
        # Archive extraction, integrity hashing and JSONL parsing can cover up
        # to TWIN_PORTABILITY_MAX_BYTES. Keep them off the ASGI event loop.
        inspection = await run_reader_io(reader.inspect)
        if not inspection.ok:
            raise BundleError(
                "bundle integrity failed: " + "; ".join(inspection.problems)
            )
        assert reader.manifest is not None
        manifest = reader.manifest
        candidate_ids = tuple(
            validate_identifier(
                requested.get(str(folder["id"]), str(folder["id"])),
                "target folder",
            )
            for folder in manifest.folders
        )
        facts = await collect_target_facts(
            manifest,
            workspace=workspace,
            candidate_folder_ids=candidate_ids,
            embedding_func=embedding_func,
            allow_unverified=allow_unverified,
        )
        compatibility = check(manifest, facts)
        folder_plan, folder_blockers = await run_reader_io(
            _folder_plan, reader, facts, requested
        )
        capacity, capacity_blockers = await run_reader_io(_capacity_plan, reader, facts)

        blockers = [
            _block(
                f"compat_{item['dimension']}",
                item["reason"],
                source=item["source"],
                target=item["target"],
            )
            for item in compatibility
            if not item["ok"]
        ]
        non_empty = {
            name: count for name, count in facts.store_counts.items() if count > 0
        }
        if non_empty:
            blockers.append(
                _block(
                    "target_not_empty",
                    "target workspace contains portable or protected state",
                    counts=non_empty,
                )
            )
        if not facts.pipeline_idle:
            blockers.append(
                _block(
                    "target_pipeline_busy",
                    "target LightRAG ingestion pipeline is busy",
                )
            )
        blockers.extend(folder_blockers)
        blockers.extend(capacity_blockers)

        report = {
            "format": REPORT_FORMAT,
            "format_version": REPORT_VERSION,
            "created_at": _utc_now(),
            "bundle": {
                "bundle_id": manifest.bundle_id,
                "manifest_hash": manifest.manifest_hash,
                "state_hash": manifest.state_hash,
                "source_workspace": manifest.source["workspace"],
            },
            "target": facts.as_dict(),
            "folders": folder_plan,
            "compat": compatibility,
            "classification": {
                "source_max": manifest.classification.max_detected,
                "unknown_present": manifest.classification.unknown_present,
                "target_ceiling": facts.classification_ceiling,
            },
            "capacity": capacity,
            "stats": {
                "counts": manifest.counts,
                "stores": {entry.store: entry.records for entry in manifest.files},
            },
            "options": {"allow_unverified": facts.allow_unverified},
            "blocking": blockers,
            "report_hash": "",
        }
        return seal_report(report)
    finally:
        await close_bundle_reader(reader)


_REPORT_KEYS = {
    "format",
    "format_version",
    "created_at",
    "bundle",
    "target",
    "folders",
    "compat",
    "classification",
    "capacity",
    "stats",
    "options",
    "blocking",
    "report_hash",
}


def load_report(
    path: str | Path, *, expected_report_hash: str | None = None
) -> dict[str, Any]:
    try:
        data = jcs_loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportError(f"cannot read import report: {exc}") from exc
    if not isinstance(data, dict) or set(data) != _REPORT_KEYS:
        raise ReportError("import report has an unsupported shape")
    if (
        data.get("format") != REPORT_FORMAT
        or data.get("format_version") != REPORT_VERSION
    ):
        raise ReportError("unsupported import report format/version")
    digest = data.get("report_hash")
    if not isinstance(digest, str) or digest != report_hash(data):
        raise ReportError("report_hash does not match the report content")
    if expected_report_hash is not None and digest != expected_report_hash:
        raise ReportError("report_hash does not match the persisted approval")
    if not isinstance(data.get("blocking"), list):
        raise ReportError("report.blocking must be a list")
    return data


def write_report(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    try:
        temporary.write_text(jcs_dumps(report) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


__all__ = [
    "DryRunRefused",
    "ReportError",
    "create_dry_run",
    "load_report",
    "report_hash",
    "seal_report",
    "write_report",
]
