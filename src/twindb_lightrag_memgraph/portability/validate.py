"""Post-import validation and semantic round-trip proof (plan T2.4)."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from .._constants import resolve_portability_batch_size
from ._io import read_rows, read_scalar
from .bundle import (
    BundleReader,
    close_bundle_reader,
    open_bundle_reader,
    run_reader_io,
)
from .canonical import jcs_dumps
from .compat import check_vector_index_contract
from .exporter import export_kb
from .jsonl import iter_jsonl
from .manifest import Manifest, state_hash_of
from .stores import Scope, portable_store, store_by_name
from .stores_graph import remap_relation_folder_json


def _map_value(value: Any, folder_map: dict[str, str]) -> Any:
    if value in (None, ""):
        return value
    return folder_map.get(str(value), str(value))


def _normalized_record(
    store: str, record: dict[str, Any], folder_map: dict[str, str]
) -> dict[str, Any]:
    mapped = dict(record)
    field_by_store = {
        "docstatus": "folder",
        "folders": "id",
        "runtime_folders": "id",
        "member_of": "folder_id",
        "tagged_with": "folder_id",
        "graph.member_of": "folder_id",
        "graph.overrides": "folder",
        "tags": "folder_id",
        "tag_categories": "folder_id",
        "activity": "folder_id",
        "procedures": "folder",
    }
    field = field_by_store.get(store)
    if field and field in mapped:
        mapped[field] = _map_value(mapped[field], folder_map)
    if store == "procedures" and isinstance(mapped.get("duplicate_requests"), list):
        requests = []
        for item in mapped["duplicate_requests"]:
            request = dict(item) if isinstance(item, dict) else item
            if isinstance(request, dict) and "folder" in request:
                request["folder"] = _map_value(request["folder"], folder_map)
            requests.append(request)
        mapped["duplicate_requests"] = requests
    if store == "graph.edges" and isinstance(mapped.get("props"), dict):
        props = dict(mapped["props"])
        if "twin_folder_json" in props:
            props["twin_folder_json"] = remap_relation_folder_json(
                props["twin_folder_json"], folder_map
            )
        mapped["props"] = props
    return mapped


def normalized_source_state_hash(
    reader: BundleReader, folder_map: dict[str, str]
) -> str:
    """Hash source JSONL after applying the import's folder mapping."""
    assert reader.manifest is not None
    hashes: dict[str, str] = {}
    for entry in reader.manifest.files:
        if not entry.path.startswith(("memgraph/", "overlay/")):
            continue
        if not entry.path.endswith(".jsonl") or not folder_map:
            hashes[entry.path] = entry.sha256
            continue
        digest = hashlib.sha256()
        for record in iter_jsonl(reader.path_of(entry.path), entry.sha256):
            normalized = _normalized_record(entry.store, record, folder_map)
            digest.update((jcs_dumps(normalized) + "\n").encode("utf-8"))
        hashes[entry.path] = digest.hexdigest()
    return state_hash_of(hashes)


async def _index_validation(
    *, workspace: str, embedding_dim: int, expected: dict[str, int]
) -> list[dict[str, Any]]:
    rows = await read_rows("SHOW VECTOR INDEX INFO")
    by_name = {str(row.get("index_name") or ""): row for row in rows}
    checks: list[dict[str, Any]] = []
    for store, records in sorted(expected.items()):
        if not store.startswith("vec."):
            continue
        name = f"vec_{workspace}_{store.removeprefix('vec.')}"
        row = by_name.get(name)
        contract = check_vector_index_contract(
            row,
            workspace=workspace,
            namespace=store.removeprefix("vec."),
            embedding_dim=embedding_dim,
            minimum_capacity=max(1, records),
            expected_size=records,
        )
        checks.append(
            {
                "index": name,
                "ok": contract["ok"],
                "expected": contract["expected"],
                "actual": contract["actual"],
                "problems": contract["problems"],
                "expected_records": records,
                "actual_records": contract["actual"]["size"],
            }
        )
    return checks


def _procedure_file_validation(
    source: Manifest, target: Manifest
) -> list[dict[str, Any]]:
    expected = {
        entry.path: entry.sha256
        for entry in source.files
        if entry.path.startswith("files/procedures/")
    }
    actual = {
        entry.path: entry.sha256
        for entry in target.files
        if entry.path.startswith("files/procedures/")
    }
    return [
        {
            "path": path,
            "ok": expected.get(path) == actual.get(path),
            "expected_sha256": expected.get(path),
            "actual_sha256": actual.get(path),
        }
        for path in sorted(set(expected) | set(actual))
    ]


def _expected_memberships(
    reader: BundleReader, folder_map: dict[str, str]
) -> dict[str, int]:
    assert reader.manifest is not None
    entry = next(item for item in reader.manifest.files if item.store == "member_of")
    counts: dict[str, int] = {}
    for record in iter_jsonl(reader.path_of(entry.path), entry.sha256):
        folder = _map_value(record["folder_id"], folder_map)
        counts[folder] = counts.get(folder, 0) + 1
    return counts


async def _folder_validation(
    expected: dict[str, int], *, workspace: str
) -> list[dict[str, Any]]:
    doc_label = store_by_name("docstatus").label(workspace)
    folder_label = store_by_name("folders").label(workspace)
    checks = []
    for folder, count in sorted(expected.items()):
        actual = int(
            await read_scalar(
                f"MATCH (d:`{doc_label}`)-[:MEMBER_OF]->(f:`{folder_label}` {{id: $folder}}) RETURN count(d)",
                folder=folder,
            )
            or 0
        )
        checks.append(
            {
                "folder": folder,
                "ok": actual == count,
                "expected_documents": count,
                "actual_documents": actual,
            }
        )
    return checks


async def validate_import(
    source: str | Path,
    *,
    workspace: str,
    folder_map: dict[str, str] | None = None,
    embedding_func: Any | None = None,
    batch: int | None = None,
) -> dict[str, Any]:
    """Validate counts, indexes, folder scope and a re-exported state hash."""
    mapping = dict(folder_map or {})
    batch_size = batch if batch is not None else resolve_portability_batch_size()
    if batch_size < 1:
        raise ValueError("batch must be >= 1")
    reader = BundleReader(Path(source))
    try:
        await open_bundle_reader(reader)
        inspection = await run_reader_io(reader.inspect)
        if not inspection.ok:
            return {"ok": False, "problems": inspection.problems}
        assert reader.manifest is not None
        manifest = reader.manifest
        effective = {
            str(folder["id"]): mapping.get(str(folder["id"]), str(folder["id"]))
            for folder in manifest.folders
        }
        scope = Scope(
            workspace=workspace,
            folder_ids=tuple(sorted(set(effective.values()))),
            folder_map=effective,
            batch_size=batch_size,
            bundle_id=manifest.bundle_id,
            embedding_dim=manifest.embedding.dim,
        )
        expected_by_store = {
            entry.store: entry.records
            for entry in manifest.files
            if entry.path.endswith(".jsonl")
        }
        counts = []
        for name, expected in sorted(expected_by_store.items()):
            actual = await portable_store(store_by_name(name)).count(scope)
            counts.append(
                {
                    "store": name,
                    "ok": actual == expected,
                    "expected": expected,
                    "actual": actual,
                }
            )

        indexes = await _index_validation(
            workspace=workspace,
            embedding_dim=manifest.embedding.dim,
            expected=expected_by_store,
        )
        expected_memberships = await run_reader_io(
            _expected_memberships, reader, effective
        )
        folders = await _folder_validation(expected_memberships, workspace=workspace)
        expected_hash = await run_reader_io(
            normalized_source_state_hash, reader, effective
        )

        with tempfile.TemporaryDirectory(prefix="twin-kb-validate-") as temporary:
            target = Path(temporary) / "reexport"
            reexport = await export_kb(
                target,
                workspace=workspace,
                include_activity=manifest.scope.include_activity,
                include_procedures=manifest.scope.include_procedures,
                batch=batch_size,
                embedding_func=embedding_func,
                actor="portability-validator",
            )
            actual_hash = reexport.state_hash
            reexport_verified = reexport.consistency.status == "verified"
            procedure_files = _procedure_file_validation(manifest, reexport)

        graph_expected = expected_by_store.get("graph.edges", 0)
        graph_actual = int(
            await read_scalar(
                f"MATCH (:`{workspace}`)-[r:DIRECTED]->(:`{workspace}`) RETURN count(r)"
            )
            or 0
        )
        graph = {
            "ok": graph_actual == graph_expected,
            "expected_edges": graph_expected,
            "actual_edges": graph_actual,
        }
        problems = [
            f"store {item['store']}: expected {item['expected']}, got {item['actual']}"
            for item in counts
            if not item["ok"]
        ]
        problems.extend(
            f"vector index {item['index']} violates: {', '.join(item['problems'])}"
            for item in indexes
            if not item["ok"]
        )
        problems.extend(
            f"procedure file {item['path']} digest differs from the bundle"
            for item in procedure_files
            if not item["ok"]
        )
        problems.extend(
            f"folder {item['folder']}: expected {item['expected_documents']} memberships, got {item['actual_documents']}"
            for item in folders
            if not item["ok"]
        )
        if expected_hash != actual_hash:
            problems.append(
                f"state_hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        if not reexport_verified:
            problems.append("validation re-export is unverified")
        if not graph["ok"]:
            problems.append(
                f"graph edge count mismatch: expected {graph_expected}, got {graph_actual}"
            )
        return {
            "ok": not problems,
            "bundle_id": manifest.bundle_id,
            "workspace": workspace,
            "expected_state_hash": expected_hash,
            "actual_state_hash": actual_hash,
            "reexport_consistency": reexport.consistency.status,
            "counts": counts,
            "indexes": indexes,
            "procedure_files": procedure_files,
            "folders": folders,
            "graph": graph,
            "problems": problems,
        }
    finally:
        await close_bundle_reader(reader)


__all__ = ["normalized_source_state_hash", "validate_import"]
