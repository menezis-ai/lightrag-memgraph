"""Shared builders for the portability tests — a valid manifest in one call."""

from __future__ import annotations

import copy
from typing import Any

from twindb_lightrag_memgraph.portability.manifest import (
    EMBEDDING_PROBE_TEXTS,
    PROBE_TEXT_SET_ID,
    state_hash_of,
)
from twindb_lightrag_memgraph.portability.stores import exportable_stores

DIM = 4
SHA_A = "a" * 64


def _canonical_files(
    *, include_activity: bool, include_procedures: bool
) -> list[dict[str, Any]]:
    files = []
    for spec in exportable_stores(
        include_activity=include_activity,
        include_procedures=include_procedures,
    ):
        assert spec.file is not None
        files.append(
            {
                "path": spec.file,
                "store": spec.name,
                "records": (
                    2 if spec.name == "docstatus" else 1 if spec.name == "tags" else 0
                ),
                "sha256": SHA_A,
                "bytes": (
                    120
                    if spec.name == "docstatus"
                    else 40 if spec.name == "tags" else 0
                ),
            }
        )
    return files


def manifest_dict(**overrides: Any) -> dict[str, Any]:
    scope_override = overrides.get("scope")
    scope_override = scope_override if isinstance(scope_override, dict) else {}
    include_activity = bool(scope_override.get("include_activity", False))
    include_procedures = bool(scope_override.get("include_procedures", False))
    files = _canonical_files(
        include_activity=include_activity,
        include_procedures=include_procedures,
    )
    data: dict[str, Any] = {
        "format": "twin-kb-bundle",
        "format_version": "1.0",
        "bundle_id": "0b6f0f5a-6c2c-4a6b-9b3a-2c0b6f0f5a6c",
        "created_at": "2026-08-25T10:00:00Z",
        "created_by": {
            "tool": "twindb_lightrag_memgraph.portability",
            "version": "1.2.0",
            "actor": "admin",
        },
        "source": {
            "workspace": "base",
            "memgraph": {
                "database": "memgraph",
                "version": "3.12.0",
                "mage": "unknown",
            },
            "lightrag_version": "1.5.6",
            "package_version": "1.2.0",
        },
        "embedding": {
            "model": "fake",
            "dim": DIM,
            "metric": "cos",
            "probe": {
                "text_set_id": PROBE_TEXT_SET_ID,
                "vectors": [
                    [0.1 * (i + 1)] * DIM for i in range(len(EMBEDDING_PROBE_TEXTS))
                ],
            },
        },
        "scope": {
            "kind": "workspace",
            "include_activity": False,
            "include_procedures": False,
        },
        "classification": {
            "max_detected": "C2",
            "ladder": ["C0", "C1", "C2", "C3", "C4"],
            "unknown_present": False,
        },
        "consistency": {
            "status": "verified",
            "pipeline_idle": True,
            "fingerprints_before": {},
            "fingerprints_after": {},
        },
        "folders": [{"id": "f1", "label": "Folder one", "kind": "team"}],
        "files": files,
        "counts": {"documents": 2, "tags": 1},
        "state_hash": state_hash_of({f["path"]: f["sha256"] for f in files}),
        "manifest_hash": "",
    }
    return _deep_merge(data, overrides)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def sealed_manifest_dict(**overrides: Any) -> dict[str, Any]:
    """A manifest dict whose manifest_hash matches its content."""
    from twindb_lightrag_memgraph.portability.canonical import jcs_sha256

    data = manifest_dict(**overrides)
    body = {k: v for k, v in data.items() if k != "manifest_hash"}
    data["manifest_hash"] = jcs_sha256(body)
    return data
