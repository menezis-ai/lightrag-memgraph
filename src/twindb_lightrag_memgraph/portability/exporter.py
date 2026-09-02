"""Workspace export orchestrator (ADR 010, decision 4).

Design record: ``docs/adr/010-kb-portability-contract.md``.
"""

from __future__ import annotations

import inspect
import math
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import __version__ as PACKAGE_VERSION
from .._constants import resolve_portability_batch_size, validate_identifier
from .bundle import BundleWriter
from .manifest import (
    CLASSIFICATION_LADDER,
    EMBEDDING_PROBE_TEXTS,
    PROBE_TEXT_SET_ID,
    TOOL_NAME,
    Classification,
    Consistency,
    Embedding,
    Manifest,
    Probe,
    Scope as ManifestScope,
    state_hash_of_entries,
)
from .stores import (
    PortabilityError,
    Scope,
    exportable_stores,
    portable_store,
)


class ExportRefused(PortabilityError):
    """The source is not in an exportable state; no manifest was produced."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _folder_manifest() -> list[dict[str, str]]:
    """Merged env + runtime catalogue without importing the FastAPI layer."""
    from .._folders import load_folder_catalog
    from ..server.folder_store import list_runtime_folders

    env_catalog = load_folder_catalog()
    env_ids = {folder.id for folder in env_catalog.folders}
    runtime = [folder for folder in list_runtime_folders() if folder.id not in env_ids]
    merged = (*env_catalog.folders, *runtime)[: env_catalog.max_folders]
    return [
        {"id": folder.id, "label": folder.label, "kind": folder.kind}
        for folder in merged
    ]


async def _pipeline_is_idle(workspace: str) -> bool:
    """Read LightRAG's status when this process is attached to shared data.

    The documented v1 CLI path runs with the API stopped.  In that offline
    process LightRAG's in-memory namespace is intentionally uninitialised; that
    means no local pipeline can be busy and the pre/post store fingerprints are
    the cross-process mutation detector.  A real status namespace, when
    present, remains authoritative.
    """
    from lightrag.kg import shared_storage

    try:
        data = await shared_storage.get_namespace_data(
            "pipeline_status", workspace=workspace
        )
    except (ValueError, shared_storage.PipelineNotInitializedError):
        return True
    return not bool(data.get("busy", False))


async def _memgraph_source() -> dict[str, str]:
    from .._capabilities import get_mage_capability_snapshot
    from ..patches.canary import installed_lightrag_version
    from ._io import read_rows

    version = ""
    rows = await read_rows("SHOW VERSION")
    if rows and rows[0].get("version") is not None:
        version = str(rows[0]["version"])
    try:
        snapshot = await get_mage_capability_snapshot()
        mage = (
            "unknown"
            if snapshot.available is None
            else "present" if snapshot.available else "absent"
        )
    except Exception:  # capability is additive; source metadata may stay unknown
        mage = "unknown"
    return {
        "database": os.environ.get("MEMGRAPH_DATABASE", "memgraph"),
        "version": version,
        "mage": mage,
        "lightrag_version": installed_lightrag_version(),
    }


def _default_embedding_func() -> tuple[Any, str, int]:
    """Build the exact embedding function used by the native LightRAG server.

    LightRAG owns binding selection, provider defaults, asymmetric prefixes,
    dimensions and binding-specific options.  The portability CLI has its own
    arguments, so temporarily hide them from the upstream ``parse_args()``;
    environment and ``.env`` resolution remain identical to production.
    """
    from lightrag.api.config import initialize_config

    argv = sys.argv
    try:
        sys.argv = [argv[0] if argv else "twin-portability"]
        args = initialize_config(force=True)
        from lightrag.api.lightrag_server import create_embedding_function_from_args

        embedding_func = create_embedding_function_from_args(args)
    except (SystemExit, TypeError, ValueError) as exc:
        raise ExportRefused(f"invalid LightRAG embedding configuration: {exc}") from exc
    except Exception as exc:
        raise ExportRefused(f"cannot build LightRAG embedding function: {exc}") from exc
    finally:
        sys.argv = argv

    dim = int(getattr(embedding_func, "embedding_dim", 0) or 0)
    if dim < 1:
        raise ExportRefused("LightRAG embedding dimension must be positive")
    model = str(
        getattr(args, "embedding_model", None)
        or getattr(embedding_func, "model_name", None)
        or f"{args.embedding_binding}:provider-default"
    )
    return embedding_func, model, dim


async def _embedding_probe(
    embedding_func: Any | None,
    embedding_model: str | None,
) -> Embedding:
    if embedding_func is None:
        embedding_func, default_model, expected_dim = _default_embedding_func()
    else:
        default_model = str(
            getattr(embedding_func, "model", None)
            or os.environ.get("LIGHTRAG_EMBEDDING_MODEL", "unknown")
        )
        expected_dim = int(
            getattr(embedding_func, "embedding_dim", 0)
            or os.environ.get("LIGHTRAG_EMBEDDING_DIM", "0")
        )

    call = getattr(embedding_func, "func", embedding_func)
    result = call(list(EMBEDDING_PROBE_TEXTS))
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "tolist"):
        result = result.tolist()
    if not isinstance(result, (list, tuple)) or len(result) != len(
        EMBEDDING_PROBE_TEXTS
    ):
        raise ExportRefused("embedding probe must return exactly three vectors")

    vectors: list[list[float]] = []
    for index, vector in enumerate(result):
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ExportRefused(f"embedding probe vector {index} is empty")
        values = [float(value) for value in vector]
        if any(not math.isfinite(value) for value in values):
            raise ExportRefused(f"embedding probe vector {index} is not finite")
        vectors.append(values)
    dim = len(vectors[0])
    if any(len(vector) != dim for vector in vectors):
        raise ExportRefused("embedding probe vectors have inconsistent dimensions")
    if expected_dim and dim != expected_dim:
        raise ExportRefused(
            f"embedding probe dimension {dim} differs from configured dimension {expected_dim}"
        )
    return Embedding(
        model=embedding_model or default_model,
        dim=dim,
        metric="cos",
        probe=Probe(text_set_id=PROBE_TEXT_SET_ID, vectors=vectors),
    )


def _observe_classification(
    record: dict[str, Any], current: str | None, unknown: bool
) -> tuple[str | None, bool]:
    metadata = record.get("metadata")
    classification = (
        metadata.get("classification") if isinstance(metadata, dict) else None
    )
    class_id = (
        classification.get("class_id") if isinstance(classification, dict) else None
    )
    if class_id in (None, ""):
        return current, unknown
    class_id = str(class_id).upper()
    if class_id == "UNKNOWN" or class_id not in CLASSIFICATION_LADDER:
        return current, True
    if current is None or CLASSIFICATION_LADDER.index(
        class_id
    ) > CLASSIFICATION_LADDER.index(current):
        current = class_id
    return current, unknown


def _logical_counts(records_by_store: dict[str, int]) -> dict[str, int]:
    return {
        "documents": records_by_store.get("docstatus", 0),
        "chunks": records_by_store.get("vec.chunks", 0),
        "entities": records_by_store.get("graph.nodes", 0),
        "relations": records_by_store.get("graph.edges", 0),
        "folders": records_by_store.get("folders", 0),
        "tags": records_by_store.get("tags", 0),
    }


async def export_kb(
    target_dir: str | Path,
    *,
    workspace: str,
    include_activity: bool = False,
    include_procedures: bool = False,
    batch: int | None = None,
    force: bool = False,
    embedding_func: Any | None = None,
    embedding_model: str | None = None,
    actor: str | None = None,
) -> Manifest:
    """Export one complete workspace into a canonical directory bundle.

    ``force`` only bypasses a busy pipeline; it never claims consistency and
    therefore seals the result as ``unverified``.  Store mutation detected by
    pre/post fingerprints has the same outcome.
    """
    workspace = validate_identifier(workspace, "workspace")
    batch_size = batch if batch is not None else resolve_portability_batch_size()
    if batch_size < 1:
        raise ValueError("batch must be >= 1")
    target = Path(target_dir)
    target_existed = target.exists()
    if target_existed and (not target.is_dir() or any(target.iterdir())):
        raise ExportRefused(f"target directory is not empty: {target}")

    bundle_id = str(uuid.uuid4())
    folders = _folder_manifest()
    scope = Scope(
        workspace=workspace,
        folder_ids=tuple(folder["id"] for folder in folders),
        batch_size=batch_size,
        bundle_id=bundle_id,
    )
    specs = exportable_stores(
        include_activity=include_activity,
        include_procedures=include_procedures,
    )
    preflight = [(spec, portable_store(spec)) for spec in specs]

    pipeline_idle = await _pipeline_is_idle(workspace)
    if not pipeline_idle and not force:
        raise ExportRefused(
            "LightRAG ingestion pipeline is busy; retry in a maintenance window or use --force"
        )
    # Fingerprint every included store before opening the output bundle.  In
    # particular, a degraded procedure store refuses here before any data file
    # can be written.
    fingerprints_before = {
        spec.name: await store.fingerprint(scope) for spec, store in preflight
    }
    embedding = await _embedding_probe(embedding_func, embedding_model)
    source_info = await _memgraph_source()

    writer: BundleWriter | None = None
    try:
        writer = BundleWriter(target)
        records_by_store: dict[str, int] = {}
        max_detected: str | None = None
        unknown_present = False
        for spec, preflight_store in preflight:
            store = (
                portable_store(spec, bundle_writer=writer)
                if spec.name == "procedures"
                else preflight_store
            )
            output = writer.open_jsonl(spec.file or "", store=spec.name)
            async for record in store.export_records(scope):
                output.write(record)
                if spec.name == "docstatus":
                    max_detected, unknown_present = _observe_classification(
                        record, max_detected, unknown_present
                    )
            entry = output.close()
            records_by_store[spec.name] = entry.records

        fingerprints_after = {
            spec.name: await store.fingerprint(scope) for spec, store in preflight
        }
        verified = pipeline_idle and fingerprints_before == fingerprints_after
        manifest = Manifest(
            bundle_id=bundle_id,
            created_at=_utc_now(),
            created_by={
                "tool": TOOL_NAME,
                "version": PACKAGE_VERSION,
                "actor": actor or os.environ.get("USER", "").strip() or "operator",
            },
            source={
                "workspace": workspace,
                "memgraph": {
                    "database": source_info["database"],
                    "version": source_info["version"],
                    "mage": source_info["mage"],
                },
                "lightrag_version": source_info["lightrag_version"],
                "package_version": PACKAGE_VERSION,
            },
            embedding=embedding,
            scope=ManifestScope(
                include_activity=include_activity,
                include_procedures=include_procedures,
            ),
            classification=Classification(
                max_detected=max_detected,
                unknown_present=unknown_present,
            ),
            consistency=Consistency(
                status="verified" if verified else "unverified",
                pipeline_idle=pipeline_idle,
                fingerprints_before=fingerprints_before,
                fingerprints_after=fingerprints_after,
            ),
            folders=folders,
            files=writer.entries,
            counts=_logical_counts(records_by_store),
            state_hash=state_hash_of_entries(writer.entries),
        ).sealed()
        # Run the strict reader once before persisting the final manifest.
        Manifest.from_json(manifest.to_json())
        return writer.finalize(manifest)
    except BaseException:
        # ``target`` was validated as empty and every descendant was created by
        # this operation, so removing it cannot touch pre-existing user data.
        if writer is not None:
            shutil.rmtree(target, ignore_errors=True)
            if target_existed:
                target.mkdir(parents=True, exist_ok=True)
        raise
