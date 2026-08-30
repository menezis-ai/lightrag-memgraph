"""CLI for canonical KB export, inspection and approved import (PR-P2)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .. import _pool
from .._constants import (
    portability_flag_enabled,
    resolve_portability_batch_size,
    resolve_portability_dir,
    validate_portability_env,
)
from .bundle import BundleError, archive_bundle, inspect_bundle
from .exporter import ExportRefused, export_kb
from .jsonl import IntegrityError
from .manifest import ManifestError
from .importer import ImportRefused, StaleReportError, apply_import
from .plan import DryRunRefused, ReportError, create_dry_run, write_report
from .stores import PortabilityError
from .validate import validate_import

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_INTEGRITY = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twindb_lightrag_memgraph.portability",
        description="Export, inspect or import a canonical Twin KB bundle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="export one complete workspace")
    export.add_argument("--workspace", default="base")
    export.add_argument(
        "--out",
        type=Path,
        help="parent directory (default: TWIN_PORTABILITY_DIR)",
    )
    export.add_argument("--archive", action="store_true")
    export.add_argument("--include-activity", action="store_true")
    export.add_argument("--include-procedures", action="store_true")
    export.add_argument("--force", action="store_true")
    export.add_argument("--batch", type=int)

    inspect_cmd = commands.add_parser("inspect", help="verify a directory or tar.gz")
    inspect_cmd.add_argument("path", type=Path)

    dry_run = commands.add_parser("dry-run", help="plan an import without writing")
    dry_run.add_argument("bundle", type=Path)
    dry_run.add_argument("--workspace", default="base")
    dry_run.add_argument("--map-folder", action="append", default=[])
    dry_run.add_argument("--report", required=True, type=Path)
    dry_run.add_argument("--allow-unverified", action="store_true")

    apply = commands.add_parser("apply", help="apply an approved dry-run report")
    apply.add_argument("bundle", type=Path)
    apply.add_argument("--report", required=True, type=Path)
    apply.add_argument("--checkpoint", type=Path)
    apply.add_argument("--batch", type=int)

    validate = commands.add_parser("validate", help="validate an imported workspace")
    validate.add_argument("--bundle", required=True, type=Path)
    validate.add_argument("--workspace", default="base")
    validate.add_argument("--map-folder", action="append", default=[])
    validate.add_argument("--out", type=Path)
    validate.add_argument("--batch", type=int)
    return parser


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


async def _export_command(args: argparse.Namespace) -> dict[str, object]:
    if not os.environ.get("MEMGRAPH_URI", "").strip():
        raise ExportRefused(
            "MEMGRAPH_URI is required for portability export; no implicit target is allowed"
        )
    validate_portability_env()
    parent = args.out or Path(resolve_portability_dir())
    parent_existed = parent.exists()
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed:
        parent.chmod(0o700)
    staging = parent / f".kb-{args.workspace}-{uuid.uuid4().hex}.part"
    try:
        manifest = await export_kb(
            staging,
            workspace=args.workspace,
            include_activity=args.include_activity
            or portability_flag_enabled("TWIN_PORTABILITY_INCLUDE_ACTIVITY"),
            include_procedures=args.include_procedures
            or portability_flag_enabled("TWIN_PORTABILITY_INCLUDE_PROCEDURES"),
            batch=(
                args.batch
                if args.batch is not None
                else resolve_portability_batch_size()
            ),
            force=args.force,
        )
        name = f"kb-{args.workspace}-{_stamp()}-{manifest.state_hash[:12]}"
        bundle_dir = parent / name
        if bundle_dir.exists():
            raise ExportRefused(
                f"bundle target already exists: {bundle_dir}; retry after the timestamp changes"
            )
        staging.replace(bundle_dir)
        archive_path: Path | None = None
        if args.archive:
            archive_path = archive_bundle(bundle_dir, parent / f"{name}.tar.gz")
        return {
            "ok": True,
            "bundle": str(bundle_dir),
            "archive": str(archive_path) if archive_path is not None else None,
            "bundle_id": manifest.bundle_id,
            "state_hash": manifest.state_hash,
            "consistency": manifest.consistency.status,
            "counts": manifest.counts,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        await _pool.close_driver()


def _inspect_command(args: argparse.Namespace) -> dict[str, object]:
    inspection = inspect_bundle(args.path)
    return inspection.as_dict()


def _require_memgraph(command: str) -> None:
    if not os.environ.get("MEMGRAPH_URI", "").strip():
        raise PortabilityError(
            f"MEMGRAPH_URI is required for portability {command}; no implicit target is allowed"
        )


def _folder_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--map-folder must be SOURCE=TARGET, got {item!r}")
        source, target = (part.strip() for part in item.split("=", 1))
        if not source or not target:
            raise ValueError(f"--map-folder must be SOURCE=TARGET, got {item!r}")
        previous = mapping.get(source)
        if previous is not None and previous != target:
            raise ValueError(f"source folder {source!r} is mapped more than once")
        mapping[source] = target
    return mapping


async def _dry_run_command(args: argparse.Namespace) -> dict[str, object]:
    _require_memgraph("dry-run")
    validate_portability_env()
    try:
        report = await create_dry_run(
            args.bundle,
            workspace=args.workspace,
            folder_map=_folder_map(args.map_folder),
            allow_unverified=args.allow_unverified or None,
        )
        write_report(args.report, report)
        return report
    finally:
        await _pool.close_driver()


async def _apply_command(args: argparse.Namespace) -> dict[str, object]:
    _require_memgraph("apply")
    validate_portability_env()
    try:
        return await apply_import(
            args.bundle,
            report_path=args.report,
            checkpoint_path=args.checkpoint,
            batch=args.batch,
        )
    finally:
        await _pool.close_driver()


async def _validate_command(args: argparse.Namespace) -> dict[str, object]:
    _require_memgraph("validate")
    validate_portability_env()
    try:
        result = await validate_import(
            args.bundle,
            workspace=args.workspace,
            folder_map=_folder_map(args.map_folder),
            batch=args.batch,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            args.out.chmod(0o600)
        return result
    finally:
        await _pool.close_driver()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = _inspect_command(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return EXIT_OK if result["ok"] else EXIT_INTEGRITY
        if args.command == "export":
            result = asyncio.run(_export_command(args))
        elif args.command == "dry-run":
            result = asyncio.run(_dry_run_command(args))
        elif args.command == "apply":
            result = asyncio.run(_apply_command(args))
        else:
            result = asyncio.run(_validate_command(args))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return EXIT_OK if result.get("ok", not result.get("blocking")) else EXIT_REFUSED
    except (BundleError, IntegrityError, ManifestError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return EXIT_INTEGRITY
    except (
        DryRunRefused,
        ExportRefused,
        ImportRefused,
        PortabilityError,
        ReportError,
        StaleReportError,
        OSError,
        ValueError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())
