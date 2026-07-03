"""Best-effort cleanup of uploaded source files after ingestion.

BNP deployments must not retain imported source documents on disk once the
document has reached LightRAG's terminal processed state. The only allowed
cleanup scope is the configured ``INPUT_DIR`` tree; ``file_path`` values stored
in DocStatus are user-controlled metadata and must never delete arbitrary paths.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Any

from lightrag.utils import logger

try:  # LightRAG 1.5.x only — utils_pipeline does NOT exist in the 1.4.x line
    from lightrag.utils_pipeline import (
        PARSED_DIR_NAME,
        canonicalize_parser_hinted_basename,
        configured_input_dir,
    )
except ImportError:  # pragma: no cover - 1.4.x (incl. BNP pin 1.4.9.11) fallback
    PARSED_DIR_NAME = "__parsed__"

    def configured_input_dir() -> Path:
        raw = os.getenv("INPUT_DIR", "").strip()
        return Path(raw) if raw else Path.cwd() / "inputs"

    def canonicalize_parser_hinted_basename(file_path: str | Path) -> str:
        return Path(file_path).name


def should_cleanup_import(props: dict[str, Any]) -> bool:
    """Return True when a DocStatus row represents a terminal import state."""
    if not props.get("file_path"):
        return False
    return str(props.get("status") or "").lower() == "processed"


async def cleanup_processed_imports(props_list: list[dict[str, Any]]) -> None:
    """Delete uploaded sources for processed DocStatus rows, best effort."""
    file_paths = [
        str(props["file_path"]) for props in props_list if should_cleanup_import(props)
    ]
    if not file_paths:
        return
    await asyncio.to_thread(_cleanup_many_sync, file_paths)


def _cleanup_many_sync(file_paths: list[str]) -> None:
    base = configured_input_dir().resolve()
    for file_path in dict.fromkeys(file_paths):
        try:
            removed = _cleanup_one_sync(base, file_path)
        except Exception as exc:  # pragma: no cover - defensive IO guard
            logger.warning(
                "twindb import cleanup failed for %s: %s",
                Path(file_path).name,
                exc,
            )
            continue
        if removed:
            logger.info(
                "twindb import cleanup removed %d file/artifact(s) for %s",
                removed,
                Path(file_path).name,
            )


def _cleanup_one_sync(base: Path, file_path: str) -> int:
    removed = 0
    seen: set[Path] = set()
    for candidate in _candidate_paths(base, file_path):
        resolved = _confined(candidate, base)
        if resolved is None or resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists() and not resolved.is_symlink():
            continue
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        removed += 1
    return removed


def _candidate_paths(base: Path, file_path: str) -> list[Path]:
    raw = Path(str(file_path))
    basename = raw.name
    names = _cleanup_names(basename)
    parsed_root = base / PARSED_DIR_NAME

    candidates: list[Path] = []
    if not raw.is_absolute():
        candidates.append(base / raw)
    candidates.extend(base / name for name in names)
    candidates.extend(parsed_root / name for name in names)
    candidates.extend(parsed_root / f"{name}.parsed" for name in names)
    candidates.extend(_canonical_name_matches(base, names))
    candidates.extend(_canonical_name_matches(parsed_root, names))

    # LightRAG may add suffixes to sidecar directories on collision. Limit the
    # sweep to the exact parser-artifact prefix under INPUT_DIR/__parsed__.
    if parsed_root.exists():
        for name in names:
            prefix = f"{name}.parsed"
            candidates.extend(parsed_root.glob(f"{prefix}_*"))
    return candidates


def _canonical_name_matches(root: Path, names: list[str]) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    targets = {cleanup_name for name in names for cleanup_name in _cleanup_names(name)}
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    return [
        entry
        for entry in entries
        if any(cleanup_name in targets for cleanup_name in _cleanup_names(entry.name))
    ]


def _cleanup_names(basename: str) -> list[str]:
    return list(
        dict.fromkeys(
            name
            for name in (
                basename,
                canonicalize_parser_hinted_basename(basename),
                _strip_final_parser_hint(basename),
            )
            if name
        )
    )


def _strip_final_parser_hint(basename: str) -> str:
    return re.sub(r"\.\[[^\]/]+\](\.[^./\\]+)$", r"\1", basename, count=1)


def _confined(path: Path, base: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=False)
    except OSError:
        return None
    return resolved if resolved == base or base in resolved.parents else None
