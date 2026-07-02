"""Runtime-mutable Twin folder store.

Sits beside `_folders.py` (env-only catalog) and persists operator
additions / edits / deletions. Two persistence modes:

- **In-memory** (default): mutations live for the lifetime of the
  process. Acceptable for dev / standalone demos where the env catalog
  is the source of truth and runtime folders are demo seeds.
- **JSON file**: when ``TWIN_FOLDERS_RUNTIME_FILE`` is set, every
  mutation rewrites the file atomically. Reads load it lazily on first
  access and on every restart. The file format is a flat list of
  ``TwinFolder.as_runtime_config()`` dicts.

The store is FastAPI-free so unit tests can drive it without spinning
the full app.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from threading import Lock
from typing import Any

from .._constants import validate_identifier
from .._folders import TwinFolder

logger = logging.getLogger(__name__)

_RUNTIME_FILE_ENV = "TWIN_FOLDERS_RUNTIME_FILE"

# Module-level state. Tests reset it via `reset_runtime_store()`.
_runtime_folders: dict[str, TwinFolder] = {}
_loaded_from_disk = False
_lock = Lock()


def _runtime_file_path() -> str | None:
    path = os.environ.get(_RUNTIME_FILE_ENV)
    return path.strip() if path and path.strip() else None


def _runtime_folder_from_item(path: str, item: dict[str, Any]) -> TwinFolder | None:
    sid_raw = str(item.get("id") or "").strip()
    if not sid_raw:
        return None
    try:
        sid = validate_identifier(sid_raw, "folder")
    except ValueError:
        logger.warning(
            "Twin folder runtime file %s: skipping invalid id %r",
            path,
            sid_raw,
        )
        return None
    return TwinFolder(
        id=sid,
        label=str(item.get("label") or sid),
        kind=str(item.get("kind") or "custom"),
        description=str(item.get("description") or ""),
        sources=int(item.get("sources") or 0),
    )


def _runtime_folders_from_payload(path: str, data: Any) -> dict[str, TwinFolder]:
    if not isinstance(data, list):
        logger.warning("Twin folder runtime file %s: not a JSON list, ignored", path)
        return {}

    folders: dict[str, TwinFolder] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        folder = _runtime_folder_from_item(path, item)
        if folder is not None:
            folders[folder.id] = folder
    return folders


def _load_from_disk_if_configured() -> None:
    """Load the JSON file on first access; idempotent within a process."""
    global _loaded_from_disk
    if _loaded_from_disk:
        return
    path = _runtime_file_path()
    _loaded_from_disk = True
    if path is None:
        return
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _runtime_folders.update(_runtime_folders_from_payload(path, data))
    except Exception:
        logger.exception(
            "Twin folder runtime file %s: failed to load; starting empty",
            path,
        )


def _persist_to_disk_if_configured() -> None:
    path = _runtime_file_path()
    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = [folder.as_runtime_config() for folder in _runtime_folders.values()]
        # Atomic write: temp file in the same directory, then rename.
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".twin-folders-", dir=dir_)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    except Exception:
        logger.exception(
            "Twin folder runtime file %s: failed to persist mutation", path
        )


def list_runtime_folders() -> list[TwinFolder]:
    """Return the current runtime-managed folders."""
    with _lock:
        _load_from_disk_if_configured()
        return list(_runtime_folders.values())


def get_runtime_folder(folder_id: str) -> TwinFolder | None:
    with _lock:
        _load_from_disk_if_configured()
        return _runtime_folders.get(folder_id)


def add_runtime_folder(
    *,
    folder_id: str,
    label: str,
    kind: str = "custom",
    description: str = "",
) -> TwinFolder:
    """Add a new runtime folder.

    Raises ``ValueError`` on invalid id; ``KeyError`` if already exists.
    """
    sid = validate_identifier(folder_id.strip(), "folder")
    with _lock:
        _load_from_disk_if_configured()
        if sid in _runtime_folders:
            raise KeyError(sid)
        folder = TwinFolder(
            id=sid,
            label=label.strip() or sid,
            kind=kind.strip() or "custom",
            description=description.strip(),
            sources=0,
        )
        _runtime_folders[sid] = folder
        _persist_to_disk_if_configured()
        return folder


def update_runtime_folder(
    folder_id: str,
    *,
    label: str | None = None,
    description: str | None = None,
    kind: str | None = None,
) -> TwinFolder | None:
    """Update an existing runtime folder. Returns ``None`` if not found."""
    with _lock:
        _load_from_disk_if_configured()
        current = _runtime_folders.get(folder_id)
        if current is None:
            return None
        updated = TwinFolder(
            id=current.id,
            label=label.strip() if label is not None else current.label,
            kind=kind.strip() if kind is not None else current.kind,
            description=(
                description.strip() if description is not None else current.description
            ),
            sources=current.sources,
        )
        _runtime_folders[folder_id] = updated
        _persist_to_disk_if_configured()
        return updated


def delete_runtime_folder(folder_id: str) -> bool:
    """Remove a runtime folder. Returns ``True`` on success, ``False`` if
    no folder with that id existed."""
    with _lock:
        _load_from_disk_if_configured()
        if folder_id not in _runtime_folders:
            return False
        del _runtime_folders[folder_id]
        _persist_to_disk_if_configured()
        return True


def reset_runtime_store() -> None:
    """Test helper — wipe in-memory state and force a re-read on next
    access. Does not delete the on-disk JSON file."""
    global _loaded_from_disk
    with _lock:
        _runtime_folders.clear()
        _loaded_from_disk = False


__all__ = [
    "add_runtime_folder",
    "delete_runtime_folder",
    "get_runtime_folder",
    "list_runtime_folders",
    "reset_runtime_store",
    "update_runtime_folder",
]
