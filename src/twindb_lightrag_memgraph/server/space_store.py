"""Runtime-mutable Twin space store.

Sits beside `_spaces.py` (env-only catalog) and persists operator
additions / edits / deletions. Two persistence modes:

- **In-memory** (default): mutations live for the lifetime of the
  process. Acceptable for dev / OVH standalone where the env catalog
  is the source of truth and runtime spaces are demo seeds.
- **JSON file**: when ``TWIN_SPACES_RUNTIME_FILE`` is set, every
  mutation rewrites the file atomically. Reads load it lazily on first
  access and on every restart. The file format is a flat list of
  ``TwinSpace.as_runtime_config()`` dicts.

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
from .._spaces import TwinSpace

logger = logging.getLogger(__name__)

_RUNTIME_FILE_ENV = "TWIN_SPACES_RUNTIME_FILE"

# Module-level state. Tests reset it via `reset_runtime_store()`.
_runtime_spaces: dict[str, TwinSpace] = {}
_loaded_from_disk = False
_lock = Lock()


def _runtime_file_path() -> str | None:
    path = os.environ.get(_RUNTIME_FILE_ENV)
    return path.strip() if path and path.strip() else None


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
        if not isinstance(data, list):
            logger.warning(
                "TWIN_SPACES_RUNTIME_FILE %s: not a JSON list, ignored", path
            )
            return
        for item in data:
            if not isinstance(item, dict):
                continue
            sid_raw = str(item.get("id") or "").strip()
            if not sid_raw:
                continue
            try:
                sid = validate_identifier(sid_raw, "space")
            except ValueError:
                logger.warning(
                    "TWIN_SPACES_RUNTIME_FILE %s: skipping invalid id %r",
                    path,
                    sid_raw,
                )
                continue
            _runtime_spaces[sid] = TwinSpace(
                id=sid,
                label=str(item.get("label") or sid),
                kind=str(item.get("kind") or "custom"),
                description=str(item.get("description") or ""),
                sources=int(item.get("sources") or 0),
            )
    except Exception:
        logger.exception(
            "TWIN_SPACES_RUNTIME_FILE %s: failed to load — starting empty",
            path,
        )


def _persist_to_disk_if_configured() -> None:
    path = _runtime_file_path()
    if path is None:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = [
            space.as_runtime_config() for space in _runtime_spaces.values()
        ]
        # Atomic write: temp file in the same directory, then rename.
        dir_ = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".twin-spaces-", dir=dir_)
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
            "TWIN_SPACES_RUNTIME_FILE %s: failed to persist mutation", path
        )


def list_runtime_spaces() -> list[TwinSpace]:
    """Return the current runtime-managed spaces."""
    with _lock:
        _load_from_disk_if_configured()
        return list(_runtime_spaces.values())


def get_runtime_space(space_id: str) -> TwinSpace | None:
    with _lock:
        _load_from_disk_if_configured()
        return _runtime_spaces.get(space_id)


def add_runtime_space(
    *,
    space_id: str,
    label: str,
    kind: str = "custom",
    description: str = "",
) -> TwinSpace:
    """Add a new runtime space.

    Raises ``ValueError`` on invalid id; ``KeyError`` if already exists.
    """
    sid = validate_identifier(space_id.strip(), "space")
    with _lock:
        _load_from_disk_if_configured()
        if sid in _runtime_spaces:
            raise KeyError(sid)
        space = TwinSpace(
            id=sid,
            label=label.strip() or sid,
            kind=kind.strip() or "custom",
            description=description.strip(),
            sources=0,
        )
        _runtime_spaces[sid] = space
        _persist_to_disk_if_configured()
        return space


def update_runtime_space(
    space_id: str,
    *,
    label: str | None = None,
    description: str | None = None,
    kind: str | None = None,
) -> TwinSpace | None:
    """Update an existing runtime space. Returns ``None`` if not found."""
    with _lock:
        _load_from_disk_if_configured()
        current = _runtime_spaces.get(space_id)
        if current is None:
            return None
        updated = TwinSpace(
            id=current.id,
            label=label.strip() if label is not None else current.label,
            kind=kind.strip() if kind is not None else current.kind,
            description=description.strip()
            if description is not None
            else current.description,
            sources=current.sources,
        )
        _runtime_spaces[space_id] = updated
        _persist_to_disk_if_configured()
        return updated


def delete_runtime_space(space_id: str) -> bool:
    """Remove a runtime space. Returns ``True`` on success, ``False`` if
    no space with that id existed."""
    with _lock:
        _load_from_disk_if_configured()
        if space_id not in _runtime_spaces:
            return False
        del _runtime_spaces[space_id]
        _persist_to_disk_if_configured()
        return True


def reset_runtime_store() -> None:
    """Test helper — wipe in-memory state and force a re-read on next
    access. Does not delete the on-disk JSON file."""
    global _loaded_from_disk
    with _lock:
        _runtime_spaces.clear()
        _loaded_from_disk = False


__all__ = [
    "add_runtime_space",
    "delete_runtime_space",
    "get_runtime_space",
    "list_runtime_spaces",
    "reset_runtime_store",
    "update_runtime_space",
]
