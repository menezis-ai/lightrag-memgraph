"""HTTP request binding for Twin folders."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Mapping

from fastapi import HTTPException, Request

from .._constants import validate_identifier
from .._folders import (
    TwinFolder,
    TwinFolderCatalog,
    load_folder_catalog as _load_env_folder_catalog,
)
from . import folder_store

_active_folder_id: ContextVar[str | None] = ContextVar(
    "twin_active_folder_id",
    default=None,
)


def load_folder_catalog() -> TwinFolderCatalog:
    """Return the merged Twin folder catalog (env seed + runtime admin
    additions). Env seed always wins on id collision so a corrupt
    runtime store can't shadow the SRE-provisioned default.

    `max_folders` is taken from the env catalog. Folders beyond that cap
    are dropped from the head (env folders first, then runtime).
    """
    env_catalog = _load_env_folder_catalog()
    env_ids = {s.id for s in env_catalog.folders}
    runtime = [s for s in folder_store.list_runtime_folders() if s.id not in env_ids]
    merged = (*env_catalog.folders, *runtime)
    merged = merged[: env_catalog.max_folders]
    return TwinFolderCatalog(
        default_folder_id=env_catalog.default_folder_id,
        max_folders=env_catalog.max_folders,
        folders=tuple(merged),
        explicit=env_catalog.explicit or bool(runtime),
    )


def is_env_seeded_folder(folder_id: str) -> bool:
    """True when ``folder_id`` came from the env seed (mutations on it
    must 403 — only the SRE provisioning channel can change those)."""
    return folder_id in {s.id for s in _load_env_folder_catalog().folders}


def build_runtime_folder_config() -> dict:
    """Mirror `_folders.build_runtime_folder_config` but consume the
    merged catalog so the React port sees runtime additions in
    `window.__twinConfig.folders`."""
    catalog = load_folder_catalog()
    folders = [folder.as_runtime_config() for folder in catalog.folders]
    return {
        "defaultFolderId": catalog.default_folder_id,
        "folders": folders,
        "maxFolders": catalog.max_folders,
    }


def resolve_folder_from_headers(headers: Mapping[str, str]) -> str:
    catalog = load_folder_catalog()
    raw = headers.get("x-twin-folder")
    candidate = (raw or catalog.default_folder_id).strip()
    try:
        folder_id = validate_identifier(candidate, "folder")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if folder_id not in catalog.ids:
        raise HTTPException(
            status_code=403,
            detail="No folder available for this KB. Please contact Twincore Team",
        )
    return folder_id


async def bind_request_folder(request: Request):
    folder_id = resolve_folder_from_headers(request.headers)
    request.state.folder = folder_id
    token = _active_folder_id.set(folder_id)
    try:
        yield folder_id
    finally:
        _active_folder_id.reset(token)


def current_folder_id() -> str:
    return _active_folder_id.get() or load_folder_catalog().default_folder_id


__all__ = [
    "TwinFolder",
    "TwinFolderCatalog",
    "bind_request_folder",
    "build_runtime_folder_config",
    "current_folder_id",
    "is_env_seeded_folder",
    "load_folder_catalog",
    "resolve_folder_from_headers",
]
