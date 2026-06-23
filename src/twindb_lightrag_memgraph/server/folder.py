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
    """Pure header-and-catalog resolution (palier 0).

    Validates that ``X-Twin-Folder`` (or the default folder when absent)
    is a known identifier inside the server-side catalog. Does NOT bind
    the result to the caller's identity — use
    :func:`resolve_folder_for_request` instead when an IdP may be
    active. Kept for unit tests and for callers that need to resolve a
    header outside of a request (e.g. the smoke runner manifest
    builder).
    """
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


def resolve_folder_for_request(request: Request) -> str:
    """Header + identity-aware folder resolution.

    Two-tier behaviour mirroring :func:`server.idp_jwt.require_admin_user`:

    - **Palier 1 — IdP dormant**: identical to
      :func:`resolve_folder_from_headers`. The route-level
      ``require_auth`` dep has already filtered anonymous; folder
      scoping is purely UX until MyAccess is wired.
    - **Palier 2 — IdP active**: header is resolved against the
      catalog, then bound to the caller's ``twin_folders`` claim
      (projected as ``user["folders"]`` by
      :func:`server.idp_jwt.claims_to_user`).

      * Folder ``∈ user.folders`` → returned.
      * ``user.folders`` empty (claim missing, e.g. early MyAccess
        rollout) → only the catalog default folder is accepted; any
        other header → 403. This is the user-chosen fallback for the
        migration period (audit 2026-06-10 C2).
      * Folder ``∉ user.folders`` and folders non-empty → 403.
    """
    from . import idp_jwt

    base_folder = resolve_folder_from_headers(request.headers)

    idp_config = idp_jwt.get_active_config()
    if idp_config is None:
        return base_folder

    user = idp_jwt.require_idp_user(request)
    if user is None:
        # Defensive: require_idp_user raises when IdP is configured.
        raise HTTPException(
            status_code=401,
            detail="Missing IdP credentials",
        )
    allowed = user.get("folders") or []
    if not allowed:
        default_id = load_folder_catalog().default_folder_id
        if base_folder != default_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Folder not in user scope. No twin_folders claim issued by "
                    "the IdP yet -- only the default folder is reachable."
                ),
            )
        return base_folder
    if base_folder not in allowed:
        raise HTTPException(
            status_code=403,
            detail="Folder not in user scope",
        )
    return base_folder


async def bind_request_folder(request: Request):  # NOSONAR - async contract.
    folder_id = resolve_folder_for_request(request)
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
    "resolve_folder_for_request",
    "resolve_folder_from_headers",
]
