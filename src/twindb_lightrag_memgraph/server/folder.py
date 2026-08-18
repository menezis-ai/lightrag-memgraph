"""HTTP request binding for Twin folders."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Annotated, Mapping

from fastapi import Header, HTTPException, Request

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
            detail=(
                "No folder is provisioned for this knowledge base. Ask your "
                "platform administrator to provision one."
            ),
        )
    return folder_id


def resolve_folder_for_request(request: Request) -> str:
    """Header + identity-aware folder resolution.

    Two-tier behaviour mirroring :func:`server.idp_jwt.require_admin_user`:

    - **IdP dormant**: the infrastructure root API key may select any catalog
      folder. Legacy JWTs, per-operator keys, and open-access requests have no
      authoritative folder claims and are confined to the default folder.
    - **IdP active**: header is resolved against the
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
        from .auth import is_infrastructure_root_request

        if is_infrastructure_root_request(request):
            return base_folder
        default_id = load_folder_catalog().default_folder_id
        if base_folder != default_id:
            raise HTTPException(
                status_code=403,
                detail="Folder claims are unavailable without the configured IdP",
            )
        return default_id

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


# Shared OpenAPI declaration for the folder-scoping header. Documentation
# only: the actual resolution reads the raw request headers so non-FastAPI
# callers (shim handlers, smoke runner) share one code path.
_FOLDER_HEADER = Header(
    alias="X-Twin-Folder",
    description=(
        "Folder to scope this request to. Must be a folder id from "
        "`GET /twin/api/folders`. Omitted: the catalog default folder "
        "is used. Unknown or out-of-scope folder ids are rejected "
        "with 403."
    ),
    examples=["general"],
)


async def document_folder_header(
    x_twin_folder: Annotated[str | None, _FOLDER_HEADER] = None,
) -> None:
    """No-op dependency that surfaces ``X-Twin-Folder`` in OpenAPI.

    For routes that resolve the folder manually inside their handler
    (``resolve_folder_for_request(request)``) instead of depending on
    :func:`bind_request_folder` — e.g. the query routes. Without it those
    operations advertise no parameters while honouring the header."""
    del x_twin_folder


async def bind_request_folder(  # NOSONAR - async contract.
    request: Request,
    x_twin_folder: Annotated[str | None, _FOLDER_HEADER] = None,
):
    # ``x_twin_folder`` is declared for OpenAPI documentation only — see
    # ``document_folder_header``.
    del x_twin_folder
    folder_id = resolve_folder_for_request(request)
    request.state.folder = folder_id
    token = _active_folder_id.set(folder_id)
    try:
        yield folder_id
    finally:
        _active_folder_id.reset(token)


def current_folder_id() -> str:
    return _active_folder_id.get() or load_folder_catalog().default_folder_id


def active_folder_id() -> str | None:
    """The folder bound for THIS request, or ``None`` when unbound.

    Unlike :func:`current_folder_id` (which falls back to the catalog default),
    this returns ``None`` off the Twin routes / when no request folder is bound,
    so storage and graph reads can apply the legacy *global* behaviour outside a
    folder-scoped request (compat). On a Twin WebUI route ``bind_request_folder``
    has set it to the resolved folder (the header value or the catalog default).
    """
    return _active_folder_id.get()


__all__ = [
    "TwinFolder",
    "TwinFolderCatalog",
    "active_folder_id",
    "bind_request_folder",
    "document_folder_header",
    "build_runtime_folder_config",
    "current_folder_id",
    "is_env_seeded_folder",
    "load_folder_catalog",
    "resolve_folder_for_request",
    "resolve_folder_from_headers",
]
