"""Folder endpoints for the Twin WebUI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import folder_store
from ..folder import current_folder_id, is_env_seeded_folder, load_folder_catalog
from ..idp_jwt import require_admin_user
from ..webui_models import Folder, FolderCreate, FolderPatch
from .events import _make_event
from .store import _stores, get_store

router = APIRouter()


@router.get("/folders", response_model=list[Folder])
async def list_folders() -> list[dict[str, Any]]:
    active = current_folder_id()
    folders = [
        folder.as_api(current=folder.id == active)
        for folder in load_folder_catalog().folders
    ]
    counts = await _folder_source_counts(folders)
    for folder in folders:
        if folder["id"] in counts:
            folder["sources"] = counts[folder["id"]]
    return folders


async def _folder_source_counts(folders: list[dict[str, Any]]) -> dict[str, int]:
    """Best-effort DocStatus totals per folder.

    The folder catalog's ``sources`` field is a static provisioning hint. In the
    live runtime the selector must show actual DocStatus membership counts, or
    operators see ``0 sources`` while the Documents table is full.
    """
    try:
        from .. import webui_router as legacy

        rag = legacy._get_rag()
        get_status_counts = getattr(rag.doc_status, "get_status_counts", None)
        if not callable(get_status_counts):
            return {}
        counts: dict[str, int] = {}
        for folder in folders:
            by_status = await get_status_counts(folder=folder["id"])
            counts[folder["id"]] = sum(int(value or 0) for value in by_status.values())
        return counts
    except Exception:
        return {}


@router.post(
    "/folders",
    response_model=Folder,
    status_code=201,
    dependencies=[Depends(require_admin_user)],
    responses={
        409: {"description": "Folder conflict"},
        422: {"description": "Invalid folder payload"},
    },
)
async def create_folder(body: FolderCreate) -> dict[str, Any]:
    """Admin: provision a new Twin folder at runtime.

    Returns 201 + the new folder. Errors:
    - 409 if an env-seeded folder already owns this id (operator cannot
      shadow an SRE-provisioned default).
    - 409 if a runtime folder with this id already exists.
    - 422 if the id fails the safe-identifier rule.
    - 422 if adding this folder would exceed `maxFolders` from the env
      configuration.
    """
    catalog = load_folder_catalog()
    if len(catalog.folders) >= catalog.max_folders:
        raise HTTPException(
            422,
            f"Cannot create folder: catalog already at max ({catalog.max_folders}). "
            "Remove an existing folder first.",
        )
    if is_env_seeded_folder(body.id):
        raise HTTPException(
            409,
            f"Folder id '{body.id}' is provisioned by the deploy env "
            "and cannot be re-created via the API.",
        )
    try:
        folder = folder_store.add_runtime_folder(
            folder_id=body.id,
            label=body.label,
            kind=body.kind,
            description=body.description,
        )
    except KeyError as exc:
        raise HTTPException(409, f"Folder '{exc.args[0]}' already exists") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    store = get_store()
    event = _make_event(
        kind="settings",
        sev="info",
        actor="operator",
        target_label=folder.label,
        summary=f"Folder '{folder.id}' created ({folder.kind})",
        meta={"folder_id": folder.id, "operation": "create"},
        target_type="folder",
    )
    await store.record_activity(event)
    return folder.as_api(current=False)


@router.patch(
    "/folders/{folder_id}",
    response_model=Folder,
    dependencies=[Depends(require_admin_user)],
    responses={
        403: {"description": "Env-seeded folder cannot be edited"},
        404: {"description": "Folder not found"},
    },
)
async def update_folder(folder_id: str, body: FolderPatch) -> dict[str, Any]:
    """Admin: edit label / kind / description of a runtime folder.

    Env-seeded folders are 403 — those changes must go through the
    deploy env (`TWIN_FOLDERS_JSON`).
    """
    if is_env_seeded_folder(folder_id):
        raise HTTPException(
            403,
            f"Folder '{folder_id}' is env-seeded and cannot be edited via the API.",
        )
    patch = body.model_dump(exclude_unset=True)
    folder = folder_store.update_runtime_folder(
        folder_id,
        label=patch.get("label"),
        kind=patch.get("kind"),
        description=patch.get("description"),
    )
    if folder is None:
        raise HTTPException(404, f"Folder '{folder_id}' not found")
    store = get_store()
    event = _make_event(
        kind="settings",
        sev="info",
        actor="operator",
        target_label=folder.label,
        summary=f"Folder '{folder.id}' updated",
        meta={
            "folder_id": folder.id,
            "operation": "update",
            "patch_keys": list(patch.keys()),
        },
        target_type="folder",
    )
    await store.record_activity(event)
    active = current_folder_id()
    return folder.as_api(current=folder.id == active)


@router.delete(
    "/folders/{folder_id}",
    status_code=204,
    dependencies=[Depends(require_admin_user)],
    responses={
        403: {"description": "Env-seeded folder cannot be deleted"},
        404: {"description": "Folder not found"},
        409: {"description": "Folder still has data"},
    },
)
async def delete_folder(folder_id: str) -> None:
    """Admin: remove a runtime folder.

    - 403 if env-seeded (only the deploy env can remove those).
    - 404 if no runtime folder with this id exists.
    - 409 if the folder still has WebUI data (tags, activity events,
      docs scoped to it). Refusing to delete avoids orphaning state.
    """
    if is_env_seeded_folder(folder_id):
        raise HTTPException(
            403,
            f"Folder '{folder_id}' is env-seeded and cannot be deleted via the API.",
        )
    # Probe the in-memory store for residual data before deleting.
    bound_store = _stores.get(folder_id)
    if bound_store is not None:
        has_docs = len(bound_store._documents) > 0  # noqa: SLF001
        # `list_tags` is sync on the in-memory backend, async on the
        # Memgraph one — normalise via inspect.iscoroutine.
        tags_result = bound_store.tags.list_tags()
        if hasattr(tags_result, "__await__"):
            tags_result = await tags_result  # type: ignore[assignment]
        has_tags = len(tags_result) > 0
        if has_docs or has_tags:
            raise HTTPException(
                409,
                f"Folder '{folder_id}' still has data (docs and/or tags). "
                "Remove the contents before deleting the folder.",
            )
    if not folder_store.delete_runtime_folder(folder_id):
        raise HTTPException(404, f"Folder '{folder_id}' not found")
    # Evict the per-folder WebUI store so future GETs don't resurrect it.
    _stores.pop(folder_id, None)
    store = get_store(load_folder_catalog().default_folder_id)
    event = _make_event(
        kind="settings",
        sev="info",
        actor="operator",
        target_label=folder_id,
        summary=f"Folder '{folder_id}' deleted",
        meta={"folder_id": folder_id, "operation": "delete"},
        target_type="folder",
    )
    await store.record_activity(event)
    return None
