"""Folder endpoints for the Twin WebUI."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .. import folder_store
from ..folder import current_folder_id, is_env_seeded_folder, load_folder_catalog
from ..idp_jwt import require_admin_user
from ..webui_models import Folder, FolderCreate, FolderPatch
from .events import _make_event
from .store import _stores, deployment_store_mode, ensure_folder_store, get_store

logger = logging.getLogger(__name__)

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
    # MG-3 (audit 2026-07-02): build the new folder's store with the
    # deployment's backend mode right away, awaiting the Memgraph index /
    # taxonomy init in the creating worker. Never fails the create: with
    # Memgraph temporarily unreachable the catalog entry stays valid and the
    # store self-heals on first access (lazy init) and at next boot.
    try:
        await ensure_folder_store(folder.id)
    except Exception:
        logger.exception(
            "Folder %r created but store backend init failed; "
            "lazy init will retry on first access",
            folder.id,
        )
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


async def _count_single(session, query: str, **params: Any) -> int:
    result = await session.run(query, **params)
    record = await result.single()
    await result.consume()
    return int(record["c"]) if record else 0


async def _memgraph_residual_data(folder_id: str) -> dict[str, int]:
    """Counts of real folder data in Memgraph (MG-4, audit 2026-07-02).

    The pre-existing guard only probed the in-process store — always empty on
    a Memgraph deployment — so a folder whose documents' ONLY membership it
    was could be deleted, making those docs invisible everywhere and
    undeletable via the membership routes. Probe the surfaces where deletion
    would orphan real data:

    - ``(:DocStatus_{ws})-[:MEMBER_OF]->(:Folder_{ws} {id})`` memberships
      (documents — sole or shared: shared docs must be unshared through the
      membership routes first, exactly like the in-process contract's
      "remove the contents before deleting");
    - operator-created graph entities ``GRAPH_MEMBER_OF`` the folder (same
      orphan class, explicit provenance — see ``graph_reader.py`` #1a);
    - ``WebuiTag_{folder}`` catalog entries (the existing contract already
      refuses on tags).

    Activity events and notifications are deliberately NOT guard-blocking:
    they are append-only audit residue, so blocking on them would make any
    folder that was ever used permanently undeletable. They are cleaned up
    after a permitted delete instead (:func:`_cleanup_memgraph_folder_residue`).
    """
    from ... import _pool
    from ..._constants import resolve_workspace, validate_identifier

    workspace = resolve_workspace()
    folder = validate_identifier(folder_id, "folder")
    doc_label = f"DocStatus_{workspace}"
    folder_label = f"Folder_{workspace}"
    tag_label = f"WebuiTag_{folder}"
    counts: dict[str, int] = {}
    async with _pool.get_read_session() as session:
        counts["documents"] = await _count_single(
            session,
            f"MATCH (d:`{doc_label}`)-[:MEMBER_OF]->"
            f"(:`{folder_label}` {{id: $folder}}) "
            "RETURN count(DISTINCT d) AS c",
            folder=folder,
        )
        counts["graph entities"] = await _count_single(
            session,
            f"MATCH (e)-[:GRAPH_MEMBER_OF]->"
            f"(:`{folder_label}` {{id: $folder}}) "
            "RETURN count(DISTINCT e) AS c",
            folder=folder,
        )
        counts["tags"] = await _count_single(
            session,
            f"MATCH (t:`{tag_label}`) RETURN count(t) AS c",
        )
    return counts


async def _cleanup_memgraph_folder_residue(folder_id: str) -> None:
    """Remove the folder's store labels + ``Folder_{ws}`` node (MG-4 (b)).

    Runs only after :func:`_memgraph_residual_data` confirmed zero
    memberships / graph entities / tags. Removes what the audit flagged as
    stranded on delete: the per-folder store labels (``WebuiTag_*``,
    ``WebuiTagCategory_*``, ``WebuiActivity_*``, ``WebuiNotification_*``) and
    the ``Folder_{ws}`` node itself. The Folder-node delete keeps its own
    NOT-EXISTS guard so a membership written between guard and cleanup can
    never be detached (fail-closed: the node survives and the folder id's
    data is reachable again if the folder is re-created).
    """
    from ... import _pool
    from ..._constants import resolve_workspace, validate_identifier

    workspace = resolve_workspace()
    folder = validate_identifier(folder_id, "folder")
    doc_label = f"DocStatus_{workspace}"
    folder_label = f"Folder_{workspace}"
    store_labels = (
        f"WebuiTag_{folder}",
        f"WebuiTagCategory_{folder}",
        f"WebuiActivity_{folder}",
        f"WebuiNotification_{folder}",
    )
    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            for label in store_labels:
                result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                await result.consume()
            result = await session.run(
                f"MATCH (f:`{folder_label}` {{id: $folder}}) "
                f"WHERE NOT EXISTS((:`{doc_label}`)-[:MEMBER_OF]->(f)) "
                "AND NOT EXISTS(()-[:GRAPH_MEMBER_OF]->(f)) "
                "DETACH DELETE f",
                folder=folder,
            )
            await result.consume()


async def _guard_seed_folder_residual(folder_id: str) -> None:
    """Seed deployment: probe the in-memory store for residual data.

    Raises 409 when the folder still holds documents and/or tags.
    """
    bound_store = _stores.get(folder_id)
    if bound_store is None:
        return
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


async def _guard_memgraph_folder_residual(folder_id: str) -> None:
    """MG-4: on a Memgraph deployment the in-process store sees nothing —
    the real data lives in the database, so the database is the guard.

    Raises 503 when the probe or the residue cleanup cannot reach the
    database (fail-closed), 409 when the folder still has data.
    """
    try:
        residual_counts = await _memgraph_residual_data(folder_id)
    except Exception as exc:
        logger.exception("Folder %r delete: residual-data probe failed", folder_id)
        raise HTTPException(
            503,
            f"Cannot verify folder '{folder_id}' is empty (storage probe "
            "failed); refusing to delete.",
        ) from exc
    residual = {kind: n for kind, n in residual_counts.items() if n > 0}
    if residual:
        detail = ", ".join(f"{n} {kind}" for kind, n in residual.items())
        raise HTTPException(
            409,
            f"Folder '{folder_id}' still has data ({detail}). "
            "Remove the contents before deleting the folder.",
        )
    try:
        await _cleanup_memgraph_folder_residue(folder_id)
    except Exception as exc:
        logger.exception("Folder %r delete: residue cleanup failed", folder_id)
        raise HTTPException(
            503,
            f"Folder '{folder_id}' residue cleanup failed; folder was "
            "not deleted, retry later.",
        ) from exc


@router.delete(
    "/folders/{folder_id}",
    status_code=204,
    dependencies=[Depends(require_admin_user)],
    responses={
        403: {"description": "Env-seeded folder cannot be deleted"},
        404: {"description": "Folder not found"},
        409: {"description": "Folder still has data"},
        503: {"description": "Residual-data probe failed; delete refused"},
    },
)
async def delete_folder(folder_id: str) -> None:
    """Admin: remove a runtime folder.

    - 403 if env-seeded (only the deploy env can remove those).
    - 404 if no runtime folder with this id exists.
    - 409 if the folder still has data: WebUI tags scoped to it, and — on a
      Memgraph deployment — real DocStatus ``MEMBER_OF`` memberships or
      operator-created graph entities. Refusing to delete avoids orphaning
      state: a doc whose only membership is this folder would become
      invisible everywhere (MG-4).
    - 503 if the deployment is Memgraph-backed but the residual-data probe or
      the residue cleanup cannot reach the database (fail-closed — never
      delete unverified; the folder stays and the delete can be retried).

    On a permitted delete in a Memgraph deployment the folder's store labels
    (``WebuiTag_*``, ``WebuiTagCategory_*``, ``WebuiActivity_*``,
    ``WebuiNotification_*``) and its ``Folder_{workspace}`` node are removed
    so nothing is stranded.
    """
    if is_env_seeded_folder(folder_id):
        raise HTTPException(
            403,
            f"Folder '{folder_id}' is env-seeded and cannot be deleted via the API.",
        )
    if folder_store.get_runtime_folder(folder_id) is None:
        raise HTTPException(404, f"Folder '{folder_id}' not found")
    if deployment_store_mode() != "memgraph":
        await _guard_seed_folder_residual(folder_id)
    else:
        await _guard_memgraph_folder_residual(folder_id)
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
