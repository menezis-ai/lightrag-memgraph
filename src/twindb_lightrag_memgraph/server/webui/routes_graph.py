"""Knowledge graph endpoints for the Twin WebUI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..idp_jwt import require_admin_user
from ..webui_models import (
    GraphEntity,
    GraphEntityCreate,
    GraphEntityPatch,
    GraphRelation,
    GraphRelationCreate,
    GraphRelationPatch,
)
from .events import _make_event
from .store import get_store

router = APIRouter()

# 409 detail for a refused mutation on a mixed-provenance (cross-folder) record.
# The physical node/edge is co-owned by another folder; a global write would
# corrupt that folder's view, so the operator must act on the record from a
# context that owns all of its provenance (or via a future folder-local override).
_MIXED_PROVENANCE_DETAIL = (
    "This {kind} is shared with another folder (mixed provenance). It cannot be "
    "edited or deleted from the current folder, because the change would alter a "
    "record another folder co-owns."
)


def _graph_memgraph_label() -> str:
    """Resolve the Cypher label LightRAG uses for entity nodes.

    Until per-folder isolation lands at the LightRAG layer (one
    workspace per Twin folder), the KG view is globally scoped to the
    single LightRAG workspace configured by the deploy (env var
    `MEMGRAPH_WORKSPACE` / `WORKSPACE`). The Twin folder catalog still
    drives UX (default vs sandbox) but the underlying graph is shared.
    """
    from ..._constants import resolve_workspace

    return resolve_workspace()


async def _validate_graph_entity_tags(
    tags: list[str] | None,
) -> None:
    """Enforce that node tags belong to the active tag catalog
    (TR-KG-03 / QA report 2026-06-12).

    Both ``PATCH /twin/api/graph/entities/{id}`` and
    ``POST /twin/api/graph/entities`` accept a ``tags`` field on the
    request body. Before this gate, the helpers in
    ``graph_reader.py`` serialised whatever list was sent into
    ``twin_tags_json`` with no check — a curl/API caller could write
    arbitrary strings onto a node, contradicting the canonical tag
    catalog the WebUI surfaces.

    Allowed = ``{entry["tag"] for entry in await store.list_tags()
                  if entry.get("status") == "active"}``. Other statuses
    (``pending-promotion``, ``pending-review``, ``deprecated``,
    ``rejected``) are intentionally rejected: a non-active tag is not
    part of the operational vocabulary, and writing it on a graph
    node would silently re-promote / re-introduce it.

    On unknown tags, raises ``HTTPException(422)`` with a detail that
    lists the rejected values plus a bounded sample of allowed ones
    so the caller has actionable feedback without leaking the entire
    catalog when it is large.
    """
    if not tags:
        return
    store = get_store()
    catalog_entries = await store.list_tags()
    allowed = {
        entry["tag"]
        for entry in catalog_entries
        if isinstance(entry, dict)
        and isinstance(entry.get("tag"), str)
        and entry.get("status") == "active"
    }
    unknown = sorted({t for t in tags if isinstance(t, str)} - allowed)
    if not unknown:
        return
    sample = sorted(allowed)[:10]
    suffix = "" if len(allowed) <= 10 else f" (+{len(allowed) - 10} more)"
    raise HTTPException(
        422,
        (
            f"Unknown node tag(s): {', '.join(unknown)}. "
            f"Allowed (active catalog): {', '.join(sample)}{suffix}."
        ),
    )


def _graph_seed_fallback_allowed() -> bool:
    """Audit C5: serve the in-memory demo graph as a fallback ONLY when
    we are explicitly in demo mode AND no IdP is configured.

    Allowed iff:
      * IdP is dormant (``get_active_config() is None``), AND
      * the active ``WebuiStore`` was built with ``webui_stores="seed"``.

    Rejected when:
      * IdP is configured (production deploy, regardless of store mode),
      * the store was built with ``webui_stores="memgraph"``,
      * the store reports an unknown mode (defensive default — refuse
        rather than guess).

    The check is **explicit on configuration**, not inferred from
    store contents. A seed store with no graph fixtures is still
    "demo mode" by construction; conversely, a memgraph store that
    happens to be empty is still production. Confusing data with
    config is exactly the bug this gate closes.
    """
    from .. import idp_jwt

    if idp_jwt.get_active_config() is not None:
        return False
    return get_store().mode == "seed"


async def _native_graph(label: str, max_nodes: int, max_depth: int):
    """Run LightRAG's native focus+context selection once.

    Returns ``(entities, relations)`` when the native graph ran (possibly an
    empty, folder-scoped result), or ``None`` when it is **unavailable** (host
    RAG absent / ``get_knowledge_graph`` failed) — the only case eligible for the
    demo seed fallback. ``_get_rag`` is resolved through the legacy shim at
    runtime so existing monkeypatches of ``server.webui_router._get_rag`` keep
    working.
    """
    from .. import graph_reader
    from .. import webui_router as legacy

    ws = _graph_memgraph_label()
    try:
        rag = legacy._get_rag()
    except Exception:
        return None
    return await graph_reader.read_graph_native(
        rag, ws, node_label=label, max_nodes=max_nodes, max_depth=max_depth
    )


def _seed_or_empty_entities() -> list[dict[str, Any]]:
    if not _graph_seed_fallback_allowed():
        return []
    return get_store().list_graph_entities()


@router.get("/graph/entities", response_model=list[GraphEntity])
async def list_graph_entities(
    label: str = "*",
    max_nodes: int = 1000,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Live Memgraph entities via LightRAG's native focus+context selection."""
    from ..folder import active_folder_id

    result = await _native_graph(label, max_nodes, max_depth)
    if result is None:
        # Native graph unavailable (no rag / call failed) → demo seed eligible.
        return _seed_or_empty_entities()
    entities, _ = result
    if entities:
        return entities
    # Native ran but empty. With a folder bound this is a legitimately empty
    # folder — NEVER fall back to the unscoped seed graph (cross-folder leak).
    if active_folder_id() is not None:
        return []
    return _seed_or_empty_entities()


@router.get("/graph/relations", response_model=list[GraphRelation])
async def list_graph_relations(
    label: str = "*",
    max_nodes: int = 1000,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """Relations for the same native subgraph as ``/graph/entities``."""
    from ..folder import active_folder_id

    result = await _native_graph(label, max_nodes, max_depth)
    if result is None:
        if not _graph_seed_fallback_allowed():
            return []
        return get_store().list_graph_relations()
    entities, relations = result
    if entities:
        return relations
    if active_folder_id() is not None:
        return []
    if not _graph_seed_fallback_allowed():
        return []
    return get_store().list_graph_relations()


@router.get("/graph/search")
async def search_graph_entities(q: str, limit: int = 50) -> list[str]:
    """Entity-label search — folder-scoped when a Twin folder is bound so the
    search box never reveals out-of-folder labels."""
    from .. import graph_reader
    from .. import webui_router as legacy

    try:
        rag = legacy._get_rag()
    except Exception:
        return []
    return await graph_reader.search_graph_labels(
        rag, q, workspace=_graph_memgraph_label(), limit=limit
    )


@router.patch(
    "/graph/entities/{entity_id}",
    response_model=GraphEntity,
    dependencies=[Depends(require_admin_user)],
    responses={
        404: {"description": "Graph entity not found"},
        409: {"description": "Entity shared with another folder (mixed provenance)"},
        422: {"description": "Invalid graph entity tags"},
    },
)
async def update_graph_entity_endpoint(
    entity_id: str, body: GraphEntityPatch
) -> dict[str, Any]:
    """Persist an edit to a graph entity in Memgraph."""
    from .. import graph_reader

    label = _graph_memgraph_label()
    patch_dict = body.model_dump(exclude_unset=True)
    await _validate_graph_entity_tags(patch_dict.get("tags"))
    try:
        updated = await graph_reader.update_graph_entity(label, entity_id, patch_dict)
    except graph_reader.MixedProvenanceError:
        raise HTTPException(409, _MIXED_PROVENANCE_DETAIL.format(kind="entity"))
    if updated is None:
        raise HTTPException(
            404, f"Graph entity '{entity_id}' not found in workspace '{label}'"
        )
    store = get_store()
    event = _make_event(
        kind="graph-entity-edited",
        sev="info",
        actor="operator",
        target_label=updated.get("name") or entity_id,
        summary=f"Graph entity '{updated.get('name') or entity_id}' updated",
        meta={
            "entity_id": entity_id,
            "patch_keys": list(patch_dict.keys()),
        },
        target_type="entity",
    )
    await store.record_activity(event)
    return updated


@router.post(
    "/graph/entities",
    response_model=GraphEntity,
    status_code=201,
    dependencies=[Depends(require_admin_user)],
    responses={
        409: {"description": "Graph entity already exists"},
        422: {"description": "Invalid graph entity tags"},
        500: {"description": "Graph projection failed"},
        503: {"description": "Graph backend rejected the write"},
    },
)
async def create_graph_entity_endpoint(
    body: GraphEntityCreate,
) -> dict[str, Any]:
    """Manually add a new entity to the KB."""
    from .. import graph_reader

    label = _graph_memgraph_label()
    payload = body.model_dump(exclude_unset=True)
    await _validate_graph_entity_tags(payload.get("tags"))
    try:
        entity = await graph_reader.create_graph_entity(label, payload)
    except graph_reader.EntityExistsError:
        raise HTTPException(
            409,
            f"Graph entity '{body.name}' already exists in workspace '{label}'",
        )
    except graph_reader.EntityProjectionError:
        raise HTTPException(
            500,
            (
                f"Graph entity '{body.name}' was created in workspace "
                f"'{label}' but the projection failed. Refresh "
                "/twin/api/graph/entities to surface it."
            ),
        )
    except graph_reader.EntityCreateBackendError:
        raise HTTPException(
            503,
            (
                f"Graph entity '{body.name}' could not be created: the "
                "Memgraph backend rejected the write. Check server logs "
                "for the underlying error."
            ),
        )
    store = get_store()
    event = _make_event(
        kind="graph-entity-edited",
        sev="info",
        actor="operator",
        target_label=entity.get("name") or body.name,
        summary=f"Graph entity '{entity.get('name') or body.name}' created",
        meta={
            "entity_id": entity["id"],
            "patch_keys": list(payload.keys()),
            "operation": "create",
        },
        target_type="entity",
    )
    await store.record_activity(event)
    return entity


@router.delete(
    "/graph/entities/{entity_id}",
    status_code=204,
    dependencies=[Depends(require_admin_user)],
    responses={
        404: {"description": "Graph entity not found"},
        409: {"description": "Entity shared with another folder (mixed provenance)"},
    },
)
async def delete_graph_entity_endpoint(entity_id: str) -> None:
    """Remove an entity from the KB and cascade-delete its edges."""
    from .. import graph_reader

    label = _graph_memgraph_label()
    try:
        ok = await graph_reader.delete_graph_entity(label, entity_id)
    except graph_reader.MixedProvenanceError:
        raise HTTPException(409, _MIXED_PROVENANCE_DETAIL.format(kind="entity"))
    if not ok:
        raise HTTPException(
            404, f"Graph entity '{entity_id}' not found in workspace '{label}'"
        )
    store = get_store()
    event = _make_event(
        kind="graph-entity-edited",
        sev="info",
        actor="operator",
        target_label=entity_id,
        summary=f"Graph entity '{entity_id}' deleted",
        meta={"entity_id": entity_id, "operation": "delete"},
        target_type="entity",
    )
    await store.record_activity(event)
    return None


@router.post(
    "/graph/relations",
    response_model=GraphRelation,
    status_code=201,
    dependencies=[Depends(require_admin_user)],
    responses={
        409: {"description": "Endpoint shared with another folder (mixed provenance)"},
        422: {"description": "Invalid graph relation"},
    },
)
async def create_graph_relation_endpoint(
    body: GraphRelationCreate,
) -> dict[str, Any]:
    """Manually add a new relation between two entities."""
    from .. import graph_reader

    label = _graph_memgraph_label()
    payload = body.model_dump(exclude_unset=True)
    try:
        relation = await graph_reader.create_graph_relation(label, payload)
    except graph_reader.MixedProvenanceError:
        raise HTTPException(409, _MIXED_PROVENANCE_DETAIL.format(kind="entity"))
    if relation is None:
        raise HTTPException(
            422,
            "Cannot create relation — one or both endpoints are missing, "
            "or the label is empty.",
        )
    store = get_store()
    event = _make_event(
        kind="graph-relation-edited",
        sev="info",
        actor="operator",
        target_label=relation.get("label") or body.label,
        summary=f"Graph relation '{relation.get('label') or body.label}' created",
        meta={
            "rel_id": relation["id"],
            "source": body.source,
            "target": body.target,
            "operation": "create",
        },
        target_type="relation",
    )
    await store.record_activity(event)
    return relation


@router.delete(
    "/graph/relations/{rel_id}",
    status_code=204,
    dependencies=[Depends(require_admin_user)],
    responses={
        404: {"description": "Graph relation not found"},
        409: {"description": "Relation shared with another folder (mixed provenance)"},
    },
)
async def delete_graph_relation_endpoint(rel_id: str) -> None:
    """Remove a relation from the KB."""
    from .. import graph_reader

    label = _graph_memgraph_label()
    try:
        ok = await graph_reader.delete_graph_relation(label, rel_id)
    except graph_reader.MixedProvenanceError:
        raise HTTPException(409, _MIXED_PROVENANCE_DETAIL.format(kind="relation"))
    if not ok:
        raise HTTPException(
            404,
            f"Graph relation '{rel_id}' not found. The relation may have "
            "been removed, or the server restarted since the last read — "
            "refresh the Graph tab to repopulate the endpoint cache.",
        )
    store = get_store()
    event = _make_event(
        kind="graph-relation-edited",
        sev="info",
        actor="operator",
        target_label=rel_id,
        summary=f"Graph relation '{rel_id}' deleted",
        meta={"rel_id": rel_id, "operation": "delete"},
        target_type="relation",
    )
    await store.record_activity(event)
    return None


@router.patch(
    "/graph/relations/{rel_id}",
    response_model=GraphRelation,
    dependencies=[Depends(require_admin_user)],
    responses={
        404: {"description": "Graph relation not found"},
        409: {"description": "Relation shared with another folder (mixed provenance)"},
    },
)
async def update_graph_relation_endpoint(
    rel_id: str, body: GraphRelationPatch
) -> dict[str, Any]:
    """Persist an edit to a graph relation in Memgraph."""
    from .. import graph_reader

    label = _graph_memgraph_label()
    patch_dict = body.model_dump(exclude_unset=True)
    try:
        updated = await graph_reader.update_graph_relation(label, rel_id, patch_dict)
    except graph_reader.MixedProvenanceError:
        raise HTTPException(409, _MIXED_PROVENANCE_DETAIL.format(kind="relation"))
    if updated is None:
        raise HTTPException(
            404,
            f"Graph relation '{rel_id}' not found. The relation may have "
            "been removed, or the server restarted since the last read — "
            "refresh the Graph tab to repopulate the endpoint cache.",
        )
    store = get_store()
    event = _make_event(
        kind="graph-relation-edited",
        sev="info",
        actor="operator",
        target_label=updated.get("label") or rel_id,
        summary=f"Graph relation '{updated.get('label') or rel_id}' updated",
        meta={
            "rel_id": rel_id,
            "patch_keys": list(patch_dict.keys()),
        },
        target_type="relation",
    )
    await store.record_activity(event)
    return updated
