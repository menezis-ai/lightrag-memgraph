"""Tag catalog and governance endpoints for the Twin WebUI."""

from __future__ import annotations

import json
import re
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Request

from ..folder import current_folder_id
from ..idp_jwt import require_admin_user
from ..webui_models import (
    AckResponse,
    TagApproveBody,
    TagCategory,
    TagDeleteBody,
    TagDeprecateBody,
    TagEditBody,
    TagEntry,
    TagReactivateBody,
    TagRejectBody,
    TagRequestBody,
    TagSuggestEditBody,
    TagSynonymsBody,
    ThesaurusEntry,
)
from ..webui_tagstore import MemgraphTagStore
from .events import _make_event, _make_notification, _request_actor, _utcnow_iso
from .store import WebuiStore, get_store

router = APIRouter(tags=["tags"])

_TAG_NAME_PATH = Path(
    description="Tag name, as listed by `GET /tags`.",
    examples=["network-segmentation"],
)


@router.get(
    "/thesaurus",
    response_model=list[ThesaurusEntry],
    summary="List thesaurus term expansions",
)
async def list_thesaurus() -> list[dict[str, Any]]:
    """Return the thesaurus of the active folder: canonical terms with
    the synonyms that queries expand through."""
    return await get_store().list_thesaurus()


@router.get(
    "/tags",
    response_model=list[TagEntry],
    summary="List the tag catalog",
)
async def list_tags() -> list[dict[str, Any]]:
    """Return every tag of the active folder with its definition,
    category, governance status (`active`, `pending-review`,
    `deprecated`, ...) and usage counters."""
    return await get_store().list_tags()


@router.get(
    "/tags/categories",
    response_model=list[TagCategory],
    summary="List tag categories",
)
async def list_tag_categories() -> list[dict[str, Any]]:
    """Return the tag categories (id, label, colour) used to group tags
    in the catalog."""
    return await get_store().list_tag_categories()


_CATEGORIES_TEMPLATE: list[dict[str, Any]] = [
    {"id": "network", "label": "Network", "color": "#1F8A7A"},
    {"id": "infra", "label": "Infrastructure", "color": "#5A7FB4"},
    {"id": "compliance", "label": "Compliance", "color": "#9C2D8E"},
    {"id": "operations", "label": "Operations", "color": "#C24A24"},
    {"id": "governance", "label": "Governance", "color": "#2C3E50"},
    {"id": "lifecycle", "label": "Lifecycle", "color": "#8A5C0E"},
]


@router.get(
    "/tags/categories/template",
    summary="Download the category template file",
)
def get_categories_template():
    """Download a starter `twin-categories.template.json` file. Edit it,
    then upload it through `POST /tags/categories/_import` to replace the
    folder's category taxonomy."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        content=_CATEGORIES_TEMPLATE,
        headers={
            "Content-Disposition": (
                'attachment; filename="twin-categories.template.json"'
            ),
        },
    )


@router.post(
    "/tags/categories/_import",
    response_model=AckResponse,
    dependencies=[Depends(require_admin_user)],
    summary="Replace the category taxonomy (admin)",
    responses={
        400: {"description": "Invalid categories payload"},
        503: {"description": "Categories backend unavailable"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "label"],
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "color": {
                                    "type": "string",
                                    "description": "Hex colour, e.g. #1F8A7A.",
                                },
                            },
                        },
                    },
                    "example": [
                        {"id": "network", "label": "Network", "color": "#1F8A7A"},
                        {"id": "compliance", "label": "Compliance", "color": "#9C2D8E"},
                    ],
                }
            },
        }
    },
)
async def import_categories(body: list[dict[str, Any]]) -> dict[str, Any]:
    """Replace (not merge) the active folder's tag categories with the
    uploaded list. Start from `GET /tags/categories/template`."""
    backend = get_store().tags
    if not isinstance(backend, MemgraphTagStore):
        raise HTTPException(
            status_code=503,
            detail=(
                "Categories import requires the Memgraph store backend "
                "(register with webui_stores='memgraph'). The current "
                "backend does not support taxonomy mutation."
            ),
        )

    try:
        await backend.replace_categories_from_list(body, source="import")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"ok": True}


def _validate_bulk_retag(targets, adds, removes) -> None:
    """Reject malformed bulk-retag payloads (size + shape limits)."""
    if not isinstance(targets, list) or not targets:
        raise HTTPException(
            status_code=400,
            detail="targets must be a non-empty list of doc_id strings.",
        )
    if len(targets) > 500:
        raise HTTPException(
            status_code=413,
            detail="bulk-retag accepts at most 500 target documents.",
        )
    if len(adds) + len(removes) > 50:
        raise HTTPException(
            status_code=413,
            detail="bulk-retag accepts at most 50 tag mutations.",
        )
    for tag in (*adds, *removes):
        if not isinstance(tag, str) or not tag.strip():
            raise HTTPException(
                status_code=400,
                detail="tags must be non-empty strings.",
            )


def _active_tag_ids(entries: list[dict[str, Any]]) -> set[str]:
    """Return exact catalog ids whose authoritative status is active."""
    return {
        entry["tag"]
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("tag"), str)
        and entry.get("status") == "active"
    }


def _unapproved_bulk_tags(
    requested: list[str], locked_rows: list[dict[str, Any]]
) -> list[str]:
    """Validate tag JSON read while its Memgraph nodes are write-locked."""
    entries: list[dict[str, Any]] = []
    for row in locked_rows:
        raw = row.get("data")
        if not isinstance(raw, str):
            continue
        try:
            entry = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(entry, dict) or entry.get("tag") != row.get("id"):
            continue
        entries.append(entry)
    return sorted(set(requested) - _active_tag_ids(entries))


def _unapproved_bulk_tag_error(unapproved: list[str]) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={
            "message": "Only active, approved tags may be attached",
            "unapproved_tags": unapproved,
        },
    )


async def _apply_bulk_tag_mutations(
    adds, removes, doc_ids, tag_label, doc_label, now, actor
) -> None:
    """Apply tag add/remove edges in a single write transaction (or no-op)."""
    if not (adds or removes):
        return
    from ... import _pool

    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            tx = await session.begin_transaction()
            try:
                if adds:
                    # list_tags() is only a fast API pre-check. Re-read and
                    # write-lock the catalog nodes in the mutation transaction
                    # so a concurrent deprecation cannot slip through the gap.
                    lock_token = secrets.token_urlsafe(24)
                    result = await tx.run(
                        f"""
                        UNWIND $tags AS tagId
                        MATCH (t:`{tag_label}` {{id: tagId}})
                        SET t.`__bulk_retag_lock` = $lock_token
                        RETURN t.id AS id, t.data AS data
                        """,
                        tags=adds,
                        lock_token=lock_token,
                    )
                    locked_rows = [dict(record) async for record in result]
                    await result.consume()
                    unapproved = _unapproved_bulk_tags(adds, locked_rows)
                    if unapproved:
                        raise _unapproved_bulk_tag_error(unapproved)

                    result = await tx.run(
                        f"""
                        UNWIND $tags AS tagId
                        MATCH (t:`{tag_label}` {{id: tagId}})
                        WHERE t.`__bulk_retag_lock` = $lock_token
                        WITH t
                        UNWIND $docs AS docId
                        MATCH (d:`{doc_label}` {{id: docId}})
                        MERGE (d)-[r:TAGGED_WITH]->(t)
                          ON CREATE SET r.at = $now, r.actor = $actor
                        """,
                        tags=adds,
                        docs=doc_ids,
                        now=now,
                        actor=actor,
                        lock_token=lock_token,
                    )
                    await result.consume()

                if removes:
                    result = await tx.run(
                        f"""
                        UNWIND $docs AS docId
                        UNWIND $tags AS tagId
                        MATCH (d:`{doc_label}` {{id: docId}})-[r:TAGGED_WITH]->(t:`{tag_label}` {{id: tagId}})
                        DELETE r
                        """,
                        docs=doc_ids,
                        tags=removes,
                    )
                    await result.consume()

                if adds:
                    result = await tx.run(
                        f"""
                        UNWIND $tags AS tagId
                        MATCH (t:`{tag_label}` {{id: tagId}})
                        WHERE t.`__bulk_retag_lock` = $lock_token
                        REMOVE t.`__bulk_retag_lock`
                        """,
                        tags=adds,
                        lock_token=lock_token,
                    )
                    await result.consume()

                await tx.commit()
            except Exception:
                await tx.rollback()
                raise


async def _emit_bulk_retag_events(
    doc_ids, resulting_by_doc, existing, adds, removes, actor
) -> None:
    """Emit one ``doc-retagged`` activity event per affected document."""
    for doc_id in doc_ids:
        new_tags = resulting_by_doc.get(doc_id, [])
        event = _make_event(
            kind="doc-retagged",
            sev="info",
            actor=actor,
            target_label=existing[doc_id] or doc_id,
            summary=f"tags: +{','.join(adds) or '∅'} -{','.join(removes) or '∅'}",
            meta={
                "doc_id": doc_id,
                "adds": adds,
                "removes": removes,
                "resulting_tags": new_tags,
            },
            target_type="document",
        )
        await get_store().record_activity(event)


@router.post(
    "/documents/_bulk-retag",
    dependencies=[Depends(require_admin_user)],
    summary="Add/remove tags on several documents (admin)",
    responses={
        400: {"description": "Invalid bulk retag payload"},
        422: {"description": "One or more added tags are not active"},
        413: {"description": "More than 500 documents or 50 tag mutations"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["targets"],
                        "properties": {
                            "targets": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Document ids to retag (1 to 500).",
                            },
                            "adds": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Tags to add. Must be active, approved "
                                    "catalog tags."
                                ),
                            },
                            "removes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Tags to remove.",
                            },
                        },
                    },
                    "example": {
                        "targets": ["doc-a1b2c3d4", "doc-e5f6a7b8"],
                        "adds": ["gdpr"],
                        "removes": ["draft"],
                    },
                }
            },
        }
    },
)
async def bulk_retag_documents(
    body: dict[str, Any],
    request: Request,
) -> dict[str, Any]:
    """Apply tag additions and removals to up to 500 documents of the
    active folder in one operation. Only active, approved tags can be
    added (422 lists the offending ones). Documents not visible in the
    folder are reported in `failed`; each retagged document gets a
    `doc-retagged` audit event."""
    from ... import _pool
    from ..._constants import resolve_workspace

    targets = body.get("targets") or []
    adds = list(body.get("adds") or [])
    removes = list(body.get("removes") or [])
    actor = _request_actor(request)

    _validate_bulk_retag(targets, adds, removes)

    active_tags = _active_tag_ids(await get_store().list_tags())
    unapproved = sorted(set(adds) - active_tags)
    if unapproved:
        raise _unapproved_bulk_tag_error(unapproved)

    workspace = resolve_workspace()
    folder = current_folder_id()
    doc_label = f"DocStatus_{workspace}"
    folder_label = f"Folder_{workspace}"
    tag_label = f"WebuiTag_{folder}"
    now = _utcnow_iso()

    async with _pool.get_read_session() as session:
        result = await session.run(
            f"""
            UNWIND $ids AS id
            MATCH (n:`{doc_label}` {{id: id}})
            WHERE EXISTS((n)-[:MEMBER_OF]->(:`{folder_label}` {{id: $folder}}))
            RETURN n.id AS id, n.file_path AS file_path
            """,
            ids=targets,
            folder=folder,
        )
        existing: dict[str, str | None] = {}
        async for record in result:
            existing[record["id"]] = record.get("file_path")
        await result.consume()

    failed = [t for t in targets if t not in existing]
    if not existing:
        return {"updated": 0, "failed": failed}

    doc_ids = list(existing.keys())

    await _apply_bulk_tag_mutations(
        adds, removes, doc_ids, tag_label, doc_label, now, actor
    )

    async with _pool.get_read_session() as session:
        result = await session.run(
            f"""
            UNWIND $docs AS docId
            MATCH (d:`{doc_label}` {{id: docId}})
            OPTIONAL MATCH (d)-[:TAGGED_WITH]->(t:`{tag_label}`)
            RETURN docId, collect(t.id) AS tags
            """,
            docs=doc_ids,
        )
        resulting_by_doc: dict[str, list[str]] = {}
        async for record in result:
            tags = sorted(tid for tid in (record["tags"] or []) if tid)
            resulting_by_doc[record["docId"]] = tags
        await result.consume()

    await _emit_bulk_retag_events(
        doc_ids, resulting_by_doc, existing, adds, removes, actor
    )

    return {"updated": len(doc_ids), "failed": failed}


async def _emit_tag_audit(
    *,
    store: WebuiStore,
    actor: str,
    kind: str,
    sev: str,
    target_label: str,
    summary: str,
    meta: dict[str, Any],
    notification: dict[str, Any] | None,
    target_id: str | None = None,
) -> None:
    """Record an activity event and (optionally) push a notification."""
    event = _make_event(
        kind=kind,
        sev=sev,
        actor=actor,
        target_label=target_label,
        summary=summary,
        meta=meta,
        target_id=target_id or target_label,
    )
    await store.record_activity(event)
    if notification is not None:
        await store.push_notification(notification)


@router.post(
    "/tags",
    response_model=TagEntry,
    status_code=201,
    dependencies=[Depends(require_admin_user)],
    summary="Request a new tag (admin)",
    responses={409: {"description": "Tag already exists"}},
)
async def request_tag(body: TagRequestBody, request: Request) -> dict[str, Any]:
    """Propose a new tag for the catalog. The tag is created in
    `pending-review` status and only becomes usable on documents after a
    reviewer approves it (`POST /tags/{name}/approve`)."""
    store = get_store()
    existing = await store.tags.get_tag(body.tag)
    if existing is not None:
        if str(existing.get("status") or "").lower() in {"rejected", "deleted"}:
            await store.tags.delete_tag(body.tag)
        else:
            raise HTTPException(409, f"Tag '{body.tag}' already exists")
    actor = _request_actor(request)
    now = _utcnow_iso()[:10]
    entry: dict[str, Any] = {
        "tag": body.tag,
        "tier": "requested",
        "category": body.category,
        "status": "pending-review",
        "def": body.def_,
        "long_description": body.long_description or "",
        "aliases": list(body.aliases),
        "deprecates": [],
        "sources_count": 0,
        "chunks_count": 0,
        "query_freq_30d": 0,
        "created": {"by": actor, "at": now},
        "last_edit": {"by": actor, "at": now, "action": "requested"},
        "related": [],
        "examples": [],
        "requested_by": actor,
        "requested_at": now,
        "justification": body.justification or "",
    }
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=body.tag,
        summary=f"Tag {body.tag} requested for palier-3 review",
        meta={"category": body.category, "justification": body.justification},
        notification=_make_notification(
            title="Tag",
            tagname=body.tag,
            suffix="requested",
            sub=body.justification or f"category {body.category}",
        ),
    )
    return stored


_TAG_PROPOSAL_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _proposal_id_base(name: str, actor: str, now: str) -> str:
    safe_name = _TAG_PROPOSAL_ID_RE.sub("-", name).strip("-") or "tag"
    safe_actor = _TAG_PROPOSAL_ID_RE.sub("-", actor).strip("-") or "system"
    stamp = re.sub(r"\D", "", now)[:14] or "now"
    return f"{safe_name}__edit__{safe_actor}-{stamp}"


async def _next_proposal_id(store: WebuiStore, base: str) -> str:
    proposal_id = base
    suffix = 2
    while await store.tags.get_tag(proposal_id) is not None:
        proposal_id = f"{base}-{suffix}"
        suffix += 1
    return proposal_id


def _suggested_edit_fields(
    entry: dict[str, Any], body: TagSuggestEditBody
) -> dict[str, Any]:
    return {
        "def": body.def_ if body.def_ is not None else entry.get("def", ""),
        "long_description": (
            body.long_description
            if body.long_description is not None
            else entry.get("long_description", "")
        ),
        "category": (
            body.category
            if body.category is not None
            else entry.get("category", "infra")
        ),
        "aliases": (
            list(body.aliases)
            if body.aliases is not None
            else list(entry.get("aliases", []))
        ),
    }


def _changed_suggested_fields(
    entry: dict[str, Any],
    proposed: dict[str, Any],
) -> list[str]:
    changed: list[str] = []
    for field in ("def", "long_description", "category", "aliases"):
        current = entry.get(field, [] if field == "aliases" else "")
        if proposed[field] != current:
            changed.append(field)
    return changed


def _approved_edit_fields(
    entry: dict[str, Any], target: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    changed: list[str] = []
    proposed_fields = [
        field
        for field in entry.get("proposed_fields", [])
        if field in {"def", "long_description", "category", "aliases"}
    ]
    for field in proposed_fields:
        if field in entry and entry[field] != target.get(field):
            target[field] = (
                list(entry[field]) if isinstance(entry[field], list) else entry[field]
            )
            changed.append(field)
    return target, changed


async def _approve_tag_edit(
    *,
    store: WebuiStore,
    proposal_name: str,
    entry: dict[str, Any],
    actor: str,
    now: str,
) -> dict[str, Any]:
    target_name = str(entry["target_tag"])
    target = await store.tags.get_tag(target_name)
    if target is None:
        raise HTTPException(404, f"Tag '{target_name}' not found")

    target, changed = _approved_edit_fields(entry, target)
    target["last_edit"] = {"by": actor, "at": now, "action": "edit-approved"}
    stored = await store.tags.upsert_tag(target)
    await store.tags.delete_tag(proposal_name)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=target_name,
        summary=f"Tag {target_name} edit approved ({', '.join(changed)})",
        meta={"proposal_id": proposal_name, "fields": changed},
        notification=_make_notification(
            title="Tag",
            tagname=target_name,
            suffix="edit approved",
            sub=", ".join(changed) or "no field change",
        ),
        target_id=target_name,
    )
    return stored


@router.post(
    "/tags/{name}/suggest-edit",
    response_model=TagEntry,
    status_code=201,
    dependencies=[Depends(require_admin_user)],
    summary="Suggest an edit to a tag (admin)",
    responses={
        400: {"description": "No edit provided"},
        404: {"description": "Tag not found"},
    },
)
async def suggest_tag_edit(
    name: Annotated[str, _TAG_NAME_PATH],
    body: TagSuggestEditBody,
    request: Request,
) -> dict[str, Any]:
    """Queue an edit proposal (definition, description, category or
    aliases) against an existing tag. The proposal waits in
    `pending-review`; approving it applies the changed fields to the
    target tag, rejecting it discards them. At least one field must
    differ from the current values (400 otherwise)."""
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = _request_actor(request)
    now_iso = _utcnow_iso()
    now = now_iso[:10]
    proposed = _suggested_edit_fields(entry, body)
    changed = _changed_suggested_fields(entry, proposed)
    if not changed:
        raise HTTPException(400, "Suggest edit requires at least one changed field")

    proposal_id = await _next_proposal_id(
        store, _proposal_id_base(name, actor, now_iso)
    )
    proposal: dict[str, Any] = {
        "tag": proposal_id,
        "tier": "requested",
        "category": proposed["category"],
        "status": "pending-review",
        "def": proposed["def"],
        "long_description": proposed["long_description"],
        "aliases": proposed["aliases"],
        "deprecates": list(entry.get("deprecates", [])),
        "sources_count": int(entry.get("sources_count", 0) or 0),
        "chunks_count": int(entry.get("chunks_count", 0) or 0),
        "query_freq_30d": int(entry.get("query_freq_30d", 0) or 0),
        "created": {"by": actor, "at": now},
        "last_edit": {"by": actor, "at": now, "action": "edit-suggested"},
        "related": list(entry.get("related", [])),
        "examples": list(entry.get("examples", [])),
        "requested_by": actor,
        "requested_at": now,
        "justification": body.justification or "",
        "proposal_kind": "edit",
        "target_tag": name,
        "proposed_fields": changed,
    }
    stored = await store.tags.upsert_tag(proposal)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=name,
        summary=f"Tag {name} edit suggested for palier-3 review",
        meta={
            "proposal_id": proposal_id,
            "fields": changed,
            "justification": body.justification,
        },
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="edit suggested",
            sub=body.justification or ", ".join(changed),
        ),
        target_id=name,
    )
    return stored


@router.post(
    "/tags/{name}/approve",
    response_model=TagEntry,
    dependencies=[Depends(require_admin_user)],
    summary="Approve a pending tag or edit proposal (admin)",
    responses={404: {"description": "Tag not found"}},
)
async def approve_tag(
    name: Annotated[str, _TAG_NAME_PATH],
    body: TagApproveBody,
    request: Request,
) -> dict[str, Any]:
    """Approve a pending tag request (the tag becomes `active` and usable
    on documents) or a pending edit proposal (the proposed fields are
    applied to the target tag and the proposal is removed)."""
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = _request_actor(request)
    now = _utcnow_iso()[:10]
    if entry.get("proposal_kind") == "edit" and entry.get("target_tag"):
        return await _approve_tag_edit(
            store=store,
            proposal_name=name,
            entry=entry,
            actor=actor,
            now=now,
        )
    entry["tier"] = 3
    entry["status"] = "active"
    entry["last_edit"] = {"by": actor, "at": now, "action": "approved"}
    entry.pop("requested_by", None)
    entry.pop("requested_at", None)
    entry.pop("justification", None)
    entry.pop("reject_reason", None)
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=name,
        summary=f"Tag {name} approved (now Tier 3)",
        meta={"tier": 3, "status": "active"},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="approved",
            sub="Added to tag catalog · Tier 3",
        ),
    )
    return stored


@router.post(
    "/tags/{name}/reject",
    response_model=TagEntry,
    dependencies=[Depends(require_admin_user)],
    summary="Reject a pending tag or edit proposal (admin)",
    responses={404: {"description": "Tag not found"}},
)
async def reject_tag(
    name: Annotated[str, _TAG_NAME_PATH],
    body: TagRejectBody,
    request: Request,
) -> dict[str, Any]:
    """Reject a pending tag request or edit proposal with a reason. The
    entry is removed from the catalog; the reason is recorded in the
    audit feed and the returned entry."""
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = _request_actor(request)
    now = _utcnow_iso()[:10]
    entry["status"] = "rejected"
    entry["last_edit"] = {"by": actor, "at": now, "action": "rejected"}
    entry["reject_reason"] = body.reason
    await store.tags.delete_tag(name)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="warning",
        target_label=name,
        summary=f"Tag {name} rejected — {body.reason}",
        meta={"reason": body.reason},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="rejected",
            sub=body.reason,
        ),
    )
    return entry


async def _resolve_tag_rename(store, entry, body) -> str | None:
    """Apply a tag rename to ``entry`` if requested; return the old name or None.

    Raises 400 on empty new name, 409 when the target name already exists.
    """
    if body.tag is None:
        return None
    new_name = body.tag.strip()
    if not new_name:
        raise HTTPException(400, "Tag name cannot be empty")
    if new_name == entry.get("tag"):
        return None
    existing = await store.tags.get_tag(new_name)
    if existing is not None:
        raise HTTPException(409, f"Tag '{new_name}' already exists")
    old_name = entry["tag"]
    entry["tag"] = new_name
    return old_name


def _apply_scalar_tag_edits(entry, body) -> list[str]:
    """Apply the non-rename scalar field edits; return the list of changed fields."""
    changed: list[str] = []
    if body.def_ is not None and body.def_ != entry.get("def"):
        entry["def"] = body.def_
        changed.append("def")
    if body.long_description is not None and body.long_description != entry.get(
        "long_description", ""
    ):
        entry["long_description"] = body.long_description
        changed.append("long_description")
    if body.category is not None and body.category != entry.get("category"):
        entry["category"] = body.category
        changed.append("category")
    if body.aliases is not None and body.aliases != entry.get("aliases"):
        entry["aliases"] = list(body.aliases)
        changed.append("aliases")
    if body.deprecates is not None:
        entry["deprecates"] = list(body.deprecates)
        changed.append("deprecates")
    return changed


async def _cascade_tag_rename(
    store, legacy, renamed_from, new_name, actor
) -> int | None:
    """Migrate seed + graph tag edges from ``renamed_from`` to ``new_name``."""
    seed_affected = legacy._cascade_seed_document_tags(
        store, name=renamed_from, strategy="migrate", to=new_name
    )
    graph_affected = await legacy._cascade_graph_tag_edges(
        name=renamed_from,
        strategy="migrate",
        to=new_name,
        actor=actor,
        strict=isinstance(store.tags, MemgraphTagStore),
    )
    return graph_affected if graph_affected is not None else seed_affected


@router.patch(
    "/tags/{name}",
    response_model=TagEntry,
    dependencies=[Depends(require_admin_user)],
    summary="Edit a tag directly (admin)",
    responses={
        400: {"description": "Invalid tag edit"},
        404: {"description": "Tag not found"},
        409: {"description": "Tag rename conflict"},
    },
)
async def edit_tag(
    name: Annotated[str, _TAG_NAME_PATH],
    body: TagEditBody,
    request: Request,
) -> dict[str, Any]:
    """Apply changes to a tag immediately (no review step): definition,
    description, category, aliases, deprecation list, or a rename. A
    rename migrates the tag's document links to the new name; the new
    name must not collide with an existing tag (409)."""
    from .. import webui_router as legacy

    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = _request_actor(request)
    now = _utcnow_iso()[:10]

    changed: list[str] = []
    renamed_from = await _resolve_tag_rename(store, entry, body)
    if renamed_from is not None:
        changed.append("tag")
    changed.extend(_apply_scalar_tag_edits(entry, body))
    entry["last_edit"] = {"by": actor, "at": now, "action": "edited"}
    stored = await store.tags.upsert_tag(entry)

    cascade_affected: int | None = None
    if renamed_from is not None:
        cascade_affected = await _cascade_tag_rename(
            store, legacy, renamed_from, entry["tag"], actor
        )
        await store.tags.delete_tag(renamed_from)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=entry.get("tag") or name,
        summary=(
            f"Tag {entry.get('tag') or name} edited "
            f"({', '.join(changed) or 'no-op'})"
        ),
        meta={
            "fields": changed,
            "renamed_from": renamed_from,
            "cascade_affected": cascade_affected,
        },
        notification=_make_notification(
            title="Tag",
            tagname=entry.get("tag") or name,
            suffix="updated",
            sub=", ".join(changed) or "no field change",
        ),
    )
    return stored


@router.post(
    "/tags/{name}/deprecate",
    response_model=TagEntry,
    dependencies=[Depends(require_admin_user)],
    summary="Deprecate a tag (admin)",
    responses={404: {"description": "Tag not found"}},
)
async def deprecate_tag(
    name: Annotated[str, _TAG_NAME_PATH],
    body: TagDeprecateBody,
    request: Request,
) -> dict[str, Any]:
    """Mark a tag as deprecated: it stays on already-tagged documents but
    can no longer be added to new ones. Reversible with
    `POST /tags/{name}/reactivate`."""
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = _request_actor(request)
    now = _utcnow_iso()[:10]
    entry["status"] = "deprecated"
    entry["last_edit"] = {"by": actor, "at": now, "action": "deprecated"}
    if body.reason:
        entry["deprecate_reason"] = body.reason
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="warning",
        target_label=name,
        summary=f"Tag {name} deprecated" + (f" — {body.reason}" if body.reason else ""),
        meta={"reason": body.reason},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="deprecated",
            sub=body.reason or "Excluded from default retrieval",
        ),
    )
    return stored


@router.post(
    "/tags/{name}/reactivate",
    response_model=TagEntry,
    dependencies=[Depends(require_admin_user)],
    summary="Reactivate a deprecated tag (admin)",
    responses={404: {"description": "Tag not found"}},
)
async def reactivate_tag(
    name: Annotated[str, _TAG_NAME_PATH],
    body: TagReactivateBody,
    request: Request,
) -> dict[str, Any]:
    """Return a deprecated tag to `active` status so it can be added to
    documents again."""
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = _request_actor(request)
    now = _utcnow_iso()[:10]
    entry["status"] = "active"
    entry["last_edit"] = {"by": actor, "at": now, "action": "reactivated"}
    entry.pop("deprecate_reason", None)
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=name,
        summary=f"Tag {name} reactivated",
        meta={"previous_status": "deprecated"},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="reactivated",
            sub="Available again in the active catalog",
        ),
    )
    return stored


@router.post(
    "/tags/{name}/synonyms",
    response_model=TagEntry,
    dependencies=[Depends(require_admin_user)],
    summary="Replace a tag's synonyms (admin)",
    responses={404: {"description": "Tag not found"}},
)
async def update_synonyms(
    name: Annotated[str, _TAG_NAME_PATH],
    body: TagSynonymsBody,
    request: Request,
) -> dict[str, Any]:
    """Replace the tag's alias list. Aliases feed the thesaurus, so
    queries mentioning an alias also match content tagged with the
    canonical name."""
    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    actor = _request_actor(request)
    now = _utcnow_iso()[:10]
    entry["aliases"] = list(body.aliases)
    entry["last_edit"] = {"by": actor, "at": now, "action": "synonyms updated"}
    stored = await store.tags.upsert_tag(entry)
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="info",
        target_label=name,
        summary=f"Tag {name} synonyms updated ({len(body.aliases)} alias)",
        meta={"aliases": list(body.aliases)},
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix="synonyms updated",
            sub=", ".join(body.aliases) if body.aliases else "no aliases",
        ),
    )
    return stored


@router.delete(
    "/tags/{name}",
    response_model=AckResponse,
    dependencies=[Depends(require_admin_user)],
    summary="Delete a tag, untagging or migrating its documents (admin)",
    responses={
        404: {"description": "Tag (or migration target) not found"},
        422: {"description": "Invalid tag deletion strategy"},
    },
)
async def delete_tag(
    name: Annotated[str, _TAG_NAME_PATH],
    request: Request,
    body: TagDeleteBody | None = None,
) -> dict[str, bool]:
    """Delete a tag from the catalog. The `strategy` decides what happens
    to the documents that carry it: `untag` (default) removes the tag
    from them, `migrate` re-links them to the tag given in `to` (which
    must exist and differ from the deleted one)."""
    from .. import webui_router as legacy

    store = get_store()
    entry = await store.tags.get_tag(name)
    if entry is None:
        raise HTTPException(404, f"Tag '{name}' not found")
    payload = body or TagDeleteBody()
    if payload.strategy == "migrate" and not payload.to:
        raise HTTPException(422, "strategy=migrate requires 'to'")
    if payload.strategy == "migrate" and payload.to == name:
        raise HTTPException(422, "strategy=migrate requires a different target tag")
    if payload.strategy == "migrate" and payload.to:
        target = await store.tags.get_tag(payload.to)
        if target is None:
            raise HTTPException(404, f"Migration target tag '{payload.to}' not found")
    actor = _request_actor(request)

    seed_affected = legacy._cascade_seed_document_tags(
        store,
        name=name,
        strategy=payload.strategy,
        to=payload.to,
    )
    graph_affected = await legacy._cascade_graph_tag_edges(
        name=name,
        strategy=payload.strategy,
        to=payload.to,
        actor=actor,
        strict=isinstance(store.tags, MemgraphTagStore),
    )
    affected_docs = (graph_affected or 0) + seed_affected

    deleted = await store.tags.delete_tag(name)
    suffix = (
        f"migrated to {payload.to}"
        if payload.strategy == "migrate"
        else "deleted (docs untagged)"
    )
    result = (
        f"{affected_docs} docs migrated to {payload.to}"
        if payload.strategy == "migrate"
        else f"{affected_docs} docs untagged"
    )
    await _emit_tag_audit(
        store=store,
        actor=actor,
        kind="tag-mutation",
        sev="warning",
        target_label=name,
        summary=f"Tag {name} {suffix} — {result}",
        meta={
            "operation": "delete-tag",
            "strategy": payload.strategy,
            "to": payload.to,
            "result": result,
            "affected_docs": affected_docs,
            "sources_count_at_delete": entry.get("sources_count", 0),
        },
        notification=_make_notification(
            title="Tag",
            tagname=name,
            suffix=suffix,
            sub=f"{affected_docs} docs affected",
        ),
    )
    return {"ok": deleted}
