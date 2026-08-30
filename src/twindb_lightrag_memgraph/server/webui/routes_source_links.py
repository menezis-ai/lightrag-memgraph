"""Folder-aware CRUD for document provenance links."""

from __future__ import annotations

import logging
import secrets
from typing import Annotated
from urllib.parse import SplitResult, urlsplit, urlunsplit

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..idp_jwt import require_admin_user
from ..source_links_store import SourceLinkNotFound, SourceLinkVersionConflict
from .events import _make_event, _request_actor, _utcnow_iso
from .store import get_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["document-source-links"])


def normalize_source_url(value: str) -> str:
    """Validate and deterministically normalize one HTTP(S) provenance URL."""
    raw = value.strip()
    if not raw or len(raw) > 2048:
        raise ValueError("source link URL must contain 1 to 2048 characters")
    if any(char.isspace() for char in raw) or "\\" in raw:
        raise ValueError("source link URL contains unsafe characters")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("source link URL is invalid") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("source link URL must use http or https")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "source link URL needs a host and must not contain credentials"
        )
    try:
        host = parsed.hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("source link URL host is invalid") from exc
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path or "/"
    return urlunsplit(SplitResult(scheme, netloc, path, parsed.query, parsed.fragment))


class SourceLink(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    doc_id: str
    url: str
    label: str | None = None
    created_by: str
    created_at: str
    updated_by: str
    updated_at: str
    version: int = Field(ge=1)
    deleted: bool = False
    deleted_by: str | None = None
    deleted_at: str | None = None


class SourceLinkCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=2048)
    label: str | None = Field(default=None, max_length=200)

    @field_validator("url")
    @classmethod
    def _normalize_url(cls, value: str) -> str:
        return normalize_source_url(value)

    @field_validator("label")
    @classmethod
    def _normalize_label(cls, value: str | None) -> str | None:
        cleaned = value.strip() if value is not None else None
        return cleaned or None


class SourceLinkUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    url: str | None = Field(default=None, min_length=1, max_length=2048)
    label: str | None = Field(default=None, max_length=200)

    @field_validator("url")
    @classmethod
    def _normalize_url(cls, value: str | None) -> str | None:
        return normalize_source_url(value) if value is not None else None

    @field_validator("label")
    @classmethod
    def _normalize_label(cls, value: str | None) -> str | None:
        cleaned = value.strip() if value is not None else None
        return cleaned or None

    @model_validator(mode="after")
    def _require_patch_field(self) -> "SourceLinkUpdate":
        if not ({"url", "label"} & self.model_fields_set):
            raise ValueError("source link update needs url and/or label")
        return self


async def _require_visible_document(doc_id: str) -> None:
    # Dynamic import preserves the legacy monkeypatch surface used in route tests.
    from .. import webui_router as legacy

    await legacy._get_doc_for_active_folder(doc_id)


async def _record_link_activity(
    *,
    action: str,
    actor: str,
    doc_id: str,
    link_id: str,
    before: dict | None,
    after: dict | None,
) -> None:
    event = _make_event(
        kind=f"source-link-{action}",
        sev="info",
        actor=actor,
        target_label=link_id,
        summary=f"source link {action} for document {doc_id}",
        meta={
            "operation": f"source-link-{action}",
            "doc_id": doc_id,
            "link_id": link_id,
            "before": before,
            "after": after,
        },
        target_type="document-source-link",
        target_id=link_id,
    )
    try:
        await get_store().record_activity(event)
    except Exception:  # audit side effect must not roll back a committed link
        logger.exception("source-link audit emission failed for %s", link_id)


def _translate_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SourceLinkNotFound):
        return HTTPException(status_code=404, detail="Source link not found")
    return HTTPException(
        status_code=409,
        detail="Source link changed concurrently; refresh and retry.",
    )


@router.get(
    "/documents/{doc_id}/source-links",
    response_model=list[SourceLink],
    responses={404: {"description": "Document not found in active folder"}},
)
async def list_document_source_links(
    doc_id: Annotated[
        str,
        Path(min_length=1, description="Document id visible in the active folder"),
    ],
) -> list[dict]:
    """List non-deleted provenance links attached to a visible document."""
    await _require_visible_document(doc_id)
    return await get_store().source_links.list_for_document(doc_id)


@router.post(
    "/documents/{doc_id}/source-links",
    response_model=SourceLink,
    status_code=201,
    dependencies=[Depends(require_admin_user)],
    responses={404: {"description": "Document not found in active folder"}},
)
async def create_document_source_link(
    doc_id: Annotated[
        str,
        Path(min_length=1, description="Document id visible in the active folder"),
    ],
    body: SourceLinkCreate,
    request: Request,
) -> dict:
    """Attach one validated HTTP(S) provenance link to a document."""
    await _require_visible_document(doc_id)
    actor = _request_actor(request)
    now = _utcnow_iso()
    row = {
        "id": f"slink_{secrets.token_hex(12)}",
        "doc_id": doc_id,
        "url": body.url,
        "label": body.label,
        "created_by": actor,
        "created_at": now,
        "updated_by": actor,
        "updated_at": now,
        "version": 1,
        "deleted": False,
        "deleted_by": None,
        "deleted_at": None,
    }
    created = await get_store().source_links.create(row)
    await _record_link_activity(
        action="created",
        actor=actor,
        doc_id=doc_id,
        link_id=created["id"],
        before=None,
        after=created,
    )
    return created


@router.patch(
    "/documents/{doc_id}/source-links/{link_id}",
    response_model=SourceLink,
    dependencies=[Depends(require_admin_user)],
    responses={
        404: {"description": "Document or link not found"},
        409: {"description": "Optimistic version conflict"},
    },
)
async def update_document_source_link(
    doc_id: Annotated[
        str,
        Path(min_length=1, description="Document id visible in the active folder"),
    ],
    link_id: Annotated[
        str,
        Path(min_length=1, description="Stable source-link id"),
    ],
    body: SourceLinkUpdate,
    request: Request,
) -> dict:
    """Update a provenance link using optimistic version comparison."""
    await _require_visible_document(doc_id)
    current = next(
        (
            row
            for row in await get_store().source_links.list_for_document(doc_id)
            if row["id"] == link_id
        ),
        None,
    )
    if current is None:
        raise HTTPException(status_code=404, detail="Source link not found")
    desired_url = body.url if body.url is not None else current["url"]
    desired_label = (
        body.label if "label" in body.model_fields_set else current.get("label")
    )
    actor = _request_actor(request)
    try:
        before, after = await get_store().source_links.update(
            doc_id,
            link_id,
            expected_version=body.version,
            url=desired_url,
            label=desired_label,
            actor=actor,
            updated_at=_utcnow_iso(),
        )
    except (SourceLinkNotFound, SourceLinkVersionConflict) as exc:
        raise _translate_store_error(exc) from exc
    await _record_link_activity(
        action="updated",
        actor=actor,
        doc_id=doc_id,
        link_id=link_id,
        before=before,
        after=after,
    )
    return after


@router.delete(
    "/documents/{doc_id}/source-links/{link_id}",
    response_model=SourceLink,
    dependencies=[Depends(require_admin_user)],
    responses={
        404: {"description": "Document or link not found"},
        409: {"description": "Optimistic version conflict"},
    },
)
async def delete_document_source_link(
    doc_id: Annotated[
        str,
        Path(min_length=1, description="Document id visible in the active folder"),
    ],
    link_id: Annotated[
        str,
        Path(min_length=1, description="Stable source-link id"),
    ],
    version: Annotated[
        int,
        Query(ge=1, description="Expected current version for compare-and-delete"),
    ],
    request: Request,
) -> dict:
    """Tombstone a provenance link using optimistic version comparison."""
    await _require_visible_document(doc_id)
    actor = _request_actor(request)
    try:
        before, after = await get_store().source_links.delete(
            doc_id,
            link_id,
            expected_version=version,
            actor=actor,
            deleted_at=_utcnow_iso(),
        )
    except (SourceLinkNotFound, SourceLinkVersionConflict) as exc:
        raise _translate_store_error(exc) from exc
    await _record_link_activity(
        action="deleted",
        actor=actor,
        doc_id=doc_id,
        link_id=link_id,
        before=before,
        after=after,
    )
    return after


__all__ = [
    "SourceLink",
    "SourceLinkCreate",
    "SourceLinkUpdate",
    "normalize_source_url",
    "router",
]
