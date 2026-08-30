"""Bounded, metadata-only profile consumed by the central catalogue scan.

The response deliberately excludes document names, summaries, chunks and raw
metadata. It aggregates only folder identity, counters, tag names, document
formats and the most frequent graph entity names. The central LLM therefore
never receives document text through this channel.
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import logging
from pathlib import PurePosixPath
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from .. import __version__
from . import api_key_store
from .auth import is_infrastructure_root_request
from .folder import catalog_profile_folder_ids, load_folder_catalog, scoped_folder
from .webui.routes_graph import list_graph_entities
from .webui.store import get_store

_MAX_PROFILE_ITEMS = 500
_MAX_PROFILE_COUNTS = 64
_MAX_TAGS = 50
_MAX_ENTITIES = 50
_SAFE_FORMAT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789")
_profile_security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)


class ProfileCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    count: int = Field(ge=0)


class ProfileEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    mentions: int = Field(ge=0)
    sources: int = Field(ge=0)


class FolderProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    kind: str
    document_count: int = Field(ge=0)
    sampled_document_count: int = Field(ge=0)
    documents_truncated: bool
    status_counts: list[ProfileCount]
    document_formats: list[ProfileCount]
    tags: list[ProfileCount]
    graph_entity_count: int = Field(ge=0)
    graph_truncated: bool
    top_graph_entities: list[ProfileEntity]


class CatalogProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    generated_at: datetime
    instance_version: str
    folder_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    graph_entity_count: int = Field(ge=0)
    folders: list[FolderProfile]


def _bounded_label(value: object, *, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _document_format(document: dict[str, Any]) -> str:
    raw_path = str(document.get("file_path") or document.get("source") or "")
    path = urlsplit(raw_path).path if "://" in raw_path else raw_path
    suffix = PurePosixPath(path).suffix.lower().lstrip(".")
    if suffix and len(suffix) <= 10 and set(suffix) <= _SAFE_FORMAT_CHARS:
        return suffix
    source_type = str(document.get("type") or "unknown").lower()
    return (
        source_type
        if source_type in {"file", "confluence", "sharepoint", "url"}
        else "unknown"
    )


def _counts(values: Counter[str], *, limit: int | None = None) -> list[ProfileCount]:
    rows = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        rows = rows[:limit]
    return [ProfileCount(name=name, count=count) for name, count in rows]


async def require_catalog_profile_read(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_profile_security)
    ] = None,
) -> None:
    """Authorize only the infra root or a dedicated ``profile:read`` key.

    Profile credentials use the ``tcp_`` prefix and are deliberately absent
    from the generic Twin auth chain. Possessing one therefore cannot grant
    access to document, graph, query or mutation routes.
    """

    catalog = load_folder_catalog()
    if is_infrastructure_root_request(request):
        request.state.catalog_profile_folder_ids = tuple(
            folder.id for folder in catalog.folders
        )
        return

    token = credentials.credentials if credentials is not None else ""
    if not token.startswith(api_key_store.PROFILE_KEY_PREFIX):
        raise HTTPException(
            status_code=401,
            detail="A profile:read credential is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        from .._constants import resolve_workspace

        entry = await api_key_store.validate_bearer(resolve_workspace(), token)
    except Exception as exc:  # noqa: BLE001 - auth uncertainty must fail closed
        logger.exception("catalog profile credential lookup failed")
        raise HTTPException(
            503, "Profile credential validation is unavailable"
        ) from exc
    if entry is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or revoked profile credential",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if entry.get("scopes") != ["profile:read"]:
        raise HTTPException(403, "Credential lacks profile:read scope")

    requested = tuple(str(value) for value in (entry.get("folders") or ()))
    if not requested:
        allowed = (catalog.default_folder_id,)
    else:
        requested_set = set(requested)
        allowed = tuple(
            folder.id for folder in catalog.folders if folder.id in requested_set
        )
        if not allowed:
            raise HTTPException(403, "No provisioned folder is in credential scope")
    request.state.catalog_profile_folder_ids = allowed
    await api_key_store.mark_used(resolve_workspace(), str(entry.get("id")))


async def _documents_for_active_folder(
    *, max_items: int
) -> tuple[list[dict[str, Any]], int]:
    store = get_store()
    if store.mode == "seed":
        documents = store.list_documents()
        return documents[:max_items], len(documents)

    # Deferred to avoid importing the full WebUI router during module wiring.
    from .webui.router import _catalog_profile_document_sample

    return await _catalog_profile_document_sample(limit=max_items)


async def _folder_profile(folder: Any, *, max_items: int) -> FolderProfile:
    with scoped_folder(folder.id):
        documents, document_count = await _documents_for_active_folder(
            max_items=max_items
        )
        entities = await list_graph_entities(
            label="*", max_nodes=max_items, max_depth=1
        )

    sampled = documents[:max_items]
    statuses: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    for document in sampled:
        statuses[_bounded_label(document.get("status") or "unknown", limit=32)] += 1
        formats[_document_format(document)] += 1
        for raw_tag in document.get("tags") or ():
            tag = _bounded_label(raw_tag, limit=64)
            if tag:
                tags[tag] += 1

    top_entities = sorted(
        entities,
        key=lambda entity: (
            -int(entity.get("mentions") or 0),
            -int(entity.get("sources") or 0),
            str(entity.get("name") or ""),
        ),
    )[:_MAX_ENTITIES]
    return FolderProfile(
        id=folder.id,
        label=folder.label,
        kind=folder.kind,
        document_count=document_count,
        sampled_document_count=len(sampled),
        documents_truncated=document_count > len(sampled),
        status_counts=_counts(statuses, limit=_MAX_PROFILE_COUNTS),
        document_formats=_counts(formats, limit=_MAX_PROFILE_COUNTS),
        tags=_counts(tags, limit=_MAX_TAGS),
        graph_entity_count=len(entities),
        graph_truncated=len(entities) >= max_items,
        top_graph_entities=[
            ProfileEntity(
                name=_bounded_label(entity.get("name"), limit=160),
                type=_bounded_label(entity.get("type") or "UNKNOWN", limit=32),
                mentions=max(0, int(entity.get("mentions") or 0)),
                sources=max(0, int(entity.get("sources") or 0)),
            )
            for entity in top_entities
            if _bounded_label(entity.get("name"), limit=160)
        ],
    )


def build_catalog_profile_router() -> APIRouter:
    router = APIRouter(
        tags=["catalog-profile"],
        dependencies=[Depends(require_catalog_profile_read)],
    )

    @router.get(
        "/catalog-profile",
        response_model=CatalogProfile,
        summary="Build a bounded metadata-only profile for the central catalogue",
    )
    async def catalog_profile(
        request: Request,
        max_items: Annotated[
            int,
            Query(
                ge=1,
                le=_MAX_PROFILE_ITEMS,
                description="Maximum documents and graph entities sampled per folder.",
            ),
        ] = _MAX_PROFILE_ITEMS,
    ) -> CatalogProfile:
        """Aggregate an authorised, metadata-only and explicitly bounded profile.

        Storage failures propagate as 503 so an unavailable KB is never
        represented to the catalogue or LLM as an empty one.
        """
        allowed = set(catalog_profile_folder_ids(request))
        if not allowed:
            raise HTTPException(403, "No provisioned folder is in caller scope")
        folders = [
            folder for folder in load_folder_catalog().folders if folder.id in allowed
        ]
        profiles = [
            await _folder_profile(folder, max_items=max_items) for folder in folders
        ]
        return CatalogProfile(
            generated_at=datetime.now(UTC),
            instance_version=__version__,
            folder_count=len(profiles),
            document_count=sum(row.document_count for row in profiles),
            graph_entity_count=sum(row.graph_entity_count for row in profiles),
            folders=profiles,
        )

    return router


def catalog_profile_wiring_probes(api_prefix: str = "/twin/api"):
    """Conditional route-table probe imported lazily to avoid a cycle."""

    from .api_wiring import ApiWiringProbe

    prefix = api_prefix.rstrip("/")
    return (ApiWiringProbe("GET", f"{prefix}/catalog-profile", "catalog-profile:read"),)
