"""Folder-aware proxy from a Twin instance to the central KB catalogue.

The catalogue credential is deployment configuration and never reaches the
browser. Routes exist only when both ``TWIN_CATALOG_URL`` and
``TWIN_CATALOG_INSTANCE_CREDENTIAL`` are configured by the app factory.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .activity_events import emit_activity_event
from .auth import require_auth
from .folder import bind_request_folder
from .idp_jwt import require_admin_user
from .tracing import make_trace_headers

logger = logging.getLogger(__name__)

CATALOG_URL_ENV = "TWIN_CATALOG_URL"
CATALOG_CREDENTIAL_ENV = "TWIN_CATALOG_INSTANCE_CREDENTIAL"


@dataclass(frozen=True)
class CatalogProxyConfig:
    base_url: str
    credential: str
    timeout_seconds: float = 10.0

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> CatalogProxyConfig | None:
        values = env if env is not None else os.environ
        base_url = (values.get(CATALOG_URL_ENV) or "").strip().rstrip("/")
        credential = (values.get(CATALOG_CREDENTIAL_ENV) or "").strip()
        if not base_url and not credential:
            return None
        if not base_url or not credential:
            raise ValueError(
                f"{CATALOG_URL_ENV} and {CATALOG_CREDENTIAL_ENV} must be set together"
            )
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path not in {"", "/"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"{CATALOG_URL_ENV} must be an http(s) origin without credentials, "
                "query or fragment"
            )
        return cls(base_url=base_url, credential=credential)


class LinkedSourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    doc_type: Literal["di", "de", "sats", "general"]
    public: bool
    title: str | None = None
    language: str | None = None
    tags: list[str] = Field(default_factory=list)
    status: Literal["draft", "active"] = "active"


class LinkedSourcePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_type: Literal["di", "de", "sats", "general"] | None = None
    public: bool | None = None
    title: str | None = None
    language: str | None = None
    tags: list[str] | None = None
    expected_version: int = Field(ge=1)


class LinkedSourceTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    reason: str | None = None


class LinkedSourcePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["create", "patch", "transition"]
    target_id: str | None = None
    action: Literal["suspend", "activate", "disable"] | None = None
    body: dict[str, Any] = Field(default_factory=dict)


def _detail(response: httpx.Response) -> Any:
    """Preserve structured 4xx details needed for client-side recovery."""
    try:
        payload = response.json()
    except ValueError:
        return "catalogue request failed"
    if isinstance(payload, dict):
        return payload.get("detail", payload)
    return "catalogue request failed"


class CatalogProxyClient:
    def __init__(
        self,
        config: CatalogProxyConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            follow_redirects=False,
            transport=self.transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        folder_id: str | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        headers = {
            "Authorization": f"Bearer {self.config.credential}",
            **make_trace_headers(),
        }
        if folder_id is not None:
            headers["X-Twin-Folder"] = folder_id
        try:
            response = await self._client.request(
                method, path, headers=headers, json=json
            )
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            logger.warning("central catalogue request failed: %s", type(exc).__name__)
            raise HTTPException(503, "central catalogue is unavailable") from exc

        if response.status_code in {401, 403}:
            logger.error(
                "central catalogue rejected the configured instance credential"
            )
            raise HTTPException(503, "central catalogue authentication failed")
        if response.status_code >= 500:
            raise HTTPException(503, "central catalogue is unavailable")
        if response.is_error:
            raise HTTPException(response.status_code, _detail(response))
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise HTTPException(502, "central catalogue returned invalid JSON") from exc


def _actor(admin: dict[str, Any] | None) -> str:
    if not isinstance(admin, dict):
        return "operator"
    return str(
        admin.get("sso_subject") or admin.get("email") or admin.get("sub") or "operator"
    )


async def _emit_mutation_activity(
    *,
    kind: str,
    admin: dict[str, Any] | None,
    folder_id: str,
    result: dict[str, Any],
) -> None:
    link = result.get("link") if isinstance(result, dict) else None
    link = link if isinstance(link, dict) else {}
    revision = result.get("revision") if isinstance(result, dict) else None
    revision = revision if isinstance(revision, dict) else {}
    link_id = str(link.get("id") or "linked-source")
    await emit_activity_event(
        kind=kind,
        sev="info",
        actor=_actor(admin),
        target_type="linked-source",
        target_label=str(link.get("title") or link.get("url") or link_id),
        target_id=link_id,
        summary=f"RAG linked source {kind.removeprefix('linked-source-')}",
        meta={
            "folder_id": folder_id,
            "link_id": link_id,
            "auid": link.get("auid"),
            "revision_state": revision.get("state"),
        },
    )


def build_linked_sources_router(
    config: CatalogProxyConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> APIRouter:
    catalogue = CatalogProxyClient(config, transport=transport)

    @asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await catalogue.aclose()

    router = APIRouter(
        prefix="/linked-sources",
        tags=["linked-sources"],
        dependencies=[Depends(require_auth)],
        lifespan=lifespan,
    )

    @router.get(
        "",
        summary="List RAG sources declared by this KB in the active folder",
        include_in_schema=False,
    )
    async def list_linked_sources(
        folder_id: Annotated[str, Depends(bind_request_folder)],
    ) -> dict[str, Any]:
        applications = await catalogue.request("GET", "/v1/instance/applications")
        links = await catalogue.request(
            "GET", "/v1/instance/links", folder_id=folder_id
        )
        application = applications[0] if applications else None
        return {"application": application, "links": links}

    @router.post(
        "/preview",
        summary="Preview a linked-source mutation without writing",
        include_in_schema=False,
    )
    async def preview_linked_source(
        body: LinkedSourcePreview,
        folder_id: Annotated[str, Depends(bind_request_folder)],
        _admin: Annotated[dict[str, Any] | None, Depends(require_admin_user)],
    ) -> Any:
        return await catalogue.request(
            "POST",
            "/v1/instance/revisions/preview",
            folder_id=folder_id,
            json=body.model_dump(exclude_unset=True),
        )

    @router.post(
        "",
        status_code=201,
        summary="Declare a RAG source for this KB",
        include_in_schema=False,
    )
    async def create_linked_source(
        body: LinkedSourceCreate,
        folder_id: Annotated[str, Depends(bind_request_folder)],
        admin: Annotated[dict[str, Any] | None, Depends(require_admin_user)],
    ) -> Any:
        result = await catalogue.request(
            "POST",
            "/v1/instance/links",
            folder_id=folder_id,
            json=body.model_dump(exclude_unset=True),
        )
        await _emit_mutation_activity(
            kind="linked-source-declared",
            admin=admin,
            folder_id=folder_id,
            result=result,
        )
        return result

    @router.patch(
        "/{link_id}",
        summary="Update a RAG source with optimistic locking",
        include_in_schema=False,
    )
    async def patch_linked_source(
        link_id: UUID,
        body: LinkedSourcePatch,
        folder_id: Annotated[str, Depends(bind_request_folder)],
        admin: Annotated[dict[str, Any] | None, Depends(require_admin_user)],
    ) -> Any:
        result = await catalogue.request(
            "PATCH",
            f"/v1/instance/links/{link_id}",
            folder_id=folder_id,
            json=body.model_dump(exclude_unset=True),
        )
        await _emit_mutation_activity(
            kind="linked-source-updated",
            admin=admin,
            folder_id=folder_id,
            result=result,
        )
        return result

    @router.post(
        "/{link_id}/disable",
        summary="Disable a RAG source without deleting its audit history",
        include_in_schema=False,
    )
    async def disable_linked_source(
        link_id: UUID,
        body: LinkedSourceTransition,
        folder_id: Annotated[str, Depends(bind_request_folder)],
        admin: Annotated[dict[str, Any] | None, Depends(require_admin_user)],
    ) -> Any:
        result = await catalogue.request(
            "POST",
            f"/v1/instance/links/{link_id}/disable",
            folder_id=folder_id,
            json=body.model_dump(exclude_unset=True),
        )
        await _emit_mutation_activity(
            kind="linked-source-disabled",
            admin=admin,
            folder_id=folder_id,
            result=result,
        )
        return result

    return router


def linked_sources_wiring_probes(api_prefix: str = "/twin/api"):
    """Conditional route-table probes; imported lazily to avoid a cycle."""
    from .api_wiring import ApiWiringProbe

    prefix = api_prefix.rstrip("/")
    return (
        ApiWiringProbe("GET", f"{prefix}/linked-sources", "linked-sources:list"),
        ApiWiringProbe("POST", f"{prefix}/linked-sources", "linked-sources:create"),
        ApiWiringProbe(
            "POST", f"{prefix}/linked-sources/preview", "linked-sources:preview"
        ),
        ApiWiringProbe(
            "PATCH",
            f"{prefix}/linked-sources/{{link_id}}",
            "linked-sources:update",
        ),
        ApiWiringProbe(
            "POST",
            f"{prefix}/linked-sources/{{link_id}}/disable",
            "linked-sources:disable",
        ),
    )
