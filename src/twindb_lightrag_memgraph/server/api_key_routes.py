"""API key management endpoints under ``/twin/api/settings/api-keys``.

Three endpoints, all gated by :func:`require_admin_user`:

- ``GET``        — list every key (revoked included). ``hash`` never exposed.
- ``POST``       — mint a new key. Returns ``full_value`` exactly once.
- ``DELETE /{id}`` — revoke a key (sets ``revoked_at`` timestamp).

Each mutation emits an activity event (``api-key-created`` /
``api-key-revoked``) so the audit feed surfaces who minted or revoked
which key. The full secret value is **never** persisted or echoed in
activity meta — only the public ``prefix`` is.

The static ``LIGHTRAG_API_KEY`` (env-set) is **not** managed here. It
remains the infra root key, invisible from the UI by design (see
``feedback`` in PR description).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as FastapiPath
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .._constants import resolve_workspace
from . import api_key_store
from .auth import require_auth
from .folder import load_folder_catalog
from .idp_jwt import require_admin_user

logger = logging.getLogger(__name__)


# Auth + admin gates are at the router level so every route (including
# future additions) is protected by default even if the router is mounted
# outside ``create_app`` / the Forgejo overlay. When the IdP is dormant,
# ``require_admin_user`` accepts only the separately managed infrastructure
# root key; generated ``twk_`` keys and local JWTs fail closed.
#
# Handlers that also inject ``admin`` via ``Depends(require_admin_user)`` do so
# to INJECT the user dict (for audit ``actor``), not to re-enforce
# — FastAPI caches the dependency result within a single request, so the
# call resolves exactly once. The visual duplication is intentional, not a
# leak.
router = APIRouter(
    prefix="/settings/api-keys",
    tags=["api-keys"],
    dependencies=[Depends(require_auth), Depends(require_admin_user)],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ApiKeyCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"name": "reporting-script"}]}
    )

    name: str = Field(
        min_length=1,
        max_length=120,
        description="Display name identifying what the key is for.",
        examples=["reporting-script"],
    )
    scopes: list[Literal["api:*", "profile:read"]] = Field(
        default_factory=lambda: ["api:*"],
        min_length=1,
        max_length=1,
        description=(
            "Exactly one capability. profile:read mints a tcp_ credential that "
            "cannot authenticate on generic Twin routes."
        ),
    )
    folders: list[str] = Field(
        default_factory=list,
        max_length=50,
        description=(
            "Folders visible to a profile:read credential. Empty means only the "
            "provisioned default folder."
        ),
    )

    @model_validator(mode="after")
    def _scope_shape(self) -> ApiKeyCreate:
        self.scopes = list(dict.fromkeys(self.scopes))
        self.folders = list(dict.fromkeys(self.folders))
        if self.scopes != ["profile:read"] and self.folders:
            raise ValueError("folders are only valid with scope profile:read")
        return self


class ApiKeyPublic(BaseModel):
    """Public shape returned by GET / DELETE. ``hash`` is intentionally
    absent; ``full_value`` is only present in the POST response (see
    :class:`ApiKeyCreated`)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "19ab42f01d0-8d7c6b5a",
                    "name": "reporting-script",
                    "prefix": "twk_Pu6s9K2a…",
                    "created_at": 1785283200000,
                    "created_by": "operator@example.test",
                    "last_used_at": None,
                    "revoked_at": None,
                }
            ]
        }
    )

    id: str = Field(
        description="Opaque identifier used to revoke this key.",
        examples=["19ab42f01d0-8d7c6b5a"],
    )
    name: str = Field(
        description="Operator-supplied display name for the key.",
        examples=["reporting-script"],
    )
    prefix: str = Field(
        description="Non-secret preview used to identify the key in listings.",
        examples=["twk_Pu6s9K2a…"],
    )
    scopes: list[str] = Field(
        default_factory=lambda: ["api:*"],
        description="Credential capabilities; legacy entries default to api:*.",
    )
    folders: list[str] = Field(
        default_factory=list,
        description="Folder ids authorised for a profile:read credential.",
    )
    created_at: int = Field(
        description="Creation time as Unix epoch milliseconds.",
        examples=[1785283200000],
    )
    created_by: str = Field(
        description="Identity that created the key.",
        examples=["operator@example.test"],
    )
    last_used_at: int | None = Field(
        default=None,
        description=(
            "Most recent successful authentication time as Unix epoch "
            "milliseconds, or null when unused."
        ),
        examples=[1785286800000],
    )
    revoked_at: int | None = Field(
        default=None,
        description=(
            "Revocation time as Unix epoch milliseconds, or null while active."
        ),
        examples=[1785290400000],
    )


class ApiKeyCreated(ApiKeyPublic):
    """POST response. ``full_value`` is shown ONCE; store it client-side."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "19ab42f01d0-8d7c6b5a",
                    "name": "reporting-script",
                    "prefix": "twk_Pu6s9K2a…",
                    "created_at": 1785283200000,
                    "created_by": "operator@example.test",
                    "last_used_at": None,
                    "revoked_at": None,
                    "full_value": "twk_example_value_returned_only_at_creation",
                }
            ]
        }
    )

    full_value: str = Field(
        description=(
            "Complete bearer secret. This value is returned only by the "
            "creation response and cannot be retrieved later."
        ),
        examples=["twk_example_value_returned_only_at_creation"],
    )


# ---------------------------------------------------------------------------
# Activity emission (decoupled so this module does not import webui_router
# directly — avoids a circular import; we resolve the store lazily).
# ---------------------------------------------------------------------------


async def _emit_event(*, action: str, key_id: str, prefix: str, actor: str) -> None:
    """Best-effort activity emission. Failures are logged, never raised."""
    try:
        from . import webui_router  # local import: avoids circular at module load

        event = webui_router._make_event(
            kind=f"api-key-{action}",
            sev="info" if action == "created" else "warning",
            actor=actor,
            target_label=prefix,
            summary=(
                f"API key '{prefix}' created"
                if action == "created"
                else f"API key '{prefix}' revoked"
            ),
            meta={"key_id": key_id, "prefix": prefix, "operation": action},
            target_type="api-key",
            target_id=key_id,
        )
        store = webui_router.get_store()
        await store.record_activity(event)
    except Exception:  # noqa: BLE001 — audit must never break the request
        logger.exception(
            "[ApiKeyRoutes] activity event emission failed (action=%s, id=%s)",
            action,
            key_id,
        )


def _actor_from_user(user: dict[str, Any] | None) -> str:
    if not isinstance(user, dict):
        return "operator"
    return str(
        user.get("sso_subject") or user.get("email") or user.get("sub") or "operator"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[ApiKeyPublic],
    summary="List API keys (admin)",
    response_description=(
        "Generated API keys, including revoked entries, with secrets omitted."
    ),
    responses={
        401: {"description": "Authentication is missing or invalid"},
        403: {"description": "Administrator privileges are required"},
    },
)
async def list_api_keys() -> list[dict[str, Any]]:
    """Return every generated API key, revoked ones included, with
    creation and last-use timestamps. Only the public prefix is exposed —
    the secret value is shown once, at creation."""
    workspace = resolve_workspace()
    try:
        await api_key_store.initialize(workspace)
    except Exception:  # noqa: BLE001 — schema setup is best-effort here
        logger.exception("[ApiKeyRoutes] initialize failed for %s", workspace)
    return await api_key_store.list_keys(workspace)


@router.post(
    "",
    response_model=ApiKeyCreated,
    status_code=201,
    summary="Create an API key (admin)",
    response_description=(
        "API key metadata plus the complete secret, returned this one time only."
    ),
    responses={
        401: {"description": "Authentication is missing or invalid"},
        403: {"description": "Administrator privileges are required"},
    },
)
async def create_api_key(
    body: ApiKeyCreate,
    admin: Annotated[dict[str, Any], Depends(require_admin_user)],
) -> dict[str, Any]:
    """Mint a new API key. The secret (`full_value`) is returned **once,
    in this response only** — store it immediately; every later listing
    exposes only the prefix. Use the key as a `Bearer` token
    (`Authorization: Bearer twk_...`); the `X-API-Key` header is reserved
    for the deployment's static infrastructure key and does not accept
    generated keys."""
    workspace = resolve_workspace()
    catalog = load_folder_catalog()
    unknown_folders = sorted(set(body.folders) - set(catalog.ids))
    if unknown_folders:
        raise HTTPException(
            422,
            f"Unknown folder id(s): {', '.join(unknown_folders)}",
        )
    try:
        await api_key_store.initialize(workspace)
    except Exception:  # noqa: BLE001
        logger.exception("[ApiKeyRoutes] initialize failed for %s", workspace)
    actor = _actor_from_user(admin)
    entry = await api_key_store.create_key(
        workspace,
        name=body.name,
        created_by=actor,
        scopes=list(body.scopes),
        folders=body.folders,
    )
    await _emit_event(
        action="created",
        key_id=str(entry.get("id")),
        prefix=str(entry.get("prefix")),
        actor=actor,
    )
    return entry


@router.delete(
    "/{key_id}",
    response_model=ApiKeyPublic,
    summary="Revoke an API key (admin)",
    response_description="The revoked API key metadata and revocation timestamp.",
    responses={
        401: {"description": "Authentication is missing or invalid"},
        403: {"description": "Administrator privileges are required"},
        404: {"description": "API key not found"},
    },
)
async def revoke_api_key(
    key_id: Annotated[
        str,
        FastapiPath(
            description="Key id, as returned by the list endpoint.",
        ),
    ],
    admin: Annotated[dict[str, Any], Depends(require_admin_user)],
) -> dict[str, Any]:
    """Revoke a key: authentication attempts with it are rejected from
    now on. The entry stays in the listing with its `revoked_at`
    timestamp so the audit trail is preserved."""
    workspace = resolve_workspace()
    entry = await api_key_store.revoke_key(workspace, key_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    actor = _actor_from_user(admin)
    await _emit_event(
        action="revoked",
        key_id=str(entry.get("id")),
        prefix=str(entry.get("prefix")),
        actor=actor,
    )
    return entry


__all__ = ["router"]
