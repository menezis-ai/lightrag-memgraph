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
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .._constants import resolve_workspace
from . import api_key_store
from .auth import require_auth
from .idp_jwt import require_admin_user

logger = logging.getLogger(__name__)


# Auth + admin gates are at the router level so every route (including
# future additions) is protected by default even if the router is mounted
# outside ``create_app`` / the Forgejo overlay. ``require_admin_user``
# deliberately treats dormant IdP as "authenticated user is admin", so it
# must never be the only dependency on an admin router.
#
# Handlers that also declare ``admin: dict = Depends(require_admin_user)``
# do so to INJECT the user dict (for audit ``actor``), not to re-enforce
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
    name: str = Field(min_length=1, max_length=120)


class ApiKeyPublic(BaseModel):
    """Public shape returned by GET / DELETE. ``hash`` is intentionally
    absent; ``full_value`` is only present in the POST response (see
    :class:`ApiKeyCreated`)."""

    id: str
    name: str
    prefix: str
    created_at: int
    created_by: str
    last_used_at: int | None = None
    revoked_at: int | None = None


class ApiKeyCreated(ApiKeyPublic):
    """POST response. ``full_value`` is shown ONCE; store it client-side."""

    full_value: str


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
    return (
        str(user.get("sso_subject") or user.get("email") or user.get("sub") or "operator")
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ApiKeyPublic])
async def list_api_keys() -> list[dict[str, Any]]:
    """Return every key for the active workspace (revoked included)."""
    workspace = resolve_workspace()
    try:
        await api_key_store.initialize(workspace)
    except Exception:  # noqa: BLE001 — schema setup is best-effort here
        logger.exception("[ApiKeyRoutes] initialize failed for %s", workspace)
    return await api_key_store.list_keys(workspace)


@router.post("", response_model=ApiKeyCreated, status_code=201)
async def create_api_key(
    body: ApiKeyCreate,
    admin: dict[str, Any] = Depends(require_admin_user),
) -> dict[str, Any]:
    """Mint a new API key. The ``full_value`` is returned ONCE — clients
    MUST store it client-side immediately. Subsequent list/get calls
    expose only the prefix."""
    workspace = resolve_workspace()
    try:
        await api_key_store.initialize(workspace)
    except Exception:  # noqa: BLE001
        logger.exception("[ApiKeyRoutes] initialize failed for %s", workspace)
    actor = _actor_from_user(admin)
    entry = await api_key_store.create_key(
        workspace,
        name=body.name,
        created_by=actor,
    )
    await _emit_event(
        action="created",
        key_id=str(entry.get("id")),
        prefix=str(entry.get("prefix")),
        actor=actor,
    )
    return entry


@router.delete("/{key_id}", response_model=ApiKeyPublic)
async def revoke_api_key(
    key_id: str,
    admin: dict[str, Any] = Depends(require_admin_user),
) -> dict[str, Any]:
    """Mark a key as revoked. Subsequent auth attempts with that key
    reject. The entry stays in the listing with a ``revoked_at`` stamp
    so the audit trail is preserved."""
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
