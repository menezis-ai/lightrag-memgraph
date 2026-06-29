"""Best-effort server-side Activity emission helpers.

Auth and gate code should not import the WebUI router directly.  This module
keeps the dependency lazy and isolates the "audit must never break auth" rule.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request

logger = logging.getLogger(__name__)

_SENSITIVE_META_TOKENS = (
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "key",
)


def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Drop obviously sensitive keys and keep only simple JSON-ish values."""
    cleaned: dict[str, Any] = {}
    for key, value in meta.items():
        lowered = key.lower()
        if any(token in lowered for token in _SENSITIVE_META_TOKENS):
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


async def emit_activity_event(
    *,
    kind: str,
    sev: str,
    actor: str,
    target_type: str,
    target_label: str,
    summary: str,
    meta: dict[str, Any],
) -> None:
    """Append one Activity event, swallowing every failure."""
    try:
        from .webui.events import _make_event
        from .webui.store import get_store

        event = _make_event(
            kind=kind,
            sev=sev,
            actor=actor,
            target_type=target_type,
            target_label=target_label,
            summary=summary,
            meta=_clean_meta(meta),
        )
        await get_store().record_activity(event)
    except Exception:  # noqa: BLE001 - Activity must never break auth/admin flow.
        logger.exception("[activity] best-effort event emission failed")


async def emit_auth_event(
    *,
    action: str,
    sev: str,
    actor: str,
    target_type: str,
    target_label: str,
    summary: str,
    meta: dict[str, Any],
) -> None:
    await emit_activity_event(
        kind="auth",
        sev=sev,
        actor=actor or "anonymous",
        target_type=target_type,
        target_label=target_label,
        summary=summary,
        meta={"operation": action, **meta},
    )


def _route_is_authz_relevant(path: str) -> bool:
    """Limit denied-route Activity to Twin/protected/admin surfaces."""
    if path in {"/login", "/logout", "/auth-status"}:
        return False
    if path.startswith("/twin/api"):
        return True
    return path.startswith(
        (
            "/activity",
            "/chunks",
            "/documents",
            "/folders",
            "/graph",
            "/notifications",
            "/query",
            "/settings/api-keys",
            "/tags",
        )
    )


async def emit_access_denied_event(
    request: Request,
    *,
    status_code: int,
    reason: str | None = None,
) -> None:
    """Record a minimal denied-route event for protected/admin surfaces."""
    path = request.url.path
    if status_code not in {401, 403} or not _route_is_authz_relevant(path):
        return
    safe_reason = reason or ("unauthorized" if status_code == 401 else "forbidden")
    actor = "anonymous"
    if status_code == 403:
        try:
            from .auth import resolve_auth_actor

            actor = await resolve_auth_actor(request) or "anonymous"
        except Exception:  # noqa: BLE001 - attribution must not break auth flow.
            logger.debug("[activity] access-denied actor resolution failed", exc_info=True)
    await emit_auth_event(
        action="access_denied",
        sev="warning",
        actor=actor,
        target_type="route",
        target_label=path,
        summary=f"access denied on {request.method} {path}",
        meta={
            "method": request.method,
            "path": path,
            "status_code": status_code,
            "reason": safe_reason,
        },
    )
