"""Best-effort server-side Activity emission helpers.

Auth and gate code should not import the WebUI router directly.  This module
keeps the dependency lazy and isolates the "audit must never break auth" rule.
"""

from __future__ import annotations

import asyncio
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

_FOLDER_APPLICABLE_SEGMENTS = {
    "query",
    "activity",
    "chunks",
    "documents",
    "graph",
    "notifications",
    "tags",
}
_FOLDER_NOT_APPLICABLE = "not-applicable"
_AUTH_ACTIVITY_EVENT_TASKS: set[asyncio.Task[Any]] = set()


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


def _folded_auth_event_markers(request: Request) -> set[str]:
    markers = getattr(request.state, "_auth_event_markers", None)
    if markers is None:
        markers = set()
        request.state._auth_event_markers = markers
    return markers


def _mark_auth_event_once(request: Request, signature: str) -> bool:
    """Mark an auth event once per-request.

    Returns True when the same event signature already exists for this request.
    """
    markers = _folded_auth_event_markers(request)
    if signature in markers:
        return True
    markers.add(signature)
    return False


def _segment_for_path(path: str) -> str:
    if path.startswith("/twin/api/"):
        path = path[len("/twin/api/") :]
    return path.strip("/").split("/", 1)[0]


def _is_folder_scoped_path(path: str) -> bool:
    segment = _segment_for_path(path)
    return segment in _FOLDER_APPLICABLE_SEGMENTS


def _auth_event_folder(request: Request, path: str) -> str:
    if not _is_folder_scoped_path(path):
        return _FOLDER_NOT_APPLICABLE

    try:
        from .folder import active_folder_id, resolve_folder_from_headers

        folder = active_folder_id()
        if folder:
            return folder
        return resolve_folder_from_headers(request.headers)
    except Exception:
        return "invalid-folder"


def _auth_event_signature(*, action: str, request: Request, status_code: int) -> str:
    return f"{action}:{request.method}:{request.url.path}:{status_code}"


def _schedule_activity_task(
    *,
    coro: Any,
) -> None:
    try:
        task = asyncio.create_task(coro)
    except Exception:  # noqa: BLE001 - task scheduling failures must not block auth.
        if hasattr(coro, "close"):
            try:
                coro.close()
            except Exception:  # noqa: BLE001 - task close may fail on already-awaited path.
                logger.exception("[activity] failed to close unscheduled auth event coroutine")
        logger.exception("[activity] failed to schedule async auth event")
        return

    _AUTH_ACTIVITY_EVENT_TASKS.add(task)

    def _cleanup(done: asyncio.Task[Any]) -> None:
        _AUTH_ACTIVITY_EVENT_TASKS.discard(done)

    task.add_done_callback(_cleanup)


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


def emit_activity_event_async(
    *,
    kind: str,
    sev: str,
    actor: str,
    target_type: str,
    target_label: str,
    summary: str,
    meta: dict[str, Any],
) -> None:
    """Fire-and-forget auth event emission."""
    _schedule_activity_task(
        coro=emit_activity_event(
            kind=kind,
            sev=sev,
            actor=actor,
            target_type=target_type,
            target_label=target_label,
            summary=summary,
            meta=meta,
        ),
    )


async def emit_auth_event(
    *,
    action: str,
    sev: str,
    actor: str,
    target_type: str,
    target_label: str,
    summary: str,
    meta: dict[str, Any],
    request: Request | None = None,
) -> None:
    cleaned_meta = dict(meta)
    if request is not None:
        cleaned_meta["folder"] = _auth_event_folder(request, request.url.path)
    await emit_activity_event(
        kind="auth",
        sev=sev,
        actor=actor or "anonymous",
        target_type=target_type,
        target_label=target_label,
        summary=summary,
        meta={"operation": action, **cleaned_meta},
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
    marker = _auth_event_signature(
        action="access_denied",
        request=request,
        status_code=status_code,
    )
    if _mark_auth_event_once(request, marker):
        return

    path = request.url.path
    if status_code not in {401, 403} or not _route_is_authz_relevant(path):
        return
    safe_reason = reason or ("unauthorized" if status_code == 401 else "forbidden")
    actor = "anonymous"
    if status_code == 403:
        try:
            from .auth import resolve_auth_actor

            actor = resolve_auth_actor(request) or "anonymous"
        except Exception:  # noqa: BLE001 - attribution must not break auth flow.
            logger.debug("[activity] access-denied actor resolution failed", exc_info=True)

    await emit_auth_event(
        action="access_denied",
        sev="warning",
        actor=actor,
        target_type="route",
        target_label=path,
        request=request,
        summary=f"access denied on {request.method} {path}",
        meta={
            "method": request.method,
            "path": path,
            "status_code": status_code,
            "reason": safe_reason,
        },
    )


def emit_access_denied_event_background(
    request: Request,
    *,
    status_code: int,
    reason: str | None = None,
) -> None:
    _schedule_activity_task(
        coro=emit_access_denied_event(
            request,
            status_code=status_code,
            reason=reason,
        )
    )
