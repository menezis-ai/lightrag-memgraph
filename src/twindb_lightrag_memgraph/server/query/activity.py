"""Best-effort activity logging for Twin query routes."""

from __future__ import annotations

import logging

from fastapi import Request

from .models import TwinQueryBody

logger = logging.getLogger(__name__)


def _actor_from_request(body: TwinQueryBody, request: Request) -> str:
    if body.actor and body.actor.strip():
        return body.actor.strip()
    for header in (
        "x-auth-request-email",
        "x-forwarded-user",
        "x-auth-request-user",
    ):
        value = request.headers.get(header)
        if value and value.strip():
            return value.strip()
    return "system"


async def _record_retrieval_activity(
    body: TwinQueryBody,
    request: Request,
    *,
    folder: str,
    sources_count: int,
    stream: bool,
) -> None:
    """Best-effort Activity write for completed retrieval calls."""
    try:
        from ..webui_router import _make_event, get_store

        actor = _actor_from_request(body, request)
        event = _make_event(
            kind="retrieval",
            sev="info",
            actor=actor,
            target_label=body.query[:120],
            summary=f"retrieval completed ({body.mode})",
            meta={
                "query": body.query,
                "mode": body.mode,
                "top_k": body.top_k,
                "sources_count": sources_count,
                "stream": stream,
                "tag_filter": body.tag_filter,
                "doc_filter": body.doc_filter,
            },
            target_type="query",
        )
        await get_store(folder).record_activity(event)
    except Exception:
        logger.exception("twin_query: failed to record retrieval activity")


__all__ = [
    "_actor_from_request",
    "_record_retrieval_activity",
]
