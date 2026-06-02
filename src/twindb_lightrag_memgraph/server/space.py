"""HTTP request binding for Twin spaces."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Mapping

from fastapi import HTTPException, Request

from .._constants import validate_identifier
from .._spaces import (
    TwinSpace,
    TwinSpaceCatalog,
    build_runtime_space_config,
    load_space_catalog,
)

_active_space_id: ContextVar[str | None] = ContextVar(
    "twin_active_space_id",
    default=None,
)


def resolve_space_from_headers(headers: Mapping[str, str]) -> str:
    catalog = load_space_catalog()
    raw = headers.get("x-twin-space") or headers.get("x-twin-workspace")
    candidate = (raw or catalog.default_space_id).strip()
    try:
        space_id = validate_identifier(candidate, "space")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if space_id not in catalog.ids:
        raise HTTPException(
            status_code=403,
            detail="No space available for this KB. Please contact Twincore Team",
        )
    return space_id


async def bind_request_space(request: Request):
    space_id = resolve_space_from_headers(request.headers)
    request.state.space = space_id
    token = _active_space_id.set(space_id)
    try:
        yield space_id
    finally:
        _active_space_id.reset(token)


def current_space_id() -> str:
    return _active_space_id.get() or load_space_catalog().default_space_id


__all__ = [
    "TwinSpace",
    "TwinSpaceCatalog",
    "bind_request_space",
    "build_runtime_space_config",
    "current_space_id",
    "load_space_catalog",
    "resolve_space_from_headers",
]

