"""HTTP request binding for Twin spaces."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Mapping

from fastapi import HTTPException, Request

from .._constants import validate_identifier
from .._spaces import (
    TwinSpace,
    TwinSpaceCatalog,
    load_space_catalog as _load_env_space_catalog,
)
from . import space_store

_active_space_id: ContextVar[str | None] = ContextVar(
    "twin_active_space_id",
    default=None,
)


def load_space_catalog() -> TwinSpaceCatalog:
    """Return the merged Twin space catalog (env seed + runtime admin
    additions). Env seed always wins on id collision so a corrupt
    runtime store can't shadow the SRE-provisioned default.

    `max_spaces` is taken from the env catalog. Spaces beyond that cap
    are dropped from the head (env spaces first, then runtime).
    """
    env_catalog = _load_env_space_catalog()
    env_ids = {s.id for s in env_catalog.spaces}
    runtime = [s for s in space_store.list_runtime_spaces() if s.id not in env_ids]
    merged = (*env_catalog.spaces, *runtime)
    merged = merged[: env_catalog.max_spaces]
    return TwinSpaceCatalog(
        default_space_id=env_catalog.default_space_id,
        max_spaces=env_catalog.max_spaces,
        spaces=tuple(merged),
        explicit=env_catalog.explicit or bool(runtime),
    )


def is_env_seeded_space(space_id: str) -> bool:
    """True when ``space_id`` came from the env seed (mutations on it
    must 403 — only the SRE provisioning channel can change those)."""
    return space_id in {s.id for s in _load_env_space_catalog().spaces}


def build_runtime_space_config() -> dict:
    """Mirror `_spaces.build_runtime_space_config` but consume the
    merged catalog so the React port sees runtime additions in
    `window.__twinConfig.spaces`."""
    catalog = load_space_catalog()
    return {
        "defaultSpaceId": catalog.default_space_id,
        "spaces": [space.as_runtime_config() for space in catalog.spaces],
        "maxSpaces": catalog.max_spaces,
    }


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
    "is_env_seeded_space",
    "load_space_catalog",
    "resolve_space_from_headers",
]

