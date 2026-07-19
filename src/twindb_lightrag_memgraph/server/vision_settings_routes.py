"""Vision-ingestion settings endpoints under ``/twin/api/settings/vision``.

Two endpoints:

- ``GET`` — effective curation settings (runtime store value when set,
  env defaults otherwise). Auth-gated only: any authenticated operator can
  read what the pipeline is doing.
- ``PUT`` — persist new values (admin-gated, mirrors the API-keys posture).
  Applies process-wide immediately: ``_vision`` re-reads through its
  provider on every image.

Scope is deliberately limited to the two CURATION knobs (``min_ocr_chars``,
``drop_classes``). The infrastructure wiring (endpoint URL, API key, model,
timeouts, size caps) stays env-only — secrets and SSRF surface do not
belong in a UI-mutable store (MARKITDOWN-INGESTION-PLAN.md).

Each mutation emits a ``vision-settings-updated`` activity event.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .. import _vision
from .._constants import resolve_workspace
from . import vision_settings_store
from .auth import require_auth
from .idp_jwt import require_admin_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/settings/vision",
    tags=["vision-settings"],
    dependencies=[Depends(require_auth)],
)

_CLASS_PATTERN = r"^[a-z0-9][a-z0-9 _-]{0,39}$"


class VisionSettings(BaseModel):
    """PUT body — the two operator-tunable curation knobs."""

    min_ocr_chars: int = Field(
        ge=0,
        le=100_000,
        description=(
            "RapidOCR pre-filter threshold; images with less OCR text are "
            "refused without a vision-LLM call. 0 captions every image."
        ),
    )
    drop_classes: list[str] = Field(
        max_length=20,
        description="Vision classifications refused after the LLM call.",
    )

    model_config = {"str_strip_whitespace": True}


class VisionSettingsPublic(VisionSettings):
    """GET/PUT response: effective values + provenance."""

    source: str  # "runtime" | "env-default"
    updated_at: int | None = None
    updated_by: str | None = None


def _validate_classes(classes: list[str]) -> list[str]:
    import re

    cleaned: list[str] = []
    for value in classes:
        slug = value.strip().lower()
        if not slug:
            continue
        if not re.match(_CLASS_PATTERN, slug):
            from fastapi import HTTPException

            raise HTTPException(
                status_code=422,
                detail=f"invalid drop class {value!r} (letters/digits/-_ only)",
            )
        cleaned.append(slug)
    return sorted(set(cleaned))


def _env_defaults() -> dict[str, Any]:
    return {
        "min_ocr_chars": _vision.min_ocr_chars(),
        "drop_classes": sorted(_vision.drop_classes()),
    }


async def _emit_event(*, actor: str, settings: dict[str, Any]) -> None:
    """Best-effort activity emission — failures logged, never raised."""
    try:
        from . import webui_router

        event = webui_router._make_event(
            kind="vision-settings-updated",
            sev="info",
            actor=actor,
            target_label="vision",
            summary=(
                f"Vision settings updated: min OCR chars "
                f"{settings['min_ocr_chars']}, drop classes "
                f"{', '.join(settings['drop_classes']) or '(none)'}"
            ),
            meta={
                "min_ocr_chars": settings["min_ocr_chars"],
                "drop_classes": settings["drop_classes"],
            },
            target_type="settings",
        )
        store = webui_router.get_store()
        await store.record_activity(event)
    except Exception:  # noqa: BLE001 (audit must never break the request)
        logger.exception("[VisionSettings] activity emission failed")


def _actor_from_user(user: dict[str, Any] | None) -> str:
    if not isinstance(user, dict):
        return "operator"
    return str(
        user.get("sso_subject") or user.get("email") or user.get("sub") or "operator"
    )


@router.get("", response_model=VisionSettingsPublic)
async def get_vision_settings() -> dict[str, Any]:
    """Effective curation settings: runtime store value or env defaults."""
    workspace = resolve_workspace()
    stored = None
    try:
        stored = await vision_settings_store.get_settings(workspace)
    except Exception:  # noqa: BLE001 (store down = env defaults, not a 500)
        logger.exception("[VisionSettings] store read failed for %s", workspace)
    if stored:
        return {
            "min_ocr_chars": stored.get("min_ocr_chars", 0),
            "drop_classes": stored.get("drop_classes", []),
            "source": "runtime",
            "updated_at": stored.get("updated_at"),
            "updated_by": stored.get("updated_by"),
        }
    return {**_env_defaults(), "source": "env-default"}


@router.put(
    "",
    response_model=VisionSettingsPublic,
    dependencies=[Depends(require_admin_user)],
    responses={422: {"description": "Invalid drop class or out-of-range value"}},
)
async def put_vision_settings(
    body: VisionSettings,
    admin: Annotated[dict[str, Any], Depends(require_admin_user)],
) -> dict[str, Any]:
    """Persist new curation settings (admin only), effective immediately."""
    workspace = resolve_workspace()
    classes = _validate_classes(body.drop_classes)
    try:
        await vision_settings_store.initialize(workspace)
    except Exception:  # noqa: BLE001 (index creation is best-effort)
        logger.exception("[VisionSettings] initialize failed for %s", workspace)
    actor = _actor_from_user(admin)
    stored = await vision_settings_store.update_settings(
        workspace,
        min_ocr_chars=body.min_ocr_chars,
        drop_classes=classes,
        updated_by=actor,
    )
    await _emit_event(actor=actor, settings=stored)
    return {
        "min_ocr_chars": stored["min_ocr_chars"],
        "drop_classes": stored["drop_classes"],
        "source": "runtime",
        "updated_at": stored["updated_at"],
        "updated_by": stored["updated_by"],
    }


def install_settings_provider() -> None:
    """Wire ``_vision`` to read the runtime store on every image.

    Called at app mount. The provider is workspace-resolved per call so a
    workspace change (tests) is honored; store failures degrade to env
    defaults inside ``_vision._effective_settings``.
    """

    async def _provider() -> dict[str, Any] | None:
        return await vision_settings_store.get_settings(resolve_workspace())

    _vision.set_settings_provider(_provider)
    logger.info("[VisionSettings] runtime settings provider installed")


__all__ = ["router", "install_settings_provider"]
