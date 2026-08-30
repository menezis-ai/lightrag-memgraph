"""Vision-ingestion settings endpoints under ``/twin/api/settings/vision``.

Two endpoints:

- ``GET`` — effective curation settings (runtime store value when set,
  env defaults otherwise). Auth-gated only: any authenticated operator can
  read what the pipeline is doing.
- ``PUT`` — persist new values (admin-gated, mirrors the API-keys posture).
  Applies process-wide immediately: ``_vision`` re-reads through its
  provider on every image.

Scope is deliberately limited to the two CURATION knobs (``min_ocr_chars``,
``drop_classes``) plus the admin-controlled procedure-ingestion activation
flag. The infrastructure wiring (endpoint URL, API key, model, timeouts,
size caps) stays env-only — secrets and SSRF surface do not belong in a
UI-mutable store (docs/adr/005-markitdown-ingestion-supply-chain.md).

Each mutation emits a ``vision-settings-updated`` activity event.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .. import _procedure, _vision
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
    """PUT body — operator-tunable ingestion settings.

    ``procedure_enabled`` is optional only for backward compatibility with
    older WebUI clients. When omitted, PUT preserves the stored activation
    value (or the deployment default when no override exists).
    """

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
    procedure_enabled: bool | None = Field(
        default=None,
        description=(
            "Whether new procedure PDFs enter the review workflow. Existing "
            "parked bundles remain reviewable when false."
        ),
    )

    model_config = {"str_strip_whitespace": True}


class VisionSettingsPublic(BaseModel):
    """GET/PUT response: effective values + provenance."""

    min_ocr_chars: int = Field(description="Effective RapidOCR pre-filter threshold.")
    drop_classes: list[str] = Field(
        description="Effective post-analysis image classes to discard."
    )
    procedure_enabled: bool = Field(
        description="Admin-controlled activation intent for new procedure PDFs."
    )
    procedure_available: bool = Field(
        description=(
            "Whether PDF and vision prerequisites can currently run procedure "
            "ingestion."
        )
    )
    source: Literal["runtime", "env-default"] = Field(
        description="Value provenance: 'runtime' or 'env-default'."
    )
    updated_at: int | None = Field(
        default=None,
        description="Runtime update time in Unix milliseconds, if persisted.",
    )
    updated_by: str | None = Field(
        default=None,
        description="Administrator identity that last persisted the settings.",
    )


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
        "procedure_enabled": _procedure.default_enabled(),
        "procedure_available": _procedure.is_available(),
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
                f"{', '.join(settings['drop_classes']) or '(none)'}, "
                f"procedure ingestion "
                f"{'enabled' if settings['procedure_enabled'] else 'disabled'}"
            ),
            meta={
                "min_ocr_chars": settings["min_ocr_chars"],
                "drop_classes": settings["drop_classes"],
                "procedure_enabled": settings["procedure_enabled"],
            },
            target_type="settings",
            target_id="vision",
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


@router.get(
    "",
    response_model=VisionSettingsPublic,
    summary="Read the image and procedure ingestion settings",
    responses={401: {"description": "Authentication required"}},
)
async def get_vision_settings() -> dict[str, Any]:
    """Return the curation settings applied to image ingestion:
    `min_ocr_chars` (images whose OCR text is shorter are refused
    without an analysis call; 0 disables the filter) and `drop_classes`
    (image classes discarded as noise, e.g. logos or signatures).
    `procedure_enabled` is the admin-controlled activation intent for new
    procedure PDFs; `procedure_available` reports deployment readiness.
    `source` says whether the values come from a runtime update or the
    deployment defaults."""
    workspace = resolve_workspace()
    stored = None
    try:
        stored = await vision_settings_store.get_settings(workspace)
    # A store outage degrades to environment defaults rather than a 500.
    except Exception:
        logger.exception("[VisionSettings] store read failed for %s", workspace)
    if stored:
        stored_procedure_enabled = stored.get("procedure_enabled")
        return {
            "min_ocr_chars": stored.get("min_ocr_chars", 0),
            "drop_classes": stored.get("drop_classes", []),
            "procedure_enabled": (
                stored_procedure_enabled
                if isinstance(stored_procedure_enabled, bool)
                else _procedure.default_enabled()
            ),
            "procedure_available": _procedure.is_available(),
            "source": "runtime",
            "updated_at": stored.get("updated_at"),
            "updated_by": stored.get("updated_by"),
        }
    return {**_env_defaults(), "source": "env-default"}


@router.put(
    "",
    response_model=VisionSettingsPublic,
    dependencies=[Depends(require_admin_user)],
    summary="Update the image and procedure ingestion settings (admin)",
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Administrator scope required"},
        409: {"description": "Procedure prerequisites unavailable"},
        422: {"description": "Invalid drop class or out-of-range value"},
    },
)
async def put_vision_settings(
    body: VisionSettings,
    admin: Annotated[dict[str, Any], Depends(require_admin_user)],
) -> dict[str, Any]:
    """Persist ingestion settings, effective immediately (no restart).

    Procedure activation is admin-controlled, while the analysis endpoint
    configuration itself remains deployment-managed.
    """
    workspace = resolve_workspace()
    classes = _validate_classes(body.drop_classes)
    # Backward-compatible omission requires this read-modify-write. Concurrent
    # legacy/current admin PUTs can race until the store grows a CAS/version
    # contract; the route's admin-only scope keeps that accepted debt narrow.
    existing = await vision_settings_store.get_settings(workspace)
    stored_enabled = (existing or {}).get("procedure_enabled")
    previous_procedure_enabled = (
        stored_enabled
        if isinstance(stored_enabled, bool)
        else _procedure.default_enabled()
    )
    if body.procedure_enabled is None:
        procedure_enabled = previous_procedure_enabled
    else:
        procedure_enabled = body.procedure_enabled
    if (
        procedure_enabled
        and not previous_procedure_enabled
        and not _procedure.is_available()
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Procedure ingestion cannot be enabled because its PDF or "
                "vision prerequisites are not available in this deployment."
            ),
        )
    try:
        await vision_settings_store.initialize(workspace)
    except Exception:  # noqa: BLE001 (index creation is best-effort)
        logger.exception("[VisionSettings] initialize failed for %s", workspace)
    actor = _actor_from_user(admin)
    stored = await vision_settings_store.update_settings(
        workspace,
        min_ocr_chars=body.min_ocr_chars,
        drop_classes=classes,
        procedure_enabled=procedure_enabled,
        updated_by=actor,
    )
    await _emit_event(actor=actor, settings=stored)
    return {
        "min_ocr_chars": stored["min_ocr_chars"],
        "drop_classes": stored["drop_classes"],
        "procedure_enabled": stored["procedure_enabled"],
        "procedure_available": _procedure.is_available(),
        "source": "runtime",
        "updated_at": stored["updated_at"],
        "updated_by": stored["updated_by"],
    }


def install_settings_provider() -> None:
    """Wire vision and procedure ingestion to the shared runtime store.

    Called at app mount. The provider is workspace-resolved per call so a
    workspace change (tests) is honored; store failures degrade to env
    defaults inside ``_vision._effective_settings``.
    """

    async def _provider() -> dict[str, Any] | None:
        return await vision_settings_store.get_settings(resolve_workspace())

    _vision.set_settings_provider(_provider)
    _procedure.set_settings_provider(_provider)
    logger.info("[VisionSettings] runtime settings provider installed")


__all__ = ["router", "install_settings_provider"]
