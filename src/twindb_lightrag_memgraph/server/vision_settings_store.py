"""Runtime store for the operator-tunable vision-ingestion settings.

Backs ``/twin/api/settings/vision`` (see ``vision_settings_routes.py``).
Only curation and procedure activation are runtime-mutable — the
infrastructure wiring (endpoint URL, API key, model, timeouts, size caps)
stays env-only by design (secrets + SSRF surface, see
docs/adr/005-markitdown-ingestion-supply-chain.md):

- ``min_ocr_chars`` — RapidOCR pre-filter threshold,
- ``drop_classes`` — vision classifications refused after the LLM call.
- ``procedure_enabled`` — whether new procedure PDFs enter review.

One node per workspace: label ``WebuiSettings_{workspace}`` (mirrors the
``WebuiApiKey_{workspace}`` overlay namespacing), ``id="vision"``, payload
in a ``data`` JSON blob. Memgraph persistence makes the setting visible to
every gunicorn worker on its next read — ``_vision`` re-reads through the
provider on each image, so a PUT applies process-wide without restart.

The store is FastAPI-free so unit tests can drive it without the app.
Absent node → ``None`` (callers fall back to the env defaults).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .. import _pool
from .._constants import validate_identifier

logger = logging.getLogger(__name__)

SETTINGS_ID = "vision"


def _label(workspace: str) -> str:
    validate_identifier(workspace, "workspace")
    return f"WebuiSettings_{workspace}"


def _now_ms() -> int:
    return int(time.time() * 1000)


async def initialize(workspace: str) -> None:
    """Best-effort index on ``id`` for the settings label."""
    label = _label(workspace)
    async with _pool.acquire_write_slot(), _pool.get_session() as session:
        result = await session.run(f"CREATE INDEX ON :`{label}`(id)")
        try:
            await result.consume()
        except Exception as exc:  # index may already exist
            if "already exists" not in str(exc).lower():
                raise


async def get_settings(workspace: str) -> dict[str, Any] | None:
    """Return the persisted vision settings, ``None`` when never set."""
    label = _label(workspace)
    async with _pool.get_read_session() as session:
        result = await session.run(
            f"MATCH (s:`{label}` {{id: $id}}) RETURN s.data AS data",
            id=SETTINGS_ID,
        )
        record = await result.single()
        await result.consume()
    if record is None or not record["data"]:
        return None
    try:
        data = json.loads(record["data"])
    except (TypeError, ValueError):
        logger.warning("[VisionSettings] corrupt data blob for %s — ignored", label)
        return None
    return data if isinstance(data, dict) else None


async def update_settings(
    workspace: str,
    *,
    min_ocr_chars: int,
    drop_classes: list[str],
    procedure_enabled: bool,
    updated_by: str,
) -> dict[str, Any]:
    """Persist the settings node (MERGE — single node per workspace)."""
    label = _label(workspace)
    data = {
        "min_ocr_chars": int(min_ocr_chars),
        "drop_classes": sorted({c.strip().lower() for c in drop_classes if c.strip()}),
        "procedure_enabled": bool(procedure_enabled),
        "updated_at": _now_ms(),
        "updated_by": updated_by,
    }
    async with _pool.acquire_write_slot(), _pool.get_session() as session:
        result = await session.run(
            f"MERGE (s:`{label}` {{id: $id}}) SET s.data = $data",
            id=SETTINGS_ID,
            data=json.dumps(data, ensure_ascii=False),
        )
        await result.consume()
    return data


async def reset_workspace(workspace: str) -> None:
    """Test helper: drop the settings node for ``workspace``."""
    label = _label(workspace)
    async with _pool.acquire_write_slot(), _pool.get_session() as session:
        result = await session.run(f"MATCH (s:`{label}`) DETACH DELETE s")
        await result.consume()
