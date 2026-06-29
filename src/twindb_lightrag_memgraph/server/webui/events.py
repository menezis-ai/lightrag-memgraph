"""Event and notification payload helpers for WebUI routes."""

from __future__ import annotations

import datetime
import secrets
from typing import Any


def _utcnow_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def _make_event(
    *,
    kind: str,
    sev: str,
    actor: str,
    target_label: str,
    summary: str,
    meta: dict[str, Any],
    target_type: str = "tag",
    target_id: str | None = None,
) -> dict[str, Any]:
    target = {"type": target_type, "label": target_label}
    if target_id is not None:
        target["id"] = target_id
    return {
        "id": _new_id("evt"),
        "ts": _utcnow_iso(),
        "rel": "now",
        "day": "Today",
        "kind": kind,
        "sev": sev,
        "actor": {"user": actor, "role": "operator"},
        "target": target,
        "summary": summary,
        "meta": meta,
    }


def _make_notification(
    *,
    title: str,
    tagname: str | None,
    suffix: str | None,
    sub: str,
    kind: str = "tag-mutation",
) -> dict[str, Any]:
    return {
        "id": _new_id("n"),
        "kind": kind,
        "title": title,
        "tagname": tagname,
        "suffix": suffix,
        "sub": sub,
        "rel": "now",
        "read": False,
    }
