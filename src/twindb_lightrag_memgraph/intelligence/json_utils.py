"""Tolerant JSON helpers for LLM responses."""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

logger = logging.getLogger("twin_rag_intelligence.json")


def load_json_object(content: str | None, *, context: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, including fenced JSON fallback."""
    if not content:
        return {}

    text = content.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            logger.warning("%s returned non-JSON content", context)
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("%s returned malformed JSON: %s", context, exc)
            return {}

    if not isinstance(data, dict):
        logger.warning(
            "%s returned JSON %s, expected object", context, type(data).__name__
        )
        return {}
    return data


def clamp_float(
    value: Any, default: float = 0.0, low: float = 0.0, high: float = 1.0
) -> float:
    """Coerce a numeric LLM field into a bounded float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(low, min(high, number))


def coerce_str(value: Any, default: str = "") -> str:
    """Return a stripped string for scalar LLM fields."""
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return default
    return str(value).strip()
