"""Prompt boundary helpers for untrusted text."""

from __future__ import annotations

import re

_RESERVED_TAG_RE = re.compile(r"</?(UNTRUSTED_[A-Z_]+|USER_QUESTION)\b", re.IGNORECASE)


def neutralize_reserved_tags(text: object) -> str:
    """Prevent user/document text from closing or forging prompt boundary tags."""
    value = "" if text is None else str(text)
    return _RESERVED_TAG_RE.sub(lambda match: match.group(0).replace("<", "< "), value)
