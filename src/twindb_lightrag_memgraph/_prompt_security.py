"""Prompt boundary helpers for untrusted text.

Stdlib-only and dependency-light BY DESIGN: the storage backends
(``kv_impl`` / ``vector_impl`` — the public GitHub slice) import this to
neutralize chunk content at ingestion (audit 2026-08-06, R-06), so it must
not pull FastAPI/pydantic/intelligence dependencies. The intelligence
layer keeps importing the same function through
``intelligence.prompt_security`` (a re-export shim).
"""

from __future__ import annotations

import re

_RESERVED_TAG_RE = re.compile(r"</?(UNTRUSTED_[A-Z_]+|USER_QUESTION)\b", re.IGNORECASE)


def neutralize_reserved_tags(text: object) -> str:
    """Prevent user/document text from closing or forging prompt boundary tags."""
    value = "" if text is None else str(text)
    return _RESERVED_TAG_RE.sub(lambda match: match.group(0).replace("<", "< "), value)


def neutralize_chunk_payloads(data: dict) -> dict:
    """Return upsert payloads with each string ``content`` tag-neutralized.

    Audit 2026-08-06, R-06: chunk text is untrusted, attacker-writable
    content that lands verbatim in LLM prompts. Neutralizing at the storage
    boundary (ingestion) — rather than at query time — protects every
    downstream consumer (generation prompts, intelligence rerank, chunk
    display) without breaking the query cache. Items without a string
    ``content`` pass through untouched. This stops delimiter forgery only;
    it cannot stop natural-language instructions — see the system-prompt
    doctrine patch (``registry._patch_untrusted_context_doctrine``) for the
    complementary layer and its honest residual.
    """
    neutralized: dict = {}
    for key, value in data.items():
        if isinstance(value, dict) and isinstance(value.get("content"), str):
            neutralized[key] = {
                **value,
                "content": neutralize_reserved_tags(value["content"]),
            }
        else:
            neutralized[key] = value
    return neutralized
