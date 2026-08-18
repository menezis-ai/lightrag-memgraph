"""Prompt boundary helpers for untrusted text.

Compatibility shim: the implementation lives in
``twindb_lightrag_memgraph._prompt_security`` (package root, stdlib-only)
since 1.1.0 so the storage backends can neutralize chunk content at
ingestion (audit 2026-08-06, R-06) without importing the intelligence
package. This re-export keeps the historical import path working.
"""

from __future__ import annotations

from .._prompt_security import neutralize_reserved_tags

__all__ = ["neutralize_reserved_tags"]
