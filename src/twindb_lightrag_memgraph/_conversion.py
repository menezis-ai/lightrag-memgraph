"""MarkItDown-based pre-conversion of uploaded files to structured markdown.

Supply-chain doctrine (``MARKITDOWN-INGESTION-PLAN.md``): Twin owns the
extraction stage. Files whose format MarkItDown handles better than
LightRAG's native extractors are converted to markdown *before* enqueue, so
every LightRAG version of the CI matrix ingests the same clean markdown
through its simplest, most stable text path — instead of each version's own
per-format extractor (pypdf flat text, tab-separated docx tables, …).

Contract (additive / graceful degradation):

- ``TWIN_CONVERT=off``, markitdown not installed, format not covered, file
  too large, or the conversion itself failing → the caller MUST fall back to
  the untouched native path. This module never raises into the ingestion
  flow: :func:`aconvert_file` returns ``None`` on any failure.
- The MIP classification gate keeps working unchanged: the converted
  markdown is enqueued under the ORIGINAL file name by the registry patch,
  so ``_classification_hook._resolve_detection_path`` still resolves (and
  classifies) the original binary in the INPUT_DIR tree.

Security posture:

- Only :meth:`MarkItDown.convert_local` on a path already written under the
  INPUT_DIR by LightRAG's own upload/scan surface. Never the string form of
  ``convert()`` — it transparently fetches ``http(s)://`` URLs (SSRF).
- Hard size cap (``TWIN_CONVERT_MAX_BYTES``) and per-conversion timeout
  (``TWIN_CONVERT_TIMEOUT``). On timeout the worker thread cannot be killed
  (CPython limitation) but the ingestion flow falls back immediately; the
  orphan thread finishes and its result is discarded.
- ZIP is deliberately NOT in the default format set: markitdown 0.1.6 has no
  decompression-size/recursion guard (zip-bomb DoS, audit 2026-07-10 §B4).
- Conversions are serialized behind a module lock: MarkItDown's converter
  instances are not documented thread-safe, and one-at-a-time is a sane
  resource bound for a CPU-heavy stage.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path

from ._constants import (
    _FALSE_FLAG_VALUES,
    TWIN_CONVERT_ENV,
    TWIN_CONVERT_FORMATS_ENV,
    TWIN_CONVERT_MAX_BYTES_ENV,
    TWIN_CONVERT_TIMEOUT_ENV,
)

logger = logging.getLogger("twindb_lightrag_memgraph")

_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})

# Formats converted by default. Two families:
# - quality upgrade over the native extractors: pdf/docx/pptx/xlsx;
# - repair of formats the native path accepts but mangles (raw utf-8 decode
#   of tagged/binary content) or rejects: html/htm/csv/epub, xls/msg.
DEFAULT_CONVERT_FORMATS = frozenset(
    {
        "pdf",
        "docx",
        "pptx",
        "xlsx",
        "xls",
        "msg",
        "epub",
        "html",
        "htm",
        "csv",
    }
)

DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # aligned on LightRAG's upload cap
DEFAULT_TIMEOUT_SECONDS = 120.0

_availability_lock = threading.Lock()
_availability: bool | None = None
_converter_lock = threading.Lock()
_converter = None
_forced_on_warned = False


def _resolve_mode() -> bool | None:
    """Resolve ``TWIN_CONVERT``: True (on), False (off), None (auto)."""
    raw = os.environ.get(TWIN_CONVERT_ENV, "").strip().lower()
    if raw in _FALSE_FLAG_VALUES:
        return False
    if raw in _TRUE_FLAG_VALUES:
        return True
    return None


def is_available() -> bool:
    """True iff the optional markitdown dependency is importable (cached)."""
    global _availability
    if _availability is None:
        with _availability_lock:
            if _availability is None:
                try:
                    import markitdown  # noqa: F401

                    _availability = True
                except ImportError:
                    _availability = False
    return _availability


def reset_caches() -> None:
    """Test hook: forget the cached import probe and converter singleton."""
    global _availability, _converter, _forced_on_warned
    with _availability_lock:
        _availability = None
    with _converter_lock:
        _converter = None
    _forced_on_warned = False


def is_enabled() -> bool:
    """Whether the pre-conversion tier is active for this process."""
    global _forced_on_warned
    mode = _resolve_mode()
    if mode is False:
        return False
    if not is_available():
        if mode is True and not _forced_on_warned:
            _forced_on_warned = True
            logger.warning(
                "twindb: %s=on but markitdown is not importable — install the "
                "[convert] extra; falling back to native extraction",
                TWIN_CONVERT_ENV,
            )
        return False
    return True


def conversion_formats() -> frozenset[str]:
    """Active format set (bare lowercase extensions, no dot)."""
    raw = os.environ.get(TWIN_CONVERT_FORMATS_ENV, "").strip()
    if not raw:
        return DEFAULT_CONVERT_FORMATS
    formats = frozenset(
        part.strip().lstrip(".").lower() for part in raw.split(",") if part.strip()
    )
    return formats or DEFAULT_CONVERT_FORMATS


def max_convert_bytes() -> int:
    raw = os.environ.get(TWIN_CONVERT_MAX_BYTES_ENV, "").strip()
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_MAX_BYTES
    except ValueError:
        return DEFAULT_MAX_BYTES


def convert_timeout_seconds() -> float:
    raw = os.environ.get(TWIN_CONVERT_TIMEOUT_ENV, "").strip()
    try:
        value = float(raw)
        return value if value > 0 else DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def extra_supported_extensions() -> tuple[str, ...]:
    """Dotted extensions the upload whitelist must accept for conversion.

    Derived from the active format set; the registry patch only appends the
    ones missing from the running ``DocumentManager`` whitelist.
    """
    return tuple(sorted(f".{fmt}" for fmt in conversion_formats()))


def should_convert(file_path: Path | str) -> bool:
    """Cheap synchronous gate: enabled + covered format + under the size cap."""
    if not is_enabled():
        return False
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")
    if ext not in conversion_formats():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size > max_convert_bytes():
        logger.warning(
            "twindb convert: %s exceeds %s (%d bytes) — native path",
            path.name,
            TWIN_CONVERT_MAX_BYTES_ENV,
            size,
        )
        return False
    return True


def _get_converter():
    """Lazy MarkItDown singleton (builtins only, plugins off, no LLM/exiftool)."""
    global _converter
    if _converter is None:
        from markitdown import MarkItDown

        _converter = MarkItDown(enable_plugins=False)
    return _converter


def _convert_sync(path: Path) -> str:
    with _converter_lock:
        result = _get_converter().convert_local(path)
    markdown = getattr(result, "markdown", None) or getattr(result, "text_content", "")
    return markdown or ""


async def aconvert_file(file_path: Path | str) -> str | None:
    """Convert ``file_path`` to markdown; ``None`` on ANY failure (logged).

    ``None`` instructs the caller to run the untouched native extraction
    path — this function is the graceful-degradation boundary.
    """
    path = Path(file_path)
    try:
        markdown = await asyncio.wait_for(
            asyncio.to_thread(_convert_sync, path),
            timeout=convert_timeout_seconds(),
        )
    except asyncio.TimeoutError:
        logger.warning(
            "twindb convert: %s timed out after %.0fs — native path",
            path.name,
            convert_timeout_seconds(),
        )
        return None
    except Exception as exc:
        logger.warning(
            "twindb convert: %s failed (%s: %s) — native path",
            path.name,
            type(exc).__name__,
            exc,
        )
        return None
    if not markdown.strip():
        logger.warning(
            "twindb convert: %s produced empty markdown — native path", path.name
        )
        return None
    logger.info("twindb convert: %s → markdown (%d chars)", path.name, len(markdown))
    return markdown
