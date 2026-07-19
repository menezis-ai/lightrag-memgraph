"""Vision-based image ingestion (MARKITDOWN-INGESTION-PLAN.md, PR 2).

Knowledge-Bot pattern (audit 2026-07-10 §D): an uploaded image goes through

1. **RapidOCR pre-filter** (offline, pip-only) — cheap text extraction; an
   image whose OCR text is under ``TWIN_VISION_MIN_OCR_CHARS`` never costs a
   vision-LLM call (set 0 to disable the pre-filter and caption everything).
2. **Vision LLM** — OpenAI-compatible endpoint (BNP: Gemma 4 31B-it served
   as LLM-as-a-service on vLLM), ``response_format=json_object``,
   ``temperature=0``, image passed as a base64 data URI. The model returns
   ``{"image_classification": str, "content": str}`` — the classification
   doubles as a relevance filter.
3. **Noise drop** — classes in ``TWIN_VISION_DROP_CLASSES`` (default
   ``invalid,logo,signature``) are refused with an explicit reason.
4. The surviving ``content`` (plus the raw OCR text) becomes the markdown
   enqueued under the ORIGINAL file name by the registry seam — same
   contract as the MarkItDown tier, so MIP policy, dedup, folder membership
   and cleanup are untouched. Images carry no MIP label: they ingest under
   ``TWIN_MIP_UNLABELED_POLICY=allow`` and are refused under ``reject``.

Contract: this module never raises into the ingestion flow.
:func:`aprocess_image` returns a :class:`VisionOutcome`; ``markdown=None``
means "do not ingest" and ``reason`` says why (the registry seam turns it
into an explicit FAILED error-document — unlike the MarkItDown tier there is
no native extractor to fall back to for an image).

The LLM client is the *sync* ``openai.OpenAI`` (the async client cannot be
awaited from the worker thread); calls run in ``asyncio.to_thread`` behind a
hard timeout. RapidOCR absent → pre-filter bypassed (warn once), the vision
call still runs.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from ._constants import (
    _FALSE_FLAG_VALUES,
    TWIN_VISION_API_KEY_ENV,
    TWIN_VISION_BASE_URL_ENV,
    TWIN_VISION_DROP_CLASSES_ENV,
    TWIN_VISION_ENV,
    TWIN_VISION_FORMATS_ENV,
    TWIN_VISION_MAX_BYTES_ENV,
    TWIN_VISION_MIN_OCR_CHARS_ENV,
    TWIN_VISION_MODEL_ENV,
    TWIN_VISION_TIMEOUT_ENV,
)

logger = logging.getLogger("twindb_lightrag_memgraph")

_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})

DEFAULT_VISION_FORMATS = frozenset({"png", "jpg", "jpeg"})
DEFAULT_DROP_CLASSES = frozenset({"invalid", "logo", "signature"})
DEFAULT_MIN_OCR_CHARS = 20
DEFAULT_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0

_MIME_BY_EXT = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}

# Prompt adapted from the Knowledge-Bot vision templates (audit §D): JSON
# braces are literal here (plain string, no .format()).
VISION_SYSTEM_PROMPT = """\
You are an advanced image understanding model analyzing images ingested \
into a knowledge base.

Respond with a single valid JSON object with exactly two fields:

{"image_classification": str, "content": str}

- "image_classification": the type of image (e.g. logo, screenshot, table, \
graph, diagram, photo, scanned-document, signature, etc.). Use "invalid" \
when the image carries no informational content.
- "content": a concise but complete summary of the informational content of \
the image. Transcribe visible text faithfully. Do not describe visual style \
or colors. Keep the original language of any text in the image.
"""

VISION_USER_PROMPT = (
    "Analyze the given image and return the JSON object "
    '{"image_classification": str, "content": str}.'
)

_availability_lock = threading.Lock()
_openai_available: bool | None = None
_ocr_lock = threading.Lock()
_ocr_engine = None
_ocr_missing_warned = False
_client = None
_client_lock = threading.Lock()


@dataclass(frozen=True)
class VisionOutcome:
    """Result of the image pipeline; ``markdown=None`` => do not ingest."""

    markdown: str | None
    reason: str
    classification: str | None = None


def _resolve_mode() -> bool | None:
    raw = os.environ.get(TWIN_VISION_ENV, "").strip().lower()
    if raw in _FALSE_FLAG_VALUES:
        return False
    if raw in _TRUE_FLAG_VALUES:
        return True
    return None


def _openai_importable() -> bool:
    global _openai_available
    if _openai_available is None:
        with _availability_lock:
            if _openai_available is None:
                try:
                    import openai  # noqa: F401

                    _openai_available = True
                except ImportError:
                    _openai_available = False
    return _openai_available


def _endpoint_configured() -> bool:
    return bool(
        os.environ.get(TWIN_VISION_BASE_URL_ENV, "").strip()
        and os.environ.get(TWIN_VISION_MODEL_ENV, "").strip()
    )


def reset_caches() -> None:
    """Test hook: forget import probes, OCR engine, LLM client, provider."""
    global _openai_available, _ocr_engine, _ocr_missing_warned, _client
    global _settings_provider
    with _availability_lock:
        _openai_available = None
    with _ocr_lock:
        _ocr_engine = None
        _ocr_missing_warned = False
    with _client_lock:
        _client = None
    _settings_provider = None


def is_enabled() -> bool:
    """Vision tier active: mode + openai importable + endpoint configured."""
    mode = _resolve_mode()
    if mode is False:
        return False
    ready = _openai_importable() and _endpoint_configured()
    if not ready and mode is True:
        logger.warning(
            "twindb vision: %s=on but the tier is not usable (openai "
            "importable: %s, %s/%s set: %s) — image ingestion disabled",
            TWIN_VISION_ENV,
            _openai_importable(),
            TWIN_VISION_BASE_URL_ENV,
            TWIN_VISION_MODEL_ENV,
            _endpoint_configured(),
        )
    return ready


def vision_formats() -> frozenset[str]:
    raw = os.environ.get(TWIN_VISION_FORMATS_ENV, "").strip()
    if not raw:
        return DEFAULT_VISION_FORMATS
    formats = frozenset(
        part.strip().lstrip(".").lower() for part in raw.split(",") if part.strip()
    )
    return formats or DEFAULT_VISION_FORMATS


def drop_classes() -> frozenset[str]:
    raw = os.environ.get(TWIN_VISION_DROP_CLASSES_ENV, "").strip()
    if not raw:
        return DEFAULT_DROP_CLASSES
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def min_ocr_chars() -> int:
    raw = os.environ.get(TWIN_VISION_MIN_OCR_CHARS_ENV, "").strip()
    try:
        value = int(raw)
        return value if value >= 0 else DEFAULT_MIN_OCR_CHARS
    except ValueError:
        return DEFAULT_MIN_OCR_CHARS


def max_image_bytes() -> int:
    raw = os.environ.get(TWIN_VISION_MAX_BYTES_ENV, "").strip()
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_MAX_BYTES
    except ValueError:
        return DEFAULT_MAX_BYTES


def vision_timeout_seconds() -> float:
    raw = os.environ.get(TWIN_VISION_TIMEOUT_ENV, "").strip()
    try:
        value = float(raw)
        return value if value > 0 else DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def extra_supported_extensions() -> tuple[str, ...]:
    """Dotted image extensions the upload whitelist must accept."""
    return tuple(sorted(f".{fmt}" for fmt in vision_formats()))


# ---------------------------------------------------------------------------
# Runtime-settings provider (Settings → Vision, /twin/api/settings/vision)
# ---------------------------------------------------------------------------
#
# The server overlay registers an async provider that reads the
# operator-tunable curation knobs (min_ocr_chars, drop_classes) from the
# Memgraph-backed settings store. Reading through the provider on EVERY
# image keeps gunicorn workers consistent without restart. Infrastructure
# wiring (endpoint/key/model/timeouts) deliberately has no runtime override.

_settings_provider = None


def set_settings_provider(provider) -> None:
    """Register an async callable returning the runtime settings dict.

    The callable returns ``{"min_ocr_chars": int, "drop_classes": [str]}``
    (extra keys ignored) or ``None`` when nothing was persisted. Pass
    ``None`` to unregister (env defaults only).
    """
    global _settings_provider
    _settings_provider = provider


async def _effective_settings() -> tuple[int, frozenset[str]]:
    """Resolve (min_ocr_chars, drop_classes): runtime store then env."""
    data = None
    if _settings_provider is not None:
        try:
            data = await _settings_provider()
        except Exception as exc:
            logger.warning(
                "twindb vision: settings provider failed (%s: %s) — env defaults",
                type(exc).__name__,
                exc,
            )
    threshold = min_ocr_chars()
    classes = drop_classes()
    if isinstance(data, dict):
        raw_min = data.get("min_ocr_chars")
        if isinstance(raw_min, int) and not isinstance(raw_min, bool) and raw_min >= 0:
            threshold = raw_min
        raw_drop = data.get("drop_classes")
        if isinstance(raw_drop, list):
            classes = frozenset(
                str(c).strip().lower() for c in raw_drop if str(c).strip()
            )
    return threshold, classes


def should_process(file_path: Path | str) -> bool:
    """Cheap gate: tier enabled + image format + under the size cap."""
    if not is_enabled():
        return False
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")
    if ext not in vision_formats():
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size > max_image_bytes():
        logger.warning(
            "twindb vision: %s exceeds %s (%d bytes) — refused",
            path.name,
            TWIN_VISION_MAX_BYTES_ENV,
            size,
        )
        return False
    return True


def _ocr_text_sync(path: Path) -> str | None:
    """RapidOCR text, ``None`` when the engine is unavailable (bypass)."""
    global _ocr_engine, _ocr_missing_warned
    with _ocr_lock:
        if _ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR

                _ocr_engine = RapidOCR()
            except ImportError:
                if not _ocr_missing_warned:
                    _ocr_missing_warned = True
                    logger.warning(
                        "twindb vision: rapidocr_onnxruntime not installed — "
                        "OCR pre-filter bypassed, every image goes to the "
                        "vision LLM"
                    )
                return None
        engine = _ocr_engine
    try:
        result, _ = engine(str(path))
    except Exception as exc:
        logger.warning(
            "twindb vision: OCR failed for %s (%s) — pre-filter bypassed",
            path.name,
            exc,
        )
        return None
    if not result:
        return ""
    return " ".join(line[1] for line in result)


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            base_url=os.environ[TWIN_VISION_BASE_URL_ENV].strip(),
            api_key=os.environ.get(TWIN_VISION_API_KEY_ENV, "twin-vision").strip()
            or "twin-vision",
        )
    return _client


def _call_vision_llm_sync(path: Path) -> str:
    """One vision chat call; returns the raw model text (JSON expected)."""
    data = path.read_bytes()
    mime = _MIME_BY_EXT.get(path.suffix.lower().lstrip("."), "image/png")
    data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    response = _get_client().chat.completions.create(
        model=os.environ[TWIN_VISION_MODEL_ENV].strip(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    return response.choices[0].message.content or ""


def _parse_vision_json(raw: str) -> dict | None:
    """Tolerant parse of the model reply (bare JSON or fenced block)."""
    for candidate in (raw, _strip_code_fence(raw)):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _strip_code_fence(raw: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw or "", re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    return match.group(0) if match else None


def _compose_markdown(
    path: Path, classification: str, content: str, ocr_text: str | None
) -> str:
    parts = [
        f"# {path.name}",
        f"_Image type: {classification}_",
        content.strip(),
    ]
    if ocr_text and ocr_text.strip():
        parts.append("## Extracted text (OCR)")
        parts.append(ocr_text.strip())
    return "\n\n".join(part for part in parts if part)


async def aprocess_image(file_path: Path | str) -> VisionOutcome:
    """Full image pipeline; never raises — failures land in ``reason``."""
    path = Path(file_path)
    try:
        return await asyncio.wait_for(
            _aprocess_inner(path), timeout=vision_timeout_seconds()
        )
    except asyncio.TimeoutError:
        reason = f"vision-timeout: no result within {vision_timeout_seconds():.0f}s"
        logger.warning("twindb vision: %s — %s", path.name, reason)
        return VisionOutcome(markdown=None, reason=reason)
    except Exception as exc:  # defensive: the seam must never crash
        reason = f"vision-error: {type(exc).__name__}: {exc}"
        logger.warning("twindb vision: %s — %s", path.name, reason)
        return VisionOutcome(markdown=None, reason=reason)


async def _aprocess_inner(path: Path) -> VisionOutcome:
    threshold, active_drop_classes = await _effective_settings()
    ocr_text = await asyncio.to_thread(_ocr_text_sync, path)

    if ocr_text is not None and threshold > 0 and len(ocr_text.strip()) < threshold:
        return VisionOutcome(
            markdown=None,
            reason=(
                f"vision-prefilter: OCR text below {threshold} chars "
                f"({len(ocr_text.strip())}) — set "
                f"{TWIN_VISION_MIN_OCR_CHARS_ENV}=0 to caption everything"
            ),
        )

    try:
        raw = await asyncio.to_thread(_call_vision_llm_sync, path)
    except Exception as exc:
        return VisionOutcome(
            markdown=None,
            reason=f"vision-llm-error: {type(exc).__name__}: {exc}",
        )

    data = _parse_vision_json(raw)
    if data is None:
        return VisionOutcome(
            markdown=None, reason="vision-llm-error: unparseable JSON reply"
        )

    classification = str(data.get("image_classification") or "unknown").strip()
    content = str(data.get("content") or "").strip()

    if classification.lower() in active_drop_classes:
        return VisionOutcome(
            markdown=None,
            reason=f"image-dropped: classification {classification!r}",
            classification=classification,
        )
    if not content or content.lower() == "invalid":
        return VisionOutcome(
            markdown=None,
            reason="image-dropped: no informational content",
            classification=classification,
        )

    markdown = _compose_markdown(path, classification, content, ocr_text)
    logger.info(
        "twindb vision: %s → %s (%d chars)",
        path.name,
        classification,
        len(markdown),
    )
    return VisionOutcome(markdown=markdown, reason="ok", classification=classification)
