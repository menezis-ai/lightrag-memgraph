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
import io
import json
import logging
import os
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
    TWIN_VISION_EXTRA_BODY_ENV,
    TWIN_VISION_TIMEOUT_ENV,
)

logger = logging.getLogger("twindb_lightrag_memgraph")

_VISION_WARNING_FORMAT = "twindb vision: %s — %s"

_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})

DEFAULT_VISION_FORMATS = frozenset({"png", "jpg", "jpeg"})
DEFAULT_DROP_CLASSES = frozenset({"invalid", "logo", "signature"})
DEFAULT_MIN_OCR_CHARS = 20
DEFAULT_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60.0

# Pixel budget for the OCR re-decode retry (see ``_ocr_text_sync``). A 600 dpi
# A4 scan is ~35 MP, so no real document is excluded; a highly compressible
# decompression bomb is tiny on disk (it clears ``max_image_bytes``) yet blows
# past this, and is skipped before Pillow materialises a single pixel.
OCR_REDECODE_MAX_PIXELS = 60_000_000

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
    """Claim an existing image owned by the active vision tier.

    Size is deliberately NOT an ownership gate. Once the upload whitelist
    accepts an image extension, the vision seam must never fall through to a
    native parser merely because the image exceeds the tier cap. The full
    processor returns an explicit refusal for that case instead.
    """
    if not is_enabled():
        return False
    path = Path(file_path)
    ext = path.suffix.lower().lstrip(".")
    if ext not in vision_formats():
        return False
    try:
        path.stat()
    except OSError:
        return False
    return True


def _ocr_source_sync(source, label: str) -> str | None:
    """RapidOCR text for a path or ndarray; ``None`` means bypass.

    RapidOCR/ONNX sessions are kept behind the same lock as lazy creation.
    The standalone-image and PDF-visual tiers can run concurrently, while a
    single shared engine is not documented as thread-safe.
    """
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
            result, _ = engine(source)
        except Exception as exc:
            logger.warning(
                "twindb vision: OCR failed for %s (%s) — pre-filter bypassed",
                label,
                exc,
            )
            return None
    if not result:
        return ""
    try:
        text_parts = []
        for line in result:
            if (
                not isinstance(line, (list, tuple))
                or len(line) < 2
                or not isinstance(line[1], str)
            ):
                raise ValueError("unexpected RapidOCR line shape")
            text_parts.append(line[1])
        return " ".join(text_parts)
    except Exception as exc:
        # A successful transport with an aberrant payload is not evidence that
        # the image has no text. Degrade open exactly like an OCR engine error:
        # bypass the cheap pre-filter and let the semantic vision pass decide.
        logger.warning(
            "twindb vision: OCR returned malformed data for %s (%s) — "
            "pre-filter bypassed",
            label,
            exc,
        )
        return None


def _decode_rgb_pixels(path: Path):
    """RGB pixel array for an OCR re-decode; ``None`` when it must be skipped.

    ``Image.open`` is lazy — the size guard below reads the header only, so a
    decompression bomb is rejected before any pixel is materialised.
    """
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            if width * height > OCR_REDECODE_MAX_PIXELS:
                logger.warning(
                    "twindb vision: %s is %dx%d — skipping OCR re-decode above "
                    "the %d pixel budget",
                    path.name,
                    width,
                    height,
                    OCR_REDECODE_MAX_PIXELS,
                )
                return None
            return np.asarray(image.convert("RGB"))
    except Exception as exc:
        logger.debug(
            "twindb vision: cannot re-decode %s for OCR (%s: %s)",
            path.name,
            type(exc).__name__,
            exc,
        )
        return None


def _ocr_text_sync(path: Path) -> str | None:
    """RapidOCR text from a file path (standalone-image compatibility).

    RapidOCR's own file loader silently yields *nothing* for colour modes it
    cannot decode — CMYK JPEG, the standard output of print/scan and prepress
    pipelines, is the confirmed case. That empty result is indistinguishable
    from "this image genuinely has no text", so the pre-filter would refuse a
    perfectly legible document as noise and it would never reach the vision
    model: a decoder miss fails CLOSED, while a missing OCR engine correctly
    degrades open. When the fast path comes back empty we therefore decode once
    through Pillow — the exact conversion :func:`ocr_png_bytes_sync` already
    performs for PDF visuals, so both OCR entry points now agree — and retry
    before concluding the image carries no text.

    Nominal images never reach the retry: it only runs where the current result
    is empty, i.e. on a document that would otherwise have been dropped.
    """
    text = _ocr_source_sync(str(path), path.name)
    if text:
        return text
    pixels = _decode_rgb_pixels(path)
    if pixels is None:
        return text
    retried = _ocr_source_sync(pixels, f"{path.name} (re-decoded)")
    return retried or text


def ocr_png_bytes_sync(png_bytes: bytes, *, label: str = "PDF visual") -> str | None:
    """RapidOCR text from an in-memory rendered PDF visual.

    The generic PDF tier uses OCR as enrichment only: unlike standalone
    images, a short result never prevents the subsequent semantic Vision
    classification.
    """
    try:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(png_bytes)) as image:
            pixels = np.asarray(image.convert("RGB"))
    except Exception as exc:
        logger.warning("twindb vision: cannot decode %s for OCR (%s)", label, exc)
        return None
    return _ocr_source_sync(pixels, label)


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from openai import OpenAI

                # asyncio cancellation cannot stop a sync request already
                # running in a worker thread.  The transport must therefore
                # enforce the same deadline itself, with SDK retries disabled
                # so one advertised attempt cannot silently become three.
                _client = OpenAI(
                    base_url=os.environ[TWIN_VISION_BASE_URL_ENV].strip(),
                    api_key=os.environ.get(
                        TWIN_VISION_API_KEY_ENV, "twin-vision"
                    ).strip()
                    or "twin-vision",
                    timeout=vision_timeout_seconds(),
                    max_retries=0,
                )
    return _client


def vision_extra_body() -> dict:
    """Vendor-specific request extensions, as JSON in ``TWIN_VISION_EXTRA_BODY``.

    Deliberately opaque: the product must not know about any one gateway. It
    exists because the same model id can be served by several backends with
    different behaviour, and only the deployment knows which it trusts.

    Concrete case (measured 2026-07-25): the CI gate points at OpenRouter,
    which routes ``google/gemma-4-31b-it`` across providers. Cerebras returned
    valid JSON 6/6 times on an identical request; DeepInfra returned
    unparseable output 3/3 and, on one run, a 54525-character repetition loop
    that blew the document timeout. Four CI failures with four different
    symptoms were that routing roulette, not a pipeline defect. The gate now
    pins its provider through this variable; BNP's own vLLM sets nothing.
    """
    raw = os.environ.get(TWIN_VISION_EXTRA_BODY_ENV, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        logger.warning(
            "twindb vision: %s is not valid JSON (%s) — ignored",
            TWIN_VISION_EXTRA_BODY_ENV,
            exc,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "twindb vision: %s must be a JSON object — ignored",
            TWIN_VISION_EXTRA_BODY_ENV,
        )
        return {}
    return data


def vision_chat_sync(messages: list[dict], *, max_tokens: int | None = None) -> str:
    """One JSON-mode chat call against the configured vision endpoint.

    Shared entry point for every tier that talks to the vision LLM (image
    ingestion here, per-schematic passes in ``_procedure``): the client, the
    model resolution and the strict-JSON posture (``temperature=0``,
    ``response_format=json_object``) stay defined in exactly one place.
    Returns the raw model text; callers parse with :func:`_parse_vision_json`.

    ``max_tokens`` is opt-in and left unset by default so the short-output
    image tier keeps its current behaviour. Callers that ask for a large
    structured object must set it: with no explicit cap the provider's own
    default decides, and a completion cut mid-object yields an unparseable
    reply *deterministically* — retrying the identical request then fails
    the same way, which is exactly what the procedure passes hit.
    """
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    extra = vision_extra_body()
    if extra:
        kwargs["extra_body"] = extra
    response = _get_client().chat.completions.create(
        model=os.environ[TWIN_VISION_MODEL_ENV].strip(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=messages,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def _call_vision_llm_sync(path: Path) -> str:
    """One vision chat call; returns the raw model text (JSON expected)."""
    data = path.read_bytes()
    mime = _MIME_BY_EXT.get(path.suffix.lower().lstrip("."), "image/png")
    data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    return vision_chat_sync(
        [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
    )


def _widest_embedded_object(raw: str) -> dict | None:
    """Return the widest decodable object embedded in ``raw``."""
    decoder = json.JSONDecoder()
    best: dict | None = None
    best_span = -1
    empty_fallback: dict | None = None
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            data, end = decoder.raw_decode(raw, index)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        if not data:
            if empty_fallback is None:
                empty_fallback = data
            continue
        # Widest span wins, NOT the first hit. Returning the first decodable
        # object looks right until the outer one is truncated: the scan then
        # walks into the object's own body and happily returns an inner
        # fragment — one entry of a `tasks` array — which parses cleanly and
        # fails the contract check downstream. That turns a loud "unparseable
        # reply" into a quiet "wrong shape", which is worse. The outermost
        # object always spans the most characters.
        span = end - index
        if span > best_span:
            best, best_span = data, span
    return best if best is not None else empty_fallback


def _parse_vision_json(raw: str) -> dict | None:
    """Tolerant parse of the model reply (bare JSON, fenced block, or noise).

    Real replies are not always the clean object the prompt asks for. The live
    gate observed one opening with a junk prefix before the fence::

        {";}```json\\n{\\n  "title": "Qualify and resolve the incident", ...

    The previous fallback was a single greedy ``\\{.*\\}`` search, which anchors
    on the FIRST brace — the junk one — and therefore produced an invalid span
    while the real object sat intact a few characters later. It also required a
    CLOSING fence, so a reply cut at the token cap was unrecoverable even when
    its JSON object had completed.

    Both are fixed by scanning every ``{`` and letting the decoder tell us where
    a real value starts: ``raw_decode`` parses one value and ignores whatever
    trails it, so a junk prefix, an unterminated fence and trailing prose all
    stop mattering. A non-empty object wins over an empty one — ``{}`` stays
    reachable so the downstream contract validator keeps reporting it as a
    malformed payload rather than a parse failure.
    """
    if not raw:
        return None

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (TypeError, ValueError):
        pass

    return _widest_embedded_object(raw)


def validate_vision_payload(data: object, *, stage: str = "vision") -> tuple[str, str]:
    """Validate the shared ``{image_classification, content}`` contract.

    Parsing JSON is not sufficient: coercing arrays, objects or numbers with
    ``str()`` would turn a malformed model response into trusted document
    content.  Both generic image tiers use this validator before applying
    semantic drop rules.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{stage}: reply does not match the expected shape")
    classification = data.get("image_classification")
    content = data.get("content")
    if not isinstance(classification, str) or not isinstance(content, str):
        raise ValueError(
            f"{stage}: reply must contain string image_classification and content"
        )
    # Keep both image pipelines consistent when the model returns a blank
    # classification. The content remains usable and gets an explicit,
    # stable label instead of producing ``_Image type: _`` in markdown.
    return classification.strip() or "unknown", content.strip()


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
        logger.warning(_VISION_WARNING_FORMAT, path.name, reason)
        return VisionOutcome(markdown=None, reason=reason)
    except Exception as exc:  # defensive: the seam must never crash
        reason = f"vision-error: {type(exc).__name__}: {exc}"
        logger.warning(_VISION_WARNING_FORMAT, path.name, reason)
        return VisionOutcome(markdown=None, reason=reason)


async def _aprocess_inner(path: Path) -> VisionOutcome:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return VisionOutcome(
            markdown=None,
            reason=f"vision-input-error: {type(exc).__name__}: {exc}",
        )

    size_limit = max_image_bytes()
    if size > size_limit:
        reason = (
            "vision-size-limit: image rejected because file size is "
            f"{size} bytes; configured maximum is {size_limit} bytes"
        )
        logger.warning(_VISION_WARNING_FORMAT, path.name, reason)
        return VisionOutcome(markdown=None, reason=reason)

    threshold, active_drop_classes = await _effective_settings()
    ocr_text = await asyncio.to_thread(_ocr_text_sync, path)

    if ocr_text is not None and threshold > 0 and len(ocr_text.strip()) < threshold:
        detected_chars = len(ocr_text.strip())
        return VisionOutcome(
            markdown=None,
            reason=(
                "vision-prefilter: image rejected before vision analysis; "
                f"OCR detected {detected_chars} text characters, below "
                f"configured minimum {threshold}"
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

    try:
        classification, content = validate_vision_payload(data, stage="vision-image")
    except ValueError as exc:
        return VisionOutcome(
            markdown=None,
            reason=f"vision-llm-error: {exc}",
        )

    classification_slug = classification.lower()
    if classification_slug in active_drop_classes:
        return VisionOutcome(
            markdown=None,
            reason=(
                "image-dropped: image rejected by active Vision policy; "
                f"classified as {classification_slug!r}, an excluded class"
            ),
            classification=classification,
        )
    if not content or content.lower() == "invalid":
        return VisionOutcome(
            markdown=None,
            reason=(
                "image-dropped: image rejected by active Vision policy; "
                "no informational content was detected"
            ),
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
