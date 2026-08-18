"""Procedure-PDF ingestion profile (PROCEDURE-PROFILE-PLAN.md, PR 1).

BNP "IT Group" level-2 procedures (calibrated 2026-07-20 on ITG0162 /
ITG0160) are text pages + 3-4 SIPOC swimlane schematics. The standard
pipeline extracts the text and silently loses the schematics — but the flow
IS the procedure. This profile:

1. **Detects** the template deterministically on the text of the first two
   pages (``ITG\\d{4}`` reference AND every required marker) — no LLM. The
   operator can force either way with the ``X-Twin-Doc-Type`` upload header
   (``procedure`` / ``standard``), bound by the registry middleware to
   :func:`_constants.get_active_doc_type`. Detection is calibrated on
   photographed samples, not on a real PDF text layer (the real files never
   leave BNP), so the forcing header is a first-class control, not a
   convenience.
2. **Renders** each page whose text carries the template's ``Schematic:``
   heading (pypdfium2 — Apache/BSD; PyMuPDF is AGPL, licence red line).
3. Runs the **dual vision pass per schematic** against the same
   ``gemma-4-31b-it`` endpoint as the image tier: pass A describes the
   diagram *blind* (measurement instrument — it cannot be biased toward
   false coherence), a comparator confronts A with the full unchunked
   document text (divergence report), pass B describes the diagram *with*
   the full text in context (canonical description to index).
4. **Parks an approval bundle** (``_procedure_store``) instead of enqueueing:
   a human approves before anything reaches LightRAG (PR 2 routes). Vision
   or render failures park a ``failed`` bundle — visible and retryable,
   never a silent drop.

Contract: :func:`aprocess_procedure` never raises into the ingestion flow.
It returns ``None`` when the document is not a procedure (the seam continues
on the untouched standard path) or a :class:`ProcedureOutcome` when a bundle
was parked (the seam reports success and stops — no enqueue). On server
surfaces, Settings → Vision supplies the authoritative admin-controlled
runtime activation flag (off by default unless ``TWIN_PROCEDURE=on`` seeds
the unsaved default). Direct legacy surfaces still honour
``TWIN_PROCEDURE=off``. Missing PDF dependencies or an unconfigured Vision
tier disable new procedure selection everywhere; already-parked paths remain
claimed so a rescan cannot bypass approval.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from . import _procedure_store, _vision
from ._constants import (
    _FALSE_FLAG_VALUES,
    TWIN_PROCEDURE_ENV,
    TWIN_PROCEDURE_MAX_BYTES_ENV,
    TWIN_PROCEDURE_MAX_SCHEMATICS_ENV,
    TWIN_PROCEDURE_MAX_TOKENS_ENV,
    TWIN_PROCEDURE_RENDER_SCALE_ENV,
    get_active_doc_type,
    get_active_operator_classification,
    get_active_storage_folder,
)

logger = logging.getLogger("twindb_lightrag_memgraph")

_PROCEDURE_ERROR_FORMAT = "twindb procedure: %s — %s"

_TRUE_FLAG_VALUES = frozenset({"1", "true", "yes", "on"})

DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # aligned on the conversion tier
DEFAULT_RENDER_SCALE = 2.0  # ~144 dpi — legible task boxes, bounded PNGs
DEFAULT_MAX_SCHEMATICS = 8
DETECTION_PAGES = 2

# Template markers (photographed ITG0162/ITG0160 samples). Detection is the
# CONJUNCTION of the reference regex (`ITG0162-…_Procedure`,
# `ITGC0094-PRO-CIM-…`) and every required marker — both live in the cover
# metadata table the template mandates on page 1 ("Level * Level 2",
# "Procedure type * 4- Operational procedures"). A document merely *quoting*
# an ITG code (e.g. another IT Group template family with the same footer)
# must NOT select the profile; the operator forcing header covers the
# false-negative side. The "IT GROUP / Classification : Internal" footer is
# deliberately NOT required: footer text extraction order/reliability on the
# real (never-seen) PDF text layer is unknown.
_REFERENCE_RE = re.compile(r"\bITGC?\d{4}\b")
_REQUIRED_MARKERS = (
    "level 2",
    "operational procedure",
)
_SCHEMATIC_MARKER = "schematic:"

# SIPOC grammar shared by both vision passes: the model decodes the template
# instead of guessing it. Plain strings (no .format) — JSON braces literal.
_SIPOC_CONTEXT = """\
The image is one process schematic page from a BNP Paribas "IT Group" \
level-2 procedure document. Its grammar is fixed:
- SIPOC swimlane: columns Suppliers / Inputs / Process Tasks / Outputs / \
Clients-Customers; horizontal lanes are activities (lane title on the left \
edge).
- Task boxes read "Tn.m - Task title" with the responsible function on a \
green band and the mandatory actors on a blue band.
- Red text on arrows states conditions (e.g. "If major incident").
- Lettered circles (A, B, C, D) are connectors continuing on another \
schematic of the same document.
- Green pills carry three-letter trigrams referencing OTHER procedures. \
Common trigrams: CHG=Manage IT Operational Changes, CONF=Manage \
Configurations, INC=Manage Incidents, PBM=Manage Problems, CTY=Manage IT \
continuity, RIS=Manage IT Risks, SEC=Manage Cyber Security (the document \
may print its own trigram table — trust the document over this list).
"""

_TASKS_JSON_SHAPE = """\
Respond with a single valid JSON object:
{"title": str, "description": str, "tasks": [{"id": str, "title": str, \
"responsible": str, "actors": str, "inputs": str, "outputs": str, \
"conditions": str, "links": str}]}
- "title": the schematic heading as printed (after "Schematic:").
- "description": the complete flow in prose, activity by activity, in the \
document's language.
- "tasks": one entry per task box, transcribing identifiers and labels \
faithfully; "links" lists connector letters and trigram pills touching the \
task. Use "" for fields not visible.
Treat conditions and links as independent attributes unless a drawn connector \
explicitly joins them. Never claim that a condition triggers a linked \
procedure merely because both touch the same task.
Describe ONLY what the image shows. Never invent tasks or actors.
"""

BLIND_SYSTEM_PROMPT = (
    _SIPOC_CONTEXT
    + "\nYou see ONLY the image — no document text is provided.\n\n"
    + _TASKS_JSON_SHAPE
)

INFORMED_SYSTEM_PROMPT = (
    _SIPOC_CONTEXT + "\nThe user message also carries the FULL TEXT of the procedure "
    "document. Use it to resolve names, roles and task wording, but "
    "describe the flow the DIAGRAM shows — the text explains, the diagram "
    "is the process.\n\n" + _TASKS_JSON_SHAPE
)

COMPARATOR_SYSTEM_PROMPT = """\
You audit the ingestion of a BNP Paribas level-2 procedure document. You \
receive (1) the full text of the document and (2) a description of one of \
its process schematics produced by a vision model that saw ONLY the image. \
Confront the two.

Respond with a single valid JSON object:
{"coherent": bool, "divergences": [str], "summary": str}
- "coherent": true when the blind description matches the document text on \
tasks, ordering, roles and conditions (cosmetic wording differences are \
fine).
- "divergences": one entry per concrete mismatch (task missing or invented, \
wrong responsible/actor, wrong condition, wrong sequence), quoting the task \
id when possible. Empty when coherent.
- "summary": 2-3 sentences for a human reviewer, in the document's language.
"""


@dataclass(frozen=True)
class ProcedureOutcome:
    """The profile claimed the document; the seam must NOT enqueue it.

    ``state`` mirrors the bundle state when one was parked or reused
    (``processing``/``pending``/``failed``/``approved``/``rejected`` — a
    reuse returns the existing bundle's state as-is), or ``"error"`` when
    the store itself was unusable (``bundle_id`` may then be ``None``) —
    the seam surfaces an explicit FAILED error-document. A claimed
    procedure NEVER falls back to the standard enqueue path: the approval
    gate must fail closed, not open.
    """

    bundle_id: str | None
    state: str
    reason: str


_availability_lock = threading.Lock()
_pdfium_available: bool | None = None
_pypdf_available: bool | None = None
_forced_on_warned = False

# Overlay event sink (family of the vision settings provider): PR 2 wires
# activity/notification emission here; PR 1 logs when nothing is registered.
_event_sink = None
# The Settings → Vision provider returns the workspace-scoped runtime
# activation flag. It is async because the shared setting lives in Memgraph
# and must be observed by every worker without a restart.
_settings_provider = None


def set_event_sink(sink) -> None:
    """Register ``callable(kind: str, payload: dict)`` (None to unregister)."""
    global _event_sink
    _event_sink = sink


def set_settings_provider(provider) -> None:
    """Register an async callable returning the runtime settings dict."""
    global _settings_provider
    _settings_provider = provider


def _emit(kind: str, payload: dict) -> None:
    if _event_sink is None:
        return
    try:
        _event_sink(kind, payload)
    except Exception as exc:  # the sink must never break ingestion
        logger.warning(
            "twindb procedure: event sink failed for %s (%s: %s)",
            kind,
            type(exc).__name__,
            exc,
        )


def _resolve_mode() -> bool | None:
    raw = os.environ.get(TWIN_PROCEDURE_ENV, "").strip().lower()
    if raw in _FALSE_FLAG_VALUES:
        return False
    if raw in _TRUE_FLAG_VALUES:
        return True
    return None


def _pdfium_importable() -> bool:
    global _pdfium_available
    if _pdfium_available is None:
        with _availability_lock:
            if _pdfium_available is None:
                try:
                    import pypdfium2  # noqa: F401

                    _pdfium_available = True
                except ImportError:
                    _pdfium_available = False
    return _pdfium_available


def _pypdf_importable() -> bool:
    global _pypdf_available
    if _pypdf_available is None:
        with _availability_lock:
            if _pypdf_available is None:
                try:
                    import pypdf  # noqa: F401

                    _pypdf_available = True
                except ImportError:
                    _pypdf_available = False
    return _pypdf_available


def reset_caches() -> None:
    """Test hook: forget import probes, warnings, sinks and providers."""
    global _pdfium_available, _pypdf_available, _forced_on_warned, _event_sink
    global _settings_provider
    with _availability_lock:
        _pdfium_available = None
        _pypdf_available = None
    _forced_on_warned = False
    _event_sink = None
    _settings_provider = None


def is_available() -> bool:
    """Whether the deployment has every prerequisite for the profile."""
    return _pdfium_importable() and _pypdf_importable() and _vision.is_enabled()


def is_enabled() -> bool:
    """Legacy deployment default: env mode plus profile prerequisites."""
    global _forced_on_warned
    mode = _resolve_mode()
    if mode is False:
        return False
    ready = is_available()
    if not ready and mode is True and not _forced_on_warned:
        _forced_on_warned = True
        logger.warning(
            "twindb procedure: %s=on but the profile is not usable "
            "(pypdfium2: %s, pypdf: %s, vision tier: %s) — install the "
            "[procedure] extra and configure the vision endpoint; every "
            "document follows the standard path",
            TWIN_PROCEDURE_ENV,
            _pdfium_importable(),
            _pypdf_importable(),
            _vision.is_enabled(),
        )
    return ready


def default_enabled() -> bool:
    """Admin-surface default before a runtime choice has been persisted.

    Mere infrastructure readiness must not activate procedure ingestion.
    ``TWIN_PROCEDURE=on`` remains an explicit deployment opt-in for
    migrations and non-WebUI launch surfaces.
    """
    return _resolve_mode() is True and is_available()


async def is_effectively_enabled() -> bool:
    """Resolve the admin-controlled runtime flag, then the env default.

    A stored boolean is the activation intent. Infrastructure prerequisites
    remain deployment-managed, so an enabled toggle cannot make an unusable
    profile claim documents.
    """
    if _settings_provider is not None:
        try:
            data = await _settings_provider()
            if isinstance(data, dict) and isinstance(
                data.get("procedure_enabled"), bool
            ):
                return data["procedure_enabled"] and is_available()
        except Exception as exc:
            logger.warning(
                "twindb procedure: settings provider failed (%s: %s) "
                "— new procedure ingestion disabled",
                type(exc).__name__,
                exc,
            )
            return False
        return default_enabled()
    return is_enabled()


async def _forced_selection_block_reason() -> str | None:
    """Explain why an explicit procedure request cannot run, if applicable.

    The WebUI server always installs the workspace settings provider, so its
    admin choice is authoritative (and provider failures fail closed). Direct
    library callers predate that provider: preserve their detailed forced-file
    validation unless the deployment explicitly set ``TWIN_PROCEDURE=off``.
    """
    disabled_reason = (
        "procedure-disabled: an administrator must enable procedure "
        "ingestion in Settings > Vision"
    )
    if _settings_provider is None:
        return disabled_reason if _resolve_mode() is False else None
    if not is_available():
        return (
            "procedure-unavailable: PDF extraction or Vision prerequisites "
            "are not configured; check Settings > Vision"
        )
    if not await is_effectively_enabled():
        return disabled_reason
    return None


def max_procedure_bytes() -> int:
    raw = os.environ.get(TWIN_PROCEDURE_MAX_BYTES_ENV, "").strip()
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_MAX_BYTES
    except ValueError:
        return DEFAULT_MAX_BYTES


def render_scale() -> float:
    raw = os.environ.get(TWIN_PROCEDURE_RENDER_SCALE_ENV, "").strip()
    try:
        value = float(raw)
        return value if 0.5 <= value <= 8.0 else DEFAULT_RENDER_SCALE
    except ValueError:
        return DEFAULT_RENDER_SCALE


def procedure_max_tokens() -> int:
    """Completion cap for the vision passes; env override, sane floor."""
    raw = os.environ.get(TWIN_PROCEDURE_MAX_TOKENS_ENV, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_PROCEDURE_MAX_TOKENS
    # A cap below the floor can only truncate a legitimate reply.
    return value if value >= 2048 else DEFAULT_PROCEDURE_MAX_TOKENS


def max_schematics() -> int:
    raw = os.environ.get(TWIN_PROCEDURE_MAX_SCHEMATICS_ENV, "").strip()
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_MAX_SCHEMATICS
    except ValueError:
        return DEFAULT_MAX_SCHEMATICS


def should_consider(file_path: Path | str) -> bool:
    """Cheap sync gate deciding whether the profile examines a file.

    A **forced** document (``X-Twin-Doc-Type: procedure``) is always
    claimed when the tier is enabled: format/size problems must produce an
    explicit failed bundle inside :func:`aprocess_procedure`, never a
    silent fall-through to the standard (unapproved) enqueue. Auto mode
    applies the cheap gates (PDF extension, size cap) before paying any IO.
    """
    if not is_enabled():
        return False
    return _should_consider_while_enabled(file_path)


def _should_consider_while_enabled(file_path: Path | str) -> bool:
    """Cheap file gate after activation has already been resolved."""
    doc_type = get_active_doc_type()
    if doc_type == "standard":
        return False
    if doc_type == "procedure":
        return True
    path = Path(file_path)
    if path.suffix.lower() != ".pdf":
        return False
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size > max_procedure_bytes():
        logger.warning(
            "twindb procedure: %s exceeds %s (%d bytes) — standard path",
            path.name,
            TWIN_PROCEDURE_MAX_BYTES_ENV,
            size,
        )
        return False
    return True


def detect_procedure(first_pages_text: str) -> bool:
    """Deterministic template detection: reference AND every required marker."""
    text = (first_pages_text or "").lower()
    if not _REFERENCE_RE.search(first_pages_text or ""):
        return False
    return all(marker in text for marker in _REQUIRED_MARKERS)


def _extract_pages_text_sync(path: Path, limit: int | None = None) -> list[str] | None:
    """Per-page text via pypdf; ``None`` on any failure (logged).

    ``limit`` bounds the pages read — detection only needs the first two, so
    a standard (non-procedure) PDF never pays a full-document extraction on
    this path before falling through to the native pipeline.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = reader.pages if limit is None else reader.pages[:limit]
        return [(page.extract_text() or "") for page in pages]
    except Exception as exc:
        logger.warning(
            "twindb procedure: text extraction failed for %s (%s: %s)",
            path.name,
            type(exc).__name__,
            exc,
        )
        return None


def _content_hash_sync(path: Path) -> str:
    """SHA-256 of the file bytes — the bundle's stable identity (idempotence:
    a rescan of the still-parked original must reuse the active bundle, not
    re-burn render + LLM calls into a duplicate)."""
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_schematic_pages(pages: list[str]) -> list[int]:
    """0-based indexes of pages carrying the template's Schematic: heading."""
    return [i for i, text in enumerate(pages) if _SCHEMATIC_MARKER in text.lower()]


def _capped_render_scale(page_width: float, page_height: float) -> float:
    """Render scale bounded so the bitmap stays under MAX_RENDER_PIXELS.

    Audit 2026-08-06, R-08b: the MediaBox geometry is fully attacker-
    controlled, and the procedure profile used to render at the raw
    configured scale — a crafted 10000x10000pt page at scale 8 is a
    ~25 GB bitmap. Shares the cap with the standard PDF profile
    (``_pdf_vision.MAX_RENDER_PIXELS``); the degrading loop mirrors the
    region-render guard there. Raises ValueError when the page cannot be
    rendered within the cap.
    """
    import math

    from ._pdf_vision import MAX_RENDER_PIXELS

    page_area = page_width * page_height
    if not math.isfinite(page_area) or page_area <= 0:
        raise ValueError("invalid procedure page geometry")
    scale = min(render_scale(), math.sqrt(MAX_RENDER_PIXELS / page_area))
    for _attempt in range(4):
        pixel_width = max(1, math.ceil(page_width * scale))
        pixel_height = max(1, math.ceil(page_height * scale))
        rounded_pixels = pixel_width * pixel_height
        if rounded_pixels <= MAX_RENDER_PIXELS:
            break
        scale *= math.sqrt(MAX_RENDER_PIXELS / rounded_pixels)
    else:
        raise ValueError("procedure page cannot be rendered within pixel cap")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("procedure page requires an invalid render scale")
    return scale


def _render_page_png_sync(path: Path, page_index: int) -> bytes:
    """Render one page to PNG bytes (pypdfium2 + Pillow)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        page = pdf[page_index]
        width, height = page.get_size()
        bitmap = page.render(scale=_capped_render_scale(width, height))
        image = bitmap.to_pil()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        pdf.close()


def _png_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


#: Default completion cap for the procedure passes, overridable with
#: ``TWIN_PROCEDURE_MAX_TOKENS``.
#:
#: Sized for the WORST realistic schematic, not the average, because hitting
#: the cap is not a graceful degradation: a JSON object cut mid-structure is
#: unparseable, so the whole schematic is lost and the retry re-fails
#: identically. Runaway protection is the transport timeout's job
#: (``TWIN_VISION_TIMEOUT``), not this number's.
#:
#: History: shipped at 4096 on 2026-07-25 and that value CAUSED a failure the
#: same day — the informed pass returned 15130 chars, exactly the cap, cut mid
#: object with an unterminated fence. A three-box fixture schematic produced
#: that; a real BNP procedure carries far more task boxes.
DEFAULT_PROCEDURE_MAX_TOKENS = 16384

#: How much of an unparseable reply to keep in the error message.
_REPLY_EXCERPT_CHARS = 400


async def _vision_json_call(messages: list[dict], stage: str, validate=None) -> dict:
    """JSON vision call retried once when the model reply is unparseable.

    ``validate`` is applied INSIDE the retry loop, so a reply that parses but
    violates the contract is retried like an unparseable one. Both are the same
    transient endpoint noise: measured 2026-07-25, the model returned a
    corrupted key (``{@title`` instead of ``title``) on an otherwise perfect
    object. Validating only after the loop made one mangled character lose the
    whole schematic permanently — on BNP's vLLM as much as on the CI gateway.

    When every attempt fails the raw reply is carried into the error instead of
    being discarded: "unparseable JSON reply" with no sample is undiagnosable
    from a CI log, and that is precisely how the first occurrence was lost.
    """
    last_raw = ""
    last_error: Exception | None = None
    for _ in range(2):
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                _vision.vision_chat_sync, messages, max_tokens=procedure_max_tokens()
            ),
            timeout=_vision.vision_timeout_seconds(),
        )
        data = _vision._parse_vision_json(raw)
        if data is not None:
            if validate is None:
                return data
            try:
                return validate(data)
            except ValueError as exc:
                last_error = exc
        last_raw = raw or ""
    if last_error is not None:
        raise last_error
    excerpt = last_raw[:_REPLY_EXCERPT_CHARS].replace("\n", " ")
    suffix = "…" if len(last_raw) > _REPLY_EXCERPT_CHARS else ""
    logger.warning(
        "twindb procedure: %s got an unparseable reply (%d chars): %s%s",
        stage,
        len(last_raw),
        excerpt,
        suffix,
    )
    raise ValueError(
        f"{stage}: unparseable JSON reply ({len(last_raw)} chars): {excerpt}{suffix}"
    )


#: The eight string fields every task entry must carry (prompt contract).
_TASK_FIELDS = (
    "id",
    "title",
    "responsible",
    "actors",
    "inputs",
    "outputs",
    "conditions",
    "links",
)


def _validate_pass_payload(data: dict, stage: str) -> dict:
    """Shape-check a blind/informed reply; raises ``ValueError`` on mismatch.

    Downstream consumers (bundle store, review UI, PR 2 markdown composition)
    rely on these fields existing with these types — an LLM reply that
    "parsed as JSON" is not yet a usable description. Tasks are validated
    down to the eight string fields the prompt contract announces.
    """
    title = data.get("title")
    description = data.get("description")
    tasks = data.get("tasks")
    if (
        not isinstance(title, str)
        or not isinstance(description, str)
        or not description.strip()
        or not isinstance(tasks, list)
    ):
        # Name what actually arrived. "does not match the expected shape" with
        # no sample is the same dead end "unparseable JSON reply" was: it cost
        # a paid CI run to learn nothing. Keys and types are enough to tell a
        # truncated reply from a renamed field from an inner fragment, and
        # they cannot leak much of the document.
        got = ", ".join(
            f"{key}:{type(value).__name__}" for key, value in list(data.items())[:12]
        )
        raise ValueError(
            f"{stage}: reply does not match the expected shape "
            f"(got {len(data)} key(s): {got or 'none'})"
        )
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or any(
            not isinstance(task.get(field), str) for field in _TASK_FIELDS
        ):
            raise ValueError(
                f"{stage}: task #{index} does not carry the eight string "
                "fields of the contract"
            )
    return {"title": title, "description": description, "tasks": tasks}


def _validate_comparator_payload(data: dict) -> dict:
    coherent = data.get("coherent")
    divergences = data.get("divergences")
    summary = data.get("summary")
    if (
        not isinstance(coherent, bool)
        or not isinstance(divergences, list)
        or any(not isinstance(d, str) for d in divergences)
        or not isinstance(summary, str)
    ):
        raise ValueError("comparator: reply does not match the expected shape")
    return {"coherent": coherent, "divergences": divergences, "summary": summary}


async def _process_schematic(
    path: Path, page_index: int, full_text: str
) -> tuple[dict, str | None]:
    """Render + dual pass + comparator for one page.

    Returns ``(schematic_entry, error)`` — on error the entry keeps whatever
    stage succeeded (the review UI shows partial results with the reason).
    """
    entry: dict = {
        "page": page_index + 1,
        "png_base64": None,
        "blind": None,
        "informed": None,
        "divergence": None,
        "error": None,
    }
    try:
        png = await asyncio.to_thread(_render_page_png_sync, path, page_index)
        entry["png_base64"] = base64.b64encode(png).decode("ascii")
        data_url = _png_data_url(png)

        image_part = {"type": "image_url", "image_url": {"url": data_url}}
        blind_task = _vision_json_call(
            [
                {"role": "system", "content": BLIND_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this process schematic."},
                        image_part,
                    ],
                },
            ],
            "blind-pass",
            functools.partial(_validate_pass_payload, stage="blind-pass"),
        )
        informed_task = _vision_json_call(
            [
                {"role": "system", "content": INFORMED_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Full document text:\n\n"
                                + full_text
                                + "\n\nDescribe this process schematic."
                            ),
                        },
                        image_part,
                    ],
                },
            ],
            "informed-pass",
            functools.partial(_validate_pass_payload, stage="informed-pass"),
        )
        # The two passes are independent by design (the blind one must not
        # see anything the informed one received) — run them concurrently.
        # ``return_exceptions`` so one pass failing never discards the
        # other's successful result (the review/retry UI shows partials).
        blind_res, informed_res = await asyncio.gather(
            blind_task, informed_task, return_exceptions=True
        )
        pass_errors: list[BaseException] = []
        for field, result, stage in (
            ("blind", blind_res, "blind-pass"),
            ("informed", informed_res, "informed-pass"),
        ):
            if isinstance(result, BaseException):
                pass_errors.append(result)
                continue
            try:
                entry[field] = result
            except ValueError as invalid:
                pass_errors.append(invalid)

        # The comparator depends ONLY on the blind pass — run it and keep
        # its report even when the informed pass failed: the divergence
        # report is exactly what the reviewer needs to judge the retry.
        if entry["blind"] is not None:
            try:
                entry["divergence"] = await _vision_json_call(
                    [
                        {"role": "system", "content": COMPARATOR_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Full document text:\n\n" + full_text + "\n\n"
                                "Blind schematic description (JSON):\n\n"
                                + json.dumps(entry["blind"], ensure_ascii=False)
                            ),
                        },
                    ],
                    "comparator",
                    _validate_comparator_payload,
                )
            except Exception as comparator_exc:
                pass_errors.append(comparator_exc)

        if pass_errors:
            raise pass_errors[0]
    except Exception as exc:  # uniform failure funnel: entry stays partial
        reason = (
            f"schematic-timeout after {_vision.vision_timeout_seconds():.0f}s"
            if isinstance(exc, asyncio.TimeoutError)
            else f"{type(exc).__name__}: {exc}"
        )
        entry["error"] = f"page {page_index + 1}: {reason}"
        return entry, entry["error"]
    return entry, None


def _advisory_classification(path: Path) -> dict | None:
    """Best-effort MIP probe for the review UI — NOT an enforcement point.

    The enqueue-time gate stays the single enforcement point (ING-1); this
    only lets the reviewer see the detected class before approving.
    """
    try:
        from .classification import detect_classification

        result = detect_classification(path)
        return {
            "class_id": result.class_id,
            "class_name": result.class_name,
            "reason": getattr(result, "reason", None),
        }
    except Exception as exc:
        logger.warning(
            "twindb procedure: advisory classification failed for %s (%s: %s)",
            path.name,
            type(exc).__name__,
            exc,
        )
        return None


async def aroute_check(file_path: Path | str) -> bool:
    """Seam-side routing decision, including the parked-bundle rescan guard.

    The runtime activation flag controls NEW procedure selection. A file
    whose bundle already exists remains claimed even after an admin turns
    ingestion off; otherwise a rescan could silently enqueue an unapproved
    procedure through the standard pipeline.

    The cheap file gate alone is not enough: a file whose bundle already
    exists may fail the cheap auto gates on a later scan (an oversized or
    non-PDF *forced* document rescanned without its header, an
    auto-undetectable forced document) and would silently fall through to
    the standard — unapproved — enqueue. An explicit
    ``X-Twin-Doc-Type: standard`` remains the only operator override.
    """
    path = Path(file_path)
    effectively_enabled = await is_effectively_enabled()
    if effectively_enabled and _should_consider_while_enabled(path):
        return True
    doc_type = get_active_doc_type()
    if doc_type == "standard":
        return False
    # An explicit procedure request must never bypass approval merely because
    # the profile is disabled or unavailable. Route it to the profile, which
    # parks an actionable failed bundle without running extraction.
    if doc_type == "procedure":
        return True
    try:
        if str(path) in await asyncio.to_thread(_procedure_store.claimed_paths):
            return True
        # Reading may have just quarantined a corrupt store — and a
        # degraded store means the claim index is LOST: every file routes
        # to the profile, which refuses the enqueue until an operator
        # explicitly recovers the .corrupt-* files.
        return await asyncio.to_thread(_procedure_store.is_degraded)
    except Exception:
        logger.exception(
            "twindb procedure: rescan guard cannot read the store for %s "
            "— failing CLOSED, the file routes to the profile",
            path.name,
        )
        return True


def _find_existing_for_path_sync(path: Path) -> dict | None:
    """Pre-selection rescan guard body (worker thread).

    Matches a bundle that already claimed this path when its recorded
    content hash equals the file's current hash — or when the bundle has no
    hash (parked by a validation/fail-closed branch): reuse conservatively,
    the gate must stay closed. A path whose content CHANGED since the
    bundle (corrected re-upload) does not match — the new content goes
    through the normal flow (and its own hash-keyed reservation).
    """
    candidates = _procedure_store.find_bundles_by_path(str(path))
    if not candidates:
        return None
    try:
        file_hash = _content_hash_sync(path)
    except OSError:
        file_hash = None
    for bundle in candidates:  # newest first
        bundle_hash = bundle.get("content_hash")
        if bundle_hash is None or (file_hash is not None and bundle_hash == file_hash):
            return bundle
    return None


async def _record_duplicate_request(
    bundle: dict, path: Path, track_id: str | None
) -> None:
    """Persist the reusing request's context (folder, track, classification).

    A pre-guard reuse can come from another folder than the bundle's (same
    file name uploaded into folder B): PR 2 needs that folder to apply
    membership at approve time and the operator classification to keep the
    strictest gate. FAIL-CLOSED: raises when the request cannot be
    persisted (store down, bundle vanished) — silently dropping the folder
    or a stricter classification is a policy loss, not a cosmetic one.
    """
    await asyncio.to_thread(
        functools.partial(
            _procedure_store.record_request,
            bundle["id"],
            path=str(path),
            folder=get_active_storage_folder(),
            track_id=track_id,
            operator_classification=get_active_operator_classification(),
            file_name=path.name,
        )
    )


def _reuse_outcome(bundle: dict, path: Path) -> ProcedureOutcome:
    logger.info(
        "twindb procedure: %s already claimed by bundle %s (state=%s) — "
        "reusing, no reprocessing",
        path.name,
        bundle["id"],
        bundle["state"],
    )
    return ProcedureOutcome(
        bundle_id=bundle["id"],
        state=bundle["state"],
        reason=f"already-parked: {bundle.get('reason', '')}",
    )


async def _guard_store_and_find_existing(
    path: Path,
) -> dict | ProcedureOutcome | None:
    degraded_reason = (
        "store-degraded: the bundle claim index was quarantined — refusing "
        "every enqueue until the .corrupt-* files next to the store are "
        "explicitly recovered and removed"
    )
    try:
        if await asyncio.to_thread(_procedure_store.is_degraded):
            logger.error(_PROCEDURE_ERROR_FORMAT, path.name, degraded_reason)
            return ProcedureOutcome(
                bundle_id=None, state="error", reason=degraded_reason
            )
        return await asyncio.to_thread(_find_existing_for_path_sync, path)
    except _procedure_store.StoreDegradedError:
        logger.error(_PROCEDURE_ERROR_FORMAT, path.name, degraded_reason)
        return ProcedureOutcome(bundle_id=None, state="error", reason=degraded_reason)
    except Exception as exc:
        reason = (
            f"store-unreadable: {type(exc).__name__}: {exc} — refusing the "
            "enqueue (the claim index cannot be consulted)"
        )
        logger.error(_PROCEDURE_ERROR_FORMAT, path.name, reason)
        return ProcedureOutcome(bundle_id=None, state="error", reason=reason)


async def _reuse_existing_request(
    existing: dict,
    path: Path,
    track_id: str | None,
    *,
    from_scan: bool,
) -> ProcedureOutcome:
    if from_scan:
        return _reuse_outcome(existing, path)
    try:
        await _record_duplicate_request(existing, path, track_id)
    except Exception as exc:
        reason = (
            f"duplicate-request-persist-failed: {type(exc).__name__}: "
            f"{exc} — refusing the enqueue"
        )
        logger.error(_PROCEDURE_ERROR_FORMAT, path.name, reason)
        return ProcedureOutcome(
            bundle_id=existing.get("id"), state="error", reason=reason
        )
    return _reuse_outcome(existing, path)


async def _auto_detected_procedure(path: Path) -> bool:
    try:
        head_pages = await asyncio.to_thread(
            _extract_pages_text_sync, path, DETECTION_PAGES
        )
    except Exception as exc:
        logger.warning(
            "twindb procedure: detection probe failed for %s (%s: %s) — "
            "standard path",
            path.name,
            type(exc).__name__,
            exc,
        )
        return False
    return head_pages is not None and detect_procedure("\n".join(head_pages))


def _forced_validation_problem(path: Path) -> str | None:
    if path.suffix.lower() != ".pdf":
        return (
            "unsupported-extension: the procedure profile handles PDF "
            "only — reroute as a standard document"
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"unreadable-file: {exc}"
    if size > max_procedure_bytes():
        return f"file-too-large: {size} bytes exceeds {TWIN_PROCEDURE_MAX_BYTES_ENV}"
    return None


async def aprocess_procedure(
    file_path: Path | str, track_id: str | None, *, from_scan: bool = False
) -> ProcedureOutcome | None:
    """Full profile; never raises. ``None`` => not a procedure, standard path.

    ``from_scan`` marks the global /documents/scan surface: a rescan is not
    an operator ingestion request, so a reuse never records the scan's
    captured folder as a duplicate request (that would silently grant a
    future membership in whatever folder the scan happened to run under).

    Phases, each with its own failure semantics:

    - **Rescan guard** (path + content hash, ANY state, ANY folder): a file
      whose bundle already exists is reused as-is — ``rejected`` included,
      it is terminal until the PR 2 retry action.
    - **Selection** (auto-detect on the first two pages): an error here in
      auto mode means "cannot claim the document" -> ``None``, standard
      path. ``None`` is reserved for the certain not-a-procedure verdict.
    - **Forced validation**: an explicitly declared procedure that the
      profile cannot handle (non-PDF, oversized) parks an explicit failed
      bundle — never a silent standard enqueue.
    - **Processing** (atomic hash-keyed reservation, then vision): fail
      closed — unexpected errors settle the reservation to ``failed``; if
      even the store is unusable, an ``"error"`` outcome makes the seam
      surface a FAILED error-document.
    """
    path = Path(file_path)
    forced = get_active_doc_type() == "procedure"

    store_result = await _guard_store_and_find_existing(path)
    if isinstance(store_result, ProcedureOutcome):
        return store_result
    if store_result is not None:
        return await _reuse_existing_request(
            store_result, path, track_id, from_scan=from_scan
        )

    forced_block_reason = await _forced_selection_block_reason() if forced else None
    if forced_block_reason is not None:
        return await _fail_closed_park(
            path,
            track_id,
            "forced",
            forced_block_reason,
        )

    # --- Selection phase -------------------------------------------------
    if not forced and not await _auto_detected_procedure(path):
        return None

    source = "forced" if forced else "detected"

    # --- Forced-document validation (fail closed, no silent downgrade) ---
    if forced:
        problem = _forced_validation_problem(path)
        if problem is not None:
            return await _fail_closed_park(path, track_id, source, problem)

    # --- Processing phase (fail closed from here on) ---------------------
    try:
        return await _aprocess_selected(path, track_id, source, from_scan)
    except Exception as exc:
        reason = f"procedure-error: {type(exc).__name__}: {exc}"
        logger.exception(
            "twindb procedure: %s — unexpected processing error", path.name
        )
        return await _fail_closed_park(path, track_id, source, reason)


async def _fail_closed_park(
    path: Path, track_id: str | None, source: str, reason: str
) -> ProcedureOutcome:
    """Park a plain failed bundle; store down -> ``error`` (seam refuses)."""
    try:
        return await _park(
            path,
            track_id,
            state="failed",
            reason=reason,
            source=source,
            content_hash=None,
            full_text="",
            schematics=[],
            schematics_total=0,
            classification=None,
        )
    except Exception:
        logger.exception(
            "twindb procedure: %s — could not even park a failed bundle "
            "; refusing the enqueue",
            path.name,
        )
        return ProcedureOutcome(bundle_id=None, state="error", reason=reason)


async def _aprocess_selected(
    path: Path, track_id: str | None, source: str, from_scan: bool
) -> ProcedureOutcome:
    """Process a claimed document through an atomic reservation.

    ``reserve_bundle`` is a get-or-create in ONE store transaction keyed on
    the content hash: two workers racing on the same document cannot both
    spend vision calls — exactly one owns the ``processing`` reservation,
    the other reuses it (same-content-other-path is recorded as a duplicate
    request on the existing bundle — with no folder when it came from a
    scan, which is not an operator ingestion request).
    """
    content_hash = await asyncio.to_thread(_content_hash_sync, path)
    reserved, created = await asyncio.to_thread(
        _reserve_sync, path, track_id, source, content_hash, from_scan
    )
    if not created:
        return _reuse_outcome(reserved, path)

    bundle_id = reserved["id"]
    try:
        fields = await _run_profile(path)
    except Exception as exc:
        reason = f"procedure-error: {type(exc).__name__}: {exc}"
        logger.exception(
            "twindb procedure: %s — unexpected processing error", path.name
        )
        return await _settle(bundle_id, path, source, state="failed", reason=reason)
    return await _settle(bundle_id, path, source, **fields)


def _reserve_sync(
    path: Path,
    track_id: str | None,
    source: str,
    content_hash: str,
    from_scan: bool,
) -> tuple[dict, bool]:
    return _procedure_store.reserve_bundle(
        content_hash=content_hash,
        file_name=path.name,
        original_path=str(path),
        track_id=track_id,
        source=source,
        folder=get_active_storage_folder(),
        # The operator's X-Twin-Classification choice dies with the request
        # context — persist it so the PR 2 approve-time enqueue can rebind
        # it and the MIP gate reproduces the upload-time policy exactly.
        operator_classification=get_active_operator_classification(),
        via_scan=from_scan,
    )


async def _run_profile(path: Path) -> dict:
    """Vision phase: returns the settle fields for the owned reservation."""
    pages = await asyncio.to_thread(_extract_pages_text_sync, path)
    if pages is None:
        return {
            "state": "failed",
            "reason": "text-extraction-failed: cannot read the PDF text layer",
            "classification": await asyncio.to_thread(_advisory_classification, path),
        }

    full_text = "\n\n".join(pages)
    found_pages = find_schematic_pages(pages)
    total = len(found_pages)
    cap = max_schematics()
    truncated = total > cap
    selected_pages = found_pages[:cap]

    schematics: list[dict] = []
    problems: list[str] = []
    for page_index in selected_pages:
        entry, error = await _process_schematic(path, page_index, full_text)
        schematics.append(entry)
        if error:
            problems.append(error)

    if not found_pages:
        # A procedure without a located schematic is exactly the silent-loss
        # case this profile exists to prevent — never approvable as-is. The
        # operator can retry (after fixing) or reroute standard (PR 2).
        problems.append(
            "no-schematic-found: the template's Schematic pages were not "
            "located — retry, or reroute as a standard document"
        )
    if truncated:
        # Same family: an incomplete bundle must never look approvable.
        problems.append(
            f"schematics-truncated: {total} schematic pages found, cap "
            f"{cap} — raise {TWIN_PROCEDURE_MAX_SCHEMATICS_ENV} and retry"
        )
        logger.warning(
            "twindb procedure: %s has %d schematic pages, cap %d (%s)",
            path.name,
            total,
            cap,
            TWIN_PROCEDURE_MAX_SCHEMATICS_ENV,
        )

    return {
        "state": "failed" if problems else "pending",
        "reason": "; ".join(problems) if problems else "ok",
        "full_text": full_text,
        "schematics": schematics,
        "schematics_total": total,
        "classification": await asyncio.to_thread(_advisory_classification, path),
    }


async def _settle(
    bundle_id: str,
    path: Path,
    source: str,
    *,
    state: str,
    reason: str,
    **fields,
) -> ProcedureOutcome:
    """Finalize an owned reservation; store down -> ``error`` (seam refuses)."""
    try:
        settled = await asyncio.to_thread(
            functools.partial(
                _procedure_store.update_bundle,
                bundle_id,
                state=state,
                reason=reason,
                **fields,
            )
        )
    except Exception as exc:
        persist_reason = (
            f"settle-persist-failed: {type(exc).__name__}: {exc} — "
            "refusing the enqueue"
        )
        logger.error(
            "twindb procedure: %s — could not settle bundle %s: %s",
            path.name,
            bundle_id,
            persist_reason,
        )
        return ProcedureOutcome(
            bundle_id=bundle_id, state="error", reason=persist_reason
        )
    if settled is None:
        # The reservation vanished mid-flight (concurrent delete, store
        # swap): nothing persisted this document's claim — announcing it
        # as parked would be a lie the approval gate cannot afford.
        vanished = (
            f"settle-lost: bundle {bundle_id} disappeared before its "
            "results could be persisted — refusing the enqueue"
        )
        logger.error(_PROCEDURE_ERROR_FORMAT, path.name, vanished)
        return ProcedureOutcome(bundle_id=bundle_id, state="error", reason=vanished)
    schematics = fields.get("schematics") or []
    logger.info(
        "twindb procedure: %s parked for approval (bundle %s, state=%s, "
        "%d schematic(s), %s)",
        path.name,
        bundle_id,
        state,
        len(schematics),
        reason,
    )
    _emit(
        "procedure-parked" if state == "pending" else "procedure-failed",
        {
            "bundle_id": bundle_id,
            "file_name": path.name,
            "state": state,
            "reason": reason,
            "source": source,
            "schematics": len(schematics),
        },
    )
    return ProcedureOutcome(bundle_id=bundle_id, state=state, reason=reason)


async def _park(
    path: Path,
    track_id: str | None,
    *,
    state: str,
    reason: str,
    source: str,
    content_hash: str | None,
    full_text: str,
    schematics: list[dict],
    schematics_total: int,
    classification: dict | None,
) -> ProcedureOutcome:
    """Direct bundle creation for pre-reservation failures (store IO in a
    worker thread — JSON with base64 PNGs must not block the event loop)."""
    bundle_id = await asyncio.to_thread(
        functools.partial(
            _procedure_store.create_bundle,
            file_name=path.name,
            original_path=str(path),
            track_id=track_id,
            state=state,
            reason=reason,
            source=source,
            folder=get_active_storage_folder(),
            content_hash=content_hash,
            full_text=full_text,
            schematics=schematics,
            schematics_total=schematics_total,
            classification=classification,
            operator_classification=get_active_operator_classification(),
        )
    )
    logger.info(
        "twindb procedure: %s parked for approval (bundle %s, state=%s, "
        "%d schematic(s), %s)",
        path.name,
        bundle_id,
        state,
        len(schematics),
        reason,
    )
    _emit(
        "procedure-parked" if state == "pending" else "procedure-failed",
        {
            "bundle_id": bundle_id,
            "file_name": path.name,
            "state": state,
            "reason": reason,
            "source": source,
            "schematics": len(schematics),
        },
    )
    return ProcedureOutcome(bundle_id=bundle_id, state=state, reason=reason)


# ---------------------------------------------------------------------------
# Approval-workflow helpers (PR 2 routes call these; FastAPI-free)
# ---------------------------------------------------------------------------


def bundle_folders(bundle: dict) -> list[str]:
    """Every folder that requested this document, primary first, deduped.

    ``None`` folders (scan-created bundles / scan-discovered duplicates) are
    skipped: a scan is not an operator ingestion request and grants no
    membership.
    """
    folders: list[str] = []
    primary = bundle.get("folder")
    if primary:
        folders.append(str(primary))
    for request in bundle.get("duplicate_requests") or []:
        if not isinstance(request, dict):
            continue
        folder = request.get("folder")
        if folder and str(folder) not in folders:
            folders.append(str(folder))
    return folders


def strictest_operator_classification(bundle: dict) -> str | None:
    """Fold the primary + duplicate-request operator classes to the max."""
    strictest = bundle.get("operator_classification")
    for request in bundle.get("duplicate_requests") or []:
        if isinstance(request, dict):
            strictest = _procedure_store._stricter_classification(
                request.get("operator_classification"), strictest
            )
    return strictest


_TASK_DETAIL_FIELDS = (
    "responsible",
    "actors",
    "inputs",
    "outputs",
    "conditions",
    "links",
)


def _task_markdown(task: dict) -> str:
    line = f"- {task.get('id') or '?'} — {task.get('title') or ''}".rstrip(" —")
    details = "; ".join(
        f"{field}: {task[field]}" for field in _TASK_DETAIL_FIELDS if task.get(field)
    )
    return f"{line} ({details})" if details else line


def _schematic_markdown(entry: dict) -> list[str]:
    informed = entry["informed"]
    title = str(informed.get("title") or "").strip()
    heading = f"## Schematic (page {entry.get('page')})"
    if title:
        heading += f": {title}"
    tasks = [task for task in informed.get("tasks") or [] if isinstance(task, dict)]
    return [
        heading,
        str(informed.get("description") or "").strip(),
        *(_task_markdown(task) for task in tasks),
    ]


def compose_approved_markdown(bundle: dict) -> str:
    """The markdown an approved procedure enqueues under its ORIGINAL name.

    Full unchunked text first, then one section per schematic carrying the
    INFORMED (canonical) description and its task inventory. The blind pass
    and the divergence report are review instruments — they never index.
    """
    parts = [str(bundle.get("full_text") or "").strip()]
    described = [
        entry
        for entry in bundle.get("schematics") or []
        if isinstance(entry, dict) and isinstance(entry.get("informed"), dict)
    ]
    if described:
        parts.append("---")
        parts.append("# Process schematics (vision descriptions)")
        for entry in described:
            parts.extend(_schematic_markdown(entry))
    return "\n\n".join(part for part in parts if part)


RETRYABLE_STATES = ("failed", "rejected")


async def aretry_bundle(bundle_id: str) -> ProcedureOutcome | None:
    """Re-run the vision profile on a failed/rejected bundle (PR 2 action).

    The ONLY relaunch path — rescans never resurrect a bundle. Takes the
    optimistic lock by transitioning to ``processing`` first; returns
    ``None`` when the bundle is missing or not retryable (route -> 409).
    Never raises: failures settle the bundle back to ``failed``.
    """
    try:
        reserved = await asyncio.to_thread(
            functools.partial(
                _procedure_store.transition_bundle,
                bundle_id,
                RETRYABLE_STATES,
                state="processing",
                reason="processing (retry)",
            )
        )
    except Exception as exc:
        reason = f"retry-error: {type(exc).__name__}: {exc}"
        logger.error("twindb procedure: retry of %s — %s", bundle_id, reason)
        return ProcedureOutcome(bundle_id=bundle_id, state="error", reason=reason)
    if reserved is None:
        return None

    path = Path(str(reserved.get("original_path") or ""))
    source = str(reserved.get("source") or "detected")
    if not path.is_file():
        return await _settle(
            bundle_id,
            path,
            source,
            state="failed",
            reason=(
                "original-missing: the source file left the input directory "
                "— re-upload the document"
            ),
        )
    try:
        fields = await _run_profile(path)
    except Exception as exc:
        reason = f"procedure-error: {type(exc).__name__}: {exc}"
        logger.exception("twindb procedure: %s — unexpected retry error", path.name)
        return await _settle(bundle_id, path, source, state="failed", reason=reason)
    return await _settle(bundle_id, path, source, **fields)
