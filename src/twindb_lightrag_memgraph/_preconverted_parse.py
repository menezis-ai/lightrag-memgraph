"""Preconverted-markdown native parsing seam (PARAGRAPH-CITATION-PLAN §6, B1).

Twin's ingestion tiers (`_conversion`, `_pdf_vision`, `_procedure`) hold the
final markdown in memory while the ORIGINAL binary must remain the identity,
MIP source, dedup key and citation name. On the 1.4.x line that markdown is
enqueued ``raw`` — LightRAG runs no parser, so no block provenance exists.
LightRAG 1.5.x adds the seam qualified by B0.1 (GO, see
``docs/audits/paragraph-citation/b01-qualification-2026-07-28.md``):
``apipeline_enqueue_documents(markdown, file_paths=<original name>,
docs_format="pending_parse", parse_engine="twinmarkdown")`` parses the
in-memory body through the native markdown parser while LightRAG still
resolves, validates and archives the original binary.

This module owns everything 1.5.x-only about that seam:

- capability detection (:func:`is_available`) — import-guarded and
  signature-checked, cached, with the ``TWIN_PRECONVERTED_PARSE=off``
  kill-switch. On the 1.4.9.11 BNP pin every probe fails closed and the
  ``raw`` enqueue path stays byte-identical (compat doctrine);
- the :class:`TwinPreconvertedMarkdownParser` adapter (engine
  ``twinmarkdown`` — an atom without dashes: 1.5.4 engine-name
  normalization truncates at the first dash);
- :func:`derive_block_boundaries` — the PURE intersection math that turns
  block spans + a chunk's ``_source_span`` into minimal, body-free
  ``twin_block_boundaries`` (``{block_id, start, end}``, code points,
  half-open, chunk-local). Unit-testable on every matrix version;
- :func:`install_backfill_boundaries` — wraps LightRAG's
  ``backfill_chunk_sidecars`` so boundaries are derived at the ONE moment
  they can be (before ``build_chunks_dict_from_chunking_result`` strips
  ``_source_span`` and cleanup deletes ``blocks.jsonl``). Fail-soft per
  chunk: a derivation problem yields no boundaries for that chunk, never a
  broken ingestion.

Compliance invariant (plan §8): boundaries only, NEVER block bodies — the
persisted chunk gains offsets, not content.
"""

from __future__ import annotations

import inspect
import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from ._constants import TWIN_PRECONVERTED_PARSE_ENV

logger = logging.getLogger("twindb_lightrag_memgraph.preconverted")

PARSER_ENGINE = "twinmarkdown"

#: Every suffix Twin's tiers can feed through the converted-markdown enqueue:
#: the MarkItDown conversion set, the PDF paths (vision + procedure) and the
#: standalone-image vision tier (its markdown rides the same seam).
PARSER_SUFFIXES = frozenset(
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
        "png",
        "jpg",
        "jpeg",
        "webp",
        "gif",
        "bmp",
        "tiff",
    }
)

_markdown_body: ContextVar[str | None] = ContextVar(
    "twin_preconverted_markdown_body", default=None
)

_capability_cache: bool | None = None
_registered = False
_backfill_installed = False


try:  # pragma: no cover - exercised only on the 1.5.x canary line
    from lightrag.parser.base import ParseContext, ParseResult
    from lightrag.parser.markdown.parser import NativeMarkdownParser

    class TwinPreconvertedMarkdownParser(NativeMarkdownParser):
        """Native markdown parsing over the in-memory preconverted body.

        The B0.1-qualified adapter, verbatim in behavior: LightRAG resolves
        and validates the ORIGINAL binary (identity, MIP, archive), while
        extraction runs on the markdown carried by the ``pending_parse``
        record — no staging ``.md`` file ever exists.
        """

        engine_name = PARSER_ENGINE

        async def parse(self, ctx: ParseContext) -> ParseResult:
            markdown = ctx.content_data.get("content")
            if not isinstance(markdown, str) or not markdown.strip():
                raise ValueError("preconverted markdown body is empty")
            token = _markdown_body.set(markdown)
            try:
                return await super().parse(ctx)
            finally:
                _markdown_body.reset(token)

        def validate_source(self, source: Path, file_path: str) -> None:
            if not (source.exists() and source.is_file()):
                raise FileNotFoundError(
                    f"preconverted source file not found: {file_path}"
                )

        def extract(
            self,
            source: Path,
            *,
            parsed_dir: Path,
            asset_dir: Path,
            base_name: str,
            runtime: Any | None = None,
        ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
            # LightRAG 1.5.5 passes its extraction runtime as a keyword.
            # Twin consumes an already converted markdown body, so the runtime
            # is intentionally unused, but accepting it keeps this adapter
            # substitutable for NativeMarkdownParser.extract.
            del source, parsed_dir, asset_dir, base_name, runtime
            markdown = _markdown_body.get()
            if markdown is None:
                raise RuntimeError("preconverted markdown context is missing")
            return self._extract_text(markdown, bundle_root=None)

except Exception:  # noqa: BLE001 - any 1.4.x import shape means "absent"
    TwinPreconvertedMarkdownParser = None  # type: ignore[assignment,misc]


def _mode_off() -> bool:
    return os.environ.get(TWIN_PRECONVERTED_PARSE_ENV, "").strip().lower() == "off"


def is_available() -> bool:
    """Whether the full 1.5.x seam exists AND the kill-switch is not set.

    Every leg is probed — a partial upstream surface (e.g. a parser
    registry without the sidecar backfill) must fail closed to the raw
    path rather than half-engage.
    """
    global _capability_cache
    if _mode_off():
        return False
    if _capability_cache is not None:
        return _capability_cache

    available = TwinPreconvertedMarkdownParser is not None
    if available:
        try:
            from lightrag import LightRAG
            from lightrag.parser.registry import (  # noqa: F401
                ParserSpec,
                register_parser,
            )
            from lightrag.sidecar import backfill as _backfill  # noqa: F401

            # The MIP classification hook may already have wrapped the
            # enqueue with a generic (*args, **kwargs) signature — probe the
            # ORIGINAL underneath it, whatever the install order was.
            target = (
                getattr(LightRAG, "_twin_original_enqueue", None)
                or LightRAG.apipeline_enqueue_documents
            )
            while hasattr(target, "__wrapped__"):
                target = target.__wrapped__
            parameters = inspect.signature(target).parameters
            available = "docs_format" in parameters and "parse_engine" in parameters
        except Exception:  # noqa: BLE001 - probe failure = capability absent
            available = False
    _capability_cache = available
    if not available:
        logger.debug(
            "twindb preconverted: 1.5.x parse seam unavailable — converted "
            "markdown stays on the raw enqueue path"
        )
    return available


def _reset_capability_cache_for_tests() -> None:
    # _backfill_installed is deliberately NOT reset: the module-level wrap
    # over the real lightrag is not uninstallable, so pretending otherwise
    # would only let a test re-wrap the wrapper.
    global _capability_cache, _registered
    _capability_cache = None
    _registered = False


def supports_suffix(path: Path) -> bool:
    """Whether the twinmarkdown engine may be requested for this file.

    ``TWIN_CONVERT_FORMATS``/``TWIN_VISION_FORMATS`` are operator-extensible
    while :data:`PARSER_SUFFIXES` is frozen into the registered ParserSpec —
    and 1.5.x rejects an explicitly requested engine for a suffix outside
    its capabilities (doc FAILED). Out-of-list suffixes therefore stay on
    the raw path (fail closed to raw, per the module doctrine).
    """
    return path.suffix.lower().lstrip(".") in PARSER_SUFFIXES


def ensure_parser_registered() -> bool:
    """Idempotently register the ``twinmarkdown`` engine. True when usable."""
    global _registered
    if not is_available():
        return False
    if _registered:
        return True
    try:
        from lightrag.parser.registry import ParserSpec, register_parser

        register_parser(
            ParserSpec(
                engine_name=PARSER_ENGINE,
                impl=f"{__name__}:TwinPreconvertedMarkdownParser",
                suffixes=PARSER_SUFFIXES,
            )
        )
        _registered = True
    except Exception:  # noqa: BLE001 - registration failure degrades to raw
        logger.exception(
            "twindb preconverted: parser registration failed — converted "
            "markdown stays on the raw enqueue path"
        )
        return False
    return True


def derive_block_boundaries(
    content: str,
    source_span: tuple[int, int],
    ref_ids: list[str],
    block_spans: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Intersect a chunk's document span with its referenced block spans.

    Pure math, no LightRAG import: offsets are Unicode CODE POINTS
    (Python string indices), half-open, LOCAL to the chunk content. A ref
    whose block span is unknown or whose intersection is empty or fails
    the exactness bound is skipped — a wrong boundary is worse than none.
    """
    source_start, source_end = source_span
    if not (0 <= source_start < source_end):
        return []
    boundaries: list[dict[str, Any]] = []
    for block_id in dict.fromkeys(ref_ids):
        span = block_spans.get(block_id)
        if span is None:
            continue
        block_start, block_end = span
        overlap_start = max(source_start, block_start)
        overlap_end = min(source_end, block_end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - source_start
        local_end = overlap_end - source_start
        if not (0 <= local_start < local_end <= len(content)):
            continue
        boundaries.append(
            {"block_id": block_id, "start": local_start, "end": local_end}
        )
    return boundaries


def _backfill_with_boundaries(original: Any, chunks: Any, blocks_path: str) -> None:
    """Run the upstream backfill, then enrich chunks with boundaries.

    Depends on three PRIVATE upstream helpers (``_load_content_blocks``,
    ``_build_block_spans``, ``_chunk_source_span``) — acceptable because
    the whole seam is capability-gated to the canary line and every
    failure here degrades to "no boundaries" (phase A lexical anchors
    still apply), never to a broken ingestion.
    """
    original(chunks, blocks_path)
    try:
        from lightrag.sidecar.backfill import (
            _build_block_spans,
            _chunk_source_span,
            _load_content_blocks,
        )

        merged, raw_spans = _build_block_spans(_load_content_blocks(blocks_path))
        block_spans = {block_id: (start, end) for start, end, block_id in raw_spans}
    except Exception:  # noqa: BLE001 - upstream drift: skip enrichment
        logger.exception(
            "twindb preconverted: block-span helpers unavailable or failed — "
            "chunks keep sidecar refs only, no twin_block_boundaries"
        )
        return

    for chunk in chunks:
        try:
            sidecar = chunk.get("sidecar")
            if not isinstance(sidecar, dict):
                continue
            source_span = _chunk_source_span(chunk, merged)
            if source_span is None:
                continue
            content = str(chunk.get("content") or "")
            ref_ids = [
                str(ref.get("id") or "")
                for ref in sidecar.get("refs") or []
                if isinstance(ref, dict)
            ]
            boundaries = derive_block_boundaries(
                content, source_span, ref_ids, block_spans
            )
            if boundaries:
                chunk["twin_block_boundaries"] = boundaries
        except Exception:  # noqa: BLE001 - one bad chunk must not break the doc
            logger.exception(
                "twindb preconverted: boundary derivation failed for one "
                "chunk — it ships without twin_block_boundaries"
            )


def install_backfill_boundaries() -> bool:
    """Wrap ``backfill_chunk_sidecars`` (module + package re-export).

    Idempotent; returns True when the wrapper is (already) in place.
    """
    global _backfill_installed
    if not is_available():
        return False
    if _backfill_installed:
        return True
    try:
        import lightrag.sidecar as sidecar_package
        from lightrag.sidecar import backfill as backfill_module

        original = backfill_module.backfill_chunk_sidecars
        if getattr(original, "_twindb_boundaries_installed", False):
            _backfill_installed = True
            return True

        def backfill_chunk_sidecars_with_boundaries(
            chunks: Any, blocks_path: str
        ) -> None:
            _backfill_with_boundaries(original, chunks, blocks_path)

        backfill_chunk_sidecars_with_boundaries._twindb_boundaries_installed = (  # type: ignore[attr-defined]
            True
        )
        backfill_chunk_sidecars_with_boundaries.__wrapped__ = original  # type: ignore[attr-defined]
        backfill_module.backfill_chunk_sidecars = (
            backfill_chunk_sidecars_with_boundaries
        )
        if hasattr(sidecar_package, "backfill_chunk_sidecars"):
            sidecar_package.backfill_chunk_sidecars = (
                backfill_chunk_sidecars_with_boundaries
            )
        _backfill_installed = True
    except Exception:  # noqa: BLE001 - degrade: refs persist, boundaries don't
        logger.exception(
            "twindb preconverted: could not install the boundary backfill — "
            "chunks keep sidecar refs only"
        )
        return False
    return True


def activate() -> bool:
    """Register the parser and install the boundary backfill (fail-soft).

    Called from ``register()``: on the 1.4.x line this is a no-op and the
    raw enqueue path is untouched. Deliberate asymmetry with the enqueue
    switch: ``_enqueue_converted`` gates on ``ensure_parser_registered()``
    ALONE, so a failed backfill install still takes ``pending_parse`` —
    native parsing without boundaries (refs only, lexical anchors intact)
    beats falling all the way back to raw.
    """
    return ensure_parser_registered() and install_backfill_boundaries()
