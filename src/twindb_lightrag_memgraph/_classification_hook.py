"""
Pre-ingestion classification hook for LightRAG.

When :func:`twindb_lightrag_memgraph.register` is called with
``classify=True`` (default if `TWIN_MIP_LABEL_MAP` is set), this module
patches LightRAG's document ingestion paths to:

1. Run :func:`classification.detect_classification` on the source file
   BEFORE LightRAG starts chunking / embedding.
2. Persist the resulting `ClassificationResult.as_dict()` payload into
   the DocStatus's ``metadata.classification`` field, so the WebUI's
   ``DocDetailPanel`` (Twin overlay) can gate the chunks tab and the
   "View raw" notice on it.
3. Optionally REJECT the document when its class outranks a configured
   maximum (env var ``TWIN_MIP_MAX_CLASSIFICATION``, default ``"C2"``).
   Rejected documents never enter the index. The rejection is logged
   and emitted as a Twin audit event (kind=``classification-rejected``)
   for the operator's review.

Wiring
------
The hook is opt-in. It is enabled by importing
``install_classification_hook()`` from this module and calling it after
``register()``. A more invasive flag-in-register() integration can be
added later; for now keep the surface explicit.

Patch points (audit 2026-07-02, finding ING-1)
----------------------------------------------
Two LightRAG entry points are patched by
:func:`install_lightrag_ingestion_hook`:

- ``LightRAG.apipeline_enqueue_documents`` — the **primary gate**. The
  native HTTP ingestion routes (``POST /documents/upload``, ``/text``,
  ``/texts``, ``/scan``) never call ``ainsert``; on the BNP pin
  (1.4.9.11) they call ``rag.apipeline_enqueue_documents`` +
  ``rag.apipeline_process_enqueue_documents`` directly
  (``document_routes.py:1563,1724``). Gating at enqueue level means a
  rejected document is never written to ``full_docs`` / DocStatus
  PENDING, so the pipeline's consistency pass preserves the FAILED
  rejection row instead of resetting it (``lightrag.py:1493-1513`` on
  the pin; same preserve logic in 1.5.x ``pipeline.py``). The patch is
  hasattr-guarded: on a LightRAG without that method, the hook falls
  back to ainsert-only gating with a loud warning.
- ``LightRAG.ainsert`` — kept for SDK/overlay callers and as the
  version fallback. Because ``ainsert`` internally calls
  ``self.apipeline_enqueue_documents`` (1.4.9.11 ``lightrag.py:1178``,
  1.5.4 ``lightrag.py:1486``), the ainsert gate sets a per-call
  ContextVar (:data:`_AINSERT_GATE_ACTIVE`) so the enqueue gate
  passes straight through underneath it — one classification run, one
  ``classification-rejected`` event per document, never two.

Design notes
------------
- In-memory ingestion without a source file is rejected while the hook is
  active: there is no trustworthy classification signal to enforce against
  the workspace ceiling.
- Native upload routes enqueue with the bare file name while the binary
  sits in the configured ``INPUT_DIR`` — :func:`_resolve_detection_path`
  re-joins the two (confined to the input tree) before probing.
- The FAILED rejection row deliberately does NOT persist an excerpt of
  the rejected content (audit finding PIPE-6b): ``content_summary`` is
  a fixed redaction placeholder, because the summary would surface the
  very document the gate refused into the WebUI.
- The hook NEVER raises into the caller — rejections are converted to FAILED
  DocStatus rows. What rejects depends on ``TWIN_MIP_UNLABELED_POLICY``
  (default ``allow``, decision 2026-07-10): readable-above-ceiling and
  untrusted UNKNOWN labels always reject; UNLABELED documents (unsupported
  format, no label, missing optional dep) are ingested with class None
  traced in metadata — set the policy to ``reject`` to restore the tier-1
  fail-closed posture.
- Audit event emission is delegated to a callback so this module stays
  decoupled from the Twin overlay's specific store implementation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable

from ._constants import get_active_operator_classification
from ._import_cleanup import _confined, configured_input_dir
from .classification import (
    ClassificationResult,
    apply_operator_classification,
    detect_classification,
    is_above,
    load_label_map,
    unlabeled_ingest_allowed,
)

log = logging.getLogger("twin.classification.hook")

# Type aliases for the callbacks the host passes in.
AuditEmitter = Callable[[str, dict[str, Any]], None]

# True while the patched ``ainsert`` gate owns the current call chain, so the
# enqueue-level gate underneath it (``ainsert`` → ``apipeline_enqueue_documents``
# on every supported LightRAG) does not classify/reject the same batch twice.
_AINSERT_GATE_ACTIVE: ContextVar[bool] = ContextVar(
    "twin_ainsert_gate_active", default=False
)

# Placeholder "path" recorded for rejections of in-memory ingestion calls
# (no source file on disk, hence no trustworthy classification signal).
_IN_MEMORY_INGESTION_PATH = "<in-memory-ingestion>"
_CLASSIFICATION_METADATA_KEYS = (
    "classification",
    "classification_rejected",
    "classification_ceiling",
)


class ClassificationRejection(Exception):
    """Raised internally when a document exceeds the configured ceiling.

    The hook catches this and converts it to a DocStatus update with
    ``status='FAILED'`` + ``error_msg`` carrying the human-readable reason,
    so the operator sees the rejection in DocumentsTab next to other
    ingestion errors.
    """

    def __init__(self, path: str, result: ClassificationResult, ceiling: str):
        self.path = path
        self.result = result
        self.ceiling = ceiling
        super().__init__(
            f"Classification {result.class_id!r} (raw: {result.raw_name!r}) "
            f"outranks workspace ceiling {ceiling!r} — refused ingestion of {path}"
        )


def classify_for_ingestion(
    file_path: str | os.PathLike[str],
    *,
    label_map: dict[str, str] | None = None,
    ceiling: str | None = None,
    audit_emit: AuditEmitter | None = None,
) -> dict[str, Any]:
    """Run the classifier and produce a payload for DocStatus.metadata.

    Parameters
    ----------
    file_path:
        Path to the source document. Required; source-less ingestion is
        rejected by the patched LightRAG entry points before this helper.
    label_map:
        Override for the loaded ``{guid → class_id}`` map. Defaults to
        :func:`load_label_map` (reads ``TWIN_MIP_LABEL_MAP``).
    ceiling:
        Maximum class allowed through. Strings outranking this trigger
        :class:`ClassificationRejection`. Defaults to env
        ``TWIN_MIP_MAX_CLASSIFICATION`` then ``"C2"``.
    audit_emit:
        Optional callback receiving ``(event_kind, payload_dict)``. When
        provided, the hook emits:
          - ``"classification-detected"`` for every successful extraction
          - ``"classification-rejected"`` when the ceiling is exceeded
        and the rejection is about to be raised.

    Returns
    -------
    dict
        The ``ClassificationResult.as_dict()`` payload, ready to be
        written into ``DocStatus.metadata['classification']``.

    Raises
    ------
    ClassificationRejection
        When the detected class outranks the configured ceiling, when the
        label is readable but untrusted (``UNKNOWN``), or — only under
        ``TWIN_MIP_UNLABELED_POLICY=reject`` — when no MIP label could be
        read at all (default policy is ``allow``; decision 2026-07-10).
    """
    path_str = os.fspath(file_path)
    if label_map is None:
        label_map = load_label_map()
    if ceiling is None:
        ceiling = os.environ.get("TWIN_MIP_MAX_CLASSIFICATION", "C2")

    result = _probe_classification(path_str, label_map)
    return _evaluate_classification_for_ingestion(
        path_str,
        result,
        ceiling=ceiling,
        audit_emit=audit_emit,
    )


def _probe_classification(
    path_str: str, label_map: dict[str, str]
) -> ClassificationResult:
    """Run the potentially blocking file/classification probe."""
    try:
        return detect_classification(path_str, label_map=label_map)
    except Exception as exc:  # belt-and-suspenders — classifier should never raise
        log.warning("Classifier raised for %s: %s — treating as UNKNOWN", path_str, exc)
        return ClassificationResult(
            class_id="UNKNOWN",
            source_format="unknown",
            reason=f"extraction-failed: {exc.__class__.__name__}",
        )


def _evaluate_classification_for_ingestion(
    path_str: str,
    result: ClassificationResult,
    *,
    ceiling: str,
    audit_emit: AuditEmitter | None,
) -> dict[str, Any]:
    """Apply policy and emit audit events on the caller's thread."""
    # Validate the immutable source signal before considering mutable operator
    # input. Otherwise an operator-selected C1/C2 could turn a missing or
    # unparseable label into an apparently authorised classification before the
    # ceiling check below.
    #
    # Unlabeled documents (class_id=None) follow TWIN_MIP_UNLABELED_POLICY
    # (default: allow — decision 2026-07-10). UNKNOWN (readable-but-unmapped
    # label, extraction crash) stays fail-closed through is_above regardless
    # of the policy; ``is_above(None, ceiling)`` is False by contract.
    source_payload = result.as_dict()
    if (result.class_id is None and not unlabeled_ingest_allowed()) or is_above(
        result.class_id, ceiling
    ):
        if audit_emit is not None:
            audit_emit(
                "classification-detected",
                {
                    "path": path_str,
                    "classification": source_payload,
                    "ceiling": ceiling,
                },
            )
            audit_emit(
                "classification-rejected",
                {
                    "path": path_str,
                    "classification": source_payload,
                    "ceiling": ceiling,
                },
            )
        raise ClassificationRejection(path_str, result, ceiling)

    # An operator-selected class (upload UI -> X-Twin-Classification header,
    # bound into the ingestion context) may only raise a trusted source class.
    # The embedded label stays a floor: operators can never downgrade below it.
    result = apply_operator_classification(result, get_active_operator_classification())

    payload = result.as_dict()

    if audit_emit is not None:
        audit_emit(
            "classification-detected",
            {"path": path_str, "classification": payload, "ceiling": ceiling},
        )

    # The source has already passed the trust/ceiling gate. This second check
    # enforces the same ceiling after a legitimate operator raise.
    if is_above(result.class_id, ceiling):
        if audit_emit is not None:
            audit_emit(
                "classification-rejected",
                {"path": path_str, "classification": payload, "ceiling": ceiling},
            )
        raise ClassificationRejection(path_str, result, ceiling)

    return payload


def install_classification_hook(
    *,
    label_map_path: str | os.PathLike[str] | None = None,
    ceiling: str | None = None,
    audit_emit: AuditEmitter | None = None,
) -> Callable[[str], dict[str, Any]]:
    """Build a closure suitable as a LightRAG pre-insert hook.

    The returned callable takes a file path and returns the classification
    payload (already validated against the ceiling — raises
    :class:`ClassificationRejection` when refused).

    This is intentionally a factory rather than a module-level singleton:
    different workspaces may have different ceilings (Public-facing one
    can be ``C1``-only, internal one ``C3``, etc.), and the host wires
    them per workspace.
    """
    label_map = load_label_map(label_map_path)
    resolved_ceiling = ceiling or os.environ.get("TWIN_MIP_MAX_CLASSIFICATION", "C2")

    def _probe(file_path: str) -> ClassificationResult:
        return _probe_classification(file_path, label_map)

    def _evaluate(file_path: str, result: ClassificationResult) -> dict[str, Any]:
        return _evaluate_classification_for_ingestion(
            file_path,
            result,
            ceiling=resolved_ceiling,
            audit_emit=audit_emit,
        )

    def _hook(file_path: str) -> dict[str, Any]:
        return _evaluate(file_path, _probe(file_path))

    # The patched LightRAG entry points need the same explicitly configured
    # ceiling when constructing a rejection for source-less ingestion.
    setattr(_hook, "_twin_classification_ceiling", resolved_ceiling)
    # The async ingestion path offloads only this probe. Policy evaluation,
    # operator ContextVar access, and the public audit callback stay on the
    # caller's event-loop thread.
    setattr(_hook, "_twin_classification_probe", _probe)
    setattr(_hook, "_twin_classification_evaluate", _evaluate)

    log.info(
        "Classification hook installed (ceiling=%s, label_map_size=%d)",
        resolved_ceiling,
        len(label_map),
    )
    return _hook


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _doc_id_for_insert(
    content: str, explicit_id: str | None, file_path: str | None = None
) -> str:
    """Mirror the installed LightRAG enqueue's deterministic document id.

    LightRAG 1.5.x keys every known-source document on its canonical file
    path, including RAW inserts. The supported 1.4.x line keys non-empty RAW
    inserts on content instead. Detect the 1.5.x enqueue capability rather
    than the package version so accepted metadata and rejected status rows
    use the same id as the active runtime.

    The blank-content file-path fallback remains for runtimes whose upload
    path defers parsing without exposing the 1.5.x ``docs_format`` argument.
    """
    if explicit_id:
        return explicit_id
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

    if file_path:
        try:
            import inspect

            from lightrag import LightRAG
            from lightrag.utils_pipeline import (
                has_known_document_source,
                normalize_document_file_path,
            )

            enqueue = getattr(LightRAG, "_twin_original_enqueue", None)
            if enqueue is None:
                enqueue = getattr(LightRAG, "apipeline_enqueue_documents", None)
            supports_document_formats = (
                callable(enqueue)
                and "docs_format" in inspect.signature(enqueue).parameters
            )
            canonical_path = normalize_document_file_path(file_path)
            if supports_document_formats and has_known_document_source(canonical_path):
                return compute_mdhash_id(canonical_path, prefix="doc-")
        except (ImportError, TypeError, ValueError):
            # Older LightRAG builds do not expose these helpers/signature.
            pass

    if not content.strip() and file_path:
        return compute_mdhash_id(file_path, prefix="doc-")
    return compute_mdhash_id(sanitize_text_for_encoding(content), prefix="doc-")


def _resolve_detection_path(path_str: str) -> str:
    """Map a DocStatus-style ``file_path`` to an on-disk file for label probing.

    The native upload routes enqueue with the bare file name while the binary
    lives in the configured ``INPUT_DIR`` (1.4.9.11
    ``document_routes.py:1563`` passes ``file_path.name``). Absolute or
    cwd-resolvable paths are used as-is; a bare name is joined to the input
    dir only when that lands on an existing file confined to the input tree
    (same confinement helper as the import cleanup). Falls back to the raw
    string, in which case detection degrades gracefully to a no-label result.
    """
    try:
        raw = Path(path_str)
        if raw.is_absolute() or raw.exists():
            return path_str
        base = configured_input_dir().resolve()
        candidate = _confined(base / raw, base)
        if candidate is not None and candidate.is_file():
            return str(candidate)
    except (OSError, ValueError) as exc:
        log.debug("Detection path resolution failed for %r: %s", path_str, exc)
    return path_str


def _failed_status_for_rejection(
    *,
    content: str,
    file_path: str,
    track_id: str,
    exc: ClassificationRejection,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    from lightrag.base import DocStatus

    now = datetime.now(timezone.utc).isoformat()
    # PIPE-6b: never persist an excerpt of the over-classified content — the
    # summary lands in Memgraph and is displayed by the WebUI, defeating the
    # gate. A fixed placeholder carries the rejection context instead.
    redacted_summary = (
        f"[content withheld: classification {exc.result.class_id} "
        f"exceeds ceiling {exc.ceiling}]"
    )
    return {
        "status": DocStatus.FAILED,
        "content_summary": redacted_summary,
        "content_length": len(content),
        "chunks_count": 0,
        "chunks_list": [],
        "created_at": now,
        "updated_at": now,
        "file_path": file_path,
        "track_id": track_id,
        "error_msg": str(exc),
        "metadata": {
            "classification": exc.result.as_dict(),
            "classification_rejected": True,
            "classification_ceiling": exc.ceiling,
        },
    }


async def _emit_rejection_event(
    *,
    actor: str,
    doc_id: str,
    file_path: str,
    exc: ClassificationRejection,
) -> None:
    try:
        from .server.webui_router import _make_event, get_store

        event = _make_event(
            kind="classification-rejected",
            sev="warning",
            actor=actor,
            target_label=file_path,
            summary=str(exc),
            meta={
                "doc_id": doc_id,
                "reason": str(exc),
                "path": file_path,
                "classification": exc.result.as_dict(),
                "ceiling": exc.ceiling,
            },
            target_type="document",
            target_id=doc_id,
        )
        await get_store().record_activity(event)
    except Exception as event_exc:
        log.warning("Failed to emit classification rejection event: %s", event_exc)


async def _merge_classification_metadata(
    rag: Any,
    doc_id: str,
    payload: dict[str, Any],
) -> None:
    try:
        existing = await rag.doc_status.get_by_id(doc_id)
        if not existing:
            return
        doc = _doc_status_to_dict(existing)
        metadata = dict(doc.get("metadata") or {})
        metadata["classification"] = payload
        doc["metadata"] = metadata
        await rag.doc_status.upsert({doc_id: doc})
    except Exception as exc:
        log.warning("Failed to persist classification metadata for %s: %s", doc_id, exc)


def _doc_status_to_dict(existing: Any) -> dict[str, Any]:
    if isinstance(existing, dict):
        return dict(existing)
    if dataclasses.is_dataclass(existing):
        return dataclasses.asdict(existing)
    return {}


async def _merge_classification_metadata_batch(
    rag: Any,
    payloads_by_doc_id: dict[str, dict[str, Any]],
) -> None:
    """Merge accepted-document classification metadata with bounded I/O.

    Memgraph's DocStatus backend exposes ``get_by_ids`` and a set-based
    ``upsert``. Use both once for the whole accepted batch instead of issuing a
    read plus two write queries per document. Alternate LightRAG stores keep
    the old per-document path through capability/error fallbacks.
    """
    if not payloads_by_doc_id:
        return

    get_by_ids = getattr(rag.doc_status, "get_by_ids", None)
    if not callable(get_by_ids):
        for doc_id, payload in payloads_by_doc_id.items():
            await _merge_classification_metadata(rag, doc_id, payload)
        return

    doc_ids = list(payloads_by_doc_id)
    try:
        existing_rows = await get_by_ids(doc_ids)
    except Exception as exc:
        log.warning(
            "Batch classification metadata read failed; retrying individually: %s",
            exc,
        )
        for doc_id, payload in payloads_by_doc_id.items():
            await _merge_classification_metadata(rag, doc_id, payload)
        return

    if isinstance(existing_rows, dict):
        identified_rows = [
            (str(doc_id), _doc_status_to_dict(existing))
            for doc_id, existing in existing_rows.items()
            if existing
        ]
    else:
        identified_rows = []
        for existing in existing_rows or []:
            doc = _doc_status_to_dict(existing)
            doc_id = doc.get("id")
            if not doc_id:
                # An alternate store returned records without stable ids. The
                # result cannot be safely aligned to payloads, so preserve the
                # original capability-neutral path.
                for fallback_id, payload in payloads_by_doc_id.items():
                    await _merge_classification_metadata(rag, fallback_id, payload)
                return
            identified_rows.append((str(doc_id), doc))

    updates: dict[str, dict[str, Any]] = {}
    for doc_id, doc in identified_rows:
        payload = payloads_by_doc_id.get(doc_id)
        if payload is None:
            continue
        metadata = dict(doc.get("metadata") or {})
        metadata["classification"] = payload
        doc["metadata"] = metadata
        updates[doc_id] = doc
    if not updates:
        log.warning(
            "Classification metadata targets were not found in DocStatus: %s",
            ", ".join(payloads_by_doc_id),
        )
        return

    try:
        await rag.doc_status.upsert(updates)
    except Exception as exc:
        log.warning(
            "Batch classification metadata write failed; retrying individually: %s",
            exc,
        )
        for doc_id, payload in payloads_by_doc_id.items():
            await _merge_classification_metadata(rag, doc_id, payload)


def _partition_inputs(
    active_hook, paths: list[Any]
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, ClassificationRejection]]]:
    """Run the classification hook per path; split accepted vs rejected.

    The hook probes the file resolved by :func:`_resolve_detection_path`;
    the raw path value stays untouched for DocStatus rows and events.
    """
    accepted: list[tuple[int, dict[str, Any]]] = []
    rejected: list[tuple[int, ClassificationRejection]] = []
    for idx, path in enumerate(paths):
        path_str = str(path or "").strip() or "unknown_source"
        try:
            payload = active_hook(_resolve_detection_path(path_str))
        except ClassificationRejection as exc:
            rejected.append((idx, exc))
            continue
        accepted.append((idx, payload))
    return accepted, rejected


def _probe_inputs(
    probe: Callable[[str], Any], paths: list[Any]
) -> list[tuple[int, str, Any]]:
    """Resolve paths and probe one ordered batch in a worker thread."""
    probed: list[tuple[int, str, Any]] = []
    for idx, path in enumerate(paths):
        path_str = str(path or "").strip() or "unknown_source"
        resolved_path = _resolve_detection_path(path_str)
        probed.append((idx, resolved_path, probe(resolved_path)))
    return probed


async def _partition_inputs_async(
    active_hook, paths: list[Any]
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, ClassificationRejection]]]:
    """Offload file probes while keeping policy/audit on the event loop."""
    probe = getattr(active_hook, "_twin_classification_probe", None)
    evaluate = getattr(active_hook, "_twin_classification_evaluate", None)
    if not callable(probe) or not callable(evaluate):
        # Third-party hooks are opaque: running them in a worker could change
        # callback/thread affinity. Preserve their synchronous caller-thread
        # behavior unless they explicitly expose the split protocol above.
        return _partition_inputs(active_hook, paths)

    probed = await asyncio.to_thread(_probe_inputs, probe, paths)
    accepted: list[tuple[int, dict[str, Any]]] = []
    rejected: list[tuple[int, ClassificationRejection]] = []
    for idx, path_str, result in probed:
        try:
            payload = evaluate(path_str, result)
        except ClassificationRejection as exc:
            rejected.append((idx, exc))
            continue
        accepted.append((idx, payload))
    return accepted, rejected


def _source_required_rejections(
    active_hook: Callable[..., Any], count: int
) -> list[tuple[int, ClassificationRejection]]:
    """Build fail-closed rejections for documents without source files."""
    ceiling = str(
        getattr(
            active_hook,
            "_twin_classification_ceiling",
            os.environ.get("TWIN_MIP_MAX_CLASSIFICATION", "C2"),
        )
    )
    result = ClassificationResult(
        class_id="UNKNOWN",
        source_format="in-memory",
        reason="source-file-required",
    )
    return [
        (
            idx,
            ClassificationRejection(_IN_MEMORY_INGESTION_PATH, result, ceiling),
        )
        for idx in range(count)
    ]


def _raw_path_at(paths, idx) -> str | None:
    """The raw (unresolved) path string at ``idx``, or None when blank."""
    if not paths or idx >= len(paths):
        return None
    return str(paths[idx] or "").strip() or None


async def _apply_classification_metadata(
    self, accepted, inputs, explicit_ids, ids_provided, paths=None
) -> None:
    """Write ``metadata.classification`` for each accepted document."""
    payloads_by_doc_id: dict[str, dict[str, Any]] = {}
    for idx, payload in accepted:
        doc_id = _doc_id_for_insert(
            str(inputs[idx]),
            str(explicit_ids[idx]) if ids_provided else None,
            file_path=_raw_path_at(paths, idx),
        )
        payloads_by_doc_id[doc_id] = payload
    await _merge_classification_metadata_batch(self, payloads_by_doc_id)


async def _record_rejections(
    self, rejected, inputs, paths, explicit_ids, ids_provided, resolved_track_id
) -> None:
    """Persist failed DocStatus rows for rejected docs and emit audit events."""
    failed_docs: dict[str, dict[str, Any]] = {}
    for idx, exc in rejected:
        doc_id = _doc_id_for_insert(
            str(inputs[idx]),
            str(explicit_ids[idx]) if ids_provided else None,
            file_path=_raw_path_at(paths, idx),
        )
        failed_docs[doc_id] = _failed_status_for_rejection(
            content=str(inputs[idx]),
            file_path=str(paths[idx] or "unknown_source"),
            track_id=resolved_track_id,
            exc=exc,
        )
        await _emit_rejection_event(
            actor="system",
            doc_id=doc_id,
            file_path=str(paths[idx] or "unknown_source"),
            exc=exc,
        )
    if failed_docs:
        await self.doc_status.upsert(failed_docs)


async def _reinsert_accepted(
    self,
    accepted,
    inputs,
    paths,
    explicit_ids,
    input,
    ids,
    file_paths,
    split_by_character,
    split_by_character_only,
    resolved_track_id,
) -> str:
    """Re-run the original ainsert on the accepted subset, then tag metadata."""
    original_ainsert = self.__class__._twin_original_ainsert
    accepted_indices = [idx for idx, _payload in accepted]
    accepted_inputs = [inputs[idx] for idx in accepted_indices]
    accepted_paths = [paths[idx] for idx in accepted_indices]
    accepted_ids = (
        [explicit_ids[idx] for idx in accepted_indices] if ids is not None else None
    )
    if isinstance(ids, list):
        accepted_ids_arg = accepted_ids
    elif accepted_ids:
        accepted_ids_arg = accepted_ids[0]
    else:
        accepted_ids_arg = None
    result_track_id = await original_ainsert(
        self,
        accepted_inputs if isinstance(input, list) else accepted_inputs[0],
        split_by_character,
        split_by_character_only,
        accepted_ids_arg,
        accepted_paths if isinstance(file_paths, list) else accepted_paths[0],
        resolved_track_id,
    )
    await _apply_classification_metadata(
        self, accepted, inputs, explicit_ids, ids is not None, paths=paths
    )
    return result_track_id


async def _patched_ainsert(
    self,
    input: str | list[str],
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    track_id: str | None = None,
) -> str:
    """Classification-aware replacement for ``LightRAG.ainsert``.

    Reads the active hook and the captured original ainsert from class
    attributes (set by ``install_lightrag_ingestion_hook``) rather than a
    closure, so the function can live at module scope.

    Sets :data:`_AINSERT_GATE_ACTIVE` for the whole call so the enqueue-level
    gate (``ainsert`` internally calls ``apipeline_enqueue_documents``) does
    not classify, reject, or emit events a second time for the same batch.
    """
    token = _AINSERT_GATE_ACTIVE.set(True)
    try:
        return await _gated_ainsert(
            self,
            input,
            split_by_character,
            split_by_character_only,
            ids,
            file_paths,
            track_id,
        )
    finally:
        _AINSERT_GATE_ACTIVE.reset(token)


async def _gated_ainsert(
    self,
    input: str | list[str],
    split_by_character: str | None,
    split_by_character_only: bool,
    ids: str | list[str] | None,
    file_paths: str | list[str] | None,
    track_id: str | None,
) -> str:
    """Body of the ainsert-level gate (see :func:`_patched_ainsert`)."""
    original_ainsert = self.__class__._twin_original_ainsert
    active_hook = self.__class__._twin_classification_hook
    inputs = _as_list(input)
    paths = _as_list(file_paths)
    explicit_ids = _as_list(ids)

    if file_paths is not None and len(paths) != len(inputs):
        raise ValueError("Number of file paths must match the number of documents")
    if ids is not None and len(explicit_ids) != len(inputs):
        raise ValueError("Number of IDs must match the number of documents")

    if file_paths is None:
        from lightrag.utils import generate_track_id

        resolved_track_id = track_id or generate_track_id("insert")
        rejection_paths = [_IN_MEMORY_INGESTION_PATH] * len(inputs)
        await _record_rejections(
            self,
            _source_required_rejections(active_hook, len(inputs)),
            inputs,
            rejection_paths,
            explicit_ids,
            ids is not None,
            resolved_track_id,
        )
        return resolved_track_id

    accepted, rejected = await _partition_inputs_async(active_hook, paths)

    if not rejected:
        result_track_id = await original_ainsert(
            self,
            input,
            split_by_character,
            split_by_character_only,
            ids,
            file_paths,
            track_id,
        )
        await _apply_classification_metadata(
            self, accepted, inputs, explicit_ids, ids is not None, paths=paths
        )
        return result_track_id

    from lightrag.utils import generate_track_id

    resolved_track_id = track_id or generate_track_id("insert")
    await _record_rejections(
        self, rejected, inputs, paths, explicit_ids, ids is not None, resolved_track_id
    )

    if not accepted:
        return resolved_track_id

    return await _reinsert_accepted(
        self,
        accepted,
        inputs,
        paths,
        explicit_ids,
        input,
        ids,
        file_paths,
        split_by_character,
        split_by_character_only,
        resolved_track_id,
    )


def _resolve_enqueue_track_id(track_id: str | None) -> str:
    """Mirror the original enqueue's track-id rule (blank string == absent)."""
    if isinstance(track_id, str) and track_id.strip():
        return track_id
    from lightrag.utils import generate_track_id

    return generate_track_id("enqueue")


def _slice_aligned(value: Any, indices: list[int], total: int) -> Any:
    """Subset a per-document aligned list; pass scalars/broadcasts through.

    LightRAG 1.5.x enqueue takes extra per-document parameters
    (``parse_engine`` / ``process_options`` / ``chunk_options``) that may be
    lists aligned with ``input``. When the gate drops rejected documents it
    must drop the matching positions of every aligned extra, or the original
    enqueue would raise on the length mismatch (or worse, misalign them).
    """
    if isinstance(value, list) and len(value) == total:
        return [value[idx] for idx in indices]
    return value


async def _enqueue_accepted(
    self,
    original_enqueue,
    accepted,
    inputs,
    paths,
    explicit_ids,
    input,
    ids,
    file_paths,
    resolved_track_id,
    args,
    kwargs,
):
    """Re-run the original enqueue on the accepted subset, then tag metadata."""
    accepted_indices = [idx for idx, _payload in accepted]
    accepted_inputs = [inputs[idx] for idx in accepted_indices]
    accepted_paths = [paths[idx] for idx in accepted_indices]
    accepted_ids = (
        [explicit_ids[idx] for idx in accepted_indices] if ids is not None else None
    )
    if isinstance(ids, list):
        accepted_ids_arg = accepted_ids
    elif accepted_ids:
        accepted_ids_arg = accepted_ids[0]
    else:
        accepted_ids_arg = None
    total = len(inputs)
    sliced_args = tuple(_slice_aligned(a, accepted_indices, total) for a in args)
    sliced_kwargs = {
        k: _slice_aligned(v, accepted_indices, total) for k, v in kwargs.items()
    }
    result = await original_enqueue(
        self,
        accepted_inputs if isinstance(input, list) else accepted_inputs[0],
        accepted_ids_arg,
        accepted_paths if isinstance(file_paths, list) else accepted_paths[0],
        resolved_track_id,
        *sliced_args,
        **sliced_kwargs,
    )
    await _apply_classification_metadata(
        self, accepted, inputs, explicit_ids, ids is not None, paths=paths
    )
    return result


async def _patched_apipeline_enqueue_documents(
    self,
    input: str | list[str],
    ids: str | list[str] | None = None,
    file_paths: str | list[str] | None = None,
    track_id: str | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Classification-aware replacement for ``apipeline_enqueue_documents``.

    This is the primary gate: the native HTTP ingestion routes call this
    method directly and never go through ``ainsert`` (audit 2026-07-02,
    ING-1). Rejected documents are never handed to the original enqueue, so
    they never reach ``full_docs`` or a PENDING DocStatus row — the FAILED
    rejection row written here is then preserved untouched by LightRAG's
    consistency pass (which only resets FAILED docs that *do* have content).

    Passthrough (no classification, exact original call) when:
    - the ainsert-level gate already owns this call chain
      (:data:`_AINSERT_GATE_ACTIVE` — prevents double gating/events);
    - the hook is not installed / was uninstalled (stale class patch);
    - argument shapes are invalid (the original raises its own native
      ``ValueError``, unchanged).

    Extra positional/keyword arguments (1.5.x ``docs_format`` /
    ``parse_engine`` / ``process_options`` / ``chunk_options`` /
    ``from_scan``) are forwarded verbatim; per-document aligned lists are
    subset alongside the inputs when documents are rejected.
    """
    original_enqueue = self.__class__._twin_original_enqueue
    active_hook = getattr(self.__class__, "_twin_classification_hook", None)
    inputs = _as_list(input)
    paths = _as_list(file_paths)
    explicit_ids = _as_list(ids)

    gate_off = (
        _AINSERT_GATE_ACTIVE.get()
        or active_hook is None
        or not inputs
        or (file_paths is not None and len(paths) != len(inputs))
        or (ids is not None and len(explicit_ids) != len(inputs))
    )
    if gate_off:
        return await original_enqueue(
            self, input, ids, file_paths, track_id, *args, **kwargs
        )

    if file_paths is None:
        resolved_track_id = _resolve_enqueue_track_id(track_id)
        rejection_paths = [_IN_MEMORY_INGESTION_PATH] * len(inputs)
        await _record_rejections(
            self,
            _source_required_rejections(active_hook, len(inputs)),
            inputs,
            rejection_paths,
            explicit_ids,
            ids is not None,
            resolved_track_id,
        )
        return resolved_track_id

    accepted, rejected = await _partition_inputs_async(active_hook, paths)

    if not rejected:
        result = await original_enqueue(
            self, input, ids, file_paths, track_id, *args, **kwargs
        )
        await _apply_classification_metadata(
            self, accepted, inputs, explicit_ids, ids is not None, paths=paths
        )
        return result

    resolved_track_id = _resolve_enqueue_track_id(track_id)
    await _record_rejections(
        self, rejected, inputs, paths, explicit_ids, ids is not None, resolved_track_id
    )

    if not accepted:
        return resolved_track_id

    return await _enqueue_accepted(
        self,
        original_enqueue,
        accepted,
        inputs,
        paths,
        explicit_ids,
        input,
        ids,
        file_paths,
        resolved_track_id,
        args,
        kwargs,
    )


def _install_enqueue_gate(cls) -> None:
    """Patch ``apipeline_enqueue_documents`` — the primary ingestion surface.

    hasattr-guarded and non-raising by contract: on a LightRAG without that
    method the hook degrades to ainsert-only gating with a loud warning
    (native HTTP ingestion routes would then be unguarded on that version).
    Idempotent via the ``_twin_enqueue_patched`` flag.
    """
    try:
        if getattr(cls, "_twin_enqueue_patched", False):
            return
        original = getattr(cls, "apipeline_enqueue_documents", None)
        if original is None or not callable(original):
            log.warning(
                "LightRAG.apipeline_enqueue_documents not found — classification "
                "gate covers ainsert only; native HTTP ingestion routes are NOT "
                "gated on this LightRAG version"
            )
            return
        if original is _patched_apipeline_enqueue_documents:
            # Already patched but the flag drifted (e.g. test fixtures snapping
            # attributes independently) — never capture the gate as "original".
            cls._twin_enqueue_patched = True
            return
        cls._twin_original_enqueue = original
        cls.apipeline_enqueue_documents = _patched_apipeline_enqueue_documents
        cls._twin_enqueue_patched = True
        log.info(
            "Classification gate installed on LightRAG.apipeline_enqueue_documents"
        )
    except Exception as exc:  # never break register()/boot on version skew
        log.warning(
            "Failed to install enqueue-level classification gate: %s — "
            "falling back to ainsert-only gating",
            exc,
        )


def _install_classification_metadata_carry_over() -> None:
    """Keep Twin security metadata across LightRAG 1.5.x state transitions.

    LightRAG 1.5.x rebuilds ``doc_status.metadata`` from a private carry-over
    tuple at every pipeline transition. Without registering Twin's fields,
    accepted classification is present at PENDING and silently disappears at
    PARSING. Older LightRAG versions do not expose this mechanism and need no
    patch.
    """
    from importlib import import_module

    try:
        utils_pipeline = import_module("lightrag.utils_pipeline")
    except ModuleNotFoundError:
        return

    try:
        attribute = "_DOC_STATUS_METADATA_CARRY_OVER_KEYS"
        keys = getattr(utils_pipeline, attribute, None)
        if not isinstance(keys, tuple):
            return
        missing = tuple(key for key in _CLASSIFICATION_METADATA_KEYS if key not in keys)
        if missing:
            setattr(utils_pipeline, attribute, (*keys, *missing))
            log.info(
                "Registered Twin classification fields for DocStatus "
                "metadata carry-over: %s",
                ", ".join(missing),
            )
    except Exception as exc:
        # This is compatibility wiring: installation must retain the module's
        # historical non-raising contract, but the omission must stay visible.
        log.warning(
            "Failed to register classification metadata carry-over: %s",
            exc,
        )


def install_lightrag_ingestion_hook(
    *,
    label_map_path: str | os.PathLike[str] | None = None,
    ceiling: str | None = None,
    audit_emit: AuditEmitter | None = None,
) -> None:
    """Patch LightRAG ingestion so file ingests are classified pre-index.

    Two patch points (see module docstring): ``apipeline_enqueue_documents``
    (primary — the native HTTP routes' only entry) and ``ainsert``
    (SDK/overlay + fallback). Rejected files are never passed to LightRAG's
    pipeline. Instead, a failed DocStatus row is written with
    ``metadata.classification`` (content summary redacted) and a
    ``classification-rejected`` activity event is emitted best-effort.

    Never raises: a LightRAG version missing the enqueue method keeps the
    historical ainsert-only behavior.
    """
    from lightrag import LightRAG

    hook = install_classification_hook(
        label_map_path=label_map_path,
        ceiling=ceiling,
        audit_emit=audit_emit,
    )
    setattr(LightRAG, "_twin_classification_hook", hook)
    _install_classification_metadata_carry_over()

    if not getattr(LightRAG, "_twin_classification_patched", False):
        setattr(LightRAG, "_twin_original_ainsert", LightRAG.ainsert)
        LightRAG.ainsert = _patched_ainsert
        setattr(LightRAG, "_twin_classification_patched", True)

    _install_enqueue_gate(LightRAG)
