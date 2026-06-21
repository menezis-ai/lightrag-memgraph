"""
Pre-ingestion classification hook for LightRAG.

When :func:`twindb_lightrag_memgraph.register` is called with
``classify=True`` (default if `TWIN_MIP_LABEL_MAP` is set), this module
patches LightRAG's document insertion path to:

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

Design notes
------------
- The hook works only when ingestion receives a file path (the most
  common case). If LightRAG is fed raw text in-memory, classification
  is unavailable and the hook deliberately passes the insertion through
  without adding classification metadata.
- The hook NEVER raises into the caller — a failed extraction yields a
  ``classification = {class_id: 'UNKNOWN', reason: 'extraction-failed'}``
  payload and the document is allowed through (or blocked, depending on
  ``TWIN_MIP_MAX_CLASSIFICATION``).
- Audit event emission is delegated to a callback so this module stays
  decoupled from the Twin overlay's specific store implementation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from .classification import (
    ClassificationResult,
    detect_classification,
    is_above,
    load_label_map,
)

log = logging.getLogger("twin.classification.hook")

# Type aliases for the callbacks the host passes in.
AuditEmitter = Callable[[str, dict[str, Any]], None]


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
        Path to the source document. Required — in-memory ingestion
        bypasses this hook.
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
        When the detected class outranks the configured ceiling.
    """
    path_str = os.fspath(file_path)
    if label_map is None:
        label_map = load_label_map()
    if ceiling is None:
        ceiling = os.environ.get("TWIN_MIP_MAX_CLASSIFICATION", "C2")

    try:
        result = detect_classification(path_str, label_map=label_map)
    except Exception as exc:  # belt-and-suspenders — classifier should never raise
        log.warning("Classifier raised for %s: %s — treating as UNKNOWN", path_str, exc)
        result = ClassificationResult(
            class_id="UNKNOWN",
            source_format="unknown",
            reason=f"extraction-failed: {exc.__class__.__name__}",
        )

    payload = result.as_dict()

    if audit_emit is not None:
        audit_emit(
            "classification-detected",
            {"path": path_str, "classification": payload, "ceiling": ceiling},
        )

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

    def _hook(file_path: str) -> dict[str, Any]:
        return classify_for_ingestion(
            file_path,
            label_map=label_map,
            ceiling=resolved_ceiling,
            audit_emit=audit_emit,
        )

    log.info(
        "Classification hook installed (ceiling=%s, label_map_size=%d)",
        resolved_ceiling, len(label_map),
    )
    return _hook


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _doc_id_for_insert(content: str, explicit_id: str | None) -> str:
    if explicit_id:
        return explicit_id
    from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

    return compute_mdhash_id(sanitize_text_for_encoding(content), prefix="doc-")


def _failed_status_for_rejection(
    *,
    content: str,
    file_path: str,
    track_id: str,
    exc: ClassificationRejection,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    from lightrag.base import DocStatus
    from lightrag.utils import get_content_summary

    now = datetime.now(timezone.utc).isoformat()
    return {
        "status": DocStatus.FAILED,
        "content_summary": get_content_summary(content),
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
                "path": file_path,
                "classification": exc.result.as_dict(),
                "ceiling": exc.ceiling,
            },
            target_type="document",
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
        if isinstance(existing, dict):
            doc = dict(existing)
        else:
            import dataclasses

            doc = dataclasses.asdict(existing) if dataclasses.is_dataclass(existing) else {}
        metadata = dict(doc.get("metadata") or {})
        metadata["classification"] = payload
        doc["metadata"] = metadata
        await rag.doc_status.upsert({doc_id: doc})
    except Exception as exc:
        log.warning("Failed to persist classification metadata for %s: %s", doc_id, exc)


def _partition_inputs(
    active_hook, paths: list[Any]
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, ClassificationRejection]]]:
    """Run the classification hook per path; split accepted vs rejected."""
    accepted: list[tuple[int, dict[str, Any]]] = []
    rejected: list[tuple[int, ClassificationRejection]] = []
    for idx, path in enumerate(paths):
        path_str = str(path or "").strip() or "unknown_source"
        try:
            payload = active_hook(path_str)
        except ClassificationRejection as exc:
            rejected.append((idx, exc))
            continue
        accepted.append((idx, payload))
    return accepted, rejected


async def _apply_classification_metadata(
    self, accepted, inputs, explicit_ids, ids_provided
) -> None:
    """Write ``metadata.classification`` for each accepted document."""
    for idx, payload in accepted:
        doc_id = _doc_id_for_insert(
            str(inputs[idx]),
            str(explicit_ids[idx]) if ids_provided else None,
        )
        await _merge_classification_metadata(self, doc_id, payload)


async def _record_rejections(
    self, rejected, inputs, paths, explicit_ids, ids_provided, resolved_track_id
) -> None:
    """Persist failed DocStatus rows for rejected docs and emit audit events."""
    failed_docs: dict[str, dict[str, Any]] = {}
    for idx, exc in rejected:
        doc_id = _doc_id_for_insert(
            str(inputs[idx]),
            str(explicit_ids[idx]) if ids_provided else None,
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
    result_track_id = await original_ainsert(
        self,
        accepted_inputs if isinstance(input, list) else accepted_inputs[0],
        split_by_character,
        split_by_character_only,
        accepted_ids if isinstance(ids, list) else (accepted_ids[0] if accepted_ids else None),
        accepted_paths if isinstance(file_paths, list) else accepted_paths[0],
        resolved_track_id,
    )
    await _apply_classification_metadata(self, accepted, inputs, explicit_ids, ids is not None)
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
    """
    original_ainsert = self.__class__._twin_original_ainsert
    active_hook = self.__class__._twin_classification_hook
    inputs = _as_list(input)
    paths = _as_list(file_paths)
    explicit_ids = _as_list(ids)

    if file_paths is None:
        return await original_ainsert(
            self,
            input,
            split_by_character,
            split_by_character_only,
            ids,
            file_paths,
            track_id,
        )
    if len(paths) != len(inputs):
        raise ValueError("Number of file paths must match the number of documents")
    if ids is not None and len(explicit_ids) != len(inputs):
        raise ValueError("Number of IDs must match the number of documents")

    accepted, rejected = _partition_inputs(active_hook, paths)

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
            self, accepted, inputs, explicit_ids, ids is not None
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


def install_lightrag_ingestion_hook(
    *,
    label_map_path: str | os.PathLike[str] | None = None,
    ceiling: str | None = None,
    audit_emit: AuditEmitter | None = None,
) -> None:
    """Patch ``LightRAG.ainsert`` so file ingests are classified pre-index.

    Rejected files are never passed to LightRAG's insert pipeline. Instead,
    a failed DocStatus row is written with ``metadata.classification`` and a
    ``classification-rejected`` activity event is emitted best-effort.
    """
    from lightrag import LightRAG

    hook = install_classification_hook(
        label_map_path=label_map_path,
        ceiling=ceiling,
        audit_emit=audit_emit,
    )
    setattr(LightRAG, "_twin_classification_hook", hook)

    if getattr(LightRAG, "_twin_classification_patched", False):
        return

    setattr(LightRAG, "_twin_original_ainsert", LightRAG.ainsert)
    LightRAG.ainsert = _patched_ainsert
    setattr(LightRAG, "_twin_classification_patched", True)
