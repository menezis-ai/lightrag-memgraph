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
  is unavailable and the document is marked
  ``metadata.classification.reason = "no-source-path"``.
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
