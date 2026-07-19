"""Adapter for LightRAG-specific assumptions that leak into the Twin
query surface.

Why this module exists
----------------------

Twin's ``/twin/api/query`` and ``/stream`` routes consume two LightRAG
contracts: the no-context fail signal and the ``aquery_llm`` retrieval
envelope. Both leak shape details that should not be scattered through
the route file — this module is the single auditable place for them.

1. **No-context marker** (``[no-context]``) appended to
   ``PROMPTS["fail_response"]`` (``lightrag/prompt.py:220``). Set in
   ``operate.py`` ×4 + ``lightrag.py`` ×1. Used as a defense-in-depth
   classification signal alongside the structured ``failure_reason``.
2. **``aquery_llm`` envelope** (``lightrag.py:2684``): single-call API
   returning ``status``, ``message``, ``data.entities/relationships/
   chunks/references``, ``metadata.failure_reason``, and
   ``llm_response.{content, response_iterator, is_streaming}``. The
   helper functions below project this into a ``(content, status)``
   tuple and a ``list[TwinRetrievalSource]`` that aligns with the
   ``[N]`` citations LightRAG asks the LLM to emit.

Doctrine
--------

These helpers fail loudly on unexpected shape (``GraphAnswerEnvelopeError``)
so the route code can decide to log + degrade gracefully (``sources=[]``)
**without** silently falling back to a separate vector retrieval pass.
The previous ``_build_sources`` helper has been demoted to
``_build_sources_legacy_fallback`` precisely because reusing it on the
nominal path was the structural lie of TR-RET-02 / audit C3.

Compatibility
-------------

- Tested against LightRAG ``1.4.9.11 / 1.4.11 / 1.4.12`` (the CI
  matrix). If LightRAG changes the marker placement or the
  ``aquery_llm`` envelope shape, the contract tests in
  ``tests/test_server/test_lightrag_compat.py`` and
  ``test_twin_query_routes.py`` will start failing on the integration
  matrix.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, Literal

LIGHTRAG_NO_CONTEXT_MARKER = "[no-context]"
"""Suffix appended to the canned fail response by LightRAG.

Source: ``PROMPTS["fail_response"]`` in ``lightrag/prompt.py:220``.
"""

AnswerStatus = Literal[
    "grounded",
    "insufficient_information",
    "source_projection_failed",
    "no_retrieval",
    "query_failed",
]

ANSWER_STATUS_GROUNDED: AnswerStatus = "grounded"
ANSWER_STATUS_INSUFFICIENT: AnswerStatus = "insufficient_information"
# The mode is sourceless by design: ``bypass`` calls the LLM directly with no
# retrieval, and ``only_need_context`` / ``only_need_prompt`` return the
# retrieved context / assembled prompt rather than a grounded answer. Distinct
# from ``grounded`` (which promises a sourced answer) and from
# ``insufficient_information`` (retrieval ran but found nothing usable): here no
# grounding was attempted, so the empty sources panel is expected, not a defect.
ANSWER_STATUS_NO_RETRIEVAL: AnswerStatus = "no_retrieval"
# The answer was produced from real retrieval context (grounded), but its
# ``data.references`` could not be projected into the Twin sources contract
# (LightRAG envelope shape broke). The answer is still returned; sources are
# empty and the UI shows a "sources unavailable" cue. Distinct from
# ``insufficient_information`` (no usable context) and from a hard 500 (which
# would hide a usable answer behind a display-layer failure).
ANSWER_STATUS_SOURCE_PROJECTION_FAILED: AnswerStatus = "source_projection_failed"
# A generic backend failure occurred (aquery_llm raised, or returned a
# ``status=failure`` envelope for any reason other than ``no_results``). On the
# non-stream ``/query`` this surfaces as a real HTTP 500; on ``/query/stream``
# the HTTP status is already committed to 200, so the failure is reported as a
# ``[query failed: …]`` token plus this status. Distinct from ``grounded`` (the
# answer is NOT trustworthy — it is an error notice, not a sourced answer),
# from ``insufficient_information`` (retrieval ran, found nothing usable), and
# from ``source_projection_failed`` (a real answer was produced, only the
# sources projection broke). Sources are always empty; the UI shows an error
# cue, never a citation affordance.
ANSWER_STATUS_QUERY_FAILED: AnswerStatus = "query_failed"


def classify_answer(answer: str) -> tuple[str, AnswerStatus]:
    """Classify a fully-materialised answer and strip the marker.

    Returns a ``(clean_answer, status)`` tuple. ``status`` is
    ``"insufficient_information"`` iff LightRAG's no-context marker
    appears anywhere in the string; the marker is removed from the
    cleaned answer so it never reaches the operator UI.

    ``str.replace`` covers all occurrences defensively — the canonical
    placement is a single suffix, but a future LightRAG variant
    putting the marker mid-string would still strip cleanly.
    """
    if LIGHTRAG_NO_CONTEXT_MARKER in answer:
        cleaned = answer.replace(LIGHTRAG_NO_CONTEXT_MARKER, "")
        return cleaned, ANSWER_STATUS_INSUFFICIENT
    return answer, ANSWER_STATUS_GROUNDED


class AnswerMarkerStripper:
    """Stateful chunk processor for the LightRAG no-context marker.

    Used during streaming so we never have to materialise the full
    answer to detect the marker (which would defeat streaming). Holds
    a rolling buffer of ``len(marker) - 1`` characters so the marker
    can land across a chunk boundary without being missed.

    Usage::

        stripper = AnswerMarkerStripper()
        async for chunk in upstream:
            for safe_text in stripper.feed(chunk):
                yield safe_text
        for safe_text in stripper.flush():
            yield safe_text
        status = stripper.status()
    """

    _BUFFER_KEEP = len(LIGHTRAG_NO_CONTEXT_MARKER) - 1

    def __init__(self) -> None:
        self._buffer = ""
        self._detected = False

    @property
    def detected(self) -> bool:
        """True once the marker has been observed."""
        return self._detected

    def status(self) -> AnswerStatus:
        """``"insufficient_information"`` iff the marker was seen, else
        ``"grounded"``."""
        return ANSWER_STATUS_INSUFFICIENT if self._detected else ANSWER_STATUS_GROUNDED

    def feed(self, chunk: str) -> Iterator[str]:
        """Consume the next chunk and yield the safe-to-emit slice.

        Holds back the last ``len(marker) - 1`` characters so a
        marker straddling two chunks is not lost. When the marker is
        found, it is dropped from the yielded text and the trailing
        text (after the marker) is buffered for the next call.
        """
        if not chunk:
            return
        combined = self._buffer + chunk
        idx = combined.find(LIGHTRAG_NO_CONTEXT_MARKER)
        if idx != -1:
            self._detected = True
            if idx > 0:
                yield combined[:idx]
            self._buffer = combined[idx + len(LIGHTRAG_NO_CONTEXT_MARKER) :]
            return
        keep = self._BUFFER_KEEP
        if len(combined) > keep:
            yield combined[:-keep]
            self._buffer = combined[-keep:]
        else:
            self._buffer = combined

    def flush(self) -> Iterator[str]:
        """Drain the rolling buffer at end-of-stream.

        Re-checks the residual buffer one last time: when the upstream
        ends mid-marker (unlikely but trivial to guard) we still detect
        and strip.
        """
        if not self._buffer:
            return
        idx = self._buffer.find(LIGHTRAG_NO_CONTEXT_MARKER)
        if idx != -1:
            self._detected = True
            if idx > 0:
                yield self._buffer[:idx]
            trailing = self._buffer[idx + len(LIGHTRAG_NO_CONTEXT_MARKER) :]
            if trailing:
                yield trailing
        else:
            yield self._buffer
        self._buffer = ""


# ----------------------------------------------------------------------
# aquery_llm envelope adapter (TR-RET-02 step 2 / audit C3)
# ----------------------------------------------------------------------


_FAILURE_REASON_NO_RESULTS = "no_results"


class GraphAnswerEnvelopeError(Exception):
    """Raised when ``aquery_llm`` returns a shape we cannot project.

    The route catches this and degrades to ``sources=[]`` with a logged
    warning — it MUST NOT fall back to ``_build_sources_legacy_fallback``
    because that reintroduces the structural lie TR-RET-02 step 2 is
    closing (sources reconstructed by a second vector pass instead of
    being the chunks LightRAG actually retrieved).
    """


_SCORE_KEYS = ("score", "similarity", "cosine_similarity")


def _first_numeric_metric(d: dict[str, Any], keys) -> float | None:
    """Return the first finite, non-boolean numeric metric, as float."""
    for key in keys:
        value = d.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                return numeric
    return None


def _reference_score(
    chunks: list[dict[str, Any]],
) -> float | None:
    """Return an explicit retrieval metric, never a rank-derived proxy.

    ``data.references`` does not carry a score of its own.  When LightRAG
    exposes a numeric metric on one of the chunks behind the reference, the
    first such metric (in retrieval-envelope order) is safe to project.  When
    it does not, ``None`` is the only epistemically honest value: synthesising
    ``0.95 .. 0.5`` from display rank made the wire field look like cosine
    similarity even though no similarity measurement existed.
    """
    for chunk in chunks:
        direct = _first_numeric_metric(chunk, _SCORE_KEYS)
        if direct is not None:
            return direct
        metrics = chunk.get("__metrics__")
        if isinstance(metrics, dict):
            nested = _first_numeric_metric(metrics, _SCORE_KEYS)
            if nested is not None:
                return nested
    return None


def classify_aquery_llm_result(
    result: dict[str, Any],
) -> tuple[str, AnswerStatus]:
    """Classify an ``aquery_llm`` envelope and extract the answer text.

    Returns ``(clean_answer, status)`` where:

    - ``status == "insufficient_information"`` iff the envelope says
      ``status == "failure"`` with ``metadata.failure_reason ==
      "no_results"`` — the canonical structured "no usable context"
      signal — OR the ``[no-context]`` marker is present (defense in
      depth for environments where LightRAG sets the marker without
      populating the failure_reason).
    - ``status == "grounded"`` otherwise, provisionally.  The route must still
      project at least one source from ``data.references`` before publishing a
      final ``grounded`` status; an empty or inconsistent projection is
      classified as ``source_projection_failed`` by
      :func:`query.response_sources._build_envelope_sources`.

    A generic backend failure (``status == "failure"`` with any other
    ``failure_reason`` — empty, ``query_failed``, …) raises
    ``GraphAnswerEnvelopeError`` so the route turns it into a real
    HTTP 500 instead of masquerading as "insufficient_information"
    (which would hide the backend issue from the operator).

    The marker is stripped from the returned content so the operator
    UI never sees ``[no-context]`` mid-bubble.
    """
    if not isinstance(result, dict):
        raise GraphAnswerEnvelopeError(
            f"aquery_llm returned non-dict {type(result).__name__}"
        )

    llm_response = result.get("llm_response") or {}
    raw_content = llm_response.get("content")
    answer: str = raw_content if isinstance(raw_content, str) else ""

    metadata = result.get("metadata") or {}
    failure_reason = (
        metadata.get("failure_reason") if isinstance(metadata, dict) else None
    )

    envelope_status = result.get("status")
    if envelope_status == "failure":
        if failure_reason == _FAILURE_REASON_NO_RESULTS:
            cleaned, _ = classify_answer(answer)
            return cleaned, ANSWER_STATUS_INSUFFICIENT
        raise GraphAnswerEnvelopeError(
            "aquery_llm reported failure: "
            f"reason={failure_reason!r} message={result.get('message')!r}"
        )

    # Even on a "success" envelope, the marker can still appear if
    # LightRAG injected the fail response without flipping
    # ``status``. Defense in depth.
    cleaned, marker_status = classify_answer(answer)
    if marker_status == ANSWER_STATUS_INSUFFICIENT:
        return cleaned, ANSWER_STATUS_INSUFFICIENT
    return cleaned, ANSWER_STATUS_GROUNDED


def _parse_envelope_references(result: dict[str, Any]):
    """Validate the envelope and return ``(references, chunks)`` or None.

    None means "nothing to project" (no ``data`` dict, or both references and
    chunks absent) — the caller returns empty sources. Raises
    ``GraphAnswerEnvelopeError`` on malformed (non-list) blocks.
    """
    if not isinstance(result, dict):
        raise GraphAnswerEnvelopeError(
            f"aquery_llm returned non-dict {type(result).__name__}"
        )
    data = result.get("data")
    if not isinstance(data, dict):
        # Legitimate envelopes (e.g. bypass mode with no retrieval) sit here.
        return None
    references = data.get("references")
    chunks = data.get("chunks")
    if references is None and chunks is None:
        return None
    if not isinstance(references, list):
        raise GraphAnswerEnvelopeError(
            f"data.references is {type(references).__name__}, expected list"
        )
    if chunks is None:
        chunks = []
    if not isinstance(chunks, list):
        raise GraphAnswerEnvelopeError(
            f"data.chunks is {type(chunks).__name__}, expected list"
        )
    return references, chunks


def _index_chunks_by_ref(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Pre-index chunks by reference_id for O(1) enrichment + chunk count."""
    chunks_by_ref: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        ref_id = str(chunk.get("reference_id") or "")
        if not ref_id:
            continue
        chunks_by_ref.setdefault(ref_id, []).append(chunk)
    return chunks_by_ref


def _first_chunk_id(matching_chunks: list[dict[str, Any]]) -> str | None:
    first_chunk = matching_chunks[0] if matching_chunks else None
    if isinstance(first_chunk, dict):
        chunk_id_raw = first_chunk.get("chunk_id")
        if isinstance(chunk_id_raw, str) and chunk_id_raw:
            return chunk_id_raw
    return None


def _first_doc_id(matching_chunks: list[dict[str, Any]]) -> str | None:
    """Return the first doc identifier carried by the retrieval envelope.

    LightRAG's ``data.chunks`` rows already include ``full_doc_id`` on the
    chunk-vdb path Twin filters in storage. The separate ``chunk_id -> doc_id``
    lookup is only a fallback for older/looser envelopes; relying on it first
    can make filtered retrieval appear source-less when that enrichment fails.
    """
    for chunk in matching_chunks:
        if not isinstance(chunk, dict):
            continue
        for key in ("full_doc_id", "doc_id"):
            doc_id_raw = chunk.get(key)
            if isinstance(doc_id_raw, str) and doc_id_raw:
                return doc_id_raw
    return None


def _chunk_count_label(chunk_count: int) -> str | None:
    if chunk_count > 1:
        return f"{chunk_count} chunks"
    if chunk_count == 1:
        return "1 chunk"
    return None


def _build_source_entry(reference, chunks_by_ref, chunk_id_to_doc_id) -> dict[str, Any]:
    """Project one ``data.references`` entry into a WebUI source dict."""
    if not isinstance(reference, dict):
        raise GraphAnswerEnvelopeError(
            f"reference entry is {type(reference).__name__}, expected dict"
        )
    ref_id_raw = reference.get("reference_id")
    if ref_id_raw is None:
        raise GraphAnswerEnvelopeError("reference missing reference_id")
    try:
        n_value = int(str(ref_id_raw))
    except (TypeError, ValueError) as exc:
        raise GraphAnswerEnvelopeError(
            f"reference_id={ref_id_raw!r} is not int-coercible"
        ) from exc

    file_path = reference.get("file_path") or ""
    matching_chunks = chunks_by_ref.get(str(ref_id_raw), [])
    chunk_id = _first_chunk_id(matching_chunks)
    doc_id = _first_doc_id(matching_chunks) or (
        chunk_id_to_doc_id.get(chunk_id) if (chunk_id and chunk_id_to_doc_id) else None
    )
    source_name = file_path or (chunk_id or f"reference-{n_value}")
    source_name_is_fallback = not file_path and not chunk_id
    return {
        "n": n_value,
        "type": "file",
        "name": source_name,
        "_lightrag_reference_name_fallback": source_name_is_fallback,
        "meta": _chunk_count_label(len(matching_chunks)),
        "score": _reference_score(matching_chunks),
        "doc_id": doc_id,
        "chunk_id": chunk_id,
    }


def build_sources_from_raw_data(
    result: dict[str, Any],
    chunk_id_to_doc_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Project ``aquery_llm`` ``data.references`` into the WebUI source list.

    Doctrine (audit C3): each LightRAG reference ID becomes one Twin
    source. We do NOT deduplicate by ``file_path`` because the LLM is
    prompted to cite as ``[N]`` where N is the reference_id — destroying
    a reference here would silently misalign the React port's
    ``parseAnswer`` citation parser with the source list.

    ``meta`` carries the count of chunks that contributed to that
    reference (informational), and ``chunk_id`` (best-effort) is the
    first chunk that produced this reference so the React port can
    drill into chunk views. ``doc_id`` is resolved via the optional
    ``chunk_id_to_doc_id`` map if provided.

    Raises ``GraphAnswerEnvelopeError`` on a missing or malformed
    ``data.references``/``data.chunks`` block so the route can choose
    to log + return empty sources, not silently rebuild them.
    """
    parsed = _parse_envelope_references(result)
    if parsed is None:
        return []
    references, chunks = parsed
    chunks_by_ref = _index_chunks_by_ref(chunks)
    entries = [
        _build_source_entry(reference, chunks_by_ref, chunk_id_to_doc_id)
        for reference in references
    ]
    # Defense-in-depth: ``n`` mirrors LightRAG's ``reference_id`` and the React
    # port collapses/keys sources by it. A duplicate would silently merge two
    # distinct sources into one and make a ``[N]`` citation ambiguous, so reject
    # the envelope here rather than project a corrupt sources list. Local
    # LightRAG mints unique reference_ids per file_path; this guards a future
    # upstream regression, not a path seen today.
    seen_n: set[int] = set()
    for entry in entries:
        n_value = entry["n"]
        if n_value in seen_n:
            raise GraphAnswerEnvelopeError(
                f"duplicate reference_id projected to n={n_value}"
            )
        seen_n.add(n_value)
    return entries


def collect_chunk_ids(result: dict[str, Any]) -> list[str]:
    """Best-effort list of chunk_ids referenced by the envelope.

    Used by the route to resolve ``chunk_id -> doc_id`` via the
    LightRAG DocStatus index, exactly like the legacy path did.
    Returns ``[]`` on unexpected shape rather than raising — the
    enrichment is informational, not a contract guarantee.
    """
    if not isinstance(result, dict):
        return []
    data = result.get("data")
    if not isinstance(data, dict):
        return []
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        return []
    out: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        chunk_id = chunk.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            out.append(chunk_id)
    return out


def is_streaming_envelope(result: dict[str, Any]) -> bool:
    """True iff the envelope is configured for streaming.

    A streaming envelope has ``llm_response.is_streaming`` true and a
    ``response_iterator`` to drain.
    """
    if not isinstance(result, dict):
        return False
    llm = result.get("llm_response")
    if not isinstance(llm, dict):
        return False
    return bool(llm.get("is_streaming")) and llm.get("response_iterator") is not None


__all__ = [
    "ANSWER_STATUS_GROUNDED",
    "ANSWER_STATUS_INSUFFICIENT",
    "AnswerMarkerStripper",
    "AnswerStatus",
    "GraphAnswerEnvelopeError",
    "LIGHTRAG_NO_CONTEXT_MARKER",
    "build_sources_from_raw_data",
    "classify_answer",
    "classify_aquery_llm_result",
    "collect_chunk_ids",
    "is_streaming_envelope",
]
