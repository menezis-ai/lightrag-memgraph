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

from collections.abc import Iterator
from typing import Any, Literal

LIGHTRAG_NO_CONTEXT_MARKER = "[no-context]"
"""Suffix appended to the canned fail response by LightRAG.

Source: ``PROMPTS["fail_response"]`` in ``lightrag/prompt.py:220``.
"""

AnswerStatus = Literal[
    "grounded", "insufficient_information", "source_projection_failed"
]

ANSWER_STATUS_GROUNDED: AnswerStatus = "grounded"
ANSWER_STATUS_INSUFFICIENT: AnswerStatus = "insufficient_information"
# The answer was produced from real retrieval context (grounded), but its
# ``data.references`` could not be projected into the Twin sources contract
# (LightRAG envelope shape broke). The answer is still returned; sources are
# empty and the UI shows a "sources unavailable" cue. Distinct from
# ``insufficient_information`` (no usable context) and from a hard 500 (which
# would hide a usable answer behind a display-layer failure).
ANSWER_STATUS_SOURCE_PROJECTION_FAILED: AnswerStatus = "source_projection_failed"


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
        return (
            ANSWER_STATUS_INSUFFICIENT
            if self._detected
            else ANSWER_STATUS_GROUNDED
        )

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
            self._buffer = combined[idx + len(LIGHTRAG_NO_CONTEXT_MARKER):]
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
            trailing = self._buffer[idx + len(LIGHTRAG_NO_CONTEXT_MARKER):]
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
    """Return the first key in ``keys`` whose value is numeric, as float."""
    for key in keys:
        value = d.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _reference_score(
    chunks: list[dict[str, Any]],
    *,
    rank: int,
    total: int,
) -> float:
    for chunk in chunks:
        direct = _first_numeric_metric(chunk, _SCORE_KEYS)
        if direct is not None:
            return direct
        metrics = chunk.get("__metrics__")
        if isinstance(metrics, dict):
            nested = _first_numeric_metric(metrics, _SCORE_KEYS)
            if nested is not None:
                return nested
    if total <= 0:
        return 0.5
    return round(0.95 - 0.45 * (rank / max(total - 1, 1)), 3)


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
    - ``status == "grounded"`` otherwise.

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
    failure_reason = metadata.get("failure_reason") if isinstance(
        metadata, dict
    ) else None

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


def _chunk_count_label(chunk_count: int) -> str | None:
    if chunk_count > 1:
        return f"{chunk_count} chunks"
    if chunk_count == 1:
        return "1 chunk"
    return None


def _build_source_entry(
    reference, rank, total, chunks_by_ref, chunk_id_to_doc_id
) -> dict[str, Any]:
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
    doc_id = (
        chunk_id_to_doc_id.get(chunk_id)
        if (chunk_id and chunk_id_to_doc_id)
        else None
    )
    return {
        "n": n_value,
        "type": "file",
        "name": file_path or (chunk_id or f"reference-{n_value}"),
        "meta": _chunk_count_label(len(matching_chunks)),
        "score": _reference_score(matching_chunks, rank=rank, total=total),
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
    return [
        _build_source_entry(
            reference, rank, len(references), chunks_by_ref, chunk_id_to_doc_id
        )
        for rank, reference in enumerate(references)
    ]


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
