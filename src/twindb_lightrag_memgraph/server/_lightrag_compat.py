"""Adapter for LightRAG-specific assumptions that leak into the Twin
query surface.

Why this module exists
----------------------

LightRAG signals a "no usable context found" answer by appending a
deterministic marker string ``[no-context]`` to its canned fail
response (``PROMPTS["fail_response"]`` in
``lightrag/prompt.py:220``). The marker is set by LightRAG's own code
in five paths (``operate.py`` ×4, ``lightrag.py`` ×1) and is the
canonical machine-readable signal that distinguishes "the retrieval
pipeline gave up" from "the LLM produced a hedged but grounded
answer".

We use that marker to set ``answer_status`` on the Twin response so
the React port can suppress the Sources panel honestly instead of
parsing the LLM prose (TR-RET-02).

This module isolates the dependency on that upstream detail behind
two helpers so the rest of the route code reads as Twin doctrine
("classify, strip, set status") and the LightRAG-specific knowledge
lives in one place that can be revisited if/when LightRAG renames or
removes the marker.

Compatibility
-------------

- Tested against LightRAG ``1.4.9.11 / 1.4.11 / 1.4.12`` (the CI
  matrix). If LightRAG removes or relocates the marker, the
  classification silently defaults to ``"grounded"`` — the Twin
  contract tests at ``tests/test_server/test_twin_query_routes.py``
  will start failing on the integration matrix because the response
  no longer ships ``answer_status="insufficient_information"`` for
  queries that have no context. Bump the matrix exclusion list in
  ``.forgejo/workflows/ci.yml`` accordingly if that happens.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

LIGHTRAG_NO_CONTEXT_MARKER = "[no-context]"
"""Suffix appended to the canned fail response by LightRAG.

Source: ``PROMPTS["fail_response"]`` in ``lightrag/prompt.py:220``.
"""

AnswerStatus = Literal["grounded", "insufficient_information"]

ANSWER_STATUS_GROUNDED: AnswerStatus = "grounded"
ANSWER_STATUS_INSUFFICIENT: AnswerStatus = "insufficient_information"


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


__all__ = [
    "ANSWER_STATUS_GROUNDED",
    "ANSWER_STATUS_INSUFFICIENT",
    "AnswerMarkerStripper",
    "AnswerStatus",
    "LIGHTRAG_NO_CONTEXT_MARKER",
    "classify_answer",
]
