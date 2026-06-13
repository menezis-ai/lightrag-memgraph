"""Unit tests for the LightRAG compat adapter (TR-RET-02).

The adapter lives in ``server/_lightrag_compat.py`` and isolates the
``[no-context]`` marker assumption so the route code reads as Twin
doctrine. These tests pin two things:

1. ``classify_answer`` round-trips the marker into a clean answer +
   the right status, deterministically.
2. ``AnswerMarkerStripper`` survives the chunk-boundary case Codex
   flagged — a marker split across two or three chunks must still be
   detected and stripped, not slipped through to the operator UI.
"""

from __future__ import annotations

from typing import Iterable

from twindb_lightrag_memgraph.server._lightrag_compat import (
    ANSWER_STATUS_GROUNDED,
    ANSWER_STATUS_INSUFFICIENT,
    AnswerMarkerStripper,
    LIGHTRAG_NO_CONTEXT_MARKER,
    classify_answer,
)


class TestClassifyAnswer:
    def test_grounded_when_no_marker(self):
        answer = "Memgraph supports vector search via MAGE."
        clean, status = classify_answer(answer)
        assert clean == answer
        assert status == ANSWER_STATUS_GROUNDED

    def test_insufficient_when_marker_present(self):
        # Canonical LightRAG fail response.
        answer = (
            "Sorry, I'm not able to provide an answer to that question."
            + LIGHTRAG_NO_CONTEXT_MARKER
        )
        clean, status = classify_answer(answer)
        assert clean == "Sorry, I'm not able to provide an answer to that question."
        assert status == ANSWER_STATUS_INSUFFICIENT
        # Marker is gone — the operator never sees the upstream
        # implementation detail.
        assert LIGHTRAG_NO_CONTEXT_MARKER not in clean

    def test_marker_anywhere_in_string_still_detected(self):
        # Defensive: a future variant putting the marker mid-string
        # should still be classified as insufficient.
        answer = f"Some prefix {LIGHTRAG_NO_CONTEXT_MARKER} some suffix"
        clean, status = classify_answer(answer)
        assert status == ANSWER_STATUS_INSUFFICIENT
        assert LIGHTRAG_NO_CONTEXT_MARKER not in clean
        assert clean == "Some prefix  some suffix"

    def test_idempotent_on_already_clean_string(self):
        answer = "Plain answer, no marker."
        clean, status = classify_answer(answer)
        # Re-classify the cleaned output — no change.
        clean2, status2 = classify_answer(clean)
        assert (clean, status) == (clean2, status2)


def _collect(it: Iterable[str]) -> str:
    return "".join(it)


class TestAnswerMarkerStripper:
    def test_marker_in_single_chunk_detected_and_stripped(self):
        stripper = AnswerMarkerStripper()
        out = _collect(
            stripper.feed(f"Sorry, no usable retrieval{LIGHTRAG_NO_CONTEXT_MARKER}")
        )
        out += _collect(stripper.flush())
        assert out == "Sorry, no usable retrieval"
        assert stripper.detected is True
        assert stripper.status() == ANSWER_STATUS_INSUFFICIENT

    def test_marker_split_across_two_chunks(self):
        # Codex's primary worry: the marker straddles a chunk
        # boundary. The rolling buffer must hold the partial prefix.
        stripper = AnswerMarkerStripper()
        # "Sorry, no answer.[no-co" + "ntext]"
        first = "Sorry, no answer.[no-co"
        second = "ntext]"
        chunks = [first, second]
        emitted = []
        for c in chunks:
            emitted.extend(stripper.feed(c))
        emitted.extend(stripper.flush())
        assert "".join(emitted) == "Sorry, no answer."
        assert stripper.detected is True

    def test_marker_split_across_three_chunks(self):
        # Even nastier: the marker is split into three sub-pieces.
        stripper = AnswerMarkerStripper()
        chunks = ["nothing here ", "[no-co", "n", "text]"]
        emitted = []
        for c in chunks:
            emitted.extend(stripper.feed(c))
        emitted.extend(stripper.flush())
        assert "".join(emitted) == "nothing here "
        assert stripper.detected is True

    def test_no_marker_across_many_chunks(self):
        stripper = AnswerMarkerStripper()
        chunks = [
            "Memgraph supports ",
            "vector search ",
            "via MAGE ",
            "with cosine similarity.",
        ]
        emitted = []
        for c in chunks:
            emitted.extend(stripper.feed(c))
        emitted.extend(stripper.flush())
        assert (
            "".join(emitted)
            == "Memgraph supports vector search via MAGE with cosine similarity."
        )
        assert stripper.detected is False
        assert stripper.status() == ANSWER_STATUS_GROUNDED

    def test_marker_at_chunk_start_does_not_leak(self):
        stripper = AnswerMarkerStripper()
        chunks = [
            "Sorry, no answer.",
            f"{LIGHTRAG_NO_CONTEXT_MARKER}",
        ]
        emitted = []
        for c in chunks:
            emitted.extend(stripper.feed(c))
        emitted.extend(stripper.flush())
        assert "".join(emitted) == "Sorry, no answer."
        assert stripper.detected is True

    def test_text_after_marker_still_yielded(self):
        # Defensive: if a future LightRAG appends a paragraph after
        # the marker, we want that paragraph to make it to the UI.
        stripper = AnswerMarkerStripper()
        out = _collect(
            stripper.feed(
                f"head {LIGHTRAG_NO_CONTEXT_MARKER} tail"
            )
        )
        out += _collect(stripper.flush())
        assert "head" in out
        assert "tail" in out
        assert LIGHTRAG_NO_CONTEXT_MARKER not in out

    def test_marker_in_residual_buffer_at_flush(self):
        # Edge case: the upstream emits exactly the buffer-keep tail
        # and ends. The marker shouldn't be the last thing in the
        # canned response (it's always followed by EOS in LightRAG),
        # but the helper handles it for robustness.
        stripper = AnswerMarkerStripper()
        chunks = ["a", LIGHTRAG_NO_CONTEXT_MARKER]
        emitted = []
        for c in chunks:
            emitted.extend(stripper.feed(c))
        emitted.extend(stripper.flush())
        assert "".join(emitted) == "a"
        assert stripper.detected is True

    def test_one_character_at_a_time(self):
        # Stress: the smallest possible chunks. Catches off-by-one in
        # the buffer-keep arithmetic.
        stripper = AnswerMarkerStripper()
        text = f"hi.{LIGHTRAG_NO_CONTEXT_MARKER}end"
        emitted = []
        for char in text:
            emitted.extend(stripper.feed(char))
        emitted.extend(stripper.flush())
        assert "".join(emitted) == "hi.end"
        assert stripper.detected is True

    def test_empty_chunk_is_a_noop(self):
        stripper = AnswerMarkerStripper()
        out = list(stripper.feed(""))
        assert out == []
        assert stripper.detected is False
