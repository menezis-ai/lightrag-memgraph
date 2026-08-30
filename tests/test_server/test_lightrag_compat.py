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
        out = _collect(stripper.feed(f"head {LIGHTRAG_NO_CONTEXT_MARKER} tail"))
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


# ----------------------------------------------------------------------
# aquery_llm envelope helpers (TR-RET-02 step 2 / audit C3)
# ----------------------------------------------------------------------


import pytest

from twindb_lightrag_memgraph.server._lightrag_compat import (
    GraphAnswerEnvelopeError,
    build_sources_from_raw_data,
    classify_aquery_llm_result,
    collect_chunk_ids,
    is_streaming_envelope,
)


def _envelope(
    *,
    status: str = "success",
    content: str | None = "An answer.",
    failure_reason: str | None = None,
    references: list[dict] | None = None,
    chunks: list[dict] | None = None,
    is_streaming: bool = False,
    response_iterator: object | None = None,
) -> dict:
    return {
        "status": status,
        "message": "Query processed",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": chunks if chunks is not None else [],
            "references": references if references is not None else [],
        },
        "metadata": ({"failure_reason": failure_reason} if failure_reason else {}),
        "llm_response": {
            "content": content,
            "response_iterator": response_iterator,
            "is_streaming": is_streaming,
        },
    }


class TestClassifyAqueryLlmResult:
    def test_grounded_on_success_envelope(self):
        result = _envelope(content="An answer.")
        cleaned, status = classify_aquery_llm_result(result)
        assert cleaned == "An answer."
        assert status == ANSWER_STATUS_GROUNDED

    def test_insufficient_on_failure_no_results(self):
        result = _envelope(
            status="failure",
            content=(
                "Sorry, I'm not able to provide an answer to that question."
                + LIGHTRAG_NO_CONTEXT_MARKER
            ),
            failure_reason="no_results",
        )
        cleaned, status = classify_aquery_llm_result(result)
        assert status == ANSWER_STATUS_INSUFFICIENT
        # Marker stripped from the cleaned answer.
        assert LIGHTRAG_NO_CONTEXT_MARKER not in cleaned

    def test_raises_on_generic_backend_failure(self):
        # ``failure_reason != "no_results"`` must NOT be masked as
        # insufficient — the route turns this into a real 500.
        result = _envelope(
            status="failure",
            failure_reason="query_failed",
            content=None,
        )
        with pytest.raises(GraphAnswerEnvelopeError):
            classify_aquery_llm_result(result)

    def test_raises_on_failure_with_empty_reason(self):
        # Defensive: a failure envelope without ``failure_reason`` is
        # an honest backend error, not a "no usable context" signal.
        result = _envelope(
            status="failure",
            failure_reason=None,
            content=None,
        )
        with pytest.raises(GraphAnswerEnvelopeError):
            classify_aquery_llm_result(result)

    def test_defense_in_depth_marker_overrides_success_status(self):
        # If LightRAG returns ``status=success`` but injects the
        # marker (older code paths), we still mark insufficient.
        result = _envelope(
            content=(
                "Sorry, I'm not able to provide an answer." + LIGHTRAG_NO_CONTEXT_MARKER
            ),
        )
        cleaned, status = classify_aquery_llm_result(result)
        assert status == ANSWER_STATUS_INSUFFICIENT
        assert LIGHTRAG_NO_CONTEXT_MARKER not in cleaned

    def test_raises_on_non_dict_input(self):
        with pytest.raises(GraphAnswerEnvelopeError):
            classify_aquery_llm_result("not a dict")  # type: ignore[arg-type]


class TestBuildSourcesFromRawData:
    """The mapping is intentionally non-deduplicating: every LightRAG
    reference_id becomes one Twin source so the ``[N]`` citations
    LightRAG asks the LLM to emit stay aligned with ``sources[i].n``."""

    def test_one_source_per_reference_id_in_order(self):
        result = _envelope(
            references=[
                {"reference_id": "1", "file_path": "/demo/runbooks/oracle.pdf"},
                {"reference_id": "2", "file_path": "/demo/runbooks/rhel.pdf"},
            ],
            chunks=[
                {
                    "reference_id": "1",
                    "chunk_id": "c-aa",
                    "file_path": "/demo/runbooks/oracle.pdf",
                },
                {
                    "reference_id": "2",
                    "chunk_id": "c-bb",
                    "file_path": "/demo/runbooks/rhel.pdf",
                },
            ],
        )
        sources = build_sources_from_raw_data(result)
        assert [s["n"] for s in sources] == [1, 2]
        assert sources[0]["name"] == "/demo/runbooks/oracle.pdf"
        assert sources[0]["chunk_id"] == "c-aa"
        assert sources[1]["chunk_id"] == "c-bb"

    def test_does_not_dedup_when_two_references_share_file_path(self):
        # Audit guardrail: dropping a reference because its file_path
        # is already present would misalign the LLM's ``[N]`` markers
        # with the source list. Two references for the same PDF must
        # produce two source entries with stable ``n`` ids.
        result = _envelope(
            references=[
                {"reference_id": "1", "file_path": "/a.pdf"},
                {"reference_id": "2", "file_path": "/a.pdf"},
            ],
            chunks=[
                {"reference_id": "1", "chunk_id": "c-1", "file_path": "/a.pdf"},
                {"reference_id": "2", "chunk_id": "c-2", "file_path": "/a.pdf"},
            ],
        )
        sources = build_sources_from_raw_data(result)
        assert len(sources) == 2
        assert [s["n"] for s in sources] == [1, 2]
        assert {s["chunk_id"] for s in sources} == {"c-1", "c-2"}

    def test_meta_carries_chunks_count(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {"reference_id": "1", "chunk_id": "c-1", "file_path": "/a.pdf"},
                {"reference_id": "1", "chunk_id": "c-2", "file_path": "/a.pdf"},
                {"reference_id": "1", "chunk_id": "c-3", "file_path": "/a.pdf"},
            ],
        )
        sources = build_sources_from_raw_data(result)
        assert sources[0]["meta"] == "3 chunks"

    def test_meta_one_chunk_uses_singular(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {"reference_id": "1", "chunk_id": "c-1", "file_path": "/a.pdf"},
            ],
        )
        sources = build_sources_from_raw_data(result)
        assert sources[0]["meta"] == "1 chunk"

    def test_missing_metric_is_null_not_a_rank_derived_similarity(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {"reference_id": "1", "chunk_id": "c-1", "file_path": "/a.pdf"},
            ],
        )

        sources = build_sources_from_raw_data(result)

        assert sources[0]["score"] is None
        assert sources[0]["retrieval_origin"] is None

    def test_request_trace_restores_measured_vector_similarity(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {"reference_id": "1", "chunk_id": "c-1", "file_path": "/a.pdf"},
            ],
        )

        sources = build_sources_from_raw_data(result, None, {"c-1": 0.837})

        assert sources[0]["score"] == 0.837
        assert sources[0]["retrieval_origin"] == "vector"

    def test_empty_request_trace_marks_unscored_reference_as_graph_sourced(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {"reference_id": "1", "chunk_id": "c-1", "file_path": "/a.pdf"},
            ],
        )

        sources = build_sources_from_raw_data(result, None, {})

        assert sources[0]["score"] is None
        assert sources[0]["retrieval_origin"] == "graph"

    def test_envelope_metric_takes_precedence_over_request_trace(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {
                    "reference_id": "1",
                    "chunk_id": "c-1",
                    "file_path": "/a.pdf",
                    "similarity": 0.75,
                },
            ],
        )

        sources = build_sources_from_raw_data(result, None, {"c-1": 0.62})

        assert sources[0]["score"] == 0.75
        assert sources[0]["retrieval_origin"] == "vector"

    @pytest.mark.parametrize("invalid_score", [True, float("nan"), float("inf")])
    def test_invalid_metric_is_not_projected_as_a_score(self, invalid_score):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {
                    "reference_id": "1",
                    "chunk_id": "c-1",
                    "file_path": "/a.pdf",
                    "score": invalid_score,
                },
            ],
        )

        sources = build_sources_from_raw_data(result)

        assert sources[0]["score"] is None

    def test_chunk_id_to_doc_id_lookup(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {"reference_id": "1", "chunk_id": "c-1", "file_path": "/a.pdf"},
            ],
        )
        sources = build_sources_from_raw_data(result, {"c-1": "doc-A"})
        assert sources[0]["doc_id"] == "doc-A"

    def test_full_doc_id_enriches_doc_id_without_lookup(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": "/a.pdf"}],
            chunks=[
                {
                    "reference_id": "1",
                    "chunk_id": "c-1",
                    "full_doc_id": "doc-A",
                    "file_path": "/a.pdf",
                },
            ],
        )
        sources = build_sources_from_raw_data(result)
        assert sources[0]["doc_id"] == "doc-A"

    def test_missing_data_block_returns_empty_list(self):
        # Bypass mode produces an envelope without ``data.references``
        # — legitimate, surface empty sources, do NOT raise.
        sources = build_sources_from_raw_data(
            {"status": "success", "metadata": {}, "llm_response": {}}
        )
        assert sources == []

    def test_data_without_references_or_chunks_returns_empty(self):
        result = {
            "status": "success",
            "data": {"entities": [], "relationships": []},
            "metadata": {},
            "llm_response": {},
        }
        assert build_sources_from_raw_data(result) == []

    def test_raises_when_references_is_not_a_list(self):
        # Defensive — the caller logs + degrades to empty rather than
        # silently calling _build_sources_legacy_fallback.
        result = _envelope(references=[], chunks=[])
        result["data"]["references"] = "not a list"  # type: ignore[index]
        with pytest.raises(GraphAnswerEnvelopeError):
            build_sources_from_raw_data(result)

    def test_raises_when_reference_id_is_missing(self):
        result = _envelope(
            references=[{"file_path": "/a.pdf"}],
            chunks=[],
        )
        with pytest.raises(GraphAnswerEnvelopeError):
            build_sources_from_raw_data(result)

    def test_raises_when_reference_id_is_not_int_coercible(self):
        result = _envelope(
            references=[{"reference_id": "abc", "file_path": "/a.pdf"}],
            chunks=[],
        )
        with pytest.raises(GraphAnswerEnvelopeError):
            build_sources_from_raw_data(result)

    def test_raises_on_duplicate_reference_id(self):
        # Defense-in-depth: two references coercing to the same ``n`` would
        # silently collapse in the React port (it keys sources by ``n``) and
        # make a ``[N]`` citation ambiguous. Reject the envelope so the route
        # degrades to source_projection_failed rather than ship a corrupt list.
        # NB: distinct ids sharing a file_path is the legitimate case kept by
        # test_does_not_dedup_when_two_references_share_file_path — this guards
        # only genuinely duplicate reference_ids.
        result = _envelope(
            references=[
                {"reference_id": "1", "file_path": "/a.pdf"},
                {"reference_id": "1", "file_path": "/b.pdf"},
            ],
            chunks=[],
        )
        with pytest.raises(GraphAnswerEnvelopeError):
            build_sources_from_raw_data(result)

    def test_name_falls_back_to_chunk_id_when_file_path_empty(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": ""}],
            chunks=[
                {"reference_id": "1", "chunk_id": "c-aa", "file_path": ""},
            ],
        )
        sources = build_sources_from_raw_data(result)
        assert sources[0]["name"] == "c-aa"
        assert sources[0]["_lightrag_reference_name_fallback"] is False

    def test_name_falls_back_to_reference_id_when_no_file_path_nor_chunk_id(self):
        result = _envelope(
            references=[{"reference_id": "1", "file_path": ""}],
            chunks=[{"reference_id": "1", "file_path": ""}],
        )
        sources = build_sources_from_raw_data(result)
        assert sources[0]["name"] == "reference-1"
        assert sources[0]["_lightrag_reference_name_fallback"] is True


class TestCollectChunkIds:
    def test_collects_unique_chunk_ids(self):
        result = _envelope(
            chunks=[
                {"reference_id": "1", "chunk_id": "c-1"},
                {"reference_id": "1", "chunk_id": "c-2"},
                {"reference_id": "2", "chunk_id": "c-3"},
            ],
        )
        assert collect_chunk_ids(result) == ["c-1", "c-2", "c-3"]

    def test_returns_empty_on_unexpected_shape(self):
        # Non-raising: enrichment is informational, the route still
        # works without doc_id resolution.
        assert collect_chunk_ids({"data": "not a dict"}) == []
        assert collect_chunk_ids("not a dict") == []  # type: ignore[arg-type]


class TestIsStreamingEnvelope:
    def test_true_when_is_streaming_and_iterator_present(self):
        result = _envelope(is_streaming=True, response_iterator=iter(["chunk"]))
        assert is_streaming_envelope(result) is True

    def test_false_when_is_streaming_false(self):
        result = _envelope(is_streaming=False)
        assert is_streaming_envelope(result) is False

    def test_false_when_iterator_missing(self):
        result = _envelope(is_streaming=True, response_iterator=None)
        assert is_streaming_envelope(result) is False
