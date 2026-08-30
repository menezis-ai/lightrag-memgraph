"""Paragraph-anchor contract (docs/adr/008-paragraph-citation-anchor.md, phase A).

Covers the required batteries:

- splitter offsets: exact, verifiable, trimmed, on a varied corpus
  (markdown, tables, CRLF, single paragraph, empty chunk);
- collector §5.1 caps: bounded memory whatever the answer length,
  citation rate or absence of sentence separators, and fail-soft
  (incomplete evidence => no anchor, never a scored truncation);
- scorer: single anchor per citation, top-2 proximity penalty;
- route wiring: /query and /query/stream publish the SAME anchor
  (the parity test that would have caught scoring only the non-stream
  path), the marker-stripped text is what gets scored, enrichment
  failures never flip the fail-closed projection verdict, the payload
  carries offsets only, and a disconnected stream never scores.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph.server._lightrag_compat import AnswerMarkerStripper
from twindb_lightrag_memgraph.server.query import (
    paragraph_anchor as pa,
    response_sources,
)
from twindb_lightrag_memgraph.server.query.models import TwinQueryBody
from twindb_lightrag_memgraph.server.query.paragraph_anchor import (
    MAX_EVIDENCE_TOKENS_TOTAL,
    MAX_TOKEN_CHARS,
    MAX_TOKENS_PER_SENTENCE,
    CitationEvidence,
    CitationEvidenceCollector,
    collect_citation_evidence,
    compute_anchor,
    split_paragraphs,
)
from twindb_lightrag_memgraph.server.query.router import (
    _generate_twin_query_stream,
)
from twindb_lightrag_memgraph.server.query.streaming import _emit_answer_tokens
from twindb_lightrag_memgraph.server.twin_query_routes import (
    build_twin_query_router,
)

from .test_twin_query_routes import FakeDisconnectRequest, FakeRag

CHUNK_CONTENT = (
    "Supplier onboarding overview and scope of this runbook.\n"
    "\n"
    "The approval process requires two signatures from procurement leads "
    "before any purchase order is issued.\n"
    "\n"
    "Archival policy: purchase records are retained for ten fiscal years.\n"
    "\n"
    "Contact the tooling team for portal access problems."
)

CITED_ANSWER = (
    "The approval process requires two signatures from procurement leads [1]."
)


def _assert_span_invariants(content: str, paragraphs) -> None:
    for p in paragraphs:
        assert 0 <= p.start < p.end <= len(content)
        span = content[p.start : p.end]
        assert span
        assert span.strip() == span, "span must be trimmed to paragraph bounds"


class TestSplitParagraphs:
    def test_blank_line_split_offsets_are_exact(self):
        content = "alpha\n\nbeta gamma\n\ndelta"
        paragraphs = split_paragraphs(content)
        assert [(p.start, p.end) for p in paragraphs] == [(0, 5), (7, 17), (19, 24)]
        assert [content[p.start : p.end] for p in paragraphs] == [
            "alpha",
            "beta gamma",
            "delta",
        ]

    def test_crlf_offsets_index_the_original_string(self):
        content = "alpha\r\n\r\nbeta"
        paragraphs = split_paragraphs(content)
        assert [(p.start, p.end) for p in paragraphs] == [(0, 5), (9, 13)]
        assert content[9:13] == "beta"

    @pytest.mark.parametrize(
        "content",
        [
            CHUNK_CONTENT,
            # markdown: headings and list items
            "# Title\n\n- first item\n- second item\n\nClosing paragraph.",
            # markdown table kept as one block
            "| a | b |\n| - | - |\n| 1 | 2 |\n\nNext paragraph after table.",
            # CRLF document
            "first block\r\n\r\nsecond block\r\nstill second\r\n\r\nthird",
            # blank line polluted with spaces/tabs
            "one\n \t \ntwo",
            # single paragraph, no separator at all
            "a single paragraph without any blank line",
            # leading/trailing blank lines
            "\n\n  padded paragraph  \n\n",
        ],
    )
    def test_invariants_on_varied_corpus(self, content: str):
        _assert_span_invariants(content, split_paragraphs(content))

    def test_empty_and_whitespace_only(self):
        assert split_paragraphs("") == []
        assert split_paragraphs(" \n \n\t\n ") == []

    def test_single_paragraph_is_whole_trimmed_content(self):
        content = "  only one paragraph here  "
        (p,) = split_paragraphs(content)
        assert content[p.start : p.end] == "only one paragraph here"


class TestCitationEvidenceCollector:
    def test_evidence_per_citation_and_fragmentation_parity(self):
        fragments = [
            "The approval process requires two sig",
            "natures from procurement leads [",
            "1]. See also the archival policy [2].",
        ]
        whole = collect_citation_evidence("".join(fragments))
        collector = CitationEvidenceCollector()
        for fragment in fragments:
            collector.feed(fragment)
        fragmented = collector.finalize()

        assert set(whole) == {1, 2}
        assert {n: e.tokens for n, e in whole.items()} == {
            n: e.tokens for n, e in fragmented.items()
        }
        assert "signatures" in whole[1].tokens
        assert "archival" in whole[2].tokens
        assert not whole[1].incomplete and not whole[2].incomplete

    def test_every_sentence_cites_structures_stay_bounded(self):
        # More unique tokens than the global cap, every sentence citing [1].
        collector = CitationEvidenceCollector()
        for i in range(0, MAX_EVIDENCE_TOKENS_TOTAL + 500, 5):
            sentence = " ".join(f"tok{i + j}" for j in range(5)) + " [1]. "
            collector.feed(sentence)
            assert len(collector._sentence_tokens) <= MAX_TOKENS_PER_SENTENCE
            assert collector._total_tokens <= MAX_EVIDENCE_TOKENS_TOTAL
            assert len(collector._tail) <= pa._TAIL_CAP
        evidence = collector.finalize()
        assert evidence[1].incomplete
        assert len(evidence[1].tokens) <= MAX_EVIDENCE_TOKENS_TOTAL
        # Incomplete evidence must never produce an anchor (fail-soft).
        assert compute_anchor(CHUNK_CONTENT, evidence[1]) is None

    def test_no_sentence_separator_stays_bounded_no_anchor(self):
        # One endless "sentence": a citation followed by an unbroken token
        # run far longer than any cap, fed in small fragments.
        collector = CitationEvidenceCollector()
        collector.feed("cited [1] ")
        for _ in range(200):
            collector.feed("x" * 1024)
            assert len(collector._tail) <= pa._TAIL_CAP
            assert len(collector._sentence_tokens) <= MAX_TOKENS_PER_SENTENCE
        evidence = collector.finalize()
        assert evidence[1].incomplete
        assert compute_anchor(CHUNK_CONTENT, evidence[1]) is None

    def test_token_longer_than_cap_marks_citation_incomplete(self):
        answer = "a" * (MAX_TOKEN_CHARS + 1) + " overlong token near [1]."
        evidence = collect_citation_evidence(answer)
        assert evidence[1].incomplete
        assert compute_anchor(CHUNK_CONTENT, evidence[1]) is None

    def test_sentences_without_citations_collect_nothing(self):
        assert collect_citation_evidence("No citations at all. Nothing here.") == {}


class TestComputeAnchor:
    def test_anchor_slices_the_expected_paragraph(self):
        evidence = collect_citation_evidence(CITED_ANSWER)[1]
        anchor = compute_anchor(CHUNK_CONTENT, evidence)
        assert anchor is not None
        assert CHUNK_CONTENT[anchor["start"] : anchor["end"]] == (
            "The approval process requires two signatures from procurement "
            "leads before any purchase order is issued."
        )
        assert anchor["paragraph_idx"] == 1
        assert anchor["paragraph_count"] == 4
        assert anchor["method"] == pa.ANCHOR_METHOD_LEXICAL
        assert 0.0 < anchor["confidence"] <= 1.0

    def test_dispersed_citation_single_anchor_lower_confidence(self):
        mono = compute_anchor(CHUNK_CONTENT, collect_citation_evidence(CITED_ANSWER)[1])
        dispersed_answer = (
            "The approval process requires two signatures [1]. "
            "Archival policy retains purchase records for ten fiscal years [1]."
        )
        dispersed = compute_anchor(
            CHUNK_CONTENT, collect_citation_evidence(dispersed_answer)[1]
        )
        assert mono is not None
        # Contract (§5.1): ONE anchor at most, and when the evidence spreads
        # over two close paragraphs the published confidence must be
        # measurably lower than the single-paragraph case.
        if dispersed is not None:
            assert dispersed["confidence"] < mono["confidence"] - 0.2

    def test_unrelated_evidence_below_threshold_yields_none(self):
        evidence = CitationEvidence(frozenset({"zebra", "quasar", "kayak"}))
        assert compute_anchor(CHUNK_CONTENT, evidence) is None

    def test_empty_content_or_empty_evidence_yields_none(self):
        evidence = collect_citation_evidence(CITED_ANSWER)[1]
        assert compute_anchor("", evidence) is None
        assert compute_anchor(CHUNK_CONTENT, CitationEvidence(frozenset())) is None

    def test_single_paragraph_content(self):
        content = "The approval process requires two signatures."
        anchor = compute_anchor(content, collect_citation_evidence(CITED_ANSWER)[1])
        assert anchor is not None
        assert (anchor["paragraph_idx"], anchor["paragraph_count"]) == (0, 1)
        assert content[anchor["start"] : anchor["end"]] == content


OVERVIEW_CHUNK = (
    "Overview: the approval process is summarised below.\n"
    "\n"
    "See the detailed sections for specifics."
)


class TestComputeBestAnchor:
    """Cross-chunk election (PR #418 review, finding 1)."""

    def test_equal_scores_elect_the_first_candidate_in_envelope_order(self):
        evidence = CitationEvidence(frozenset(f"tok{i}" for i in range(10)))
        supporting = " ".join(f"tok{i}" for i in range(10))

        elected = pa.compute_best_anchor(
            [("c1", supporting), ("c2", supporting)], evidence
        )

        assert elected is not None
        assert elected[0] == "c1"

    def test_elects_the_chunk_actually_supporting_the_citation(self):
        evidence = collect_citation_evidence(CITED_ANSWER)[1]
        # Alone, the overview chunk clears the floor — the plausible-but-
        # wrong anchor the election exists to prevent.
        assert compute_anchor(OVERVIEW_CHUNK, evidence) is not None

        elected = pa.compute_best_anchor(
            [("chunk-1", OVERVIEW_CHUNK), ("chunk-2", CHUNK_CONTENT)], evidence
        )
        assert elected is not None
        chunk_id, anchor = elected
        assert chunk_id == "chunk-2"
        # Offsets index the WINNING chunk's content, and idx/count describe
        # that chunk's own paragraph list.
        assert CHUNK_CONTENT[anchor["start"] : anchor["end"]].startswith(
            "The approval process requires two signatures"
        )
        assert anchor["paragraph_count"] == 4

    def test_top2_penalty_applies_across_chunks(self):
        evidence = collect_citation_evidence(CITED_ANSWER)[1]
        _, alone = pa.compute_best_anchor([("c1", CHUNK_CONTENT)], evidence)
        # The same supporting paragraph duplicated in a second chunk makes
        # the election ambiguous: the runner-up in the OTHER chunk scores
        # ~equal, so the §5.1 penalty must collapse the confidence exactly
        # as an in-chunk runner-up would.
        duplicated = pa.compute_best_anchor(
            [("c1", CHUNK_CONTENT), ("c2", CHUNK_CONTENT)], evidence
        )
        assert duplicated is None or (
            duplicated[1]["confidence"] < alone["confidence"] - 0.3
        )

    def test_empty_or_incomplete_inputs(self):
        evidence = collect_citation_evidence(CITED_ANSWER)[1]
        assert pa.compute_best_anchor([], evidence) is None
        assert pa.compute_best_anchor([("c1", "")], evidence) is None
        assert (
            pa.compute_best_anchor(
                [("c1", CHUNK_CONTENT)],
                CitationEvidence(frozenset({"approval"}), incomplete=True),
            )
            is None
        )


@pytest.fixture()
async def make_client():
    async def _make(rag: FakeRag):
        app = FastAPI()
        app.include_router(build_twin_query_router(lambda: rag))
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://test")

    return _make


def _anchored_rag(**kwargs: Any) -> FakeRag:
    return FakeRag(
        chunks=[
            {
                "id": "chunk-1",
                "file_path": "/kb/supplier-policy.pdf",
                "content": CHUNK_CONTENT,
                "score": 0.9,
            }
        ],
        **kwargs,
    )


class SharedReferenceRag(FakeRag):
    """Realistic envelope shape: ONE reference_id for the whole file.

    LightRAG mints reference ids per *file*, so every chunk of a document
    shares the same ``[n]``. The default FakeRag one-reference-per-chunk
    shape masked PR #418 finding 1 — an anchor scored only against the
    first chunk of the reference.
    """

    def _build_envelope(self, *, is_streaming: bool) -> dict[str, Any]:
        envelope = super()._build_envelope(is_streaming=is_streaming)
        data = envelope["data"]
        file_path = ""
        for chunk in data["chunks"]:
            file_path = chunk.get("file_path") or file_path
            chunk["reference_id"] = "1"
        data["references"] = [{"reference_id": "1", "file_path": file_path}]
        return envelope


class TestAnchorRouteWiring:
    async def test_query_publishes_verifiable_anchor(self, make_client):
        rag = _anchored_rag(answer=CITED_ANSWER)
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "approval process?"})
        assert r.status_code == 200
        body = r.json()
        assert body["answer_status"] == "grounded"
        anchor = body["sources"][0]["anchor"]
        assert anchor is not None
        assert CHUNK_CONTENT[anchor["start"] : anchor["end"]].startswith(
            "The approval process"
        )
        assert anchor["method"] == "lexical_overlap"

    async def test_shared_reference_elects_the_supporting_chunk(self, make_client):
        # PR #418 review, finding 1: one reference, two chunks of the same
        # file, textual support ONLY in the second. The source must repoint
        # chunk_id at the winning chunk atomically with its anchor — never
        # anchor a lookalike paragraph in the first chunk.
        rag = SharedReferenceRag(
            answer=CITED_ANSWER,
            chunks=[
                {
                    "id": "chunk-overview",
                    "file_path": "/kb/supplier-policy.pdf",
                    "content": OVERVIEW_CHUNK,
                    "score": 0.9,
                },
                {
                    "id": "chunk-support",
                    "file_path": "/kb/supplier-policy.pdf",
                    "content": CHUNK_CONTENT,
                },
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "approval process?"})
        assert r.status_code == 200
        (source,) = r.json()["sources"]
        assert source["n"] == 1
        assert source["chunk_id"] == "chunk-support"
        anchor = source["anchor"]
        assert anchor is not None
        assert CHUNK_CONTENT[anchor["start"] : anchor["end"]].startswith(
            "The approval process requires two signatures"
        )

    async def test_shared_reference_without_anchor_keeps_first_chunk(self, make_client):
        # No citation in the answer -> no election: the projected chunk_id
        # stays LightRAG's first-chunk default, untouched.
        rag = SharedReferenceRag(
            answer="An answer citing nothing.",
            chunks=[
                {
                    "id": "chunk-overview",
                    "file_path": "/kb/supplier-policy.pdf",
                    "content": OVERVIEW_CHUNK,
                    "score": 0.9,
                },
                {
                    "id": "chunk-support",
                    "file_path": "/kb/supplier-policy.pdf",
                    "content": CHUNK_CONTENT,
                },
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "q"})
        (source,) = r.json()["sources"]
        assert source["chunk_id"] == "chunk-overview"
        assert source["anchor"] is None

    async def test_stream_and_non_stream_publish_the_same_anchor(self, make_client):
        # §5.1: the test that would have caught wiring the scorer only into
        # the path the UI does not use. Same envelope fixture, the streaming
        # answer arrives in deliberately awkward fragments.
        non_stream = _anchored_rag(answer=CITED_ANSWER)
        client = await make_client(non_stream)
        async with client:
            r = await client.post("/query", json={"query": "q"})
        anchor_non_stream = r.json()["sources"][0]["anchor"]

        streamed = _anchored_rag(
            stream_chunks=[
                "The approval process requires two sig",
                "natures from procurement leads [",
                "1].",
            ]
        )
        client = await make_client(streamed)
        async with client:
            r = await client.post("/query/stream", json={"query": "q"})
        events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        (sources_event,) = [e for e in events if e["type"] == "sources"]
        anchor_stream = sources_event["value"][0].get("anchor")

        assert anchor_non_stream is not None
        assert anchor_stream == anchor_non_stream

    async def test_collector_sees_stripped_text_only(self):
        # The no-context marker must never enter scorer evidence — the
        # collector is fed the stripper output, not the raw stream (§5.1).
        rag = _anchored_rag(
            stream_chunks=[
                "The approval process cited [1] here [no-co",
                "ntext] end.",
            ]
        )
        envelope = rag._build_envelope(is_streaming=True)
        collector = CitationEvidenceCollector()
        stripper = AnswerMarkerStripper()
        async for _ in _emit_answer_tokens(envelope, stripper, collector=collector):
            pass
        evidence = collector.finalize()
        assert stripper.detected
        assert "context" not in evidence[1].tokens
        assert "no" not in evidence[1].tokens
        assert "approval" in evidence[1].tokens

    async def test_enrichment_failure_never_flips_projection(
        self, make_client, monkeypatch
    ):
        def _boom(*_args: Any, **_kwargs: Any):
            raise RuntimeError("anchor scorer exploded")

        monkeypatch.setattr(response_sources, "compute_best_anchor", _boom)
        rag = _anchored_rag(answer=CITED_ANSWER)
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "q"})
        assert r.status_code == 200
        body = r.json()
        # Fail-soft: the grounded verdict and the sources list are intact,
        # only the anchors are missing.
        assert body["answer_status"] == "grounded"
        assert len(body["sources"]) == 1
        assert body["sources"][0]["anchor"] is None

    async def test_answer_without_citations_keeps_native_shape(self, make_client):
        rag = _anchored_rag(answer="An answer citing nothing at all.")
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "q"})
        body = r.json()
        assert body["answer_status"] == "grounded"
        assert body["sources"][0]["anchor"] is None

    async def test_anchor_payload_is_offsets_only(self, make_client):
        # §5 retention guard: a big chunk must not inflate sources — the
        # anchor carries offsets, never paragraph text (the WebUI persists
        # whole threads to localStorage under a quota-swallowing catch).
        big_content = "\n\n".join(
            f"Paragraph {i} filler text " + ("lorem ipsum " * 40) for i in range(20)
        )
        big_content += "\n\nThe approval process requires two signatures [end]."
        rag = FakeRag(
            answer="The approval process requires two signatures [1].",
            chunks=[
                {
                    "id": "chunk-big",
                    "file_path": "/kb/big.pdf",
                    "content": big_content,
                    "score": 0.9,
                }
            ],
        )
        client = await make_client(rag)
        async with client:
            r = await client.post("/query", json={"query": "q"})
        source = r.json()["sources"][0]
        anchor = source["anchor"]
        assert anchor is not None
        assert set(anchor) == {
            "start",
            "end",
            "paragraph_idx",
            "paragraph_count",
            "confidence",
            "method",
        }
        # The serialized source stays small whatever the chunk size.
        assert len(big_content) > 10_000
        assert len(json.dumps(source)) < 500

    async def test_disconnected_stream_never_scores(self, monkeypatch):
        finalize_calls: list[int] = []
        original_finalize = CitationEvidenceCollector.finalize

        def _spy(self):  # noqa: ANN001 - test spy
            finalize_calls.append(1)
            return original_finalize(self)

        monkeypatch.setattr(CitationEvidenceCollector, "finalize", _spy)

        class HangingRag(FakeRag):
            async def aquery_llm(self, query: str, *, param):
                import asyncio

                await asyncio.Event().wait()

        from lightrag.base import QueryParam

        rag = HangingRag()
        body = TwinQueryBody(query="q")
        request = FakeDisconnectRequest([True])
        events = [
            line
            async for line in _generate_twin_query_stream(
                rag, body, request, "default", QueryParam
            )
        ]
        # The retrieval stage is intentionally flushed before the expensive
        # lookup so the stream opens immediately. ClientDisconnectedDuringQuery
        # then exits without generation/status/sources events, and the anchor
        # pipeline is never invoked (§5.1).
        assert [json.loads(event) for event in events] == [
            {"type": "stage", "value": "retrieval"}
        ]
        assert finalize_calls == []
