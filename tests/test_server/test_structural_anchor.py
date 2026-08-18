"""Structural anchor election + projection (phase B1), fail-soft contracts.

The lexical election keeps its own suite (test_paragraph_anchor.py); here:
the shared election over ingestion-persisted ``twin_block_boundaries``, the
malformed-sidecar refusal, the batch boundary read, and the
structural-first / lexical-fallback order inside source enrichment.
"""

from __future__ import annotations

from typing import Any

from twindb_lightrag_memgraph.server.query import response_sources
from twindb_lightrag_memgraph.server.query.paragraph_anchor import (
    ANCHOR_METHOD_LEXICAL,
    ANCHOR_METHOD_STRUCTURAL,
    CitationEvidence,
    collect_citation_evidence,
    compute_best_structural_anchor,
)

_CHUNK = (
    "Atlas failover requires a manual switchback approval.\n\n"
    "Unrelated tail material about the cafeteria menu and parking rules."
)
_BOUNDARIES = [
    {"block_id": "b-1", "start": 0, "end": 53},
    {"block_id": "b-2", "start": 55, "end": len(_CHUNK)},
]


def _evidence() -> CitationEvidence:
    evidence = collect_citation_evidence(
        "Atlas failover requires a manual switchback approval [1]."
    )
    assert 1 in evidence
    return evidence[1]


def test_structural_election_publishes_block_offsets_and_method():
    elected = compute_best_structural_anchor(
        [("chunk-1", _CHUNK, _BOUNDARIES)], _evidence()
    )
    assert elected is not None
    chunk_id, anchor = elected
    assert chunk_id == "chunk-1"
    assert anchor["method"] == ANCHOR_METHOD_STRUCTURAL
    assert (anchor["start"], anchor["end"]) == (0, 53)
    assert anchor["paragraph_idx"] == 0
    assert anchor["paragraph_count"] == 2
    assert _CHUNK[anchor["start"] : anchor["end"]].startswith("Atlas failover")


def test_malformed_boundaries_yield_no_structural_anchor():
    # Plan §6 point 3: malformed sidecar data must never fabricate an
    # anchor. One bad entry poisons the whole chunk's boundary list.
    for bad in (
        [{"block_id": "b", "start": 5, "end": 2}],
        [{"block_id": "b", "start": 0, "end": 10_000}],
        [{"block_id": "b", "start": "0", "end": 10}],
        [{"block_id": "b", "start": 0, "end": 20}, "not-a-dict"],
    ):
        assert (
            compute_best_structural_anchor([("chunk-1", _CHUNK, bad)], _evidence())
            is None
        )


class _FakeTextChunks:
    def __init__(self, rows: Any):
        self._rows = rows

    async def get_by_ids(self, chunk_ids):
        if isinstance(self._rows, Exception):
            raise self._rows
        return self._rows


class _FakeRag:
    def __init__(self, rows: Any):
        self.text_chunks = _FakeTextChunks(rows)


async def test_fetch_chunk_boundaries_maps_and_filters():
    rag = _FakeRag(
        {
            "c1": {"twin_block_boundaries": _BOUNDARIES},
            "c2": {"content": "no boundaries here"},
        }
    )
    out = await response_sources._fetch_chunk_boundaries(rag, ["c1", "c2", "c3"])
    assert out == {"c1": _BOUNDARIES}


async def test_fetch_chunk_boundaries_fails_soft():
    rag = _FakeRag(RuntimeError("storage down"))
    assert await response_sources._fetch_chunk_boundaries(rag, ["c1"]) == {}
    assert await response_sources._fetch_chunk_boundaries(rag, []) == {}


def _envelope_with_chunk() -> dict[str, Any]:
    return {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "reference_id": "1",
                    "chunk_id": "chunk-1",
                    "content": _CHUNK,
                    "file_path": "atlas.docx",
                }
            ],
            "references": [{"reference_id": "1", "file_path": "atlas.docx"}],
        },
    }


def _source() -> dict[str, Any]:
    return {"n": 1, "chunk_id": "chunk-1", "doc": "atlas.docx"}


def test_enrichment_prefers_structural_then_falls_back_to_lexical():
    evidence = {1: _evidence()}

    structural = [_source()]
    response_sources._enrich_sources_with_anchors(
        structural, _envelope_with_chunk(), evidence, {"chunk-1": _BOUNDARIES}
    )
    assert structural[0]["anchor"]["method"] == ANCHOR_METHOD_STRUCTURAL

    # No boundaries for the elected chunk: phase A behavior, unchanged.
    lexical = [_source()]
    response_sources._enrich_sources_with_anchors(
        lexical, _envelope_with_chunk(), evidence, {}
    )
    assert lexical[0]["anchor"]["method"] == ANCHOR_METHOD_LEXICAL

    # Malformed boundaries: structural refuses, lexical still anchors.
    fallback = [_source()]
    response_sources._enrich_sources_with_anchors(
        fallback,
        _envelope_with_chunk(),
        evidence,
        {"chunk-1": [{"block_id": "b", "start": 9, "end": 1}]},
    )
    assert fallback[0]["anchor"]["method"] == ANCHOR_METHOD_LEXICAL
