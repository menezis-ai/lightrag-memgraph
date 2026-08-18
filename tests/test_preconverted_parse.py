"""Unit contract of the B1 preconverted-parse seam (no Memgraph, no 1.5.x).

The 1.5.4 end-to-end proof lives in tests/test_lightrag_15_sidecar_probe.py
(canary line). Here: the pure boundary math, the capability gates, and the
raw-vs-pending_parse switch of ``registry._enqueue_converted`` on every
matrix version.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from twindb_lightrag_memgraph import _preconverted_parse
from twindb_lightrag_memgraph.patches import registry


@pytest.fixture(autouse=True)
def _fresh_capability(monkeypatch):
    monkeypatch.delenv("TWIN_PRECONVERTED_PARSE", raising=False)
    _preconverted_parse._reset_capability_cache_for_tests()
    yield
    _preconverted_parse._reset_capability_cache_for_tests()


class TestDeriveBlockBoundaries:
    BLOCKS = {"b1": (0, 40), "b2": (42, 90), "b3": (200, 240)}

    def test_intersections_are_chunk_local_half_open(self):
        # Chunk spans document [30, 80): overlaps the tail of b1 and the
        # head of b2; b3 lies fully outside.
        content = "x" * 50
        out = _preconverted_parse.derive_block_boundaries(
            content, (30, 80), ["b1", "b2", "b3"], self.BLOCKS
        )
        assert out == [
            {"block_id": "b1", "start": 0, "end": 10},
            {"block_id": "b2", "start": 12, "end": 50},
        ]

    def test_unknown_ref_and_bad_span_are_skipped(self):
        content = "x" * 10
        assert (
            _preconverted_parse.derive_block_boundaries(
                content, (0, 10), ["missing"], self.BLOCKS
            )
            == []
        )
        # Inverted source span: nothing derivable.
        assert (
            _preconverted_parse.derive_block_boundaries(
                content, (10, 0), ["b1"], self.BLOCKS
            )
            == []
        )

    def test_boundary_exceeding_content_is_dropped(self):
        # Span claims 40 code points but the content only has 5: a wrong
        # boundary is worse than none.
        out = _preconverted_parse.derive_block_boundaries(
            "x" * 5, (0, 40), ["b1"], self.BLOCKS
        )
        assert out == []


class TestCapabilityGates:
    def test_parser_extract_accepts_lightrag_runtime_keyword(self):
        parser_class = _preconverted_parse.TwinPreconvertedMarkdownParser
        if parser_class is None:
            pytest.skip("LightRAG native parser is unavailable on this matrix line")

        signature = inspect.signature(parser_class.extract)
        assert "runtime" in signature.parameters
        assert signature.parameters["runtime"].kind is inspect.Parameter.KEYWORD_ONLY

    def test_parser_extract_ignores_runtime_and_uses_preconverted_body(
        self, monkeypatch, tmp_path
    ):
        parser_class = _preconverted_parse.TwinPreconvertedMarkdownParser
        if parser_class is None:
            pytest.skip("LightRAG native parser is unavailable on this matrix line")

        parser = object.__new__(parser_class)
        expected = ([{"text": "converted"}], {"assets": []}, {"meta": True})
        monkeypatch.setattr(
            parser,
            "_extract_text",
            lambda markdown, bundle_root: (
                expected if (markdown, bundle_root) == ("# converted", None) else None
            ),
        )
        token = _preconverted_parse._markdown_body.set("# converted")
        try:
            result = parser.extract(
                Path("original.pdf"),
                parsed_dir=tmp_path,
                asset_dir=tmp_path,
                base_name="original",
                runtime=object(),
            )
        finally:
            _preconverted_parse._markdown_body.reset(token)

        assert result == expected

    def test_kill_switch_wins_over_capability(self, monkeypatch):
        monkeypatch.setenv("TWIN_PRECONVERTED_PARSE", "off")
        assert _preconverted_parse.is_available() is False
        assert _preconverted_parse.ensure_parser_registered() is False

    def test_absent_parser_class_means_unavailable(self, monkeypatch):
        monkeypatch.setattr(_preconverted_parse, "TwinPreconvertedMarkdownParser", None)
        assert _preconverted_parse.is_available() is False

    def test_activate_is_noop_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(_preconverted_parse, "is_available", lambda: False)
        assert _preconverted_parse.activate() is False


class _FakeEnqueueRag:
    def __init__(self):
        self.calls = []

    async def apipeline_enqueue_documents(self, markdown, **kwargs):
        self.calls.append((markdown, kwargs))


class TestEnqueueConvertedSwitch:
    async def test_raw_contract_when_seam_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            registry._preconverted_parse, "ensure_parser_registered", lambda: False
        )
        rag = _FakeEnqueueRag()
        path = tmp_path / "report.docx"
        ok, track = await registry._enqueue_converted(rag, path, "# md", "t1", False)
        assert (ok, track) == (True, "t1")
        assert rag.calls == [("# md", {"file_paths": "report.docx", "track_id": "t1"})]

    async def test_out_of_list_suffix_stays_raw_even_with_seam_active(
        self, monkeypatch, tmp_path
    ):
        # TWIN_CONVERT_FORMATS is operator-extensible while PARSER_SUFFIXES
        # is frozen into the ParserSpec: 1.5.x rejects an explicit engine on
        # an unsupported suffix (doc FAILED), so out-of-list stays raw.
        monkeypatch.setattr(
            registry._preconverted_parse, "ensure_parser_registered", lambda: True
        )
        rag = _FakeEnqueueRag()
        path = tmp_path / "legacy.rtf"
        ok, track = await registry._enqueue_converted(rag, path, "# md", "t3", False)
        assert (ok, track) == (True, "t3")
        assert rag.calls == [("# md", {"file_paths": "legacy.rtf", "track_id": "t3"})]

    async def test_pending_parse_contract_when_seam_active(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            registry._preconverted_parse, "ensure_parser_registered", lambda: True
        )
        rag = _FakeEnqueueRag()
        path = tmp_path / "report.docx"
        ok, track = await registry._enqueue_converted(rag, path, "# md", "t2", False)
        assert (ok, track) == (True, "t2")
        assert rag.calls == [
            (
                "# md",
                {
                    "file_paths": "report.docx",
                    "track_id": "t2",
                    "docs_format": "pending_parse",
                    "parse_engine": _preconverted_parse.PARSER_ENGINE,
                },
            )
        ]
