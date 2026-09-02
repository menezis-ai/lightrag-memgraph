"""Unit contract of the B1 preconverted-parse seam (no Memgraph).

Covers the pure boundary math, the LightRAG 1.5.6 capability gates, and the
raw-vs-pending_parse switch of ``registry._enqueue_converted``.
"""

from __future__ import annotations

import inspect
import json
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


# ── Boundary backfill (ADR 008 phase B1) ───────────────────────────────
#
# ``_backfill_with_boundaries`` is the ONE place where twin_block_boundaries
# are produced, and it was the largest uncovered surface of this module. It
# runs against the REAL upstream backfill and the REAL private span helpers:
# the whole point of the seam is that it stays glued to those three private
# functions, so a fake would assert nothing about the coupling that can drift.

_BLOCK_ONE = "Block one body."
_BLOCK_TWO = "Block two body."
_MERGED = f"{_BLOCK_ONE}\n\n{_BLOCK_TWO}"


def _blocks_file(tmp_path: Path) -> str:
    """A blocks.jsonl in ``writer.write_sidecar`` shape: meta row then content."""
    path = tmp_path / "blocks.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "meta", "version": 1}),
                json.dumps({"type": "content", "blockid": "b1", "content": _BLOCK_ONE}),
                json.dumps({"type": "content", "blockid": "b2", "content": _BLOCK_TWO}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _whole_document_chunk() -> dict:
    return {
        "chunk_order_index": 0,
        "content": _MERGED,
        "_source_span": {"start": 0, "end": len(_MERGED)},
    }


class TestBackfillWithBoundaries:
    def test_boundaries_are_derived_for_every_referenced_block(self, tmp_path):
        from lightrag.sidecar.backfill import backfill_chunk_sidecars

        chunks = [_whole_document_chunk()]
        _preconverted_parse._backfill_with_boundaries(
            backfill_chunk_sidecars, chunks, _blocks_file(tmp_path)
        )

        # The upstream refs still land (we wrap, we do not replace)...
        assert [ref["id"] for ref in chunks[0]["sidecar"]["refs"]] == ["b1", "b2"]
        # ...and each ref gains a chunk-local, half-open span carrying NO body.
        assert chunks[0]["twin_block_boundaries"] == [
            {"block_id": "b1", "start": 0, "end": len(_BLOCK_ONE)},
            {"block_id": "b2", "start": len(_BLOCK_ONE) + 2, "end": len(_MERGED)},
        ]
        content = chunks[0]["content"]
        for boundary in chunks[0]["twin_block_boundaries"]:
            assert "content" not in boundary and "text" not in boundary
            assert content[boundary["start"] : boundary["end"]]

    def test_upstream_runs_first_even_when_enrichment_is_impossible(
        self, tmp_path, monkeypatch
    ):
        """Upstream drift in the private helpers must cost the boundaries only —
        the sidecar refs the pipeline depends on still have to be attached.

        The drift is injected on the span builder, which the REAL upstream
        backfill also calls; ``original`` is therefore a stand-in that records
        the call and attaches the refs, so the assertion is about ordering
        (upstream first, enrichment second) and not about upstream's own math.
        """
        import lightrag.sidecar.backfill as upstream

        calls = []

        def _original(chunk_list, blocks_path):
            calls.append(blocks_path)
            chunk_list[0]["sidecar"] = {
                "type": "block",
                "id": "b1",
                "refs": [{"type": "block", "id": "b1"}],
            }

        def _boom(_blocks):
            raise RuntimeError("upstream renamed its span builder")

        monkeypatch.setattr(upstream, "_build_block_spans", _boom)
        chunks = [_whole_document_chunk()]
        _preconverted_parse._backfill_with_boundaries(
            _original, chunks, _blocks_file(tmp_path)
        )

        assert calls  # upstream ran before the failure
        assert chunks[0]["sidecar"]["refs"]
        assert "twin_block_boundaries" not in chunks[0]

    def test_a_chunk_without_a_usable_span_simply_gets_no_boundaries(self, tmp_path):
        from lightrag.sidecar.backfill import backfill_chunk_sidecars

        # Pre-set sidecar: upstream skips the chunk (P/multimodal path), so no
        # _source_span is ever attached and _chunk_source_span returns None.
        chunk = {
            "chunk_order_index": 0,
            "content": _MERGED,
            "sidecar": {
                "type": "block",
                "id": "b1",
                "refs": [{"type": "block", "id": "b1"}],
            },
        }
        _preconverted_parse._backfill_with_boundaries(
            backfill_chunk_sidecars, [chunk], _blocks_file(tmp_path)
        )
        assert "twin_block_boundaries" not in chunk

    def test_one_unusable_chunk_does_not_cost_the_others_their_boundaries(
        self, tmp_path
    ):
        """Per-chunk fail-soft: a chunk that is not even a mapping must not
        abort the loop for the well-formed ones behind it."""
        from lightrag.sidecar.backfill import backfill_chunk_sidecars

        good = _whole_document_chunk()
        chunks = ["not-a-chunk", good]
        _preconverted_parse._backfill_with_boundaries(
            backfill_chunk_sidecars, chunks, _blocks_file(tmp_path)
        )
        assert good["twin_block_boundaries"]

    def test_a_non_dict_sidecar_is_skipped(self, tmp_path):
        chunk = dict(_whole_document_chunk(), sidecar="block:b1")
        _preconverted_parse._backfill_with_boundaries(
            lambda _chunks, _path: None, [chunk], _blocks_file(tmp_path)
        )
        assert "twin_block_boundaries" not in chunk


class TestInstallBackfillBoundaries:
    @pytest.fixture(autouse=True)
    def _restore_upstream(self, monkeypatch):
        """Keep the wrapper out of the rest of the session: it is installed on
        an upstream module global that no later test could unwind."""
        import lightrag.sidecar as sidecar_package
        import lightrag.sidecar.backfill as backfill_module

        monkeypatch.setattr(
            backfill_module,
            "backfill_chunk_sidecars",
            backfill_module.backfill_chunk_sidecars,
        )
        monkeypatch.setattr(
            sidecar_package,
            "backfill_chunk_sidecars",
            sidecar_package.backfill_chunk_sidecars,
        )
        monkeypatch.setattr(_preconverted_parse, "_backfill_installed", False)

    def test_install_wraps_both_the_module_and_the_package_re_export(self, tmp_path):
        import lightrag.sidecar as sidecar_package
        import lightrag.sidecar.backfill as backfill_module

        original = backfill_module.backfill_chunk_sidecars
        assert _preconverted_parse.install_backfill_boundaries() is True

        wrapper = backfill_module.backfill_chunk_sidecars
        assert wrapper is not original
        assert wrapper.__wrapped__ is original
        assert sidecar_package.backfill_chunk_sidecars is wrapper

        # The wrapper is the production entry point: exercise it, not just its
        # identity — an installed wrapper that derives nothing is the failure
        # mode this seam actually has.
        chunks = [_whole_document_chunk()]
        wrapper(chunks, _blocks_file(tmp_path))
        assert chunks[0]["twin_block_boundaries"]

    def test_install_is_idempotent_and_never_wraps_a_wrapper(self):
        import lightrag.sidecar.backfill as backfill_module

        assert _preconverted_parse.install_backfill_boundaries() is True
        wrapper = backfill_module.backfill_chunk_sidecars

        # Same process, flag still set: short-circuit.
        assert _preconverted_parse.install_backfill_boundaries() is True
        assert backfill_module.backfill_chunk_sidecars is wrapper

        # Flag lost (a reload, another import path) but the marker is on the
        # installed callable: still no second layer.
        _preconverted_parse._backfill_installed = False
        assert _preconverted_parse.install_backfill_boundaries() is True
        assert backfill_module.backfill_chunk_sidecars is wrapper

    def test_install_is_refused_when_the_seam_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(_preconverted_parse, "is_available", lambda: False)
        assert _preconverted_parse.install_backfill_boundaries() is False

    def test_install_failure_degrades_to_refs_only(self, monkeypatch):
        import lightrag.sidecar.backfill as backfill_module

        monkeypatch.delattr(backfill_module, "backfill_chunk_sidecars")
        assert _preconverted_parse.install_backfill_boundaries() is False
        assert _preconverted_parse._backfill_installed is False


class TestParserRegistration:
    def test_registration_is_idempotent(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "lightrag.parser.registry.register_parser", lambda spec: calls.append(spec)
        )
        assert _preconverted_parse.ensure_parser_registered() is True
        assert _preconverted_parse.ensure_parser_registered() is True
        assert len(calls) == 1

    def test_registration_failure_falls_back_to_raw(self, monkeypatch):
        def _boom(_spec):
            raise RuntimeError("registry refused the spec")

        monkeypatch.setattr("lightrag.parser.registry.register_parser", _boom)
        assert _preconverted_parse.ensure_parser_registered() is False
        assert _preconverted_parse.activate() is False

    def test_the_probe_unwraps_the_classification_hook(self, monkeypatch):
        """The MIP hook wraps the enqueue with a generic ``(*args, **kwargs)``
        signature. Probing the wrapper instead of the original would read as
        "no docs_format" and silently strand every converted document on the
        raw path — unwrap until the real signature shows."""
        from lightrag import LightRAG

        def _generic(*args, **kwargs):  # the shape the hook installs
            raise AssertionError("probe must not call the enqueue")

        _generic.__wrapped__ = LightRAG.apipeline_enqueue_documents
        monkeypatch.setattr(
            LightRAG, "apipeline_enqueue_documents", _generic, raising=False
        )

        assert _preconverted_parse.is_available() is True

    def test_a_failing_capability_probe_means_unavailable(self, monkeypatch):
        def _boom(_target):
            raise RuntimeError("signature introspection failed")

        monkeypatch.setattr(inspect, "signature", _boom)
        assert _preconverted_parse.is_available() is False
