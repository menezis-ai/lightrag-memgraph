"""LightRAG 1.5.4 sidecar qualification against the real Twin ingestion seams.

This is the executable evidence for ``PARAGRAPH-CITATION-PLAN.md`` phase B0.
It is intentionally version-gated: the supported 1.4.x matrix must collect and
skip it, while the dedicated 1.5.4 Memgraph canary runs the complete probe.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
import zipfile
from contextvars import ContextVar
from importlib.metadata import version
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable

import numpy as np
import pytest
from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.utils import EmbeddingFunc, Tokenizer

import twindb_lightrag_memgraph
from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph import docstatus_impl
from twindb_lightrag_memgraph._classification_hook import (
    _patched_apipeline_enqueue_documents,
    install_lightrag_ingestion_hook,
)
from twindb_lightrag_memgraph.server._lightrag_compat import (
    build_sources_from_raw_data,
    collect_chunk_ids,
)

twindb_lightrag_memgraph.register()

LIGHTRAG_VERSION = version("lightrag-hku")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        LIGHTRAG_VERSION != "1.5.4",
        reason="phase B0 sidecar probe targets LightRAG 1.5.4 exactly",
    ),
]

EMBEDDING_DIM = 32
MIP_C2_GUID = "11111111-2222-3333-4444-555555555555"
_CLASSIFICATION_ATTRS = (
    "ainsert",
    "apipeline_enqueue_documents",
    "_twin_classification_hook",
    "_twin_classification_patched",
    "_twin_original_ainsert",
    "_twin_original_enqueue",
    "_twin_enqueue_patched",
)
_B01_PARSER_ENGINE = "twinmarkdown"
_B01_MARKDOWN_BODY: ContextVar[str | None] = ContextVar(
    "twin_b01_markdown_body", default=None
)
_B01_RESOLVED_SOURCES: dict[str, Path] = {}
_B01_SOURCE_SPANS: dict[int, tuple[int, int]] = {}
_B01_EXPECTED_BOUNDARIES: dict[int, list[dict[str, Any]]] = {}


if LIGHTRAG_VERSION == "1.5.4":
    from lightrag.parser.base import ParseContext, ParseResult
    from lightrag.parser.markdown.parser import NativeMarkdownParser

    class _B01PreconvertedMarkdownParser(NativeMarkdownParser):
        """Probe adapter: native Markdown parsing over an in-memory body."""

        engine_name = _B01_PARSER_ENGINE

        async def parse(self, ctx: ParseContext) -> ParseResult:
            markdown = ctx.content_data.get("content")
            if not isinstance(markdown, str) or not markdown.strip():
                raise ValueError("preconverted Markdown body is empty")
            token = _B01_MARKDOWN_BODY.set(markdown)
            try:
                return await super().parse(ctx)
            finally:
                _B01_MARKDOWN_BODY.reset(token)

        def validate_source(self, source: Path, file_path: str) -> None:
            if not (source.exists() and source.is_file()):
                raise FileNotFoundError(
                    f"preconverted source file not found: {file_path}"
                )
            _B01_RESOLVED_SOURCES[file_path] = source.resolve()

        def extract(
            self,
            source: Path,
            *,
            parsed_dir: Path,
            asset_dir: Path,
            base_name: str,
        ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
            del source, parsed_dir, asset_dir, base_name
            markdown = _B01_MARKDOWN_BODY.get()
            if markdown is None:
                raise RuntimeError("preconverted Markdown context is missing")
            return self._extract_text(markdown, bundle_root=None)


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    """Return stable, non-zero vectors so every probe chunk is retrievable."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vector = np.frombuffer(
            digest * (EMBEDDING_DIM // len(digest) + 1), dtype=np.uint8
        )[:EMBEDDING_DIM].astype(np.float32)
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
    return np.array(vectors)


async def _mock_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Serve extraction and query calls without external model traffic."""
    prompt_text = prompt.lower() if isinstance(prompt, str) else ""
    if (
        "entity_types" in prompt_text
        or "continue extracting" in prompt_text
        or "extract" in prompt_text
    ):
        return "\n".join(
            [
                "entity<|#|>Atlas<|#|>service<|#|>Atlas is an internal service.",
                "<|COMPLETE|>",
            ]
        )
    return "Atlas is documented by the retrieved source [1]."


def _snapshot_classification_attrs() -> dict[str, Any]:
    return {name: getattr(LightRAG, name, None) for name in _CLASSIFICATION_ATTRS}


def _restore_classification_attrs(saved: dict[str, Any]) -> None:
    for name, value in saved.items():
        if value is None:
            if hasattr(LightRAG, name):
                try:
                    delattr(LightRAG, name)
                except AttributeError:
                    # 1.5.x provides pipeline methods through a mixin.
                    pass
        else:
            setattr(LightRAG, name, value)


async def _cleanup_workspace(workspace: str) -> None:
    try:
        async with _pool.get_session() as session:
            for prefix in ("KV_", "Vec_", "DocStatus_", "Folder_"):
                result = await session.run(
                    "MATCH (n) "
                    "WHERE ANY(label IN labels(n) WHERE label STARTS WITH $prefix) "
                    "DETACH DELETE n",
                    prefix=f"{prefix}{workspace}",
                )
                await result.consume()
            result = await session.run(f"MATCH (n:`{workspace}`) DETACH DELETE n")
            await result.consume()
    except Exception:
        # Cleanup must not hide the probe's actual assertion failure.
        pass


def _build_labeled_docx(path: Path) -> None:
    docx = pytest.importorskip("docx", reason="LightRAG [api] extra is required")
    document = docx.Document()
    document.add_heading("Converted control", level=1)
    document.add_paragraph(
        "Atlas conversion proof. "
        + "Alpha " * 24
        + "The astral marker 🚀 precedes TARGETAFTERASTRAL. "
        + "Beta " * 24
    )
    document.save(path)

    custom_xml = dedent(f"""
        <?xml version="1.0"?>
        <Properties
          xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
          xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
          <property fmtid="x" pid="2" name="MSIP_Label_{MIP_C2_GUID}_Name">
            <vt:lpwstr>C2 Confidentiel</vt:lpwstr>
          </property>
        </Properties>
        """).strip()
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("docProps/custom.xml", custom_xml)


def _build_pdf(path: Path) -> None:
    pypdf = pytest.importorskip("pypdf", reason="LightRAG [api] extra is required")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as stream:
        writer.write(stream)


def _row_field(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _row_metadata(row: Any) -> dict[str, Any]:
    metadata = _row_field(row, "metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return metadata if isinstance(metadata, dict) else {}


async def _single_track_document(
    rag: LightRAG, track_id: str
) -> tuple[str, Any, list[dict[str, Any]]]:
    tracked = await rag.doc_status.get_docs_by_track_id(track_id)
    assert isinstance(tracked, dict) and len(tracked) == 1, tracked
    doc_id, row = next(iter(tracked.items()))
    chunk_ids = list(_row_field(row, "chunks_list", []) or [])
    assert chunk_ids
    chunks = await rag.text_chunks.get_by_ids(chunk_ids)
    if isinstance(chunks, dict):
        chunks = [chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in chunks]
    chunks = [chunk for chunk in chunks or [] if isinstance(chunk, dict)]
    assert len(chunks) == len(chunk_ids)
    return doc_id, row, chunks


def _content_block_spans(
    block_rows: list[dict[str, Any]],
) -> tuple[str, dict[str, tuple[int, int]]]:
    contents = [
        row
        for row in block_rows
        if row.get("type") == "content"
        and isinstance(row.get("content"), str)
        and row["content"].strip()
    ]
    merged_parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for row in contents:
        if merged_parts:
            cursor += 2
        content = row["content"]
        start = cursor
        cursor += len(content)
        spans[str(row["blockid"])] = (start, cursor)
        merged_parts.append(content)
    return "\n\n".join(merged_parts), spans


def _assert_exact_codepoint_projection(
    chunks: list[dict[str, Any]],
    block_rows: list[dict[str, Any]],
    *,
    target: str = "TARGET_AFTER_ASTRAL",
    source_spans: dict[int, tuple[int, int]] | None = None,
) -> None:
    """Prove the pre-cleanup span conversion on overlap and astral Unicode."""
    merged, block_spans = _content_block_spans(block_rows)
    assert "🚀" in merged
    target_start = merged.index(target)
    assert len(merged[:target_start].encode("utf-16-le")) // 2 > target_start

    chunk_ranges: list[tuple[int, int]] = []
    target_projected = False
    for chunk in chunks:
        content = str(chunk["content"])
        if source_spans is None:
            document_start = merged.find(content)
            assert document_start >= 0, content
            document_end = document_start + len(content)
        else:
            document_start, document_end = source_spans[_chunk_probe_key(chunk)]
            assert merged[document_start:document_end] == content
        chunk_ranges.append((document_start, document_end))

        sidecar = chunk.get("sidecar")
        assert isinstance(sidecar, dict)
        assert set(sidecar) == {"type", "id", "refs"}
        assert "_source_span" not in chunk
        refs = sidecar["refs"]
        assert isinstance(refs, list) and refs

        for ref in refs:
            block_start, block_end = block_spans[ref["id"]]
            overlap_start = max(document_start, block_start)
            overlap_end = min(document_end, block_end)
            assert overlap_start < overlap_end
            local_start = overlap_start - document_start
            local_end = overlap_end - document_start
            assert content[local_start:local_end] == merged[overlap_start:overlap_end]

        if (
            document_start <= target_start
            and target_start + len(target) <= document_end
        ):
            local_target = target_start - document_start
            assert content[local_target : local_target + len(target)] == target
            target_projected = True

    ordered_ranges = sorted(chunk_ranges)
    assert any(
        current_start < previous_end
        for (_, previous_end), (current_start, _) in zip(
            ordered_ranges, ordered_ranges[1:]
        )
    ), ordered_ranges
    assert target_projected


def _chunk_probe_key(chunk: dict[str, Any]) -> int:
    """Return the per-document key shared by pre-persist and stored chunks."""
    return int(chunk["chunk_order_index"])


def _backfill_boundaries(
    original_backfill: Callable[[list[dict[str, Any]], str], None],
    chunks: list[dict[str, Any]],
    blocks_path: str,
) -> None:
    """Probe the B0.1 derivation seam before LightRAG drops source spans."""
    from lightrag.sidecar.backfill import (
        _build_block_spans,
        _chunk_source_span,
        _load_content_blocks,
    )

    original_backfill(chunks, blocks_path)
    merged, block_spans = _build_block_spans(_load_content_blocks(blocks_path))
    spans_by_id = {block_id: (start, end) for start, end, block_id in block_spans}

    for chunk in chunks:
        sidecar = chunk.get("sidecar")
        if not isinstance(sidecar, dict):
            continue
        raw_source_span = chunk.get("_source_span")
        assert isinstance(raw_source_span, dict)
        source_span = _chunk_source_span(chunk, merged)
        assert source_span is not None
        source_start, source_end = source_span
        assert source_span == (
            int(raw_source_span["start"]),
            int(raw_source_span["end"]),
        )
        content = str(chunk["content"])
        assert merged[source_start:source_end] == content

        boundaries: list[dict[str, Any]] = []
        for ref in sidecar.get("refs") or []:
            block_id = str(ref.get("id") or "")
            block_start, block_end = spans_by_id[block_id]
            overlap_start = max(source_start, block_start)
            overlap_end = min(source_end, block_end)
            assert overlap_start < overlap_end
            local_start = overlap_start - source_start
            local_end = overlap_end - source_start
            assert content[local_start:local_end] == merged[overlap_start:overlap_end]
            boundaries.append(
                {
                    "block_id": block_id,
                    "start": local_start,
                    "end": local_end,
                }
            )
        assert boundaries
        chunk["twin_block_boundaries"] = boundaries
        chunk_key = _chunk_probe_key(chunk)
        _B01_SOURCE_SPANS[chunk_key] = source_span
        _B01_EXPECTED_BOUNDARIES[chunk_key] = [
            dict(boundary) for boundary in boundaries
        ]


def _assert_persisted_boundaries(
    chunks: list[dict[str, Any]], block_rows: list[dict[str, Any]]
) -> None:
    """Verify the body-free boundaries persisted after parser cleanup."""
    block_ids = {
        str(row["blockid"])
        for row in block_rows
        if row.get("type") == "content" and row.get("blockid")
    }
    assert block_ids
    for chunk in chunks:
        content = str(chunk["content"])
        boundaries = chunk.get("twin_block_boundaries")
        assert isinstance(boundaries, list) and boundaries
        assert boundaries == _B01_EXPECTED_BOUNDARIES[_chunk_probe_key(chunk)]
        sidecar_refs = {str(ref["id"]) for ref in chunk["sidecar"].get("refs") or []}
        assert {str(item["block_id"]) for item in boundaries} == sidecar_refs
        for boundary in boundaries:
            assert set(boundary) == {"block_id", "start", "end"}
            assert str(boundary["block_id"]) in block_ids
            local_start = boundary["start"]
            local_end = boundary["end"]
            assert isinstance(local_start, int) and isinstance(local_end, int)
            assert 0 <= local_start < local_end <= len(content)
            assert content[local_start:local_end]


@pytest.fixture
def b01_probe_seam(monkeypatch):
    """Register the isolated parser adapter and boundary derivation wrapper."""
    if LIGHTRAG_VERSION != "1.5.4":
        yield
        return

    import lightrag.sidecar as sidecar_package
    from lightrag.parser import registry as parser_registry
    from lightrag.parser.registry import ParserSpec
    from lightrag.sidecar import backfill as backfill_module

    previous_spec = parser_registry._REGISTRY.get(_B01_PARSER_ENGINE)
    _B01_RESOLVED_SOURCES.clear()
    _B01_SOURCE_SPANS.clear()
    _B01_EXPECTED_BOUNDARIES.clear()
    parser_registry.register_parser(
        ParserSpec(
            engine_name=_B01_PARSER_ENGINE,
            impl=f"{__name__}:_B01PreconvertedMarkdownParser",
            suffixes=frozenset({"docx", "pdf"}),
        )
    )
    original_backfill = backfill_module.backfill_chunk_sidecars

    def backfill_with_boundaries(
        chunks: list[dict[str, Any]], blocks_path: str
    ) -> None:
        _backfill_boundaries(original_backfill, chunks, blocks_path)

    monkeypatch.setattr(
        backfill_module, "backfill_chunk_sidecars", backfill_with_boundaries
    )
    monkeypatch.setattr(
        sidecar_package,
        "backfill_chunk_sidecars",
        backfill_with_boundaries,
        raising=False,
    )

    try:
        yield
    finally:
        _B01_RESOLVED_SOURCES.clear()
        _B01_SOURCE_SPANS.clear()
        _B01_EXPECTED_BOUNDARIES.clear()
        if previous_spec is None:
            parser_registry._REGISTRY.pop(_B01_PARSER_ENGINE, None)
        else:
            parser_registry._REGISTRY[_B01_PARSER_ENGINE] = previous_spec
        for cache_key in list(parser_registry._INSTANCE_CACHE):
            if cache_key[0] == _B01_PARSER_ENGINE:
                parser_registry._INSTANCE_CACHE.pop(cache_key, None)


@pytest.fixture
async def b0_runtime(monkeypatch, tmp_path):
    """Real Memgraph runtime plus artifact capture immediately before cleanup."""
    from lightrag.kg.shared_storage import (
        finalize_share_data,
        initialize_pipeline_status,
        initialize_share_data,
    )

    workspace = f"paragraph_b0_{uuid.uuid4().hex[:8]}"
    input_dir = tmp_path / "inputs"
    working_dir = tmp_path / "working"
    input_dir.mkdir()

    monkeypatch.setenv("MEMGRAPH_WORKSPACE", workspace)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("TWIN_CONVERT", "on")
    monkeypatch.setenv("TWIN_CONVERT_FORMATS", "docx")
    monkeypatch.setenv("TWIN_MIP_UNLABELED_POLICY", "allow")
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    label_map = tmp_path / "labels.json"
    label_map.write_text(json.dumps({MIP_C2_GUID: "C2"}), encoding="utf-8")

    saved_classification_attrs = _snapshot_classification_attrs()
    install_lightrag_ingestion_hook(label_map_path=label_map, ceiling="C2")

    artifact_snapshots: dict[str, list[dict[str, Any]]] = {}
    real_cleanup = docstatus_impl.cleanup_processed_imports

    async def capture_then_cleanup(props_list: list[dict[str, Any]]) -> None:
        parsed_root = input_dir / "__parsed__"
        for props in props_list:
            if str(props.get("status") or "").lower() != "processed":
                continue
            file_name = Path(str(props.get("file_path") or "")).name
            rows: list[dict[str, Any]] = []
            if parsed_root.exists():
                for blocks_path in parsed_root.rglob("*.blocks.jsonl"):
                    rows.extend(
                        json.loads(line)
                        for line in blocks_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
            artifact_snapshots[file_name] = rows
        await real_cleanup(props_list)

    monkeypatch.setattr(
        docstatus_impl, "cleanup_processed_imports", capture_then_cleanup
    )

    finalize_share_data()
    initialize_share_data()
    await _cleanup_workspace(workspace)

    rag = LightRAG(
        working_dir=str(working_dir),
        kv_storage="MemgraphKVStorage",
        vector_storage="MemgraphVectorDBStorage",
        doc_status_storage="MemgraphDocStatusStorage",
        graph_storage="MemgraphStorage",
        workspace=workspace,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=_mock_embedding,
        ),
        llm_model_func=_mock_llm,
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False,
        chunk_token_size=90,
        chunk_overlap_token_size=30,
        tokenizer=Tokenizer("paragraph-b0-char", _CharTokenizer()),
        vector_db_storage_cls_kwargs={"cosine_better_than_threshold": -1.0},
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    try:
        yield rag, input_dir, artifact_snapshots
    finally:
        await _cleanup_workspace(workspace)
        await rag.finalize_storages()
        _restore_classification_attrs(saved_classification_attrs)
        finalize_share_data()


async def test_lightrag_1_5_4_sidecar_qualification(b0_runtime, monkeypatch):
    """Close B0-1..B0-5 with observed success or an explicit incompatibility.

    Since B1 the production seam takes ``pending_parse`` on 1.5.x, so this
    historical qualification pins the RAW path explicitly through the
    kill-switch: it remains the executable proof of what ``raw`` enqueue
    does NOT produce, and of the fallback contract behind
    ``TWIN_PRECONVERTED_PARSE=off``.
    """
    monkeypatch.setenv("TWIN_PRECONVERTED_PARSE", "off")
    pytest.importorskip(
        "markitdown", reason="phase B0 integration canary requires [convert]"
    )
    from lightrag.api.routers import document_routes

    from twindb_lightrag_memgraph import _pdf_vision
    from twindb_lightrag_memgraph import _procedure
    from twindb_lightrag_memgraph.patches import registry

    rag, input_dir, artifact_snapshots = b0_runtime
    assert (
        getattr(rag.apipeline_enqueue_documents, "__func__", None)
        is _patched_apipeline_enqueue_documents
    )

    # B0-2/B0-3 control: a real native Markdown parse produces blocks, backfills
    # sidecar refs on overlapping chunks, persists those refs, then Twin cleanup
    # removes the source and the raw parser artifact.
    native_path = input_dir / "native-control.[native].md"
    native_canonical_name = "native-control.md"
    native_path.write_text(
        "\n\n".join(
            [
                "# Native control",
                "Alpha " * 24,
                "## Astral section",
                "The astral marker 🚀 precedes TARGET_AFTER_ASTRAL in this block. "
                + "Beta " * 22,
                "## Tail section",
                "Gamma " * 26,
            ]
        ),
        encoding="utf-8",
    )
    native_track = f"b0-native-{uuid.uuid4().hex[:8]}"
    native_enqueue = document_routes.pipeline_enqueue_file
    while hasattr(native_enqueue, "__wrapped__"):
        native_enqueue = native_enqueue.__wrapped__
    success, returned_track = await native_enqueue(
        rag, native_path, track_id=native_track
    )
    assert success is True
    assert returned_track == native_track
    await rag.apipeline_process_enqueue_documents()

    _, native_row, native_chunks = await _single_track_document(rag, native_track)
    native_metadata = _row_metadata(native_row)
    assert native_metadata["parse_engine"] == "native"
    assert artifact_snapshots[native_canonical_name]
    assert not native_path.exists()
    assert not list((input_dir / "__parsed__").rglob("*.blocks.jsonl"))
    _assert_exact_codepoint_projection(
        native_chunks, artifact_snapshots[native_canonical_name]
    )

    # B0-1: run the real MarkItDown conversion through Twin's production
    # wrapper. The original .docx name reaches the MIP hook, but the converted
    # body is enqueued as RAW today, so no native Markdown parser or sidecar runs.
    converted_path = input_dir / "converted-original.docx"
    _build_labeled_docx(converted_path)
    converted_track = f"b0-converted-{uuid.uuid4().hex[:8]}"
    converted_enqueue = registry._make_converting_pipeline_enqueue_file(
        document_routes, native_enqueue
    )
    success, returned_track = await converted_enqueue(
        rag, converted_path, track_id=converted_track
    )
    assert success is True
    assert returned_track == converted_track
    await rag.apipeline_process_enqueue_documents()

    _, converted_row, converted_chunks = await _single_track_document(
        rag, converted_track
    )
    converted_metadata = _row_metadata(converted_row)
    assert converted_metadata["classification"]["class_id"] == "C2"
    assert converted_metadata["classification"]["source_format"] == "ooxml"
    assert artifact_snapshots[converted_path.name] == []
    assert all("sidecar" not in chunk for chunk in converted_chunks)
    assert not converted_path.exists()

    # B0-5: use the exact offline Markdown composers for PDF Vision and
    # Procedure, then pass their output through the shared production enqueue
    # seam. Page headings survive as text, but RAW enqueue again produces no
    # block provenance.
    vision_path = input_dir / "vision-procedure-original.pdf"
    _build_pdf(vision_path)
    page_markdown = _pdf_vision._text_fallback(
        vision_path,
        (
            "Page one contains the Atlas overview.",
            "Page two contains the operating procedure.",
        ),
    )
    assert page_markdown is not None
    procedure_markdown = _procedure.compose_approved_markdown(
        {
            "full_text": page_markdown,
            "schematics": [
                {
                    "page": 2,
                    "informed": {
                        "title": "Atlas workflow",
                        "description": "Validated procedure schematic.",
                        "tasks": [{"id": "T1", "title": "Validate Atlas"}],
                    },
                }
            ],
        }
    )
    assert "## Page 1" in procedure_markdown
    assert "## Page 2" in procedure_markdown

    vision_track = f"b0-vision-{uuid.uuid4().hex[:8]}"
    success, returned_track = await registry._enqueue_converted(
        rag,
        vision_path,
        procedure_markdown,
        vision_track,
        False,
    )
    assert success is True
    assert returned_track == vision_track
    await rag.apipeline_process_enqueue_documents()

    _, vision_row, vision_chunks = await _single_track_document(rag, vision_track)
    vision_metadata = _row_metadata(vision_row)
    assert vision_metadata["classification"]["source_format"] == "pdf"
    assert artifact_snapshots[vision_path.name] == []
    assert all("sidecar" not in chunk for chunk in vision_chunks)
    assert any("## Page 1" in chunk["content"] for chunk in vision_chunks)
    assert any("## Page 2" in chunk["content"] for chunk in vision_chunks)
    assert not vision_path.exists()

    # B0-4: capture the actual 1.5.4 retrieval envelope produced over the real
    # persisted chunks and feed it through Twin's compatibility projection.
    envelope = await rag.aquery_llm(
        "What does the Atlas source document?",
        param=QueryParam(
            mode="naive",
            top_k=20,
            chunk_top_k=20,
            enable_rerank=False,
        ),
    )
    assert envelope["status"] == "success"
    assert set(envelope["data"]) >= {
        "entities",
        "relationships",
        "chunks",
        "references",
    }
    assert envelope["data"]["chunks"]
    assert envelope["data"]["references"]
    chunk_ids = collect_chunk_ids(envelope)
    assert chunk_ids
    sources = build_sources_from_raw_data(envelope)
    assert sources
    assert {source["n"] for source in sources} == {
        int(reference["reference_id"]) for reference in envelope["data"]["references"]
    }


@pytest.mark.usefixtures("b01_probe_seam")
async def test_lightrag_1_5_4_preconverted_markdown_seam(b0_runtime):
    """Prove B0.1 without staging files or production patch wiring."""
    pytest.importorskip(
        "markitdown", reason="phase B0.1 integration canary requires [convert]"
    )
    from lightrag.utils import compute_mdhash_id

    from twindb_lightrag_memgraph import _conversion
    from twindb_lightrag_memgraph import _pdf_vision
    from twindb_lightrag_memgraph import _procedure

    rag, input_dir, artifact_snapshots = b0_runtime

    converted_path = input_dir / "b01-converted-original.docx"
    _build_labeled_docx(converted_path)
    converted_markdown = await _conversion.aconvert_file(converted_path)
    assert converted_markdown is not None
    converted_track = f"b01-converted-{uuid.uuid4().hex[:8]}"
    returned_track = await rag.apipeline_enqueue_documents(
        converted_markdown,
        file_paths=converted_path.name,
        track_id=converted_track,
        docs_format="pending_parse",
        parse_engine=_B01_PARSER_ENGINE,
    )
    assert returned_track == converted_track
    await rag.apipeline_process_enqueue_documents()

    converted_doc_id, converted_row, converted_chunks = await _single_track_document(
        rag, converted_track
    )
    assert converted_doc_id == compute_mdhash_id(converted_path.name, prefix="doc-")
    assert _row_field(converted_row, "file_path") == converted_path.name
    converted_metadata = _row_metadata(converted_row)
    assert converted_metadata["classification"]["class_id"] == "C2"
    assert converted_metadata["classification"]["source_format"] == "ooxml"
    assert converted_metadata["parse_engine"] == _B01_PARSER_ENGINE
    assert converted_metadata["parse_format"] == "lightrag"
    assert converted_metadata["source_file"] == converted_path.name
    converted_full_doc = await rag.full_docs.get_by_id(converted_doc_id)
    assert converted_full_doc["file_path"] == converted_path.name
    assert "source_file" not in converted_full_doc
    assert _B01_RESOLVED_SOURCES[converted_path.name] == converted_path.resolve()
    converted_blocks = artifact_snapshots[converted_path.name]
    assert converted_blocks
    _assert_exact_codepoint_projection(
        converted_chunks,
        converted_blocks,
        target="TARGETAFTERASTRAL",
        source_spans=_B01_SOURCE_SPANS,
    )
    _assert_persisted_boundaries(converted_chunks, converted_blocks)
    _B01_SOURCE_SPANS.clear()
    _B01_EXPECTED_BOUNDARIES.clear()
    assert not converted_path.exists()
    assert not list((input_dir / "__parsed__").rglob("*.blocks.jsonl"))

    vision_path = input_dir / "b01-vision-procedure-original.pdf"
    _build_pdf(vision_path)
    page_markdown = _pdf_vision._text_fallback(
        vision_path,
        (
            "Page one contains the Atlas overview.",
            "Page two contains the operating procedure.",
        ),
    )
    assert page_markdown is not None
    procedure_markdown = _procedure.compose_approved_markdown(
        {
            "full_text": page_markdown,
            "schematics": [
                {
                    "page": 2,
                    "informed": {
                        "title": "Atlas workflow",
                        "description": "Validated procedure schematic.",
                        "tasks": [{"id": "T1", "title": "Validate Atlas"}],
                    },
                }
            ],
        }
    )
    vision_track = f"b01-vision-{uuid.uuid4().hex[:8]}"
    returned_track = await rag.apipeline_enqueue_documents(
        procedure_markdown,
        file_paths=vision_path.name,
        track_id=vision_track,
        docs_format="pending_parse",
        parse_engine=_B01_PARSER_ENGINE,
    )
    assert returned_track == vision_track
    await rag.apipeline_process_enqueue_documents()

    vision_doc_id, vision_row, vision_chunks = await _single_track_document(
        rag, vision_track
    )
    assert vision_doc_id == compute_mdhash_id(vision_path.name, prefix="doc-")
    assert _row_field(vision_row, "file_path") == vision_path.name
    vision_metadata = _row_metadata(vision_row)
    assert vision_metadata["classification"]["source_format"] == "pdf"
    assert vision_metadata["parse_engine"] == _B01_PARSER_ENGINE
    assert vision_metadata["source_file"] == vision_path.name
    vision_full_doc = await rag.full_docs.get_by_id(vision_doc_id)
    assert vision_full_doc["file_path"] == vision_path.name
    assert "source_file" not in vision_full_doc
    assert _B01_RESOLVED_SOURCES[vision_path.name] == vision_path.resolve()
    vision_blocks = artifact_snapshots[vision_path.name]
    assert vision_blocks
    vision_block_bodies = [
        str(row["content"]) for row in vision_blocks if row.get("type") == "content"
    ]
    assert any("## Page 1" in body for body in vision_block_bodies)
    assert any("## Page 2" in body for body in vision_block_bodies)
    _assert_persisted_boundaries(vision_chunks, vision_blocks)
    assert any("## Page 1" in chunk["content"] for chunk in vision_chunks)
    assert any("## Page 2" in chunk["content"] for chunk in vision_chunks)
    assert not vision_path.exists()
    assert not list((input_dir / "__parsed__").rglob("*.blocks.jsonl"))

    # Citation and display identity remain the two original binaries; no
    # synthetic Markdown source is ever created or indexed.
    envelope = await rag.aquery_llm(
        "What does the Atlas source document?",
        param=QueryParam(
            mode="naive",
            top_k=20,
            chunk_top_k=20,
            enable_rerank=False,
        ),
    )
    reference_paths = {
        str(reference["file_path"]) for reference in envelope["data"]["references"]
    }
    assert converted_path.name in reference_paths
    assert vision_path.name in reference_paths
    assert all(not path.endswith(".md") for path in reference_paths)
    assert not list(input_dir.rglob("*.md"))


async def test_lightrag_1_5_4_b1_production_seam(b0_runtime):
    """B1: the PRODUCTION enqueue seam produces boundaries end to end.

    Unlike the B0.1 probe above (isolated fixture parser + monkeypatched
    backfill), this drives ``registry._enqueue_converted`` with the shipped
    ``_preconverted_parse`` module exactly as ``register()`` wires it:
    twinmarkdown engine, boundary backfill, cleanup — then proves the
    persisted ``twin_block_boundaries`` are consumable by the structural
    anchor election."""
    pytest.importorskip(
        "markitdown", reason="phase B1 integration canary requires [convert]"
    )
    from twindb_lightrag_memgraph import _conversion, _preconverted_parse
    from twindb_lightrag_memgraph.patches import registry
    from twindb_lightrag_memgraph.server.query.paragraph_anchor import (
        ANCHOR_METHOD_STRUCTURAL,
        collect_citation_evidence,
        compute_best_structural_anchor,
    )

    rag, input_dir, artifact_snapshots = b0_runtime
    assert _preconverted_parse.activate() is True

    source_path = input_dir / "b1-production-original.docx"
    _build_labeled_docx(source_path)
    markdown = await _conversion.aconvert_file(source_path)
    assert markdown is not None
    track_id = f"b1-prod-{uuid.uuid4().hex[:8]}"
    success, returned_track = await registry._enqueue_converted(
        rag, source_path, markdown, track_id, False
    )
    assert success is True
    assert returned_track == track_id
    await rag.apipeline_process_enqueue_documents()

    doc_id, row, chunks = await _single_track_document(rag, track_id)
    metadata = _row_metadata(row)
    assert metadata["classification"]["class_id"] == "C2"
    assert metadata["parse_engine"] == _preconverted_parse.PARSER_ENGINE
    assert _row_field(row, "file_path") == source_path.name

    anchored = False
    for chunk in chunks:
        content = str(chunk["content"])
        sidecar = chunk.get("sidecar")
        assert isinstance(sidecar, dict), chunk
        boundaries = chunk.get("twin_block_boundaries")
        assert isinstance(boundaries, list) and boundaries, chunk
        sidecar_refs = {str(ref["id"]) for ref in sidecar.get("refs") or []}
        assert {str(b["block_id"]) for b in boundaries} <= sidecar_refs
        for boundary in boundaries:
            assert set(boundary) == {"block_id", "start", "end"}
            assert 0 <= boundary["start"] < boundary["end"] <= len(content)
        # The persisted boundaries feed the structural election directly.
        evidence = collect_citation_evidence(f"{content[:60].strip()} [1].")
        if 1 in evidence:
            elected = compute_best_structural_anchor(
                [(str(chunk.get("chunk_id") or doc_id), content, boundaries)],
                evidence[1],
            )
            if elected is not None:
                assert elected[1]["method"] == ANCHOR_METHOD_STRUCTURAL
                anchored = True
    assert anchored

    # Retention contract unchanged: original gone, no raw artifacts left.
    assert artifact_snapshots[source_path.name]
    assert not source_path.exists()
    assert not list((input_dir / "__parsed__").rglob("*.blocks.jsonl"))
    assert not list(input_dir.rglob("*.md"))
