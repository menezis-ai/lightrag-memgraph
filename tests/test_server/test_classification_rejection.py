"""Integration test for the LightRAG pre-ingestion classification gate."""

from __future__ import annotations

import zipfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from twindb_lightrag_memgraph._classification_hook import (
    install_lightrag_ingestion_hook,
)
from twindb_lightrag_memgraph.server import webui_router

C3_GUID = "33333333-3333-3333-3333-333333333333"


def _build_docx_with_label(tmp_path: Path, label_name: str, guid: str) -> Path:
    custom_xml = dedent(f"""
        <?xml version="1.0"?>
        <Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties"
                    xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
          <property fmtid="x" pid="2" name="MSIP_Label_{guid}_Name">
            <vt:lpwstr>{label_name}</vt:lpwstr>
          </property>
        </Properties>
    """).strip()
    target = tmp_path / "classified-c3.docx"
    with zipfile.ZipFile(target, "w") as z:
        z.writestr("docProps/custom.xml", custom_xml)
    return target


class FakeDocStatus:
    def __init__(self) -> None:
        self.upserts: dict[str, dict[str, Any]] = {}

    async def upsert(self, docs: dict[str, dict[str, Any]]) -> None:
        self.upserts.update(docs)

    async def get_by_id(self, doc_id: str):
        return self.upserts.get(doc_id)


class FakeRag:
    def __init__(self) -> None:
        self.doc_status = FakeDocStatus()


@pytest.fixture()
def patched_lightrag(monkeypatch):
    from lightrag import LightRAG

    original_ainsert = LightRAG.ainsert

    async def forbidden_ainsert(*_args, **_kwargs):
        raise AssertionError("rejected document reached LightRAG.ainsert")

    monkeypatch.setattr(LightRAG, "ainsert", forbidden_ainsert)
    for attr in (
        "_twin_classification_hook",
        "_twin_original_ainsert",
        "_twin_classification_patched",
    ):
        monkeypatch.delattr(LightRAG, attr, raising=False)
    yield LightRAG
    monkeypatch.setattr(LightRAG, "ainsert", original_ainsert)
    for attr in (
        "_twin_classification_hook",
        "_twin_original_ainsert",
        "_twin_classification_patched",
    ):
        monkeypatch.delattr(LightRAG, attr, raising=False)
    webui_router.reset_store()


class TestClassificationRejectionIntegration:
    async def test_c3_above_ceiling_writes_failed_docstatus_and_activity(
        self,
        tmp_path,
        patched_lightrag,
    ):
        label_map = tmp_path / "labels.json"
        label_map.write_text(f'{{"{C3_GUID}": "C3"}}')
        source_path = _build_docx_with_label(tmp_path, "C3 Strict", C3_GUID)
        install_lightrag_ingestion_hook(
            label_map_path=label_map,
            ceiling="C2",
        )

        rag = object.__new__(patched_lightrag)
        rag.doc_status = FakeDocStatus()
        track_id = await patched_lightrag.ainsert(
            rag,
            "strict content",
            ids="doc-c3",
            file_paths=str(source_path),
            track_id="track-c3",
        )

        assert track_id == "track-c3"
        failed = rag.doc_status.upserts["doc-c3"]
        assert failed["status"].value == "failed"
        assert failed["track_id"] == "track-c3"
        assert "outranks workspace ceiling" in failed["error_msg"]
        assert failed["metadata"]["classification"]["class_id"] == "C3"
        assert failed["metadata"]["classification_rejected"] is True

        events, _, _ = await webui_router.get_store().list_activity()
        rejected = [e for e in events if e["kind"] == "classification-rejected"]
        assert len(rejected) == 1
        assert rejected[0]["meta"]["doc_id"] == "doc-c3"
        assert rejected[0]["meta"]["classification"]["class_id"] == "C3"
