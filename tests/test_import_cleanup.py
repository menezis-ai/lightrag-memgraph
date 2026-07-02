from contextlib import asynccontextmanager

from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph import docstatus_impl
from twindb_lightrag_memgraph._import_cleanup import cleanup_processed_imports
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage


async def test_cleanup_processed_import_removes_input_and_parsed_artifacts(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    source = tmp_path / "report.pdf"
    parsed_source = tmp_path / "__parsed__" / "report.pdf"
    parsed_artifact = tmp_path / "__parsed__" / "report.pdf.parsed"
    parsed_artifact.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    parsed_source.write_text("archived source", encoding="utf-8")
    (parsed_artifact / "page-1.txt").write_text("parsed", encoding="utf-8")

    await cleanup_processed_imports(
        [{"status": "processed", "file_path": "report.pdf"}]
    )

    assert not source.exists()
    assert not parsed_source.exists()
    assert not parsed_artifact.exists()


async def test_cleanup_processed_import_is_confined_to_input_dir(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    outside = tmp_path / "outside.txt"
    input_dir.mkdir()
    outside.write_text("keep", encoding="utf-8")
    monkeypatch.setenv("INPUT_DIR", str(input_dir))

    await cleanup_processed_imports(
        [{"status": "processed", "file_path": str(outside)}]
    )
    await cleanup_processed_imports(
        [{"status": "processed", "file_path": "../outside.txt"}]
    )

    assert outside.exists()


async def test_cleanup_processed_import_removes_parser_hinted_source(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    hinted = tmp_path / "report.[native].pdf"
    hinted.write_text("source", encoding="utf-8")

    await cleanup_processed_imports(
        [{"status": "processed", "file_path": "report.pdf"}]
    )

    assert not hinted.exists()


async def test_cleanup_processed_import_ignores_non_processed_status(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("INPUT_DIR", str(tmp_path))
    source = tmp_path / "pending.pdf"
    source.write_text("source", encoding="utf-8")

    await cleanup_processed_imports(
        [{"status": "processing", "file_path": "pending.pdf"}]
    )

    assert source.exists()


async def test_docstatus_upsert_cleans_processed_import_after_write(monkeypatch):
    cleaned = []

    @asynccontextmanager
    async def fake_write_slot():
        yield

    @asynccontextmanager
    async def fake_session():
        yield object()

    async def fake_write(*args, **kwargs):
        return None

    async def fake_cleanup(props_list):
        cleaned.extend(props_list)

    store = MemgraphDocStatusStorage.__new__(MemgraphDocStatusStorage)
    store.namespace = "doc_status"
    store.workspace = "test"
    store.global_config = {}
    store.embedding_func = None

    monkeypatch.setattr(_pool, "acquire_write_slot", fake_write_slot)
    monkeypatch.setattr(_pool, "get_session", fake_session)
    monkeypatch.setattr(store, "_run_upsert_writes", fake_write)
    monkeypatch.setattr(docstatus_impl, "cleanup_processed_imports", fake_cleanup)

    await store.upsert({"doc-1": {"status": "processed", "file_path": "done.pdf"}})

    assert len(cleaned) == 1
    assert cleaned[0]["id"] == "doc-1"
    assert cleaned[0]["status"] == "processed"
    assert cleaned[0]["file_path"] == "done.pdf"
