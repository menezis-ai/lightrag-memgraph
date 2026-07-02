from starlette.background import BackgroundTasks
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import (
    _install_storage_folder_capture,
    _patch_background_tasks_folder_context,
)
from twindb_lightrag_memgraph._constants import (
    get_active_storage_folder,
    storage_folder_context,
)
from twindb_lightrag_memgraph.docstatus_impl import MemgraphDocStatusStorage
from lightrag.base import DocProcessingStatus, DocStatus


async def test_background_task_reapplies_captured_storage_folder():
    seen: list[str | None] = []

    async def task():
        seen.append(get_active_storage_folder())

    _patch_background_tasks_folder_context()
    background = BackgroundTasks()
    with storage_folder_context("sandbox"):
        background.add_task(task)

    assert get_active_storage_folder() is None
    await background()

    assert seen == ["sandbox"]


def test_docstatus_serialization_uses_storage_folder_context():
    status = DocProcessingStatus(
        content_summary="summary",
        content_length=7,
        file_path="sandbox.md",
        status=DocStatus.PENDING,
        created_at="2026-06-20T00:00:00Z",
        updated_at="2026-06-20T00:00:00Z",
    )

    with storage_folder_context("sandbox"):
        props = MemgraphDocStatusStorage._serialize_status("doc-1", status)

    assert props["folder"] == "sandbox"


async def test_reprocess_failed_captures_request_storage_folder(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        '[{"id":"default","label":"Default","kind":"kb"},'
        '{"id":"sandbox","label":"Sandbox","kind":"kb"}]',
    )

    app = FastAPI()
    _install_storage_folder_capture(app)

    @app.post("/documents/reprocess_failed")
    async def reprocess_failed_probe():
        return {"folder": get_active_storage_folder()}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/documents/reprocess_failed",
            headers={"X-Twin-Folder": "sandbox"},
        )

    assert response.status_code == 200
    assert response.json() == {"folder": "sandbox"}
