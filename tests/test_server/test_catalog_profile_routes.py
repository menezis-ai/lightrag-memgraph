"""Metadata-only instance profile consumed by the central catalogue scan."""

from __future__ import annotations

from contextlib import asynccontextmanager, nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException

from twindb_lightrag_memgraph import _pool
from twindb_lightrag_memgraph._constants import (
    DEFAULT_TWIN_MAX_FOLDERS,
    resolve_workspace,
)
from twindb_lightrag_memgraph.server import (
    api_key_store,
    catalog_profile_routes,
    webui_router,
)
from twindb_lightrag_memgraph.server.auth import configure_auth, require_auth
from twindb_lightrag_memgraph.server.catalog_profile_routes import (
    CatalogProfile,
    build_catalog_profile_router,
)
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp
from tests._repo_only import require_repo_path
from twindb_lightrag_memgraph.server.webui import router as webui_router_impl
from twindb_lightrag_memgraph.server.webui_seed import DOCUMENTS

_PROFILE_V1_FIXTURE = (
    Path(__file__).parents[2]
    / "services"
    / "twin_catalog"
    / "tests"
    / "fixtures"
    / "catalog_profile_v1.json"
)


@pytest.fixture(autouse=True)
def _catalog_profile_env(monkeypatch):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {"id": "default", "label": "Default", "kind": "primary"},
                {"id": "sandbox", "label": "Sandbox", "kind": "custom"},
            ]
        ),
    )
    monkeypatch.setenv("TWIN_MAX_FOLDERS", "5")
    monkeypatch.delenv("TWIN_FOLDERS_RUNTIME_FILE", raising=False)
    configure_idp(None)
    configure_auth(api_key="profile-root-key")
    webui_router.reset_store()
    yield
    webui_router.reset_store()
    configure_idp(None)
    configure_auth()


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(build_catalog_profile_router(), prefix="/twin/api")
    return app


def _use_memgraph_stores() -> None:
    for folder in ("default", "sandbox"):
        webui_router.set_store(
            webui_router.WebuiStore.for_folder(folder, mode="memgraph"),
            folder=folder,
        )


async def _no_graph_tags(_documents) -> None:
    return None


async def _no_graph_entities(**_kwargs):
    return []


async def test_catalog_profile_router_rejects_anonymous_when_mounted_directly():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()), base_url="http://test"
    ) as client:
        response = await client.get("/twin/api/catalog-profile")
    assert response.status_code == 401


async def test_infrastructure_profile_aggregates_authorized_folders_without_content():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()),
        base_url="http://test",
        headers={"Authorization": "Bearer profile-root-key"},
    ) as client:
        response = await client.get(
            "/twin/api/catalog-profile", params={"max_items": 25}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["schema_version"] == "1"
    assert [folder["id"] for folder in body["folders"]] == ["default", "sandbox"]
    default = body["folders"][0]
    assert default["sampled_document_count"] <= 25
    assert default["status_counts"]
    assert default["document_formats"]
    assert default["top_graph_entities"]

    serialized = response.text
    for document in DOCUMENTS:
        for forbidden in (
            document.get("source"),
            document.get("summary"),
            document.get("content_summary"),
        ):
            if forbidden:
                assert str(forbidden) not in serialized
    assert "chunks" not in serialized.lower()
    assert "summary" not in serialized.lower()


async def test_profile_uses_scoped_total_and_bounds_count_vocabularies(monkeypatch):
    class FakeDocStatus:
        async def get_docs_paginated(
            self,
            page: int = 1,
            page_size: int = 500,
            status_filter=None,
            folder: str | None = None,
        ):
            assert page == 1 and status_filter is None
            total = 10_000 if folder == "default" else 0
            rows = [
                (
                    f"doc-{index}",
                    {
                        "status": f"status-{index}",
                        "file_path": f"document-{index}.x{index}",
                        "metadata": {"folder": folder},
                    },
                )
                for index in range(min(page_size, total))
            ]
            return rows, total

    class FakeRag:
        doc_status = FakeDocStatus()

    _use_memgraph_stores()
    monkeypatch.setattr(webui_router_impl, "_get_rag", lambda: FakeRag())
    monkeypatch.setattr(
        webui_router_impl, "_attach_graph_tags_for_documents", _no_graph_tags
    )
    monkeypatch.setattr(
        catalog_profile_routes, "list_graph_entities", _no_graph_entities
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()),
        base_url="http://test",
        headers={"Authorization": "Bearer profile-root-key"},
    ) as client:
        response = await client.get(
            "/twin/api/catalog-profile", params={"max_items": 65}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    default = body["folders"][0]
    assert default["document_count"] == 10_000
    assert default["sampled_document_count"] == 65
    assert default["documents_truncated"] is True
    assert len(default["status_counts"]) == 64
    assert len(default["document_formats"]) == 64
    assert body["document_count"] == 10_000


async def test_legacy_unscoped_storage_walks_all_pages_for_folder_total(monkeypatch):
    class FakeDocStatus:
        def __init__(self) -> None:
            self.docs = [
                (
                    f"doc-{index}",
                    {
                        "status": "processed",
                        "file_path": f"document-{index}.pdf",
                    },
                )
                for index in range(520)
            ]

        async def get_docs_paginated(
            self,
            page: int = 1,
            page_size: int = 500,
            status_filter=None,
        ):
            assert status_filter is None
            start = (page - 1) * page_size
            return self.docs[start : start + page_size], len(self.docs)

        async def get_folders_for_doc(self, doc_id: str):
            index = int(doc_id.removeprefix("doc-"))
            return ["sandbox"] if index % 10 == 0 else ["default"]

    class FakeRag:
        doc_status = FakeDocStatus()

    _use_memgraph_stores()
    monkeypatch.setattr(webui_router_impl, "_get_rag", lambda: FakeRag())
    monkeypatch.setattr(
        webui_router_impl, "_attach_graph_tags_for_documents", _no_graph_tags
    )
    monkeypatch.setattr(
        catalog_profile_routes, "list_graph_entities", _no_graph_entities
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()),
        base_url="http://test",
        headers={"Authorization": "Bearer profile-root-key"},
    ) as client:
        response = await client.get(
            "/twin/api/catalog-profile", params={"max_items": 25}
        )

    assert response.status_code == 200, response.text
    folders = {row["id"]: row for row in response.json()["folders"]}
    assert folders["default"]["document_count"] == 468
    assert folders["default"]["documents_truncated"] is True
    assert folders["sandbox"]["document_count"] == 52
    assert folders["sandbox"]["sampled_document_count"] == 25
    assert folders["sandbox"]["documents_truncated"] is True


async def test_legacy_unscoped_storage_fails_closed_at_page_cap(monkeypatch):
    class FakeDocStatus:
        async def get_docs_paginated(
            self,
            page: int = 1,
            page_size: int = 500,
            status_filter=None,
        ):
            assert page == 1 and status_filter is None
            rows = [
                (
                    f"doc-{index}",
                    {
                        "status": "processed",
                        "file_path": f"document-{index}.pdf",
                        "metadata": {"folder": "default"},
                    },
                )
                for index in range(page_size)
            ]
            return rows, page_size + 1

    class FakeRag:
        doc_status = FakeDocStatus()

    _use_memgraph_stores()
    monkeypatch.setattr(webui_router_impl, "_get_rag", lambda: FakeRag())
    monkeypatch.setattr(webui_router_impl, "_CATALOG_PROFILE_LEGACY_MAX_PAGES", 1)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()),
        base_url="http://test",
        headers={"Authorization": "Bearer profile-root-key"},
    ) as client:
        response = await client.get(
            "/twin/api/catalog-profile", params={"max_items": 1}
        )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Legacy document storage exceeds catalog profile scan limit"
    )


async def test_memgraph_document_failure_is_not_reported_as_an_empty_kb(monkeypatch):
    def unavailable_rag():
        raise HTTPException(503, "document storage unavailable")

    _use_memgraph_stores()
    monkeypatch.setattr(webui_router_impl, "_get_rag", unavailable_rag)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()),
        base_url="http://test",
        headers={"Authorization": "Bearer profile-root-key"},
    ) as client:
        response = await client.get("/twin/api/catalog-profile")

    assert response.status_code == 503
    assert response.json()["detail"] == "document storage unavailable"


async def test_catalog_profile_is_internal_but_remains_callable():
    app = _app()
    assert "/twin/api/catalog-profile" not in app.openapi()["paths"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer profile-root-key"},
    ) as client:
        response = await client.get(
            "/twin/api/catalog-profile", params={"max_items": 1}
        )

    assert response.status_code == 200, response.text


def test_catalog_profile_v1_fixture_matches_producer_contract():
    # The producer is a separate distribution: services/ is not in the BNP
    # export, which ships tests/ but not the catalogue it cross-checks.
    require_repo_path(
        "services/twin_catalog/tests/fixtures/catalog_profile_v1.json",
        module_level=False,
    )
    profile = CatalogProfile.model_validate_json(
        _PROFILE_V1_FIXTURE.read_text(encoding="utf-8")
    )
    assert profile.schema_version == "1"


def test_catalog_profile_v1_folder_limit_tracks_the_platform_limit():
    assert catalog_profile_routes._MAX_PROFILE_FOLDERS == DEFAULT_TWIN_MAX_FOLDERS


@pytest.mark.parametrize(
    ("field", "limit"),
    (("id", 128), ("label", 160), ("kind", 64)),
)
async def test_folder_profile_rejects_metadata_outside_v1_contract(
    monkeypatch, field, limit
):
    async def no_documents(*, max_items: int):
        assert max_items == 1
        return [], 0

    values = {"id": "default", "label": "Default", "kind": "primary"}
    values[field] = "x" * (limit + 1)
    monkeypatch.setattr(
        catalog_profile_routes, "_documents_for_active_folder", no_documents
    )
    monkeypatch.setattr(
        catalog_profile_routes, "list_graph_entities", _no_graph_entities
    )
    monkeypatch.setattr(
        catalog_profile_routes, "scoped_folder", lambda _folder_id: nullcontext()
    )

    with pytest.raises(HTTPException) as caught:
        await catalog_profile_routes._folder_profile(
            SimpleNamespace(**values), max_items=1
        )

    assert caught.value.status_code == 503
    assert f"folder {field} exceeds the v1 limit of {limit}" in caught.value.detail


async def test_catalog_profile_rejects_more_than_five_folders(monkeypatch):
    folders = tuple(
        SimpleNamespace(id=f"folder_{index}", label=f"Folder {index}", kind="custom")
        for index in range(6)
    )
    catalog = SimpleNamespace(folders=folders)
    monkeypatch.setattr(catalog_profile_routes, "load_folder_catalog", lambda: catalog)
    monkeypatch.setattr(
        catalog_profile_routes,
        "catalog_profile_folder_ids",
        lambda _request: tuple(folder.id for folder in folders),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()),
        base_url="http://test",
        headers={"Authorization": "Bearer profile-root-key"},
    ) as client:
        response = await client.get("/twin/api/catalog-profile")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Catalog profile exceeds the v1 limit of 5 folders"
    )


async def test_profile_credential_is_limited_to_its_folder_scope(monkeypatch):
    async def validate_bearer(_workspace: str, token: str):
        assert token == "tcp_profile-secret"
        return {
            "id": "profile-key",
            "scopes": ["profile:read"],
            "folders": ["sandbox"],
        }

    async def mark_used(_workspace: str, key_id: str):
        assert key_id == "profile-key"

    monkeypatch.setattr(api_key_store, "validate_bearer", validate_bearer)
    monkeypatch.setattr(api_key_store, "mark_used", mark_used)
    app = _app()

    @app.get("/generic", dependencies=[Depends(require_auth)])
    async def generic_route():
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer tcp_profile-secret"},
    ) as client:
        response = await client.get("/twin/api/catalog-profile")
        generic = await client.get("/generic")

    assert response.status_code == 200, response.text
    assert [folder["id"] for folder in response.json()["folders"]] == ["sandbox"]
    assert generic.status_code == 401


async def test_minted_profile_key_round_trips_through_offline_store_and_route(
    monkeypatch,
):
    records: dict[str, dict] = {}

    class MemoryResult:
        def __init__(self, row=None) -> None:
            self.row = row

        async def single(self):
            return self.row

        async def consume(self) -> None:
            return None

    class MemorySession:
        async def run(self, query: str, **params):
            if "CREATE (n:" in query:
                records[params["hash"]] = {
                    "id": params["id"],
                    "data": params["data"],
                    "last_used": None,
                }
                return MemoryResult()
            if "{hash: $h}" in query:
                record = records.get(params["h"])
                return MemoryResult(
                    None
                    if record is None
                    else {
                        "data": record["data"],
                        "last_used": record["last_used"],
                    }
                )
            if "SET n.last_used_at_ms" in query:
                for record in records.values():
                    if record["id"] == params["id"]:
                        record["last_used"] = params["now"]
                return MemoryResult()
            raise AssertionError(f"unexpected query: {query}")

    session = MemorySession()

    @asynccontextmanager
    async def memory_session():
        yield session

    @asynccontextmanager
    async def memory_write_slot():
        yield

    monkeypatch.setenv("WORKSPACE", "catalog_profile_offline")
    monkeypatch.setattr(_pool, "get_session", memory_session)
    monkeypatch.setattr(_pool, "get_read_session", memory_session)
    monkeypatch.setattr(_pool, "acquire_write_slot", memory_write_slot)

    workspace = resolve_workspace()
    created = await api_key_store.create_key(
        workspace,
        name="central-catalog",
        created_by="test",
        scopes=["profile:read"],
        folders=["sandbox"],
    )
    validated = await api_key_store.validate_bearer(workspace, created["full_value"])
    assert validated is not None
    assert validated["scopes"] == ["profile:read"]
    assert validated["folders"] == ["sandbox"]

    app = _app()

    @app.get("/generic", dependencies=[Depends(require_auth)])
    async def generic_route():
        return {"ok": True}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {created['full_value']}"},
    ) as client:
        profile = await client.get("/twin/api/catalog-profile")
        generic = await client.get("/generic")

    assert profile.status_code == 200, profile.text
    assert [row["id"] for row in profile.json()["folders"]] == ["sandbox"]
    assert generic.status_code == 401
    assert next(iter(records.values()))["last_used"] is not None


async def test_general_operator_key_cannot_read_catalog_profile(monkeypatch):
    async def validate_bearer(_workspace: str, _token: str):
        raise AssertionError("a twk_ credential must not reach profile validation")

    monkeypatch.setattr(api_key_store, "validate_bearer", validate_bearer)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=_app()),
        base_url="http://test",
        headers={"Authorization": "Bearer twk_general-secret"},
    ) as client:
        response = await client.get("/twin/api/catalog-profile")

    assert response.status_code == 401
