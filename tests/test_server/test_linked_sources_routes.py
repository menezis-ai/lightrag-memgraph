"""Central-catalogue proxy contract for PR 4 linked sources."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from twindb_lightrag_memgraph.server import linked_sources_routes as routes
from twindb_lightrag_memgraph.server import webui_router
from twindb_lightrag_memgraph.server.auth import configure_auth
from twindb_lightrag_memgraph.server.idp_jwt import configure_idp, require_admin_user
from twindb_lightrag_memgraph.server.tracing import (
    bind_trace_context,
    resolve_trace_context,
)


@pytest.fixture(autouse=True)
def _dormant_idp(monkeypatch):
    configure_auth()
    configure_idp(None)
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.delenv("TWIN_FOLDERS_JSON", raising=False)
    yield
    configure_auth()
    configure_idp(None)


def _config() -> routes.CatalogProxyConfig:
    return routes.CatalogProxyConfig(
        base_url="https://catalog.test", credential="tck_prefix_secret"
    )


def _app(transport: httpx.AsyncBaseTransport, *, admin: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(
        routes.build_linked_sources_router(_config(), transport=transport),
        prefix="/twin/api",
    )
    if admin:
        app.dependency_overrides[require_admin_user] = lambda: {
            "sso_subject": "steward@example.test"
        }
    return app


@asynccontextmanager
async def _client(
    transport: httpx.AsyncBaseTransport,
    *,
    admin: bool = True,
    include_activity: bool = False,
) -> AsyncIterator[httpx.AsyncClient]:
    app = _app(transport, admin=admin)
    if include_activity:
        app.include_router(webui_router.router, prefix="/twin/api")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client


def test_catalog_proxy_config_is_all_or_nothing():
    assert routes.CatalogProxyConfig.from_env({}) is None
    with pytest.raises(ValueError, match="must be set together"):
        routes.CatalogProxyConfig.from_env({"TWIN_CATALOG_URL": "https://catalog"})
    with pytest.raises(ValueError, match=r"http\(s\) origin"):
        routes.CatalogProxyConfig.from_env(
            {
                "TWIN_CATALOG_URL": "https://user:pass@catalog.test?token=x",
                "TWIN_CATALOG_INSTANCE_CREDENTIAL": "secret",
            }
        )
    with pytest.raises(ValueError, match=r"http\(s\) origin"):
        routes.CatalogProxyConfig.from_env(
            {
                "TWIN_CATALOG_URL": "https://catalog.test/api",
                "TWIN_CATALOG_INSTANCE_CREDENTIAL": "secret",
            }
        )


async def test_list_combines_application_and_folder_links():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.headers["authorization"] == "Bearer tck_prefix_secret"
        if request.url.path.endswith("/applications"):
            return httpx.Response(200, json=[{"auid": "AP011121"}])
        assert request.headers["x-twin-folder"] == "default"
        return httpx.Response(200, json=[{"id": "link-1", "doc_type": "de"}])

    async with _client(httpx.MockTransport(handler)) as client:
        response = await client.get("/twin/api/linked-sources")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "application": {"auid": "AP011121"},
        "links": [{"id": "link-1", "doc_type": "de"}],
    }
    assert len(seen) == 2


async def test_catalogue_requests_propagate_the_active_distributed_parent():
    seen: list[httpx.Request] = []
    context = resolve_trace_context(
        {"traceparent": ("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=[])

    with bind_trace_context(context):
        async with _client(httpx.MockTransport(handler)) as client:
            response = await client.get("/twin/api/linked-sources")

    assert response.status_code == 200
    assert len(seen) == 2
    for request in seen:
        assert request.headers["traceparent"] == context.traceparent
        assert request.headers["x-trace-id"] == context.trace_id
        assert request.headers["authorization"] == "Bearer tck_prefix_secret"


async def test_preview_and_mutations_forward_without_leaking_credential(monkeypatch):
    requests: list[tuple[str, str, dict]] = []
    activity = AsyncMock()
    monkeypatch.setattr(routes, "emit_activity_event", activity)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        requests.append((request.method, request.url.path, body))
        if request.url.path.endswith("/preview"):
            return httpx.Response(200, json={"verdict": {"safe": True}})
        return httpx.Response(
            200,
            json={
                "link": {
                    "id": "9f69535d-7f87-46e8-af39-c4a0f0e2193f",
                    "auid": "AP011121",
                    "url": "https://confluence.test/display/PF/X",
                },
                "revision": {"state": "published"},
            },
        )

    async with _client(httpx.MockTransport(handler)) as client:
        preview = await client.post(
            "/twin/api/linked-sources/preview",
            json={
                "operation": "create",
                "body": {
                    "url": "https://confluence.test/display/PF/X",
                    "doc_type": "de",
                    "public": False,
                },
            },
        )
        created = await client.post(
            "/twin/api/linked-sources",
            json={
                "url": "https://confluence.test/display/PF/X",
                "doc_type": "de",
                "public": False,
            },
        )
        patched = await client.patch(
            "/twin/api/linked-sources/9f69535d-7f87-46e8-af39-c4a0f0e2193f",
            json={"title": None, "language": None, "expected_version": 1},
        )
        disabled = await client.post(
            "/twin/api/linked-sources/9f69535d-7f87-46e8-af39-c4a0f0e2193f/disable",
            json={"expected_version": 2},
        )

    assert preview.status_code == 200
    assert created.status_code == 201
    assert patched.status_code == disabled.status_code == 200
    assert requests == [
        (
            "POST",
            "/v1/instance/revisions/preview",
            {
                "operation": "create",
                "body": {
                    "url": "https://confluence.test/display/PF/X",
                    "doc_type": "de",
                    "public": False,
                },
            },
        ),
        (
            "POST",
            "/v1/instance/links",
            {
                "url": "https://confluence.test/display/PF/X",
                "doc_type": "de",
                "public": False,
            },
        ),
        (
            "PATCH",
            "/v1/instance/links/9f69535d-7f87-46e8-af39-c4a0f0e2193f",
            {"title": None, "language": None, "expected_version": 1},
        ),
        (
            "POST",
            "/v1/instance/links/9f69535d-7f87-46e8-af39-c4a0f0e2193f/disable",
            {"expected_version": 2},
        ),
    ]
    assert [call.kwargs["kind"] for call in activity.await_args_list] == [
        "linked-source-declared",
        "linked-source-updated",
        "linked-source-disabled",
    ]
    assert {call.kwargs["target_id"] for call in activity.await_args_list} == {
        "9f69535d-7f87-46e8-af39-c4a0f0e2193f"
    }
    assert "tck_prefix_secret" not in repr(requests)


async def test_catalog_auth_failure_is_connector_503():
    transport = httpx.MockTransport(lambda _: httpx.Response(401, json={"detail": "x"}))
    async with _client(transport) as client:
        response = await client.get("/twin/api/linked-sources")
    assert response.status_code == 503
    assert response.json()["detail"] == "central catalogue authentication failed"


async def test_write_routes_remain_admin_gated():
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    async with _client(transport, admin=False) as client:
        response = await client.post(
            "/twin/api/linked-sources",
            json={
                "url": "https://confluence.test/display/PF/X",
                "doc_type": "de",
                "public": False,
            },
        )
    assert response.status_code == 403


async def test_linked_sources_router_rejects_anonymous_when_mounted_directly():
    configure_auth(api_key="test-infrastructure-key")
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=[]))
    async with _client(transport) as client:
        response = await client.get("/twin/api/linked-sources")
    assert response.status_code == 401


async def test_structured_version_conflict_survives_proxy():
    detail = {
        "message": "row_version mismatch — reload and retry",
        "current_version": 3,
    }
    transport = httpx.MockTransport(
        lambda _: httpx.Response(409, json={"detail": detail})
    )
    async with _client(transport) as client:
        response = await client.patch(
            "/twin/api/linked-sources/9f69535d-7f87-46e8-af39-c4a0f0e2193f",
            json={"title": "Runbook", "expected_version": 2},
        )
    assert response.status_code == 409
    assert response.json() == {"detail": detail}


async def test_link_id_is_validated_before_upstream_interpolation():
    transport = httpx.MockTransport(lambda _: pytest.fail("upstream must not run"))
    async with _client(transport) as client:
        response = await client.patch(
            "/twin/api/linked-sources/not-a-uuid",
            json={"title": "Runbook", "expected_version": 1},
        )
    assert response.status_code == 422


async def test_declared_source_remains_readable_in_real_activity_feed():
    webui_router.reset_store()

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "link": {
                    "id": "9f69535d-7f87-46e8-af39-c4a0f0e2193f",
                    "auid": "AP011121",
                    "url": "https://confluence.test/display/PF/X",
                },
                "revision": {"state": "published"},
            },
        )

    try:
        async with _client(
            httpx.MockTransport(handler), include_activity=True
        ) as client:
            declared = await client.post(
                "/twin/api/linked-sources",
                json={
                    "url": "https://confluence.test/display/PF/X",
                    "doc_type": "de",
                    "public": False,
                },
            )
            feed = await client.get(
                "/twin/api/activity",
                params={
                    "kind": "linked-source-declared",
                    "resource.id": "9f69535d-7f87-46e8-af39-c4a0f0e2193f",
                },
            )
    finally:
        webui_router.reset_store()

    assert declared.status_code == 201
    assert feed.status_code == 200, feed.text
    assert feed.json()["items"][0]["kind"] == "linked-source-declared"
    assert (
        feed.json()["items"][0]["target"]["id"]
        == "9f69535d-7f87-46e8-af39-c4a0f0e2193f"
    )


async def test_catalog_proxy_client_reuses_its_http_pool_until_closed():
    proxy = routes.CatalogProxyClient(
        _config(), transport=httpx.MockTransport(lambda _: httpx.Response(200, json=[]))
    )
    client = proxy._client
    await proxy.request("GET", "/v1/instance/applications")
    await proxy.request("GET", "/v1/instance/applications")
    assert proxy._client is client
    assert not client.is_closed
    await proxy.aclose()
    assert client.is_closed


async def test_internal_catalog_proxy_surface_is_hidden_but_remains_callable():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=[]))
    app = _app(transport)
    internal_paths = {
        "/twin/api/linked-sources",
        "/twin/api/linked-sources/preview",
        "/twin/api/linked-sources/{link_id}",
        "/twin/api/linked-sources/{link_id}/disable",
    }
    assert internal_paths.isdisjoint(app.openapi()["paths"])
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/twin/api/linked-sources")

    assert response.status_code == 200, response.text


def test_runtime_config_gates_catalogue_capability(monkeypatch):
    from twindb_lightrag_memgraph import _build_runtime_config

    monkeypatch.delenv("TWIN_CATALOG_URL", raising=False)
    monkeypatch.delenv("TWIN_CATALOG_INSTANCE_CREDENTIAL", raising=False)
    assert _build_runtime_config()["catalogEnabled"] is False

    monkeypatch.setenv("TWIN_CATALOG_URL", "https://catalogue.example.com")
    monkeypatch.setenv("TWIN_CATALOG_INSTANCE_CREDENTIAL", "tic_secret")
    assert _build_runtime_config()["catalogEnabled"] is True
