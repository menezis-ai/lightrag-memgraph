"""Smoke tests for the replaced /webui mount runtime config injection."""

from __future__ import annotations

import json
import re

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from httpx import ASGITransport, AsyncClient

from twindb_lightrag_memgraph import _mount_twin_ui, _replace_webui_mount


def _extract_twin_config(html: str) -> dict:
    match = re.search(r"window\.__twinConfig\s*=\s*(\{.*?\});", html, re.S)
    assert match is not None
    return json.loads(match.group(1))


async def test_webui_mount_substitutes_runtime_config(monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {
                    "id": "default",
                    "label": "Default folder",
                    "kind": "primary",
                }
            ]
        ),
    )
    monkeypatch.delenv("TWIN_IDP_JWKS_URL", raising=False)

    native_dist = tmp_path / "native"
    native_dist.mkdir()
    (native_dist / "index.html").write_text("<html>native</html>", encoding="utf-8")

    twin_dist = tmp_path / "twin"
    twin_dist.mkdir()
    (twin_dist / "index.html").write_text(
        """
        <!doctype html>
        <html>
          <head><title>Twin</title></head>
          <body>
            <div id="root"></div>
            <script>
              window.__twinConfig = __TWIN_CONFIG_JSON__;
            </script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    app = FastAPI()
    app.mount(
        "/webui",
        StaticFiles(directory=str(native_dist), html=True),
        name="webui",
    )
    _replace_webui_mount(app, str(twin_dist))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        for path in ("/webui/", "/webui/index.html"):
            response = await client.get(path)

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "__TWIN_CONFIG_JSON__" not in response.text
            assert "window.location.hash==='#/login'" in response.text

            config = _extract_twin_config(response.text)
            assert config["apiBaseUrl"] == "/twin/api"
            assert config["lightragBaseUrl"] == ""
            assert config["defaultFolderId"] == "default"
            assert config["folders"] == [
                {
                    "id": "default",
                    "label": "Default folder",
                    "kind": "primary",
                    "description": "",
                    "sources": 0,
                }
            ]
            assert config["defaultFolderId"] == "default"
            assert config["folders"] == [
                {
                    "id": "default",
                    "label": "Default folder",
                    "kind": "primary",
                    "description": "",
                    "sources": 0,
                }
            ]
            assert "debugUser" in config


async def test_twin_ui_mount_preserves_twin_api_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("TWIN_DEFAULT_FOLDER", "default")
    monkeypatch.setenv(
        "TWIN_FOLDERS_JSON",
        json.dumps(
            [
                {
                    "id": "default",
                    "label": "Default folder",
                    "kind": "primary",
                }
            ]
        ),
    )

    twin_dist = tmp_path / "twin"
    twin_dist.mkdir()
    (twin_dist / "index.html").write_text(
        """
        <!doctype html>
        <html>
          <head><title>Twin</title></head>
          <body>
            <div id="root"></div>
            <script>
              window.__twinConfig = __TWIN_CONFIG_JSON__;
            </script>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    app = FastAPI()

    @app.get("/twin/api/probe")
    async def probe():
        return {"ok": True}

    _mount_twin_ui(app, str(twin_dist), "/twin")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        api_response = await client.get("/twin/api/probe")
        assert api_response.status_code == 200
        assert api_response.json() == {"ok": True}

        ui_response = await client.get("/twin/")
        assert ui_response.status_code == 200
        assert ui_response.headers["content-type"].startswith("text/html")
        assert "__TWIN_CONFIG_JSON__" not in ui_response.text

        config = _extract_twin_config(ui_response.text)
        assert config["apiBaseUrl"] == "/twin/api"
