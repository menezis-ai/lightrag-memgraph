from __future__ import annotations

import json
from email.message import Message
from types import SimpleNamespace
from urllib.parse import urlparse

from tests.smoke import run_smoke


def _headers(values: dict[str, str]) -> Message:
    message = Message()
    for key, value in values.items():
        message[key] = value
    return message


def _response(
    status: int,
    payload: dict[str, object] | str,
    headers: dict[str, str] | None = None,
) -> run_smoke.SmokeResponse:
    if isinstance(payload, str):
        body = payload.encode("utf-8")
    else:
        body = json.dumps(payload).encode("utf-8")
    return run_smoke.SmokeResponse(status=status, headers=_headers(headers or {}), body=body)


def test_runtime_smoke_runner_executes_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("TWIN_SMOKE_BASE_URL", "https://runtime.example.intra")
    monkeypatch.setenv("ARTIFACTORY_USERNAME", "fixture-user")
    monkeypatch.setenv("ARTIFACTORY_PASSWORD", "fixture-pass")
    monkeypatch.setattr(
        run_smoke,
        "_build_opener",
        lambda base_url, ca_bundle: SimpleNamespace(cookie_jar=SimpleNamespace(clear=lambda: None)),
    )

    def fake_request(opener, url, method, payload, headers, timeout):
        path = urlparse(url).path
        authenticated = headers.get("Authorization") == "Bearer fixture-token"
        if path == "/webui":
            return _response(200, "<html><title>Twin</title></html>")
        if path == "/auth-status":
            return _response(
                200,
                {
                    "auth_enabled": True,
                    "authenticated": authenticated,
                    "login_required": not authenticated,
                },
            )
        if path == "/twin/api/documents" and not authenticated:
            return _response(401, {"detail": "Not authenticated"}, {"WWW-Authenticate": "Bearer"})
        if path == "/login":
            if payload != {"username": "fixture-user", "password": "fixture-pass"}:
                return _response(401, {"detail": "bad credentials"})
            return _response(
                200,
                {
                    "access_token": "fixture-token",
                    "token_type": "bearer",
                    "expires_in": 3600,
                },
                {"Set-Cookie": "twin_local_token=fixture; HttpOnly; Secure; SameSite=lax"},
            )
        if path in {
            "/documents",
            "/twin/api/health",
            "/twin/api/documents",
            "/twin/api/graph/entities",
        }:
            return _response(200 if authenticated else 401, {"ok": authenticated})
        if path == "/logout":
            return _response(200, {"ok": True})
        return _response(404, {"detail": "not found"})

    monkeypatch.setattr(run_smoke, "_request", fake_request)
    report_path = tmp_path / "report.json"
    trace_path = tmp_path / "trace.log"
    manifest = {
        "target": {
            "base_url_env": "TWIN_SMOKE_BASE_URL",
            "timeout_seconds": 5,
        },
        "auth": {
            "username_env": "ARTIFACTORY_USERNAME",
            "password_env": "ARTIFACTORY_PASSWORD",
            "attach_bearer_after_login": True,
        },
        "artifacts": {
            "write_json_report": str(report_path),
            "write_http_trace": str(trace_path),
        },
        "checks": [
            {
                "name": "webui",
                "method": "GET",
                "path": "/webui",
                "expect_status": 200,
                "expect_body_contains": "Twin",
            },
            {
                "name": "unauthenticated",
                "method": "GET",
                "path": "/auth-status",
                "expect_status": 200,
                "expect_json": {"authenticated": False, "login_required": True},
            },
            {
                "name": "protected",
                "method": "GET",
                "path": "/twin/api/documents",
                "expect_status": 401,
                "expect_header_contains": {"www-authenticate": "Bearer"},
            },
            {
                "name": "login",
                "method": "POST",
                "path": "/login",
                "login": True,
                "expect_status": 200,
                "expect_json_fields": ["access_token", "token_type", "expires_in"],
                "expect_set_cookie_contains": [
                    "twin_local_token=",
                    "HttpOnly",
                    "Secure",
                    "SameSite=lax",
                ],
            },
            {
                "name": "authenticated",
                "method": "GET",
                "path": "/auth-status",
                "after_login": True,
                "expect_status": 200,
                "expect_json": {"authenticated": True, "login_required": False},
            },
            {
                "name": "documents",
                "method": "GET",
                "path": "/documents",
                "after_login": True,
                "expect_status": 200,
            },
            {
                "name": "twin health",
                "method": "GET",
                "path": "/twin/api/health",
                "after_login": True,
                "expect_status": 200,
            },
            {
                "name": "twin documents",
                "method": "GET",
                "path": "/twin/api/documents",
                "after_login": True,
                "expect_status": 200,
            },
            {
                "name": "graph",
                "method": "GET",
                "path": "/twin/api/graph/entities",
                "after_login": True,
                "expect_status": 200,
            },
            {
                "name": "logout",
                "method": "POST",
                "path": "/logout",
                "after_login": True,
                "expect_status": 200,
            },
            {
                "name": "logged out",
                "method": "GET",
                "path": "/auth-status",
                "expect_status": 200,
                "expect_json": {"authenticated": False, "login_required": True},
            },
        ],
    }
    manifest_path = tmp_path / "smoke.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = run_smoke.run_manifest(str(manifest_path))

    assert report["ok"] is True
    assert len(report["checks"]) == len(manifest["checks"])
    assert report_path.exists()
    assert trace_path.read_text(encoding="utf-8").count("PASS") == len(manifest["checks"])
