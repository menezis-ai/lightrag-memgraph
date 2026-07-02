#!/usr/bin/env python3
"""Run procedural HTTP smoke checks from a JSON manifest.

The runner is intentionally stdlib-only so it can execute inside restricted
release containers without installing extra packages.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import Any


@dataclass
class SmokeResponse:
    status: int
    headers: Message
    body: bytes


class SmokeFailure(AssertionError):
    """Raised when one smoke check fails."""


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def _env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SmokeFailure(f"Missing required environment variable: {name}")
    return value


def _build_opener(
    base_url: str, ca_bundle: str | None
) -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.HTTPCookieProcessor(cookie_jar)
    ]
    if base_url.startswith("https://"):
        if ca_bundle and Path(ca_bundle).exists():
            context = ssl.create_default_context(cafile=ca_bundle)
        else:
            context = ssl.create_default_context()
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)
    opener.cookie_jar = cookie_jar  # type: ignore[attr-defined]
    return opener


def _join_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    relative = path.lstrip("/")
    return urllib.parse.urljoin(base, relative)


def _json_body(payload: dict[str, Any] | None) -> bytes | None:
    if payload is None:
        return None
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _request(
    opener: urllib.request.OpenerDirector,
    url: str,
    method: str,
    payload: dict[str, Any] | None,
    headers: dict[str, str],
    timeout: float,
) -> SmokeResponse:
    request_headers = {"Accept": "application/json, text/html;q=0.9, */*;q=0.8"}
    request_headers.update(headers)
    body = _json_body(payload)
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return SmokeResponse(
                status=response.status,
                headers=response.headers,
                body=response.read(),
            )
    except urllib.error.HTTPError as error:
        return SmokeResponse(
            status=error.code,
            headers=error.headers,
            body=error.read(),
        )


def _decode_body(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _parse_json(body: bytes, check_name: str) -> Any:
    try:
        return json.loads(_decode_body(body) or "null")
    except json.JSONDecodeError as exc:
        raise SmokeFailure(f"{check_name}: response body is not JSON: {exc}") from exc


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, expected_value in expected.items():
            if key not in actual or not _contains(actual[key], expected_value):
                return False
        return True
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(
            any(_contains(candidate, item) for candidate in actual) for item in expected
        )
    return actual == expected


def _header_value(headers: Message, name: str) -> str:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return ""


def _set_cookie_values(headers: Message) -> list[str]:
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        return list(get_all("Set-Cookie") or [])
    value = _header_value(headers, "Set-Cookie")
    return [value] if value else []


def _assert_response(
    check: dict[str, Any], response: SmokeResponse
) -> dict[str, Any] | None:
    name = check.get("name", check.get("path", "<unnamed>"))
    expected_status = check.get("expect_status")
    if expected_status is not None and response.status != expected_status:
        raise SmokeFailure(
            f"{name}: expected HTTP {expected_status}, got {response.status}"
        )

    expected_body = check.get("expect_body_contains")
    if expected_body is not None and expected_body not in _decode_body(response.body):
        raise SmokeFailure(f"{name}: response body does not contain {expected_body!r}")

    parsed_json: dict[str, Any] | None = None
    if "expect_json" in check or "expect_json_fields" in check:
        parsed = _parse_json(response.body, name)
        if not isinstance(parsed, dict):
            raise SmokeFailure(f"{name}: response JSON is not an object")
        parsed_json = parsed

    if "expect_json" in check and not _contains(parsed_json, check["expect_json"]):
        raise SmokeFailure(f"{name}: response JSON does not contain expected subset")

    for field in check.get("expect_json_fields", []):
        if parsed_json is None or field not in parsed_json:
            raise SmokeFailure(f"{name}: response JSON is missing field {field!r}")

    for header_name, expected_fragment in check.get(
        "expect_header_contains", {}
    ).items():
        value = _header_value(response.headers, header_name)
        if expected_fragment.lower() not in value.lower():
            raise SmokeFailure(
                f"{name}: header {header_name!r} does not contain {expected_fragment!r}"
            )

    if "expect_set_cookie_contains" in check:
        cookie_header = "\n".join(_set_cookie_values(response.headers))
        cookie_header_lower = cookie_header.lower()
        for fragment in check["expect_set_cookie_contains"]:
            if fragment.lower() not in cookie_header_lower:
                raise SmokeFailure(f"{name}: Set-Cookie does not contain {fragment!r}")

    return parsed_json


def _check_payload(
    check: dict[str, Any], auth: dict[str, Any]
) -> dict[str, Any] | None:
    if not check.get("login"):
        return check.get("json")

    username = _env_required(auth.get("username_env", "TWIN_SMOKE_USERNAME"))
    password = _env_required(auth.get("password_env", "TWIN_SMOKE_PASSWORD"))
    return {"username": username, "password": password}


def _write_text(path: str | None, value: str) -> None:
    if not path:
        return
    target = Path(path)
    if target.parent != Path("."):
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")


def run_manifest(manifest_path: str) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    target = manifest.get("target", {})
    auth = manifest.get("auth", {})
    artifacts = manifest.get("artifacts", {})
    base_url = _env_required(target.get("base_url_env", "TWIN_SMOKE_BASE_URL")).rstrip(
        "/"
    )
    timeout = float(target.get("timeout_seconds", 30))
    opener = _build_opener(base_url, target.get("ca_bundle"))
    bearer_token: str | None = None
    logged_in = False
    started = time.time()
    results: list[dict[str, Any]] = []
    trace_lines: list[str] = []

    for check in manifest.get("checks", []):
        name = check.get("name", check.get("path", "<unnamed>"))
        method = check.get("method", "GET").upper()
        path = check["path"]
        check_started = time.time()
        result = {
            "name": name,
            "method": method,
            "path": path,
            "ok": False,
            "status": None,
            "duration_ms": None,
            "error": None,
        }
        try:
            headers = dict(check.get("headers", {}))
            if check.get("after_login") and not logged_in:
                raise SmokeFailure(f"{name}: check requires a successful prior login")
            if (
                check.get("after_login")
                and auth.get("attach_bearer_after_login")
                and bearer_token
            ):
                headers["Authorization"] = f"Bearer {bearer_token}"
            payload = _check_payload(check, auth)
            response = _request(
                opener=opener,
                url=_join_url(base_url, path),
                method=method,
                payload=payload,
                headers=headers,
                timeout=timeout,
            )
            result["status"] = response.status
            parsed_json = _assert_response(check, response)
            result["ok"] = True
            if check.get("login"):
                logged_in = True
                if isinstance(parsed_json, dict):
                    token = parsed_json.get("access_token")
                    bearer_token = token if isinstance(token, str) else None
            if path.rstrip("/") == "/logout":
                logged_in = False
                bearer_token = None
                cookie_jar = getattr(opener, "cookie_jar", None)
                if cookie_jar is not None:
                    cookie_jar.clear()
        except Exception as exc:  # noqa: BLE001 - this is a CLI smoke runner
            result["error"] = str(exc)
        finally:
            result["duration_ms"] = round((time.time() - check_started) * 1000, 2)
            results.append(result)
            status = "PASS" if result["ok"] else "FAIL"
            trace_lines.append(
                f"{status} {method} {path} status={result['status']} "
                f"duration_ms={result['duration_ms']} error={result['error'] or ''}"
            )

    report = {
        "ok": all(result["ok"] for result in results),
        "base_url": base_url,
        "manifest": manifest_path,
        "duration_ms": round((time.time() - started) * 1000, 2),
        "checks": results,
    }
    _write_text(
        artifacts.get("write_json_report"),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _write_text(artifacts.get("write_http_trace"), "\n".join(trace_lines) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Twin runtime smoke checks")
    parser.add_argument("manifest", help="Path to the smoke JSON manifest")
    args = parser.parse_args(argv)

    try:
        report = run_manifest(args.manifest)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"smoke failed before checks: {exc}", file=sys.stderr)
        return 1

    passed = sum(1 for check in report["checks"] if check["ok"])
    total = len(report["checks"])
    print(f"smoke {'passed' if report['ok'] else 'failed'}: {passed}/{total} checks")
    for check in report["checks"]:
        if not check["ok"]:
            print(f"- {check['name']}: {check['error']}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
