"""OpenAPI documentation contract.

The generated spec is user-facing documentation; these tests pin the
guarantees it makes (PR #417 + its review):

- every operation carries a description;
- the ``X-Twin-Folder`` header is advertised on the routes that honour it,
  including the query routes that resolve the folder inside their handler;
- public routes advertise accurate security (no lock on ``/login``,
  ``/logout``, ``/ready``; anonymous-allowed on ``/auth-status``; the
  overlay ``/health`` clears its router-level bearer requirement);
- no stakeholder name leaks into the published document;
- shipped examples validate against their own constraints.
"""

from __future__ import annotations

import json

import pytest

from twindb_lightrag_memgraph._constants import validate_identifier
from twindb_lightrag_memgraph.server.app import create_app

_METHODS = {"get", "post", "put", "patch", "delete"}


@pytest.fixture(scope="module")
def spec() -> dict:
    return create_app().openapi()


def _operations(spec: dict):
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if method in _METHODS:
                yield path, method, op


def test_every_operation_is_documented(spec):
    missing = [
        f"{method.upper()} {path}"
        for path, method, op in _operations(spec)
        if not (op.get("description") or "").strip()
    ]
    assert missing == [], f"operations without a description: {missing}"


def test_every_parameter_is_documented(spec):
    missing = [
        f"{method.upper()} {path}: {parameter.get('in')} {parameter.get('name')}"
        for path, method, op in _operations(spec)
        for parameter in op.get("parameters", [])
        if not (parameter.get("description") or "").strip()
    ]
    assert missing == [], f"parameters without a description: {missing}"


def test_framework_response_placeholders_are_replaced(spec):
    placeholders = {"Successful Response", "Validation Error"}
    generic = [
        f"{method.upper()} {path}: {code} {response.get('description')}"
        for path, method, op in _operations(spec)
        for code, response in op.get("responses", {}).items()
        if response.get("description") in placeholders
    ]
    assert generic == [], f"generic response documentation: {generic}"


def test_generated_success_description_reads_as_an_action(spec):
    update = spec["paths"]["/twin/api/settings/vision"]["put"]
    assert update["responses"]["200"]["description"] == (
        "Request completed successfully: update the image and procedure "
        "ingestion settings (admin)."
    )


def test_api_key_contract_is_fully_documented(spec):
    collection = spec["paths"]["/twin/api/settings/api-keys"]
    create = collection["post"]
    assert create["responses"]["201"]["description"].startswith("API key metadata")
    assert {"401", "403", "422"}.issubset(create["responses"])
    create_schema = spec["components"]["schemas"]["ApiKeyCreate"]
    assert create_schema["examples"] == [{"name": "reporting-script"}]
    assert create_schema["properties"]["name"]["description"]

    revoke = spec["paths"]["/twin/api/settings/api-keys/{key_id}"]["delete"]
    key_id = next(p for p in revoke["parameters"] if p["name"] == "key_id")
    assert key_id["required"] is True
    assert key_id["description"]
    assert {"401", "403", "404", "422"}.issubset(revoke["responses"])


def test_query_routes_advertise_the_folder_header(spec):
    # These handlers resolve the folder via resolve_folder_for_request()
    # instead of bind_request_folder — the documentary dependency must keep
    # the header visible in the spec (review finding #2).
    targets = [
        ("post", "/query"),
        ("post", "/twin/api/query"),
        ("post", "/twin/api/query/data"),
        ("post", "/twin/api/query/stream"),
    ]
    for method, path in targets:
        op = spec["paths"][path][method]
        names = {p.get("name") for p in op.get("parameters", [])}
        assert "X-Twin-Folder" in names, f"{path} parameters: {sorted(names)}"


def test_folder_bound_routes_advertise_the_folder_header(spec):
    op = spec["paths"]["/twin/api/documents"]["get"]
    names = {p.get("name") for p in op.get("parameters", [])}
    assert "X-Twin-Folder" in names


def test_public_routes_do_not_require_auth(spec):
    for method, path in [
        ("post", "/login"),
        ("post", "/logout"),
        ("get", "/ready"),
    ]:
        assert not spec["paths"][path][method].get(
            "security"
        ), f"{path} must not advertise a security requirement"
    # Bearer-optional endpoints: anonymous access is explicitly allowed
    # (the empty security requirement is present alongside the bearer one).
    for path in ("/auth-status", "/twin/api/health"):
        security = spec["paths"][path]["get"]["security"]
        assert {} in security, f"{path} must allow anonymous access: {security}"


def test_no_stakeholder_names_in_the_spec(spec):
    text = json.dumps(spec).lower()
    for token in ("bnp", "cib", "sigilum", "menezis"):
        assert token not in text, f"stakeholder token {token!r} leaked into the spec"


def test_folder_create_example_is_a_valid_identifier(spec):
    schema = spec["components"]["schemas"]["FolderCreate"]
    examples = schema.get("examples") or []
    assert examples, "FolderCreate must ship a body example"
    # The shipped example must pass the same identifier rule the handler
    # enforces — an official example that 422s is worse than none.
    validate_identifier(examples[0]["id"], "folder")
