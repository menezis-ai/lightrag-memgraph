"""Final OpenAPI documentation pass shared by both server topologies.

FastAPI supplies technically valid defaults for responses, but labels such as
``Successful Response`` and ``Validation Error`` are framework placeholders,
not useful operator documentation.  The Twin API explorer renders this schema
verbatim, so normalize only those defaults and advertise the authentication
failure that router-level dependencies cannot add to OpenAPI automatically.
Explicit route documentation always wins.
"""

from __future__ import annotations

from typing import Any

_HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _successful_response_description(status: str, summary: str) -> str:
    action = summary[:1].lower() + summary[1:]
    if status == "201":
        return f"Request completed successfully and created a resource: {action}."
    if status == "204":
        return f"Request completed successfully with no response body: {action}."
    return f"Request completed successfully: {action}."


def _document_operation(operation: dict[str, Any]) -> None:
    summary = str(operation.get("summary") or "The operation").rstrip(".")
    responses = operation.setdefault("responses", {})
    if not isinstance(responses, dict):
        return

    for status, raw_response in responses.items():
        if not isinstance(raw_response, dict):
            continue
        description = raw_response.get("description")
        if description == "Successful Response":
            raw_response["description"] = _successful_response_description(
                str(status), summary
            )
        elif description == "Validation Error":
            raw_response["description"] = (
                "The request body or parameters failed validation; inspect "
                "`detail` for the invalid or missing fields."
            )

    security = operation.get("security")
    credentials_required = (
        isinstance(security, list)
        and bool(security)
        and all(
            isinstance(requirement, dict) and requirement for requirement in security
        )
    )
    if credentials_required:
        responses.setdefault(
            "401",
            {
                "description": (
                    "Authentication credentials are missing, invalid, or expired."
                )
            },
        )


def install_openapi_documentation(app: Any) -> None:
    """Install the final documentation pass on a FastAPI-compatible app.

    This is deliberately a wrapper around the app's existing ``openapi``
    callable so it also works on the production LightRAG host application.
    Calling it more than once is harmless.
    """

    if getattr(app, "_twindb_openapi_documentation_installed", False):
        return
    original_openapi = app.openapi

    def documented_openapi() -> dict[str, Any]:
        schema = original_openapi()
        paths = schema.get("paths", {}) if isinstance(schema, dict) else {}
        if isinstance(paths, dict):
            for path_item in paths.values():
                if not isinstance(path_item, dict):
                    continue
                for method, operation in path_item.items():
                    if method in _HTTP_METHODS and isinstance(operation, dict):
                        _document_operation(operation)
        return schema

    # A host can have generated and cached its schema before the Twin overlay
    # is mounted. Clear that cache so the next request sees every Twin route.
    app.openapi_schema = None
    app.openapi = documented_openapi
    app._twindb_openapi_documentation_installed = True


__all__ = ["install_openapi_documentation"]
