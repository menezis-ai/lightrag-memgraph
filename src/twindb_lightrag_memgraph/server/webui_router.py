"""Compatibility exports for the Twin WebUI router.

The implementation lives under :mod:`twindb_lightrag_memgraph.server.webui`.
This module preserves the historical import path used by the app wiring,
tests, and downstream callers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .folder import current_folder_id
from .webui.events import _make_event, _utcnow_iso
from .webui.router import (
    _attach_graph_tags_for_documents,
    _cascade_graph_tag_edges,
    _cascade_seed_document_tags,
    _coerce_doc_metadata,
    _delete_doc_from_rag,
    _post_delete_hygiene,
    _doc_matches_active_folder,
    _filter_doc_status_rows,
    _get_doc_for_active_folder,
    _get_rag,
    _graph_tags_for_doc,
    _graph_tags_for_doc_or_none,
    _infer_document_type,
    _list_documents_from_doc_status,
    _project_doc_status_for_webui,
    _require_auth_except_health,
    _status_filter_for_doc_status,
    _status_to_dict,
    _webui_doc_status,
    router,
)
from .webui.routes_graph import (
    _graph_memgraph_label,
    _graph_seed_fallback_allowed,
    _native_graph,
    _validate_graph_entity_tags,
)
from .webui.store import WebuiStore, _stores, get_store, reset_store, set_store
from .webui_models import OpenApiGroup

_router_module = import_module(".webui.router", __package__)
_PATCHABLE_HELPERS = (
    "_attach_graph_tags_for_documents",
    "_delete_doc_from_rag",
    "_post_delete_hygiene",
    "_get_rag",
)


def _sync_patchable_helpers() -> None:
    for name in _PATCHABLE_HELPERS:
        setattr(_router_module, name, globals()[name])


async def _get_doc_for_active_folder(doc_id: str) -> dict[str, Any]:
    _sync_patchable_helpers()
    return await _router_module._get_doc_for_active_folder(doc_id)


async def _graph_tags_for_doc(doc_id: str) -> list[str]:
    _sync_patchable_helpers()
    return await _router_module._graph_tags_for_doc(doc_id)


async def _graph_tags_for_doc_or_none(doc_id: str) -> list[str] | None:
    _sync_patchable_helpers()
    return await _router_module._graph_tags_for_doc_or_none(doc_id)


async def _list_documents_from_doc_status(
    *,
    status: str | None,
    q: str | None,
    tag: str | None,
) -> list[dict[str, Any]]:
    _sync_patchable_helpers()
    return await _router_module._list_documents_from_doc_status(
        status=status, q=q, tag=tag
    )


__all__ = [
    "router",
    "WebuiStore",
    "get_store",
    "set_store",
    "reset_store",
    "OpenApiGroup",
]
