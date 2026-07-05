"""LightRAG ``QueryParam`` construction helpers for Twin query routes."""

from __future__ import annotations

import dataclasses
from typing import Any


def _query_param_kwargs(body: Any, *, stream: bool = False) -> dict[str, Any]:
    param_kwargs: dict[str, Any] = {
        "mode": body.mode,
        "top_k": body.top_k,
        "only_need_context": body.only_need_context,
        "only_need_prompt": body.only_need_prompt,
        "stream": stream,
    }
    if body.response_type is not None:
        param_kwargs["response_type"] = body.response_type
    if body.chunk_top_k is not None:
        param_kwargs["chunk_top_k"] = body.chunk_top_k
    if body.max_entity_tokens is not None:
        param_kwargs["max_entity_tokens"] = body.max_entity_tokens
    if body.max_relation_tokens is not None:
        param_kwargs["max_relation_tokens"] = body.max_relation_tokens
    if body.max_total_tokens is not None:
        param_kwargs["max_total_tokens"] = body.max_total_tokens
    if body.hl_keywords:
        param_kwargs["hl_keywords"] = body.hl_keywords
    if body.ll_keywords:
        param_kwargs["ll_keywords"] = body.ll_keywords
    if body.conversation_history:
        param_kwargs["conversation_history"] = body.conversation_history
    if body.history_turns is not None:
        param_kwargs["history_turns"] = body.history_turns
    if body.user_prompt is not None and body.user_prompt.strip():
        param_kwargs["user_prompt"] = body.user_prompt.strip()
    if body.enable_rerank is not None:
        param_kwargs["enable_rerank"] = body.enable_rerank
    if body.tag_filter is not None:
        param_kwargs["tag_filter"] = body.tag_filter
    if body.doc_filter is not None:
        param_kwargs["doc_filter"] = body.doc_filter
    return param_kwargs


def _query_param_ctor_fields(query_param_cls: Any) -> set[str] | None:
    """Constructor-accepted field names for the installed ``QueryParam``.

    Returns ``None`` when the fields cannot be introspected (non-dataclass),
    in which case callers fall back to passing every kwarg through.
    """
    try:
        return {f.name for f in dataclasses.fields(query_param_cls)}
    except TypeError:
        return None


def _make_query_param(query_param_cls: Any, param_kwargs: dict[str, Any]) -> Any:
    """Build a ``QueryParam`` that is resilient to upstream field churn.

    LightRAG renames/removes ``QueryParam`` fields between minor releases
    (e.g. ``history_turns`` was dropped in 1.5, and ``tag_filter`` is a Twin
    extension never present upstream). Passing such a kwarg straight to the
    constructor raises ``TypeError`` and 500s the whole query endpoint. We
    instead route only constructor-known kwargs through ``__init__`` and apply
    the rest as runtime attributes, so downstream code that understands them
    still sees them and the request never crashes.
    """
    fields = _query_param_ctor_fields(query_param_cls)
    if fields is None:
        return query_param_cls(**param_kwargs)

    ctor_kwargs = {k: v for k, v in param_kwargs.items() if k in fields}
    extra_kwargs = {k: v for k, v in param_kwargs.items() if k not in fields}
    param = query_param_cls(**ctor_kwargs)
    for key, value in extra_kwargs.items():
        setattr(param, key, value)
    return param


__all__ = [
    "_make_query_param",
    "_query_param_ctor_fields",
    "_query_param_kwargs",
]
