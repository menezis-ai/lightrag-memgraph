"""Retrieval scoping and query/data fallback decisions."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..._constants import (
    RetrievalFilters,
    retrieval_score_context,
    storage_filter_context,
    storage_folder_context,
)
from .models import TwinQueryBody
from .source_filters import _doc_filter_terms, _tag_filter_groups


def _retrieval_filters_from_body(body: TwinQueryBody) -> RetrievalFilters:
    """Map request filters onto the storage-layer ``RetrievalFilters`` contract.

    Reuses the same normaliser as the post-filter guard-rail so the two stay
    in lock-step: ``tag_*`` are lower-cased (case-insensitive tag ids),
    ``doc_*`` are case-preserving. Anything that normalises to a single
    group — the flat form, or a grouped form with one effective group — is
    canonicalised into ``tag_all`` / ``tag_any`` so the storage Cypher stays
    byte-identical to the flat path; only ≥2 effective groups populate
    ``tag_groups`` (OR between groups), leaving the flat sets empty.
    """
    tag_groups = _tag_filter_groups(body.tag_filter_payload)
    doc_required, doc_optional = _doc_filter_terms(body.doc_filter)
    flat_tag_filter = tag_groups if len(tag_groups) == 1 else ()
    return RetrievalFilters(
        doc_all=frozenset(doc_required),
        doc_any=frozenset(doc_optional),
        tag_all=flat_tag_filter[0][0] if flat_tag_filter else frozenset(),
        tag_any=flat_tag_filter[0][1] if flat_tag_filter else frozenset(),
        tag_groups=tag_groups if len(tag_groups) > 1 else (),
        min_score=body.min_score,
    )


@contextmanager
def _retrieval_scope(folder: str, body: TwinQueryBody) -> Iterator[dict[str, float]]:
    """Bind folder membership and retrieval filters during grounding calls.

    Every ``aquery_llm`` / ``aquery_data`` / ``aquery`` issued under this scope
    has its vector retrievals constrained at the Memgraph storage layer to the
    active folder and requested docs/tags/``min_score``. The downstream Sources
    post-filter becomes a guard-rail that removes nothing in the nominal case.
    The yielded mapping captures measured chunk similarities from that same
    grounding call so source projection does not need a second vector query.
    """
    with (
        storage_folder_context(folder),
        storage_filter_context(_retrieval_filters_from_body(body)),
        retrieval_score_context() as retrieval_scores,
    ):
        yield retrieval_scores


def _is_no_retrieval_mode(body: TwinQueryBody) -> bool:
    """True for modes that produce no sourced answer by design."""
    # ``bypass`` and ``only_need_prompt`` are rejected by TwinQueryBody.
    return body.only_need_context


def _has_advanced_filter(body: TwinQueryBody) -> bool:
    filters = _retrieval_filters_from_body(body)
    return filters.has_doc or filters.has_tag


def _query_data_failure_reason(result: dict[str, Any]) -> str | None:
    if result.get("status") != "failure":
        return None
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return None
    reason = metadata.get("failure_reason")
    return str(reason) if reason else None


def _query_data_fallback_mode(body: TwinQueryBody) -> str | None:
    """Chunk-inclusive fallback for filtered structured retrieval.

    Upstream LightRAG's ``aquery_data(mode="hybrid")`` goes through ``kg_query``.
    In that path, no entities/relations means ``no_results`` unless the mode is
    ``mix``. For tag/doc-filtered API calls this reads as a broken filter: the
    caller asked for the tagged corpus, not specifically for "only KG rows".
    Retrying as ``mix`` preserves KG data when it exists and lets filtered chunks
    surface when the graph side is empty.
    """
    if not body.fallback_to_mix:
        return None
    if body.only_need_context:
        return None
    if not _has_advanced_filter(body):
        return None
    if body.mode in {"local", "global", "hybrid"}:
        return "mix"
    return None


def _annotate_query_data_fallback(
    result: dict[str, Any],
    *,
    requested_mode: str,
    fallback_mode: str,
) -> dict[str, Any]:
    annotated = dict(result)
    metadata = dict(result.get("metadata") or {})
    metadata.setdefault("requested_mode", requested_mode)
    metadata["fallback_mode"] = fallback_mode
    metadata["fallback_reason"] = "filtered_graph_mode_no_results"
    annotated["metadata"] = metadata
    return annotated


__all__ = [
    "_annotate_query_data_fallback",
    "_has_advanced_filter",
    "_is_no_retrieval_mode",
    "_query_data_failure_reason",
    "_query_data_fallback_mode",
    "_retrieval_filters_from_body",
    "_retrieval_scope",
]
