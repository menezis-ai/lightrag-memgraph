"""Pure source/doc/tag filter helpers for Twin query routes."""

from __future__ import annotations

from typing import Any

UNKNOWN_SOURCE_NAME = "unknown source"


def _source_file_path_candidate(source: dict[str, Any]) -> str | None:
    """Return a reliable file_path candidate from a projected source."""
    name = source.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    if source.get("_lightrag_reference_name_fallback"):
        return None
    if name == UNKNOWN_SOURCE_NAME:
        return None
    return name.strip()


def _split_source_ids(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        return []
    return [
        item.strip() for item in raw.replace("<SEP>", ",").split(",") if item.strip()
    ]


def _tag_filter_terms(
    tag_filter: dict[str, list[str]] | None,
) -> tuple[set[str], set[str]]:
    if not tag_filter:
        return set(), set()
    required = {
        tag.strip().lower()
        for tag in tag_filter.get("all", [])
        if isinstance(tag, str) and tag.strip()
    }
    optional = {
        tag.strip().lower()
        for tag in tag_filter.get("any", [])
        if isinstance(tag, str) and tag.strip()
    }
    return required, optional


def _doc_filter_terms(
    doc_filter: dict[str, list[str]] | None,
) -> tuple[set[str], set[str]]:
    if not doc_filter:
        return set(), set()
    required = {
        doc.strip()
        for doc in doc_filter.get("all", [])
        if isinstance(doc, str) and doc.strip()
    }
    optional = {
        doc.strip()
        for doc in doc_filter.get("any", [])
        if isinstance(doc, str) and doc.strip()
    }
    return required, optional


def _doc_tags_match_filter(
    doc_tags: set[str], tag_filter: dict[str, list[str]] | None
) -> bool:
    """Audit C2: doc tags come from the ``TAGGED_WITH`` graph relation,
    never from ``DocStatus.metadata.tags`` (which can lag the WebUI
    retag flow and produces a misleading filter result).
    """
    required, optional = _tag_filter_terms(tag_filter)
    if not required and not optional:
        return True
    if required and not required.issubset(doc_tags):
        return False
    if optional and doc_tags.isdisjoint(optional):
        return False
    return True


def _source_doc_candidates(source: dict[str, Any]) -> set[str]:
    # ``name`` is used as a document-level fallback only when it is a real
    # identifier-like value coming from payload metadata. Synthetic fallbacks from
    # the envelope projection cannot be trusted as filter evidence.
    # We rely on an explicit marker emitted during projection.
    synthetic_name_marker = "_lightrag_reference_name_fallback"

    out = {
        str(source[key]).strip()
        for key in ("doc_id", "name")
        if isinstance(source.get(key), str) and source.get(key).strip()
    }
    if UNKNOWN_SOURCE_NAME in out:
        out.discard(UNKNOWN_SOURCE_NAME)
    if source.get(synthetic_name_marker):
        out.discard(str(source.get("name") or "").strip())
    return out


def _source_matches_doc_filter(
    source: dict[str, Any], doc_filter: dict[str, list[str]] | None
) -> bool:
    """Mirror the storage-layer ``doc_all`` / ``doc_any`` semantics.

    Source candidates are the source's own doc identifiers (``doc_id`` + path),
    so this is the *set* form of ``vector_impl._doc_conditions_set``:
    ``all`` → requested ⊆ candidates (strict — NOT the union-as-``any`` the
    legacy post-filter conflated); ``any`` → candidates ∩ requested ≠ ∅; both
    AND-ed when present. Same shape as :func:`_doc_tags_match_filter`. The real
    exclusion lives in the Cypher; this is the last-line guard if the envelope
    shape shifts under a LightRAG bump.
    """
    required, optional = _doc_filter_terms(doc_filter)
    if not required and not optional:
        return True
    candidates = _source_doc_candidates(source)
    if not candidates:
        return False
    if required and not required.issubset(candidates):
        return False
    if optional and candidates.isdisjoint(optional):
        return False
    return True


__all__ = [
    "UNKNOWN_SOURCE_NAME",
    "_doc_filter_terms",
    "_doc_tags_match_filter",
    "_source_doc_candidates",
    "_source_file_path_candidate",
    "_source_matches_doc_filter",
    "_split_source_ids",
    "_tag_filter_terms",
]
