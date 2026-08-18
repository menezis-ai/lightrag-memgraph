"""Pure source/doc/tag filter helpers for Twin query routes."""

from __future__ import annotations

from typing import Any, TypedDict


class FlatTagFilterPayload(TypedDict, total=False):
    """Flat tag-filter wire payload: ``all`` AND ``any``, both optional."""

    all: list[str]
    any: list[str]


class GroupedTagFilterPayload(TypedDict):
    """Grouped tag-filter wire payload: OR between conjunctive groups."""

    groups: list[FlatTagFilterPayload]


# Wire payload of a tag filter, EITHER form. Named so every signature in the
# chain says both forms are welcome; the pydantic wire models in ``models.py``
# own the validation, these TypedDicts describe the validated shape statically.
TagFilter = FlatTagFilterPayload | GroupedTagFilterPayload

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
    """Extract ``(required, optional)`` terms from a FLAT tag filter.

    The grouped form must go through :func:`_tag_filter_groups`: reading a
    ``groups`` payload here would return empty sets and silently disable
    filtering, so it is refused loudly instead.
    """
    if not tag_filter:
        return set(), set()
    if "groups" in tag_filter:
        raise ValueError("grouped tag_filter must be normalised via _tag_filter_groups")
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


def _tag_filter_groups(
    tag_filter: TagFilter | None,
) -> tuple[tuple[frozenset[str], frozenset[str]], ...]:
    """Normalise both tag-filter wire forms into OR-able groups.

    Flat ``{"all": [...], "any": [...]}`` becomes a single group; the grouped
    ``{"groups": [{...}, ...]}`` form yields one group per entry. Each group is
    a ``(required, optional)`` pair with the flat semantics (``required`` ⊆
    tags AND tags ∩ ``optional`` ≠ ∅). Groups whose terms strip to nothing are
    dropped — mirroring the flat form, where a blank-only filter is inactive —
    so an all-blank filter normalises to ``()`` (no filtering) rather than a
    vacuously-true group.
    """
    if not tag_filter:
        return ()
    raw_groups = tag_filter.get("groups")
    if raw_groups is None:
        candidates: list[Any] = [tag_filter]
    elif isinstance(raw_groups, list):
        candidates = raw_groups
    else:
        return ()
    groups: list[tuple[frozenset[str], frozenset[str]]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        required, optional = _tag_filter_terms(candidate)
        if required or optional:
            groups.append((frozenset(required), frozenset(optional)))
    return tuple(groups)


def _tag_filter_active(tag_filter: TagFilter | None) -> bool:
    """True when the tag filter (flat or grouped) carries at least one term."""
    return bool(_tag_filter_groups(tag_filter))


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


def _doc_tags_match_filter(doc_tags: set[str], tag_filter: TagFilter | None) -> bool:
    """Audit C2: doc tags come from the ``TAGGED_WITH`` graph relation,
    never from ``DocStatus.metadata.tags`` (which can lag the WebUI
    retag flow and produces a misleading filter result).

    Grouped filters match when at least one group matches (OR between
    groups); within a group the flat semantics are unchanged.
    """
    groups = _tag_filter_groups(tag_filter)
    if not groups:
        return True
    for required, optional in groups:
        if required and not required.issubset(doc_tags):
            continue
        if optional and doc_tags.isdisjoint(optional):
            continue
        return True
    return False


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
    "FlatTagFilterPayload",
    "GroupedTagFilterPayload",
    "TagFilter",
    "UNKNOWN_SOURCE_NAME",
    "_doc_filter_terms",
    "_doc_tags_match_filter",
    "_source_doc_candidates",
    "_source_file_path_candidate",
    "_source_matches_doc_filter",
    "_split_source_ids",
    "_tag_filter_active",
    "_tag_filter_groups",
    "_tag_filter_terms",
]
