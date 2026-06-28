"""Memgraph → WebUI graph reader.

Reads the live LightRAG-extracted entity/relation graph straight from
Memgraph and reshapes it into the typed contract the React port consumes
(`GraphEntity` / `GraphRelation` Pydantic models in `webui_models.py`).

Three concerns are colocated here:

1. **Cypher queries** — minimal, label-scoped to the active workspace.
   Each entity has properties (entity_id, entity_type, description,
   source_id, …); edges share the synthetic `DIRECTED` relation type
   with properties (weight, keywords, description, source_id).
2. **Type mapping** — LightRAG's extractor emits free-form
   `entity_type` strings ("organization", "person", "concept", …). The
   WebUI enum is closed (`PRODUCT | TECHNOLOGY | CONCEPT | ORG | PERSON
   | LOCATION`). We do a best-effort lookup with a stable fallback.
3. **Deterministic layout** — entity ids are hash-bucketed per type and
   scattered inside a per-type rectangle so the layout is stable across
   page loads without running a force simulation in the browser.

Write persistence (M12 batch 2) lives in the "Writes" section below:
entity/relation patch, create, and delete helpers backing the
PATCH/POST/DELETE `/twin/api/graph/*` routes in `webui_router.py`.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Sequence

import json

from .._pool import acquire_write_slot, get_read_session, get_session

logger = logging.getLogger(__name__)

try:  # version-skew guard (see feedback_lightrag_version_skew)
    from lightrag.constants import GRAPH_FIELD_SEP as _GRAPH_FIELD_SEP
except Exception:  # pragma: no cover - defensive
    _GRAPH_FIELD_SEP = "<SEP>"


# ----------------------------------------------------------------------
# Exceptions raised by the write helpers (M12 batch 3 contract)
# ----------------------------------------------------------------------


class GraphEntityCreateError(Exception):
    """Base type for failures in :func:`create_graph_entity`.

    The route handler in ``webui_router.py`` discriminates these to map
    each cause to a distinct HTTP status (409 / 500 / 503). Returning
    ``None`` is no longer a valid failure signal — see TR-KG-01.
    """


class EntityExistsError(GraphEntityCreateError):
    """An entity with this canonical name already exists in the workspace."""


class EntityCreateBackendError(GraphEntityCreateError):
    """The ``CREATE`` statement failed (driver, syntax, lock, …).

    The original exception is chained via ``raise … from exc`` so the
    full traceback is preserved in logs without leaking driver details
    to the HTTP client.
    """


class EntityProjectionError(GraphEntityCreateError):
    """The entity was written, but the post-CREATE projection failed.

    Operationally: the node exists in Memgraph (a subsequent
    ``GET /graph/entities`` will surface it), but we cannot return the
    projected payload to the operator in this response.
    """


class MixedProvenanceError(Exception):
    """A folder-scoped mutation targeted an entity/relation whose provenance is
    *mixed* — at least one source chunk belongs to the active folder, but at
    least one belongs to another folder.

    The physical node/edge is shared across folders (single LightRAG namespace).
    PATCH/DELETE now handle this by writing a folder-local overlay/tombstone
    instead of touching the shared base. Operations without a folder-local model
    (notably create-relation on a mixed endpoint) still raise this and the route
    maps it to HTTP 409.
    """


# ----------------------------------------------------------------------
# Type mapping (LightRAG free-form → WebUI closed enum)
# ----------------------------------------------------------------------

# Keys are lowercased entity_type strings as they appear in
# LightRAG-extracted nodes. Values are the WebUI enum.
_TYPE_MAP: dict[str, str] = {
    # ORG
    "organization": "ORG",
    "org": "ORG",
    "company": "ORG",
    "team": "ORG",
    "department": "ORG",
    # PERSON
    "person": "PERSON",
    "people": "PERSON",
    "individual": "PERSON",
    # LOCATION
    "location": "LOCATION",
    "geo": "LOCATION",
    "place": "LOCATION",
    "site": "LOCATION",
    "datacenter": "LOCATION",
    "city": "LOCATION",
    "country": "LOCATION",
    # PRODUCT
    "product": "PRODUCT",
    "software": "PRODUCT",
    "database": "PRODUCT",
    "application": "PRODUCT",
    "service": "PRODUCT",
    "tool": "PRODUCT",
    # TECHNOLOGY
    "technology": "TECHNOLOGY",
    "tech": "TECHNOLOGY",
    "protocol": "TECHNOLOGY",
    "standard": "TECHNOLOGY",
    "framework": "TECHNOLOGY",
    "language": "TECHNOLOGY",
    # CONCEPT — also the fallback bucket
    "concept": "CONCEPT",
    "category": "CONCEPT",
    "event": "CONCEPT",
    "process": "CONCEPT",
    "procedure": "CONCEPT",
    "topic": "CONCEPT",
}
# All closed-enum lowercase names (org/person/product/...) are already
# covered above, so a value written by a PATCH and then re-read by
# `read_graph_entities` round-trips cleanly.

_DEFAULT_TYPE = "CONCEPT"


def map_entity_type(raw: str | None) -> str:
    """Map a LightRAG entity_type string to the WebUI enum.

    Unknown / empty values fall back to ``CONCEPT`` so the closed enum
    contract on the React side never breaks.
    """
    if not raw:
        return _DEFAULT_TYPE
    key = raw.strip().lower()
    if not key:
        return _DEFAULT_TYPE
    return _TYPE_MAP.get(key, _DEFAULT_TYPE)


# ----------------------------------------------------------------------
# Layout (deterministic, hash-bucketed per type)
# ----------------------------------------------------------------------

# Canvas dimensions match the SVG viewport used in the WebUI fixtures.
_CANVAS_W = 960
_CANVAS_H = 620
_MARGIN = 40

# Per-type centroid in the SVG canvas. Lays out the six WebUI types in
# a 3×2 grid so visually-distinct buckets are spatially separated.
_TYPE_CENTROIDS: dict[str, tuple[int, int]] = {
    "PRODUCT": (480, 310),
    "TECHNOLOGY": (200, 200),
    "CONCEPT": (760, 200),
    "ORG": (200, 460),
    "PERSON": (760, 460),
    "LOCATION": (480, 540),
}

# Per-type max radius around the centroid; tuned so 50+ entities per type
# still fit without too much overlap.
_TYPE_RADIUS = 160


def _hash_floats(eid: str) -> tuple[float, float]:
    """Two stable floats in [0, 1) derived from a SHA-1 of the id."""
    digest = hashlib.sha1(eid.encode("utf-8")).digest()
    a = int.from_bytes(digest[0:4], "big") / 0xFFFFFFFF
    b = int.from_bytes(digest[4:8], "big") / 0xFFFFFFFF
    return a, b


def layout_position(entity_id: str, entity_type: str) -> tuple[int, int]:
    """Deterministic (x, y) for an entity, clustered by type.

    Uses polar coordinates around the per-type centroid: angle from
    hash byte 0..4, radius from hash byte 4..8, jittered so the bucket
    looks organic. Same id always returns the same position.
    """
    cx, cy = _TYPE_CENTROIDS.get(entity_type, (_CANVAS_W // 2, _CANVAS_H // 2))
    h1, h2 = _hash_floats(entity_id)
    angle = h1 * 2 * math.pi
    radius = math.sqrt(h2) * _TYPE_RADIUS
    x = int(cx + radius * math.cos(angle))
    y = int(cy + radius * math.sin(angle))
    # Clamp to the canvas with a margin so nodes stay on-screen.
    x = max(_MARGIN, min(_CANVAS_W - _MARGIN, x))
    y = max(_MARGIN, min(_CANVAS_H - _MARGIN, y))
    return x, y


# ----------------------------------------------------------------------
# Cypher reads
# ----------------------------------------------------------------------


def _sanitize_workspace(workspace: str) -> str:
    """Backtick-quote workspace label safely (mirrors LightRAG's own
    `_get_workspace_label`: double any backtick to keep the value
    valid inside a backtick-quoted Cypher identifier)."""
    return workspace.replace("`", "``")


def _entity_id_to_node_id(entity_id: str) -> str:
    """Stable WebUI node id derived from a LightRAG entity_id.

    The React port assumes ``id`` is opaque and uses it as React key /
    selection token. We prefix to avoid collision with the seed fixtures
    (``e_oracle``…) so a mixed default-space + Memgraph workspace doesn't
    clash on the same id.
    """
    return f"kg_{entity_id}"


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if item is not None and str(item)]


def _json_str_dict(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(key): str(val)
        for key, val in parsed.items()
        if key is not None and val is not None
    }


def _resolve_entity_scope(
    all_chunks: set[str],
    chunk_to_doc: dict[str, str] | None,
    member_docs: set[str] | None,
) -> tuple[int, int, set[str], bool] | None:
    """Compute ``(mentions, sources, resolved_docs, mixed)`` for an entity.

    Folder-scoped (``member_docs`` not None) keeps only member chunks/docs and
    returns ``None`` when none survive (caller drops the entity); ``mixed`` is
    True when it stays visible but some source chunk is non-member/unresolvable.
    Global mode (``member_docs`` None) resolves docs when ``chunk_to_doc`` is
    available, else falls back to ``sources = mentions``."""
    if member_docs is not None:
        cd = chunk_to_doc or {}
        member_chunks = {c for c in all_chunks if cd.get(c) in member_docs}
        resolved_docs = {cd[c] for c in member_chunks}
        if not resolved_docs:
            return None
        mixed = len(member_chunks) < len(all_chunks)
        return len(member_chunks), len(resolved_docs), resolved_docs, mixed
    if chunk_to_doc:
        resolved_docs = {chunk_to_doc[c] for c in all_chunks if c in chunk_to_doc}
        sources = len(resolved_docs) if resolved_docs else len(all_chunks)
        return len(all_chunks), sources, resolved_docs, False
    return len(all_chunks), len(all_chunks), set(), False


def _node_record_to_entity(
    record: dict[str, Any],
    chunk_to_doc: dict[str, str] | None = None,
    member_docs: set[str] | None = None,
    direct_members: set[str] | None = None,
    override: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project a Cypher entity row into the WebUI ``GraphEntity`` shape.

    ``mentions`` is the count of distinct chunks the entity appears in
    (LightRAG joins the chunk ids in ``source_id`` with ``<SEP>``).
    ``sources`` is the count of distinct parent documents — computed by
    joining each chunk id against ``chunk_to_doc`` (built once from
    ``DocStatus.chunks_list``). When the map is missing or no chunks
    can be resolved (orphan chunks, fresh KB, lookup failure) sources
    falls back to mentions so the badge never reads 0 with non-empty
    source_id.

    **Folder cloisonnement** (``member_docs`` not None): the entity is visible
    only if ≥1 source chunk belongs to a member doc; ``mentions`` / ``sources`` /
    ``source_docs`` are scoped to member chunks/docs. Returns ``None`` when the
    entity has no member source (caller drops it). ``member_docs`` None → legacy
    global behaviour (no scoping).

    **Manual creates** (#1a): an entity whose id is in ``direct_members`` is
    visible in the active folder via explicit ``GRAPH_MEMBER_OF`` membership even
    with no chunk provenance — it shows as a pure, operator-owned node (zero
    mentions/sources, no source_docs, real description kept).

    **Folder-local override** (#1b): when an ``override`` overlay is present for
    the active folder, ``deleted`` drops the record (folder-local tombstone) and
    otherwise its fields replace the (possibly masked) base values — folder F sees
    its own edit while folder B keeps the base. The overlay only applies to an
    already-visible record (it is a view modifier, not a membership grant).
    """
    entity_id = record.get("entity_id") or ""
    raw_type = record.get("entity_type") or ""
    mapped_type = map_entity_type(str(raw_type))
    # LightRAG joins per-chunk descriptions with `<SEP>` when an entity
    # is mentioned in multiple chunks. Replace with a visible separator
    # so the WebUI summary reads cleanly instead of leaking the marker.
    summary = (
        (record.get("description") or "")
        .replace(_GRAPH_FIELD_SEP, " · ")
        .strip()
    )
    source_id = record.get("source_id") or ""
    all_chunks = {
        c.strip()
        for c in str(source_id).replace(_GRAPH_FIELD_SEP, ",").split(",")
        if c.strip()
    }
    scope = _resolve_entity_scope(all_chunks, chunk_to_doc, member_docs)
    if scope is None:
        is_direct = (
            member_docs is not None
            and direct_members is not None
            and str(entity_id) in direct_members
        )
        if not is_direct:
            return None  # not visible in this folder
        # Operator-owned in this folder, no chunk provenance.
        mentions, sources, resolved_docs, mixed = 0, 0, set(), False
    else:
        mentions, sources, resolved_docs, mixed = scope
    # #1b: a folder-local tombstone hides an otherwise-visible record in F only.
    if override is not None and override.get("deleted"):
        return None
    if mixed:
        # The description LightRAG blended across all source docs may carry
        # non-member text; the graph tab is a direct exposure surface → mask it.
        # The node + folder-scoped source_docs stay visible.
        summary = _MASKED_ENTITY_SUMMARY
    x, y = layout_position(str(entity_id), mapped_type)
    entity = {
        "id": _entity_id_to_node_id(str(entity_id)),
        "name": str(record.get("display_name") or entity_id),
        "type": mapped_type,
        "x": x,
        "y": y,
        "mentions": mentions,
        "sources": sources,
        "source_docs": sorted(resolved_docs),
        "summary": summary[:600],
        "tags": _json_list(record.get("twin_tags_json")),
        "properties": _json_str_dict(record.get("twin_props_json")),
    }
    # #1b: overlay folder-local edits (un-masks the fields F explicitly set).
    if override is not None:
        _apply_entity_override(entity, override)
    return entity


def _chunk_ids_from_record(record) -> tuple[str, list] | None:
    """Extract ``(doc_id, chunk_ids)`` from a DocStatus row, or None if unusable."""
    doc_id = str(record.get("doc_id") or "")
    if not doc_id:
        return None
    raw = record.get("chunks_list")
    if not raw:
        return None
    try:
        chunk_ids = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return None
    if not isinstance(chunk_ids, list):
        return None
    return doc_id, chunk_ids


async def _load_chunk_to_doc_index(workspace: str) -> dict[str, str]:
    """Build a ``chunk_id → doc_id`` map from ``DocStatus_{workspace}``.

    LightRAG stores the doc's chunk list as a JSON string under the
    ``chunks_list`` property. We read every DocStatus row once and
    invert the mapping so per-entity ``sources`` counts can be derived
    in O(1) lookups. On any failure returns ``{}`` and callers fall
    back to the legacy ``sources = mentions`` heuristic.
    """
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (d:`DocStatus_{label}`) "
        "WHERE d.chunks_list IS NOT NULL "
        "RETURN d.id AS doc_id, d.chunks_list AS chunks_list"
    )
    chunk_to_doc: dict[str, str] = {}
    try:
        async with get_read_session() as session:
            result = await session.run(query)
            async for record in result:
                parsed = _chunk_ids_from_record(record)
                if parsed is None:
                    continue
                doc_id, chunk_ids = parsed
                for chunk_id in chunk_ids:
                    if isinstance(chunk_id, str) and chunk_id:
                        chunk_to_doc[chunk_id] = doc_id
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader: failed to load chunk→doc index for workspace=%s",
            workspace,
        )
        return {}
    return chunk_to_doc


async def _load_member_docs(workspace: str, folder: str) -> set[str]:
    """Doc ids ``MEMBER_OF`` *folder* (folder cloisonnement of the graph view).

    On any failure returns an empty set → **fail-closed**: an empty member set
    hides every entity/relation (none can prove a member source doc) rather than
    leaking the global graph. The failure is logged loudly. Mirrors the storage
    cloisonnement posture (batch 2).
    """
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (d:`DocStatus_{label}`)"
        f"-[:MEMBER_OF]->(:`Folder_{label}` {{id: $folder}}) "
        "RETURN collect(d.id) AS ids"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, folder=folder)
            record = await result.single()
            await result.consume()
        if record and record["ids"]:
            return {str(d) for d in record["ids"]}
        return set()
    except Exception:
        logger.exception(
            "graph_reader: member-docs load failed (ws=%s, folder=%s) — "
            "fail-closed (empty)",
            workspace,
            folder,
        )
        return set()


async def _active_member_docs(workspace: str) -> set[str] | None:
    """Member-doc set for the request's active folder, or ``None`` (global).

    ``None`` when no request folder is bound (off the Twin routes / unscoped
    callers) → the graph reads keep their legacy global behaviour. On a Twin
    WebUI route ``bind_request_folder`` has bound a folder, so the graph is
    scoped to that folder's membership.
    """
    from .folder import active_folder_id

    folder = active_folder_id()
    if not folder:
        return None
    return await _load_member_docs(workspace, folder)


async def _member_context(
    workspace: str,
) -> tuple[set[str] | None, dict[str, str] | None]:
    """Resolve ``(member_docs, chunk_to_doc)`` for the request's active folder.

    Returns ``(None, None)`` when no folder is bound (unscoped / native caller)
    → graph mutations keep their legacy global behaviour. When a folder IS bound,
    returns the folder's member-doc set plus the chunk→doc index so the write
    helpers can (a) gate the mutation on folder visibility and (b) re-project the
    post-write response through the same membership masking the GET path applies.

    ``member_docs`` is fail-closed (empty set) on load failure — an empty set
    hides every entity/relation, so a transient Memgraph error refuses the write
    rather than letting it touch an out-of-folder object.
    """
    member_docs = await _active_member_docs(workspace)
    if member_docs is None:
        return None, None
    chunk_to_doc = await _load_chunk_to_doc_index(workspace)
    return member_docs, chunk_to_doc


# Folder-scoped mutation gate verdicts (see _entity_mutation_gate /
# _relation_mutation_gate). "member" → pure-member, mutation allowed;
# "mixed" → shared across folders, mutation refused (would corrupt another
# folder's view of a co-owned record); "absent" → not visible in this folder.
_GATE_MEMBER = "member"
_GATE_MIXED = "mixed"
_GATE_ABSENT = "absent"

# Manual graph authorship (#1a). LLM-extracted records are folder-scoped by chunk
# provenance (source_id chunks ∈ a folder's member docs). Operator-created records
# have no chunk provenance, so their folder membership is explicit:
#   - entities  → a relationship  (:{ws} {entity_id})-[:GRAPH_MEMBER_OF]->(:Folder_{ws} {id})
#   - relations → an edge property `twin_folder_json` (a JSON list of folder ids;
#     a Memgraph edge can't own a sub-relationship, so the relational form isn't
#     available — this is the one justified property, mirroring how the edge
#     already carries its own `source_id`).
# A future folder-local override layer (#1b) builds on the same Folder nodes.
_GRAPH_MEMBER_REL = "GRAPH_MEMBER_OF"
_REL_FOLDER_PROP = "twin_folder_json"

# Folder-local overrides (#1b). A *mixed* (cross-folder shared) record is never
# mutated on its base node/edge — instead folder F's edit/delete lands on a
# per-folder overlay, so folder B keeps seeing the untouched base. The overlay
# stores base-shaped props (description, entity_type, display_name, twin_tags_json,
# twin_props_json for entities; keywords, weight, twin_props_json for relations)
# plus a `deleted` tombstone, applied on read only inside F.
#   - entity overlay → (base:{ws} {entity_id})-[:HAS_OVERRIDE]->(:GraphOverride_{ws} {folder})
#     (relationship → a base DETACH DELETE cascades the overlay).
#   - relation overlay → standalone (:GraphRelOverride_{ws} {src, tgt, folder}),
#     linked (s)-[:HAS_REL_OVERRIDE]->(o) from the source endpoint (edges can't own
#     a sub-relationship). entity_id / endpoints stay the immutable global key —
#     a display_name override never changes identity or relations.
_HAS_OVERRIDE_REL = "HAS_OVERRIDE"
_HAS_REL_OVERRIDE_REL = "HAS_REL_OVERRIDE"
_ENTITY_OVERRIDE_FIELDS = (
    "description",
    "entity_type",
    "display_name",
    "twin_tags_json",
    "twin_props_json",
)
_REL_OVERRIDE_FIELDS = ("keywords", "weight", "twin_props_json")


def _entity_override_label(label: str) -> str:
    return f"GraphOverride_{label}"


def _rel_override_label(label: str) -> str:
    return f"GraphRelOverride_{label}"


def _entity_override_return(var: str) -> str:
    """RETURN fragment projecting an entity-overlay node's fields + tombstone."""
    cols = ", ".join(f"{var}.{f} AS {f}" for f in _ENTITY_OVERRIDE_FIELDS)
    return f"{cols}, {var}.deleted AS deleted"


def _row_to_entity_override(record) -> dict[str, Any] | None:
    """Build an entity-override dict from a Cypher record, or None if the
    overlay node is absent (all fields + deleted are null)."""
    ov = {f: record.get(f) for f in _ENTITY_OVERRIDE_FIELDS}
    deleted = record.get("deleted")
    if deleted is None and all(v is None for v in ov.values()):
        return None
    ov["deleted"] = bool(deleted)
    return ov


def _row_to_rel_override(record) -> dict[str, Any] | None:
    ov = {f: record.get(f) for f in _REL_OVERRIDE_FIELDS}
    deleted = record.get("deleted")
    if deleted is None and all(v is None for v in ov.values()):
        return None
    ov["deleted"] = bool(deleted)
    return ov


def _apply_entity_override(ent: dict[str, Any], ov: dict[str, Any]) -> None:
    """Overlay folder-local entity fields onto a projected GraphEntity, replacing
    the (possibly masked) base values. Never touches ``id`` (the global key)."""
    if ov.get("description") is not None:
        ent["summary"] = str(ov["description"])[:600]
    if ov.get("entity_type") is not None:
        ent["type"] = map_entity_type(str(ov["entity_type"]))
    if ov.get("display_name") is not None:
        ent["name"] = str(ov["display_name"])
    if ov.get("twin_tags_json") is not None:
        ent["tags"] = _json_list(ov["twin_tags_json"])
    if ov.get("twin_props_json") is not None:
        ent["properties"] = _json_str_dict(ov["twin_props_json"])


def _apply_relation_override(rel: dict[str, Any], ov: dict[str, Any]) -> None:
    """Overlay folder-local relation fields onto a projected GraphRelation. Never
    touches ``id`` / ``source`` / ``target`` (endpoint identity is immutable)."""
    if ov.get("keywords") is not None:
        label = str(ov["keywords"]).strip().upper().replace(" ", "_")
        rel["label"] = label or "RELATED_TO"
    if ov.get("weight") is not None:
        try:
            strength = float(ov["weight"])
        except (TypeError, ValueError):
            strength = rel.get("strength", 0.5)
        if strength > 1.0:
            strength = min(1.0, strength / 10.0)
        rel["strength"] = round(strength, 3)
    if ov.get("twin_props_json") is not None:
        rel["properties"] = _json_str_dict(ov["twin_props_json"])


async def _load_folder_overrides(
    workspace: str, folder: str
) -> dict[str, dict[str, Any]]:
    """All entity overlays for *folder*, keyed by ``entity_id`` (batch — one read
    per graph load, like ``chunk_to_doc``). ``{}`` on error."""
    label = _sanitize_workspace(workspace)
    ov_label = _entity_override_label(label)
    query = (
        f"MATCH (n:`{label}`)-[:`{_HAS_OVERRIDE_REL}`]->"
        f"(o:`{ov_label}` {{folder: $folder}}) "
        f"RETURN n.entity_id AS entity_id, {_entity_override_return('o')}"
    )
    out: dict[str, dict[str, Any]] = {}
    try:
        async with get_read_session() as session:
            result = await session.run(query, folder=folder)
            async for record in result:
                eid = record["entity_id"]
                ov = _row_to_entity_override(record)
                if eid and ov is not None:
                    out[str(eid)] = ov
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader: folder-override load failed (ws=%s, folder=%s)",
            workspace,
            folder,
        )
        return {}
    return out


async def _load_folder_rel_overrides(
    workspace: str, folder: str
) -> dict[tuple[str, str], dict[str, Any]]:
    """All relation overlays for *folder*, keyed by ``(src, tgt)``. ``{}`` on error."""
    label = _sanitize_workspace(workspace)
    ro_label = _rel_override_label(label)
    cols = ", ".join(f"o.{f} AS {f}" for f in _REL_OVERRIDE_FIELDS)
    query = (
        f"MATCH (o:`{ro_label}` {{folder: $folder}}) "
        f"RETURN o.src AS src, o.tgt AS tgt, {cols}, o.deleted AS deleted"
    )
    out: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        async with get_read_session() as session:
            result = await session.run(query, folder=folder)
            async for record in result:
                src, tgt = record["src"], record["tgt"]
                ov = _row_to_rel_override(record)
                if src and tgt and ov is not None:
                    out[(str(src), str(tgt))] = ov
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader: folder rel-override load failed (ws=%s, folder=%s)",
            workspace,
            folder,
        )
        return {}
    return out


async def _load_one_entity_override(
    workspace: str, folder: str | None, entity_id: str
) -> dict[str, Any] | None:
    """Targeted entity-overlay lookup for a single (entity, folder) — used by the
    post-write projection. ``None`` when no folder is bound or no overlay exists."""
    if not folder:
        return None
    label = _sanitize_workspace(workspace)
    ov_label = _entity_override_label(label)
    query = (
        f"MATCH (n:`{label}` {{entity_id: $eid}})-[:`{_HAS_OVERRIDE_REL}`]->"
        f"(o:`{ov_label}` {{folder: $folder}}) "
        f"RETURN {_entity_override_return('o')}"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, eid=entity_id, folder=folder)
            ov = None
            async for record in result:
                ov = _row_to_entity_override(record)
                break
            await result.consume()
        return ov
    except Exception:
        logger.exception(
            "graph_reader: one-entity override load failed (%s)", entity_id
        )
        return None


async def _load_one_rel_override(
    workspace: str, folder: str | None, src: str, tgt: str
) -> dict[str, Any] | None:
    if not folder:
        return None
    label = _sanitize_workspace(workspace)
    ro_label = _rel_override_label(label)
    cols = ", ".join(f"o.{f} AS {f}" for f in _REL_OVERRIDE_FIELDS)
    query = (
        f"MATCH (o:`{ro_label}` {{src: $src, tgt: $tgt, folder: $folder}}) "
        f"RETURN {cols}, o.deleted AS deleted"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, src=src, tgt=tgt, folder=folder)
            ov = None
            async for record in result:
                ov = _row_to_rel_override(record)
                break
            await result.consume()
        return ov
    except Exception:
        logger.exception(
            "graph_reader: one-rel override load failed (%s→%s)", src, tgt
        )
        return None


async def _upsert_entity_override(
    workspace: str,
    folder: str,
    entity_id: str,
    fields: dict[str, Any],
    *,
    deleted: bool,
) -> bool:
    """MERGE folder F's entity overlay and set its fields + tombstone flag. The
    base node is never modified. Returns ``False`` if the base node is missing."""
    label = _sanitize_workspace(workspace)
    ov_label = _entity_override_label(label)
    set_fields = "SET o += $fields " if fields else ""
    query = (
        f"MATCH (n:`{label}` {{entity_id: $eid}}) "
        f"MERGE (n)-[:`{_HAS_OVERRIDE_REL}`]->(o:`{ov_label}` {{folder: $folder}}) "
        f"{set_fields}"
        "SET o.deleted = $deleted "
        "RETURN o.folder AS folder"
    )
    try:
        async with acquire_write_slot():
            async with get_session() as session:
                result = await session.run(
                    query,
                    eid=entity_id,
                    folder=folder,
                    fields=fields,
                    deleted=deleted,
                )
                rows = [record async for record in result]
                await result.consume()
        return bool(rows)
    except Exception:
        logger.exception(
            "graph_reader: entity override upsert failed (%s, folder=%s)",
            entity_id,
            folder,
        )
        return False


async def _upsert_rel_override(
    workspace: str,
    folder: str,
    src: str,
    tgt: str,
    fields: dict[str, Any],
    *,
    deleted: bool,
) -> bool:
    """MERGE folder F's relation overlay (standalone, linked from the source
    endpoint for cascade) and set its fields + tombstone. Base edge untouched."""
    label = _sanitize_workspace(workspace)
    ro_label = _rel_override_label(label)
    set_fields = "SET o += $fields " if fields else ""
    query = (
        f"MATCH (s:`{label}` {{entity_id: $src}}) "
        f"MERGE (o:`{ro_label}` {{src: $src, tgt: $tgt, folder: $folder}}) "
        f"MERGE (s)-[:`{_HAS_REL_OVERRIDE_REL}`]->(o) "
        f"{set_fields}"
        "SET o.deleted = $deleted "
        "RETURN o.folder AS folder"
    )
    try:
        async with acquire_write_slot():
            async with get_session() as session:
                result = await session.run(
                    query,
                    src=src,
                    tgt=tgt,
                    folder=folder,
                    fields=fields,
                    deleted=deleted,
                )
                rows = [record async for record in result]
                await result.consume()
        return bool(rows)
    except Exception:
        logger.exception(
            "graph_reader: rel override upsert failed (%s→%s, folder=%s)",
            src,
            tgt,
            folder,
        )
        return False


async def _load_direct_member_entity_rows(
    workspace: str, folder: str
) -> list[dict[str, Any]]:
    """Full entity rows for nodes explicitly ``GRAPH_MEMBER_OF`` *folder*.

    These are operator-created entities with no chunk provenance, so the
    chunk-membership read skips them. Surfaced here so a manual create survives a
    folder-scoped refresh. Best-effort: ``[]`` on error (the chunk-membership
    result still stands)."""
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (n:`{label}`)-[:`{_GRAPH_MEMBER_REL}`]->"
        f"(:`Folder_{label}` {{id: $folder}}) "
        "RETURN n.entity_id AS entity_id, n.entity_type AS entity_type, "
        "n.description AS description, n.source_id AS source_id, "
        "n.display_name AS display_name, n.twin_tags_json AS twin_tags_json, "
        "n.twin_props_json AS twin_props_json"
    )
    rows: list[dict[str, Any]] = []
    try:
        async with get_read_session() as session:
            result = await session.run(query, folder=folder)
            async for record in result:
                rows.append(
                    {
                        "entity_id": record["entity_id"],
                        "entity_type": record["entity_type"],
                        "description": record["description"],
                        "source_id": record["source_id"],
                        "display_name": record["display_name"],
                        "twin_tags_json": record["twin_tags_json"],
                        "twin_props_json": record["twin_props_json"],
                    }
                )
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader: direct-member entity load failed (ws=%s, folder=%s)",
            workspace,
            folder,
        )
        return []
    return rows


async def _load_manual_relation_rows(workspace: str) -> list[dict[str, Any]]:
    """Edge rows for operator-created relations (those carrying a
    ``twin_folder_json`` stamp). Bounded — manual relations are few. The active
    folder is matched in Python (avoids fragile JSON-substring Cypher). ``[]`` on
    error."""
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (s:`{label}`)-[r:DIRECTED]->(t:`{label}`) "
        f"WHERE r.`{_REL_FOLDER_PROP}` IS NOT NULL "
        "RETURN s.entity_id AS source_id, t.entity_id AS target_id, "
        "r.keywords AS keywords, r.weight AS weight, "
        "r.source_id AS chunk_source_id, "
        f"r.`{_REL_FOLDER_PROP}` AS twin_folder_json, "
        "r.twin_props_json AS twin_props_json"
    )
    rows: list[dict[str, Any]] = []
    try:
        async with get_read_session() as session:
            result = await session.run(query)
            async for record in result:
                rows.append(
                    {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "keywords": record["keywords"],
                        "weight": record["weight"],
                        "chunk_source_id": record["chunk_source_id"],
                        "twin_folder_json": record["twin_folder_json"],
                        "twin_props_json": record["twin_props_json"],
                    }
                )
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader: manual relation load failed (ws=%s)", workspace
        )
        return []
    return rows


async def _load_relation_rows_between_entities(
    workspace: str,
    entity_ids: set[str],
    *,
    max_edges: int = 5000,
) -> list[dict[str, Any]] | None:
    """Load stored ``:DIRECTED`` edges whose endpoints are already visible.

    ``read_graph_native`` still delegates node selection to LightRAG, but native
    edge objects can omit relationship properties such as ``source_id``. Folder
    scoping depends on that provenance, so after selecting visible nodes we
    re-read matching edges from Memgraph and project those authoritative rows.
    ``None`` means the read failed and callers should fall back to native edges.
    """
    if not entity_ids:
        return []
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (s:`{label}`)-[r:DIRECTED]->(t:`{label}`) "
        "WHERE s.entity_id IN $entity_ids AND t.entity_id IN $entity_ids "
        "RETURN s.entity_id AS source_id, t.entity_id AS target_id, "
        "r.keywords AS keywords, r.weight AS weight, "
        "r.source_id AS chunk_source_id, "
        f"r.`{_REL_FOLDER_PROP}` AS twin_folder_json, "
        "r.twin_props_json AS twin_props_json "
        "LIMIT $max_edges"
    )
    rows: list[dict[str, Any]] = []
    try:
        async with get_read_session() as session:
            result = await session.run(
                query,
                entity_ids=sorted(entity_ids),
                max_edges=max_edges,
            )
            async for record in result:
                rows.append(
                    {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "keywords": record["keywords"],
                        "weight": record["weight"],
                        "chunk_source_id": record["chunk_source_id"],
                        "twin_folder_json": record["twin_folder_json"],
                        "twin_props_json": record["twin_props_json"],
                    }
                )
            await result.consume()
        return rows
    except Exception:
        logger.exception(
            "graph_reader: visible relation row load failed (ws=%s)", workspace
        )
        return None


async def _entity_mutation_gate(
    workspace: str,
    entity_id: str,
    chunk_to_doc: dict[str, str] | None,
    member_docs: set[str],
) -> str:
    """Classify an entity for a folder-scoped mutation: pure-member, mixed, or
    absent. A record is ``member`` when its chunk provenance is pure-member OR it
    is an operator-created entity explicitly ``GRAPH_MEMBER_OF`` the active folder
    (#1a) — so manual creates are editable/deletable in their folder. Fail-closed
    (``absent``) on read error so a transient backend fault refuses the write."""
    from .folder import active_folder_id

    folder = active_folder_id()
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (n:`{label}` {{entity_id: $eid}}) "
        f"OPTIONAL MATCH (n)-[gm:`{_GRAPH_MEMBER_REL}`]->"
        f"(:`Folder_{label}` {{id: $folder}}) "
        "RETURN n.source_id AS source_id, gm IS NOT NULL AS direct"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, eid=entity_id, folder=folder)
            row = None
            async for record in result:
                row = {
                    "source_id": record["source_id"],
                    "direct": record["direct"],
                }
                break
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader._entity_mutation_gate: read failed for %s", entity_id
        )
        return _GATE_ABSENT
    if row is None:
        return _GATE_ABSENT
    all_chunks = {
        c.strip()
        for c in str(row["source_id"] or "")
        .replace(_GRAPH_FIELD_SEP, ",")
        .split(",")
        if c.strip()
    }
    scope = _resolve_entity_scope(all_chunks, chunk_to_doc, member_docs)
    if scope is not None:
        return _GATE_MIXED if scope[3] else _GATE_MEMBER
    # No chunk membership → fall back to explicit manual-create folder membership.
    return _GATE_MEMBER if row.get("direct") else _GATE_ABSENT


async def _relation_mutation_gate(
    workspace: str,
    src: str,
    tgt: str,
    chunk_to_doc: dict[str, str] | None,
    member_docs: set[str],
) -> str:
    """Classify a relation for a folder-scoped mutation. ``member`` when the
    edge's own chunk provenance is pure-member OR it is an operator-created edge
    stamped with the active folder in ``twin_folder_json`` (#1a). Fail-closed
    (``absent``) on read error."""
    from .folder import active_folder_id

    folder = active_folder_id()
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (s:`{label}` {{entity_id: $src}})-[r:DIRECTED]->"
        f"(t:`{label}` {{entity_id: $tgt}}) "
        f"RETURN r.source_id AS chunk_source_id, "
        f"r.`{_REL_FOLDER_PROP}` AS twin_folder_json"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, src=src, tgt=tgt)
            row = None
            async for record in result:
                row = {
                    "chunk_source_id": record["chunk_source_id"],
                    "twin_folder_json": record["twin_folder_json"],
                }
                break
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader._relation_mutation_gate: read failed for %s→%s",
            src,
            tgt,
        )
        return _GATE_ABSENT
    if row is None:
        return _GATE_ABSENT
    _docs, in_folder, mixed = _resolve_source_docs(
        row["chunk_source_id"], chunk_to_doc, member_docs
    )
    if in_folder:
        return _GATE_MIXED if mixed else _GATE_MEMBER
    # No chunk membership → fall back to explicit manual-relation folder stamp.
    if folder and folder in _json_list(row.get("twin_folder_json")):
        return _GATE_MEMBER
    return _GATE_ABSENT


async def _load_member_chunks(workspace: str, folder: str) -> set[str]:
    """Chunk ids owned by docs ``MEMBER_OF`` *folder* (for Cypher pushdown).

    Read from each member doc's ``chunks_list``. Used to push the membership
    predicate **before** the ``LIMIT`` in the flat graph reads / label search, so
    a folder isn't starved by a global LIMIT that lands mostly on non-member
    nodes. Fail-closed (empty) on error — never leak a global result set.
    """
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (d:`DocStatus_{label}`)"
        f"-[:MEMBER_OF]->(:`Folder_{label}` {{id: $folder}}) "
        "WHERE d.chunks_list IS NOT NULL "
        "RETURN d.chunks_list AS chunks_list"
    )
    member_chunks: set[str] = set()
    try:
        async with get_read_session() as session:
            result = await session.run(query, folder=folder)
            async for record in result:
                raw = record.get("chunks_list")
                try:
                    ids = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError):
                    continue
                if isinstance(ids, list):
                    member_chunks.update(c for c in ids if isinstance(c, str) and c)
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader: member-chunks load failed (ws=%s, folder=%s) — "
            "fail-closed (empty)",
            workspace,
            folder,
        )
        return set()
    return member_chunks


async def _active_member_chunks(workspace: str) -> set[str] | None:
    """Member chunk-id set for the active folder, or ``None`` (global/unscoped)."""
    from .folder import active_folder_id

    folder = active_folder_id()
    if not folder:
        return None
    return await _load_member_chunks(workspace, folder)


def _membership_predicate(var: str) -> str:
    """Cypher WHERE-fragment: graph node/edge ``var`` has ≥1 member source chunk
    (params ``$sep`` + ``$mchunks``). Pushed before ``LIMIT`` so the cap lands on
    in-folder records."""
    return (
        f"any(_cid IN split(coalesce({var}.source_id, ''), $sep) "
        f"WHERE _cid IN $mchunks)"
    )


# When only PART of a record's source chunks are in-folder, the text fields
# LightRAG aggregated across all source docs (entity description, relation
# keywords/label, operator props) can encode non-member content. The graph tab
# is a direct exposure surface, so a *mixed* record stays structurally visible
# (node/edge + scoped source_docs) but its non-decomposable text payload is
# neutralised — un-blending would require per-folder re-extraction.
_MASKED_ENTITY_SUMMARY = "[description hidden: mixed-folder provenance]"
_MIXED_RELATION_LABEL = "MIXED_PROVENANCE"

# read_graph_native delegates node SELECTION to LightRAG's get_knowledge_graph,
# which ranks nodes GLOBALLY (by degree) — it cannot be told about folders. When
# a folder is bound we therefore over-fetch candidates and filter to member ones
# in Python, then truncate to the requested cap, so a folder whose nodes sort
# after max_nodes globally isn't starved. Residual: a folder that is a very thin
# slice of a huge KB can still under-return (documented, bounded — no escalation
# loop). Mirrors the batch-2 retrieval over-fetch posture.
_GRAPH_NATIVE_OVERFETCH_FACTOR = 4
_GRAPH_NATIVE_OVERFETCH_CAP = 8000


def _native_overfetch(max_nodes: int) -> int:
    return min(max(max_nodes, max_nodes * _GRAPH_NATIVE_OVERFETCH_FACTOR),
               _GRAPH_NATIVE_OVERFETCH_CAP)


def _resolve_source_docs(
    source_id: Any,
    chunk_to_doc: dict[str, str] | None,
    member_docs: set[str] | None,
) -> tuple[set[str], bool, bool]:
    """Resolve a ``source_id`` chunk list to its parent docs, folder-scoped.

    Returns ``(docs, in_folder, mixed)``:

    - ``docs`` = parent doc ids the chunks resolve to (via ``chunk_to_doc``),
      intersected with ``member_docs`` when a folder is active.
    - ``in_folder`` = visible in the active folder: ``True`` always when no folder
      is active (global), else ``True`` iff ≥1 source chunk belongs to a member.
    - ``mixed`` = visible but NOT pure: at least one source chunk is non-member
      or unresolvable (so the aggregated text payload may leak non-member
      content). Always ``False`` when no folder is active.
    """
    chunks = {
        c.strip()
        for c in str(source_id or "").replace(_GRAPH_FIELD_SEP, ",").split(",")
        if c.strip()
    }
    cd = chunk_to_doc or {}
    resolved = {cd[c] for c in chunks if c in cd}
    if member_docs is None:
        return resolved, True, False
    member_chunks = {c for c in chunks if cd.get(c) in member_docs}
    scoped = {cd[c] for c in member_chunks}
    in_folder = bool(scoped)
    mixed = in_folder and len(member_chunks) < len(chunks)
    return scoped, in_folder, mixed


# In-process cache of relation_id → (workspace, src, tgt). Populated on
# every `read_graph_relations` call so the PATCH route can reverse the
# hash back to a Cypher MATCH. Eviction is intentionally lazy — for a
# typical KB with O(10³) relations the memory cost is trivial; multi-
# process deploys are tolerable because every worker rebuilds its own
# cache as soon as the React port re-fetches.
_RELATION_ENDPOINT_CACHE: dict[str, tuple[str, str, str]] = {}


def _remember_relation(workspace: str, rel_id: str, src: str, tgt: str) -> None:
    _RELATION_ENDPOINT_CACHE[rel_id] = (workspace, src, tgt)


def lookup_relation_endpoints(rel_id: str) -> tuple[str, str, str] | None:
    """Return ``(workspace, src, tgt)`` for a relation id known from a
    previous read, or ``None`` if the cache was never primed for it."""
    return _RELATION_ENDPOINT_CACHE.get(rel_id)


def _relation_id_from_endpoints(src: str, tgt: str) -> str:
    """Stable relation id derived from the endpoint pair.

    Since LightRAG MERGEs a single :DIRECTED edge per source/target
    pair, encoding the pair as the public id gives the WebUI a key
    that PATCH can resolve back to a Cypher MATCH without storing an
    extra property on the edge.
    """
    h = hashlib.sha1(f"{src}->{tgt}".encode("utf-8")).hexdigest()[:12]
    return f"kr_{h}"


def _edge_record_to_relation(
    record: dict[str, Any],
    index: int,
    chunk_to_doc: dict[str, str] | None = None,
    member_docs: set[str] | None = None,
    active_folder: str | None = None,
    override: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Project a Cypher edge row into the WebUI ``GraphRelation`` shape.

    ``index`` is accepted for backwards compatibility but ignored; the
    relation id is now derived from the endpoint pair.

    **Folder cloisonnement** (``member_docs`` not None): the relation is visible
    only if ≥1 of its *own* source chunks (``record["chunk_source_id"]`` — the
    relationship's ``source_id`` provenance, NOT the endpoint ids) belongs to a
    member doc. Returns ``None`` when scoped out (caller drops it). On a *mixed*
    relation (some source chunk non-member) the edge stays visible but its
    blended text payload (``label`` from keywords, operator ``properties``) is
    neutralised. This is on top of the endpoint-visibility filter the callers
    already apply. ``None`` → legacy global behaviour.

    **Manual relations** (#1a): an operator-created edge has no chunk provenance
    but is stamped with ``active_folder`` in ``record["twin_folder_json"]``. When
    chunk-membership fails, the edge is still visible (pure, not masked) if the
    active folder is in that stamp.

    **Folder-local override** (#1b): an ``override`` overlay for the active folder
    tombstones (``deleted`` → ``None``) or replaces the edge's label/strength/props
    in F's view only.
    """
    del index  # ignored — id is endpoint-derived
    mixed = False
    if member_docs is not None:
        _docs, in_folder, mixed = _resolve_source_docs(
            record.get("chunk_source_id"), chunk_to_doc, member_docs
        )
        if not in_folder:
            stamped = _json_list(record.get("twin_folder_json"))
            if not (active_folder and active_folder in stamped):
                return None
            mixed = False  # operator-owned in this folder
    if override is not None and override.get("deleted"):
        return None  # #1b folder-local tombstone
    src = record.get("source_id") or ""
    tgt = record.get("target_id") or ""
    keywords = record.get("keywords") or ""
    weight = record.get("weight")
    try:
        strength = float(weight) if weight is not None else 0.5
    except (TypeError, ValueError):
        strength = 0.5
    # LightRAG sometimes returns weight on a 0..10 scale; normalize to
    # 0..1 if it overshoots.
    if strength > 1.0:
        strength = min(1.0, strength / 10.0)
    label = str(keywords).strip().upper().replace(" ", "_") or "RELATED_TO"
    properties = _json_str_dict(record.get("twin_props_json"))
    if mixed:
        # Mixed-folder provenance → mask the blended text payload (label from
        # LightRAG keywords, operator props). Endpoints + strength stay.
        label = _MIXED_RELATION_LABEL
        properties = {}
    relation = {
        "id": _relation_id_from_endpoints(str(src), str(tgt)),
        "source": _entity_id_to_node_id(str(src)),
        "target": _entity_id_to_node_id(str(tgt)),
        "label": label,
        "strength": round(strength, 3),
        "properties": properties,
    }
    # #1b: overlay folder-local edits (un-masks the fields F explicitly set).
    if override is not None:
        _apply_relation_override(relation, override)
    return relation


async def read_graph_entities(
    workspace: str,
    *,
    max_nodes: int = 200,
) -> list[dict[str, Any]]:
    """Return every entity stored under ``workspace`` as a WebUI
    ``GraphEntity`` dict, capped at ``max_nodes`` for safety.

    Returns ``[]`` if Memgraph has no nodes for this workspace yet, or
    if the read fails — callers fall back to the seed in that case.
    """
    label = _sanitize_workspace(workspace)
    # Push the folder-membership predicate BEFORE the LIMIT so the cap lands on
    # in-folder entities (a global LIMIT could otherwise starve a folder whose
    # nodes sort after $max_nodes globally). None → unscoped (legacy global).
    member_chunks = await _active_member_chunks(workspace)
    where = f"WHERE {_membership_predicate('n')} " if member_chunks is not None else ""
    query = (
        f"MATCH (n:`{label}`) "
        f"{where}"
        "RETURN n.entity_id AS entity_id, n.entity_type AS entity_type, "
        "n.description AS description, n.source_id AS source_id, "
        "n.display_name AS display_name, n.twin_tags_json AS twin_tags_json, "
        "n.twin_props_json AS twin_props_json "
        "LIMIT $max_nodes"
    )
    params: dict[str, Any] = {"max_nodes": max_nodes}
    if member_chunks is not None:
        params["sep"] = _GRAPH_FIELD_SEP
        params["mchunks"] = list(member_chunks)
    try:
        async with get_read_session() as session:
            result = await session.run(query, **params)
            rows = []
            async for record in result:
                rows.append(
                    {
                        "entity_id": record["entity_id"],
                        "entity_type": record["entity_type"],
                        "description": record["description"],
                        "source_id": record["source_id"],
                        "display_name": record["display_name"],
                        "twin_tags_json": record["twin_tags_json"],
                        "twin_props_json": record["twin_props_json"],
                    }
                )
            await result.consume()
        chunk_to_doc = await _load_chunk_to_doc_index(workspace)
        member_docs = await _active_member_docs(workspace)
        # #1a: union operator-created entities (GRAPH_MEMBER_OF the active
        # folder) — they have no chunk provenance so the predicate above skips
        # them. Dedup by id; direct rows are surfaced via `direct_members`.
        direct_members: set[str] | None = None
        from .folder import active_folder_id

        folder = active_folder_id()
        overrides: dict[str, dict[str, Any]] = {}
        if member_docs is not None and folder:
            seen = {r.get("entity_id") for r in rows}
            direct_rows = await _load_direct_member_entity_rows(workspace, folder)
            direct_members = {r["entity_id"] for r in direct_rows if r.get("entity_id")}
            rows.extend(r for r in direct_rows if r.get("entity_id") not in seen)
            overrides = await _load_folder_overrides(workspace, folder)
        out: list[dict[str, Any]] = []
        for row in rows:
            if not row.get("entity_id"):
                continue
            entity = _node_record_to_entity(
                row,
                chunk_to_doc,
                member_docs,
                direct_members,
                overrides.get(str(row["entity_id"])),
            )
            if entity is not None:
                out.append(entity)
        return out
    except Exception:
        logger.exception(
            "graph_reader: failed to read entities for workspace=%s", workspace
        )
        return []


def _native_node_to_entity(
    node: Any,
    chunk_to_doc: dict[str, str],
    member_docs: set[str] | None = None,
    direct_members: set[str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    props = getattr(node, "properties", None) or {}
    entity_id = props.get("entity_id") or getattr(node, "id", None)
    if not entity_id:
        return None
    row = {
        "entity_id": entity_id,
        "entity_type": props.get("entity_type"),
        "description": props.get("description"),
        "source_id": props.get("source_id"),
        "display_name": props.get("display_name"),
        "twin_tags_json": props.get("twin_tags_json"),
        "twin_props_json": props.get("twin_props_json"),
    }
    override = overrides.get(str(entity_id)) if overrides else None
    return _node_record_to_entity(
        row, chunk_to_doc, member_docs, direct_members, override
    )


def _native_edge_to_relation(
    edge: Any,
    index: int,
    chunk_to_doc: dict[str, str] | None = None,
    member_docs: set[str] | None = None,
    active_folder: str | None = None,
    overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    src = getattr(edge, "source", None)
    tgt = getattr(edge, "target", None)
    if not src or not tgt:
        return None
    eprops = getattr(edge, "properties", None) or {}
    row = {
        "source_id": src,
        "target_id": tgt,
        # The relationship's OWN chunk provenance (distinct from the endpoint
        # entity ids above) — drives folder-scoping of the relation.
        "chunk_source_id": eprops.get("source_id"),
        "keywords": eprops.get("keywords"),
        "weight": eprops.get("weight"),
        "twin_folder_json": eprops.get(_REL_FOLDER_PROP),
        "twin_props_json": eprops.get("twin_props_json"),
    }
    override = overrides.get((str(src), str(tgt))) if overrides else None
    return _edge_record_to_relation(
        row, index, chunk_to_doc, member_docs, active_folder, override
    )


def _build_native_entities(
    kg, chunk_to_doc, member_docs, direct_members, overrides, max_nodes
) -> list[dict]:
    """Project native KG nodes to entities, capped after membership filtering."""
    entities: list[dict[str, Any]] = []
    for node in getattr(kg, "nodes", []) or []:
        entity = _native_node_to_entity(
            node, chunk_to_doc, member_docs, direct_members, overrides
        )
        if entity is not None:
            entities.append(entity)
            if len(entities) >= max_nodes:
                break  # truncate to the requested cap after membership filtering
    return entities


def _build_native_relations(
    kg,
    workspace,
    valid_ids,
    chunk_to_doc,
    member_docs,
    active_folder,
    overrides,
) -> list[dict]:
    """Project native KG edges to relations, dropping any whose endpoints did
    not survive entity membership filtering."""
    relations: list[dict[str, Any]] = []
    for i, edge in enumerate(getattr(kg, "edges", []) or []):
        rel = _native_edge_to_relation(
            edge, i, chunk_to_doc, member_docs, active_folder, overrides
        )
        if rel is None:
            continue
        if rel["source"] not in valid_ids or rel["target"] not in valid_ids:
            continue
        _remember_relation(workspace, rel["id"], rel["source"], rel["target"])
        relations.append(rel)
    return relations


async def read_graph_native(
    rag: Any,
    workspace: str,
    *,
    node_label: str = "*",
    max_depth: int = 3,
    max_nodes: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Build the WebUI graph by delegating node/edge SELECTION to LightRAG's
    native ``rag.get_knowledge_graph`` instead of a flat ``LIMIT`` scan.

    Why: a flat ``MATCH (n:label) LIMIT 200`` returns an arbitrary, unordered
    1% slice of a large KB (17k+ entities) — the graph looked "dénutri" and a
    searched entity (e.g. schizophrenia) was simply not in the slice. The
    native API is the focus+context model the LightRAG UI itself uses:
      - ``node_label="*"`` → top nodes by degree (the meaningful hubs), capped;
      - ``node_label=X``  → BFS neighbourhood (``max_depth``) around entity X,
        so any entity reached via search brings its context with it.

    We keep the Twin ``GraphEntity``/``GraphRelation`` contract by mapping the
    returned ``KnowledgeGraph`` nodes/edges through the same projectors used by
    the legacy reader.

    Returns ``None`` when the native graph is **unavailable** (``get_knowledge_
    graph`` failed) so the caller can fall back to the demo seed; returns a
    ``(entities, relations)`` tuple — **possibly empty** — when it ran. A
    folder-scoped empty result is a legitimately empty folder, NOT "unavailable":
    the caller must NOT seed-fallback on it (that would leak the unscoped seed
    graph into a folder view).
    """
    from .folder import active_folder_id

    folder = active_folder_id()
    # Over-fetch candidates when scoping so the post-filter result still fills the
    # cap (get_knowledge_graph ranks globally, can't be folder-told).
    fetch_nodes = _native_overfetch(max_nodes) if folder else max_nodes
    try:
        kg = await rag.get_knowledge_graph(
            node_label=node_label,
            max_depth=max_depth,
            max_nodes=fetch_nodes,
        )
    except Exception:
        logger.exception(
            "graph_reader: native get_knowledge_graph failed "
            "(label=%s, workspace=%s)",
            node_label,
            workspace,
        )
        return None

    chunk_to_doc = await _load_chunk_to_doc_index(workspace)
    member_docs = await _load_member_docs(workspace, folder) if folder else None
    entity_overrides = await _load_folder_overrides(workspace, folder) if folder else {}
    rel_overrides = await _load_folder_rel_overrides(workspace, folder) if folder else {}

    # #1a: operator-created entities have no chunk provenance and the native
    # degree-ranked read won't return an isolated manual node — load them
    # explicitly and union them in so a manual create survives a folder refresh.
    direct_rows: list[dict[str, Any]] = []
    direct_members: set[str] | None = None
    if folder:
        direct_rows = await _load_direct_member_entity_rows(workspace, folder)
        direct_members = {r["entity_id"] for r in direct_rows if r.get("entity_id")}

    entities = _build_native_entities(
        kg, chunk_to_doc, member_docs, direct_members, entity_overrides, max_nodes
    )
    present = {e["id"] for e in entities}
    for row in direct_rows:
        ent = _node_record_to_entity(
            row,
            chunk_to_doc,
            member_docs,
            direct_members,
            entity_overrides.get(str(row.get("entity_id"))),
        )
        if ent is not None and ent["id"] not in present:
            entities.append(ent)
            present.add(ent["id"])

    valid_ids = {e["id"] for e in entities}
    raw_valid_ids = {_strip_node_prefix(eid) for eid in valid_ids}
    stored_relation_rows = await _load_relation_rows_between_entities(
        workspace, raw_valid_ids
    )
    if stored_relation_rows is None:
        relations = _build_native_relations(
            kg, workspace, valid_ids, chunk_to_doc, member_docs, folder, rel_overrides
        )
    else:
        relations = []
        for i, row in enumerate(stored_relation_rows):
            rel = _edge_record_to_relation(
                row,
                i,
                chunk_to_doc,
                member_docs,
                folder,
                rel_overrides.get(
                    (str(row.get("source_id")), str(row.get("target_id")))
                ),
            )
            if rel is None:
                continue
            if rel["source"] not in valid_ids or rel["target"] not in valid_ids:
                continue
            _remember_relation(
                workspace,
                rel["id"],
                rel["source"],
                rel["target"],
            )
            relations.append(rel)
    # Union operator-created relations among the visible entities (#1a) — the
    # native read only returns edges between degree-ranked nodes, so a manual
    # edge touching a manual node would otherwise never appear.
    if folder:
        rel_ids = {r["id"] for r in relations}
        for row in await _load_manual_relation_rows(workspace):
            if folder not in _json_list(row.get("twin_folder_json")):
                continue
            rel = _edge_record_to_relation(
                row,
                0,
                chunk_to_doc,
                member_docs,
                folder,
                rel_overrides.get((str(row.get("source_id")), str(row.get("target_id")))),
            )
            if rel is None:
                continue
            if rel["source"] not in valid_ids or rel["target"] not in valid_ids:
                continue
            if rel["id"] in rel_ids:
                continue
            _remember_relation(workspace, rel["id"], rel["source"], rel["target"])
            relations.append(rel)
            rel_ids.add(rel["id"])
    return entities, relations


async def _search_labels_scoped(
    workspace: str, q: str, member_chunks: set[str], limit: int
) -> list[str]:
    """Folder-aware entity-label search: substring match constrained to entities
    with ≥1 member source chunk. Loses the native fuzzy ranking but never reveals
    out-of-folder labels (the search box is an exposure surface)."""
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (n:`{label}`) "
        "WHERE toLower(n.entity_id) CONTAINS toLower($q) "
        f"AND {_membership_predicate('n')} "
        "RETURN DISTINCT n.entity_id AS eid LIMIT $limit"
    )
    out: list[str] = []
    if member_chunks:
        try:
            async with get_read_session() as session:
                result = await session.run(
                    query,
                    q=q,
                    sep=_GRAPH_FIELD_SEP,
                    mchunks=list(member_chunks),
                    limit=limit,
                )
                async for record in result:
                    eid = record.get("eid")
                    if eid:
                        out.append(str(eid))
                await result.consume()
        except Exception:
            logger.exception(
                "graph_reader: scoped label search failed (ws=%s, q=%r) — "
                "fail-closed (empty)",
                workspace,
                q,
            )
            return []
    from .folder import active_folder_id

    folder = active_folder_id()
    if not folder:
        return out
    overrides = await _load_folder_overrides(workspace, folder)
    tombstoned = {
        eid for eid, override in overrides.items() if override.get("deleted")
    }
    out = [eid for eid in out if eid not in tombstoned]

    def add_match(eid: str, *labels: Any) -> None:
        if len(out) >= limit or eid in tombstoned or eid in out:
            return
        needle = q.lower()
        if any(needle in str(label or "").lower() for label in labels):
            out.append(eid)

    # #1a + #1b: direct-member manual entities have no chunk provenance, and a
    # folder-local display_name override should be searchable in that folder.
    direct_rows = await _load_direct_member_entity_rows(workspace, folder)
    direct_members = {str(r["entity_id"]) for r in direct_rows if r.get("entity_id")}
    for row in direct_rows:
        eid = str(row.get("entity_id") or "")
        if not eid:
            continue
        override = overrides.get(eid) or {}
        add_match(
            eid,
            row.get("entity_id"),
            row.get("display_name"),
            override.get("display_name"),
        )

    # Overlay display names on chunk-backed entities may be the only text that
    # matches the query. Verify visibility through the same gate used by writes
    # so the overlay never grants cross-folder search visibility by itself.
    chunk_to_doc = await _load_chunk_to_doc_index(workspace)
    member_docs = await _load_member_docs(workspace, folder)
    for eid, override in overrides.items():
        if len(out) >= limit:
            break
        if eid in direct_members or override.get("deleted"):
            continue
        if q.lower() not in str(override.get("display_name") or "").lower():
            continue
        verdict = await _entity_mutation_gate(workspace, eid, chunk_to_doc, member_docs)
        if verdict != _GATE_ABSENT:
            add_match(eid, override.get("display_name"))
    return out


async def search_graph_labels(
    rag: Any, q: str, *, workspace: str | None = None, limit: int = 50
) -> list[str]:
    """Server-side entity-label search for the Graph search box.

    Folder-scoped when a Twin folder is bound (``workspace`` given + active
    folder): a substring match constrained to in-folder entities, so the search
    box can NOT reveal out-of-folder labels by autocompletion. Off a folder
    (``workspace`` None / unbound) it delegates to LightRAG's native fuzzy
    ``search_labels`` over the whole KB (legacy global behaviour).
    """
    member_chunks = (
        await _active_member_chunks(workspace) if workspace else None
    )
    if member_chunks is not None:
        return await _search_labels_scoped(workspace, q, member_chunks, limit)
    try:
        graph = rag.chunk_entity_relation_graph
        return list(await graph.search_labels(q, limit))
    except Exception:
        logger.exception("graph_reader: native search_labels failed (q=%r)", q)
        return []


async def read_graph_relations(
    workspace: str,
    *,
    valid_node_ids: Sequence[str] | None = None,
    max_edges: int = 600,
) -> list[dict[str, Any]]:
    """Return every relation stored under ``workspace`` as a WebUI
    ``GraphRelation`` dict, capped at ``max_edges``.

    When ``valid_node_ids`` is provided (already-projected WebUI ids
    from a preceding `read_graph_entities` call), edges whose endpoints
    aren't in that set are dropped to keep the layout consistent with
    the truncated node list.
    """
    label = _sanitize_workspace(workspace)
    # Membership predicate on the relationship's own source chunks, BEFORE the
    # LIMIT (same rationale as read_graph_entities). None → unscoped global.
    member_chunks = await _active_member_chunks(workspace)
    where = f"WHERE {_membership_predicate('r')} " if member_chunks is not None else ""
    query = (
        f"MATCH (s:`{label}`)-[r:DIRECTED]->(t:`{label}`) "
        f"{where}"
        "RETURN s.entity_id AS source_id, t.entity_id AS target_id, "
        "r.keywords AS keywords, r.weight AS weight, "
        "r.source_id AS chunk_source_id, "
        "r.twin_props_json AS twin_props_json "
        "LIMIT $max_edges"
    )
    params: dict[str, Any] = {"max_edges": max_edges}
    if member_chunks is not None:
        params["sep"] = _GRAPH_FIELD_SEP
        params["mchunks"] = list(member_chunks)
    try:
        async with get_read_session() as session:
            result = await session.run(query, **params)
            rows = []
            async for record in result:
                rows.append(
                    {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "keywords": record["keywords"],
                        "weight": record["weight"],
                        "chunk_source_id": record["chunk_source_id"],
                        "twin_props_json": record["twin_props_json"],
                    }
                )
            await result.consume()
        chunk_to_doc = await _load_chunk_to_doc_index(workspace)
        member_docs = await _active_member_docs(workspace)
        from .folder import active_folder_id

        folder = active_folder_id()
        rel_overrides = (
            await _load_folder_rel_overrides(workspace, folder) if folder else {}
        )
        valid = set(valid_node_ids) if valid_node_ids is not None else None
        out: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            if not row.get("source_id") or not row.get("target_id"):
                continue
            rel = _edge_record_to_relation(
                row,
                i,
                chunk_to_doc,
                member_docs,
                folder,
                rel_overrides.get(
                    (str(row.get("source_id")), str(row.get("target_id")))
                ),
            )
            if rel is None:
                continue  # scoped out of the active folder
            if valid is not None and (
                rel["source"] not in valid or rel["target"] not in valid
            ):
                continue
            # Prime the endpoint cache so PATCH can reverse the relation
            # id back to the underlying LightRAG entity_ids.
            _remember_relation(
                workspace, rel["id"], str(row["source_id"]), str(row["target_id"])
            )
            out.append(rel)
        return out
    except Exception:
        logger.exception(
            "graph_reader: failed to read relations for workspace=%s", workspace
        )
        return []


# ----------------------------------------------------------------------
# Writes (M12 batch 2 — PATCH persistence)
# ----------------------------------------------------------------------


def _strip_node_prefix(webui_id: str) -> str:
    """Reverse `_entity_id_to_node_id` — strip the ``kg_`` prefix so we
    can target the underlying LightRAG entity_id in Cypher.
    """
    if webui_id.startswith("kg_"):
        return webui_id[3:]
    return webui_id


def _entity_patch_to_props(patch: dict[str, Any]) -> dict[str, Any]:
    """Translate the WebUI ``GraphEntityPatch`` shape into a flat dict
    of Memgraph node properties suitable for ``SET n += $props``.

    Mapping:
    - ``summary`` → ``description``
    - ``type``    → ``entity_type``  (closed enum string; lower-cased
                                     on read by `map_entity_type`)
    - ``name``    → ``display_name`` (the immutable ``entity_id`` PK
                                     is never rewritten by an edit)
    - ``tags``    → ``twin_tags_json`` (JSON-encoded list — Memgraph
                                       doesn't store native arrays of
                                       strings through Bolt cleanly)
    - ``properties`` → ``twin_props_json`` (free-form k/v store)
    """
    props: dict[str, Any] = {}
    if "summary" in patch and patch["summary"] is not None:
        props["description"] = patch["summary"]
    if "type" in patch and patch["type"] is not None:
        props["entity_type"] = patch["type"]
    if "name" in patch and patch["name"] is not None:
        props["display_name"] = patch["name"]
    if "tags" in patch and patch["tags"] is not None:
        props["twin_tags_json"] = json.dumps(list(patch["tags"]))
    if "properties" in patch and patch["properties"] is not None:
        props["twin_props_json"] = json.dumps(dict(patch["properties"]))
    return props


def _relation_patch_to_props(patch: dict[str, Any]) -> dict[str, Any]:
    """Translate ``GraphRelationPatch`` into Memgraph edge properties."""
    props: dict[str, Any] = {}
    if "label" in patch and patch["label"] is not None:
        # Store as-is; on read we upper-snake the label.
        props["keywords"] = patch["label"]
    if "strength" in patch and patch["strength"] is not None:
        props["weight"] = float(patch["strength"])
    if "properties" in patch and patch["properties"] is not None:
        props["twin_props_json"] = json.dumps(dict(patch["properties"]))
    return props


async def update_graph_entity(
    workspace: str,
    webui_id: str,
    patch: dict[str, Any],
) -> dict[str, Any] | None:
    """MERGE the given properties onto a Memgraph entity node, then
    re-read and return the canonical WebUI shape. Returns ``None`` when
    the node doesn't exist.
    """
    entity_id = _strip_node_prefix(webui_id)
    member_docs, chunk_to_doc = await _member_context(workspace)
    props = _entity_patch_to_props(patch)
    # Folder cloisonnement (#1/#1b): a folder-scoped mutation may only touch
    # the shared base when the entity is pure-member. `absent` stays 404.
    # `mixed` is now a folder-local overlay write, so folder A can edit its view
    # without mutating folder B's shared physical node.
    if member_docs is not None:
        verdict = await _entity_mutation_gate(
            workspace, entity_id, chunk_to_doc, member_docs
        )
        if verdict == _GATE_ABSENT:
            return None
        if verdict == _GATE_MIXED:
            if not props:
                return await _read_one_entity(
                    workspace, entity_id, chunk_to_doc, member_docs
                )
            from .folder import active_folder_id

            folder = active_folder_id()
            if not folder:
                raise MixedProvenanceError(entity_id)
            ok = await _upsert_entity_override(
                workspace, folder, entity_id, props, deleted=False
            )
            if not ok:
                return None
            return await _read_one_entity(
                workspace, entity_id, chunk_to_doc, member_docs
            )
    label = _sanitize_workspace(workspace)
    if not props:
        # Nothing to write — still return the current state if the node exists.
        return await _read_one_entity(
            workspace, entity_id, chunk_to_doc, member_docs
        )
    async with acquire_write_slot():
        async with get_session() as session:
            update_query = (
                f"MATCH (n:`{label}` {{entity_id: $entity_id}}) "
                "SET n += $props "
                "RETURN n.entity_id AS entity_id"
            )
            try:
                result = await session.run(
                    update_query, entity_id=entity_id, props=props
                )
                rows = [record async for record in result]
                await result.consume()
            except Exception:
                logger.exception(
                    "graph_reader.update_graph_entity: write failed for %s",
                    entity_id,
                )
                return None
            if not rows:
                return None
    return await _read_one_entity(workspace, entity_id, chunk_to_doc, member_docs)


async def update_graph_relation(
    workspace: str,
    rel_id: str,
    patch: dict[str, Any],
) -> dict[str, Any] | None:
    """MERGE properties on the DIRECTED edge whose WebUI id is
    ``rel_id``. The endpoint pair is recovered from
    ``_RELATION_ENDPOINT_CACHE`` populated on previous reads — callers
    that haven't fetched the relations recently get ``None``.
    """
    endpoints = lookup_relation_endpoints(rel_id)
    if endpoints is None:
        return None
    cached_workspace, src, tgt = endpoints
    if cached_workspace != workspace:
        # Defensive — should not happen with our single-workspace
        # deploy, but if a future multi-tenant config swaps workspace
        # between calls we want to refuse the write rather than
        # silently update a different KB.
        return None
    member_docs, chunk_to_doc = await _member_context(workspace)
    props = _relation_patch_to_props(patch)
    # Folder cloisonnement (#1/#1b): only a pure-member relation mutates the
    # base edge. `mixed` writes a folder-local overlay; `absent` stays 404.
    if member_docs is not None:
        verdict = await _relation_mutation_gate(
            workspace, src, tgt, chunk_to_doc, member_docs
        )
        if verdict == _GATE_ABSENT:
            return None
        if verdict == _GATE_MIXED:
            if not props:
                return await _read_one_relation(
                    workspace, src, tgt, chunk_to_doc, member_docs
                )
            from .folder import active_folder_id

            folder = active_folder_id()
            if not folder:
                raise MixedProvenanceError(rel_id)
            ok = await _upsert_rel_override(
                workspace, folder, src, tgt, props, deleted=False
            )
            if not ok:
                return None
            return await _read_one_relation(
                workspace, src, tgt, chunk_to_doc, member_docs
            )
    label = _sanitize_workspace(workspace)
    if not props:
        return await _read_one_relation(
            workspace, src, tgt, chunk_to_doc, member_docs
        )
    async with acquire_write_slot():
        async with get_session() as session:
            update_query = (
                f"MATCH (s:`{label}` {{entity_id: $src}})-[r:DIRECTED]->"
                f"(t:`{label}` {{entity_id: $tgt}}) "
                "SET r += $props "
                "RETURN s.entity_id AS source_id"
            )
            try:
                result = await session.run(
                    update_query, src=src, tgt=tgt, props=props
                )
                rows = [record async for record in result]
                await result.consume()
            except Exception:
                logger.exception(
                    "graph_reader.update_graph_relation: write failed for %s→%s",
                    src,
                    tgt,
                )
                return None
            if not rows:
                return None
    return await _read_one_relation(workspace, src, tgt, chunk_to_doc, member_docs)


async def _read_one_entity(
    workspace: str,
    entity_id: str,
    chunk_to_doc: dict[str, str] | None = None,
    member_docs: set[str] | None = None,
) -> dict[str, Any] | None:
    """Re-fetch a single entity to return its canonical projection.

    When ``member_docs`` is provided (folder bound) the projection is
    folder-scoped exactly like the GET path: returns ``None`` if the entity has
    no member source chunk (so a post-write re-read of an out-of-folder entity
    surfaces nothing), and masks the blended description on mixed-folder
    provenance. A manually-created entity ``GRAPH_MEMBER_OF`` the active folder
    is surfaced via direct membership (#1a). ``member_docs=None`` keeps the
    legacy global projection.
    """
    from .folder import active_folder_id

    folder = active_folder_id()
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (n:`{label}` {{entity_id: $eid}}) "
        f"OPTIONAL MATCH (n)-[gm:`{_GRAPH_MEMBER_REL}`]->"
        f"(:`Folder_{label}` {{id: $folder}}) "
        "RETURN n.entity_id AS entity_id, n.entity_type AS entity_type, "
        "n.description AS description, n.source_id AS source_id, "
        "n.display_name AS display_name, n.twin_tags_json AS twin_tags_json, "
        "n.twin_props_json AS twin_props_json, gm IS NOT NULL AS direct"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, eid=entity_id, folder=folder)
            row = None
            direct = False
            async for record in result:
                row = {
                    "entity_id": record["entity_id"],
                    "entity_type": record["entity_type"],
                    "description": record["description"],
                    "source_id": record["source_id"],
                    "display_name": record["display_name"],
                    "twin_tags_json": record["twin_tags_json"],
                    "twin_props_json": record["twin_props_json"],
                }
                direct = bool(record["direct"])
                break
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader._read_one_entity: read failed for %s", entity_id
        )
        return None
    if not row or not row.get("entity_id"):
        return None
    direct_members = {entity_id} if direct else None
    override = await _load_one_entity_override(workspace, folder, entity_id)
    return _node_record_to_entity(
        row, chunk_to_doc, member_docs, direct_members, override
    )


async def _read_one_relation(
    workspace: str,
    src: str,
    tgt: str,
    chunk_to_doc: dict[str, str] | None = None,
    member_docs: set[str] | None = None,
) -> dict[str, Any] | None:
    """Re-fetch a single relation to return its canonical projection.

    The edge's own ``source_id`` (``chunk_source_id``) is selected so the
    folder-scoped projection can detect mixed-folder provenance — without it a
    mixed relation is undetectable and its blended label/props leak. When
    ``member_docs`` is provided the projection is folder-scoped like the GET
    path (``None`` when no member source chunk; masked label/props on mixed). A
    manually-created edge stamped with the active folder in ``twin_folder_json``
    is surfaced via that stamp (#1a). ``member_docs=None`` keeps the legacy
    global projection.
    """
    from .folder import active_folder_id

    folder = active_folder_id()
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (s:`{label}` {{entity_id: $src}})-[r:DIRECTED]->"
        f"(t:`{label}` {{entity_id: $tgt}}) "
        "RETURN s.entity_id AS source_id, t.entity_id AS target_id, "
        "r.keywords AS keywords, r.weight AS weight, "
        "r.source_id AS chunk_source_id, "
        f"r.`{_REL_FOLDER_PROP}` AS twin_folder_json, "
        "r.twin_props_json AS twin_props_json"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, src=src, tgt=tgt)
            row = None
            async for record in result:
                row = {
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "keywords": record["keywords"],
                    "weight": record["weight"],
                    "chunk_source_id": record["chunk_source_id"],
                    "twin_folder_json": record["twin_folder_json"],
                    "twin_props_json": record["twin_props_json"],
                }
                break
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader._read_one_relation: read failed for %s→%s", src, tgt
        )
        return None
    if not row:
        return None
    override = await _load_one_rel_override(workspace, folder, src, tgt)
    return _edge_record_to_relation(
        row, 0, chunk_to_doc, member_docs, folder, override
    )


# ----------------------------------------------------------------------
# Lifecycle — create + delete (M12 batch 3)
# ----------------------------------------------------------------------


async def entity_exists(workspace: str, entity_id: str) -> bool:
    """Cheap existence probe used by the POST/DELETE routes."""
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (n:`{label}` {{entity_id: $eid}}) RETURN n.entity_id AS eid LIMIT 1"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, eid=entity_id)
            found = False
            async for _ in result:
                found = True
                break
            await result.consume()
        return found
    except Exception:
        logger.exception(
            "graph_reader.entity_exists: probe failed for %s", entity_id
        )
        return False


async def create_graph_entity(
    workspace: str, payload: dict[str, Any], *, actor: str = "operator"
) -> dict[str, Any]:
    """Manually add an entity to the KB.

    Returns the projected entity on success. On failure, raises one of
    the typed exceptions defined at the top of this module so the
    route handler can map each cause to a distinct HTTP status:

    - ``EntityExistsError`` — same canonical name already in the workspace.
    - ``EntityCreateBackendError`` — the ``CREATE`` statement failed.
    - ``EntityProjectionError`` — write succeeded, re-read for the
      response payload failed.

    The previous ``None``-as-failure contract conflated all four causes
    into a single 409, which lied to the operator when the real cause
    was a backend outage. See TR-KG-01.
    """
    name = (payload.get("name") or "").strip()
    if not name:
        # Pydantic's field_validator on ``GraphEntityCreate.name`` is
        # the primary gate (returns 422 before the handler runs). This
        # fallback covers direct callers that bypass the route.
        raise EntityCreateBackendError("entity name is empty")
    entity_id = name  # LightRAG uses the canonical name as the PK
    label = _sanitize_workspace(workspace)

    if await entity_exists(workspace, entity_id):
        raise EntityExistsError(entity_id)

    # Build the property map. We seed ``source_id`` with a
    # ``manual:<actor>`` marker so the audit feed can distinguish
    # operator-added entities from LLM-extracted ones, and so the
    # mention/sources badges show a non-zero count instead of 0.
    props: dict[str, Any] = {
        "entity_id": entity_id,
        "entity_type": payload.get("type") or _DEFAULT_TYPE,
        "description": (payload.get("summary") or "").strip(),
        "source_id": f"manual:{actor}",
    }
    if "name" in payload and payload["name"] is not None:
        props["display_name"] = payload["name"]
    if payload.get("tags"):
        props["twin_tags_json"] = json.dumps(list(payload["tags"]))
    if payload.get("properties"):
        props["twin_props_json"] = json.dumps(dict(payload["properties"]))

    from .folder import active_folder_id

    folder = active_folder_id()
    async with acquire_write_slot():
        async with get_session() as session:
            query = (
                f"CREATE (n:`{label}`) "
                "SET n = $props "
                "RETURN n.entity_id AS entity_id"
            )
            try:
                result = await session.run(query, props=props)
                rows = [record async for record in result]
                await result.consume()
                if folder:
                    # #1a: stamp explicit folder membership so this chunk-less
                    # manual entity survives a folder-scoped refresh.
                    stamp = (
                        f"MATCH (n:`{label}` {{entity_id: $eid}}) "
                        f"MERGE (f:`Folder_{label}` {{id: $folder}}) "
                        f"MERGE (n)-[:`{_GRAPH_MEMBER_REL}`]->(f)"
                    )
                    await (
                        await session.run(stamp, eid=entity_id, folder=folder)
                    ).consume()
            except Exception as exc:
                logger.exception(
                    "graph_reader.create_graph_entity: insert failed for %s",
                    entity_id,
                )
                raise EntityCreateBackendError(str(exc)) from exc
            if not rows:
                raise EntityCreateBackendError(
                    f"CREATE returned no rows for {entity_id!r}"
                )
    try:
        projected = await _read_one_entity(workspace, entity_id)
    except Exception as exc:
        logger.exception(
            "graph_reader.create_graph_entity: projection failed for %s",
            entity_id,
        )
        raise EntityProjectionError(entity_id) from exc
    if projected is None:
        raise EntityProjectionError(entity_id)
    return projected


async def delete_graph_entity(workspace: str, webui_id: str) -> bool:
    """Remove the entity and cascade its edges. Returns ``True`` if a
    node was deleted, ``False`` if nothing matched."""
    entity_id = _strip_node_prefix(webui_id)
    label = _sanitize_workspace(workspace)

    # Folder cloisonnement (#1/#1b): a pure-member entity is physically deleted.
    # `mixed` becomes a folder-local tombstone so folder A hides it without
    # deleting folder B's shared physical node. `absent` → 404. Off Twin routes
    # (no folder), the global path is unchanged.
    member_docs, chunk_to_doc = await _member_context(workspace)
    if member_docs is not None:
        verdict = await _entity_mutation_gate(
            workspace, entity_id, chunk_to_doc, member_docs
        )
        if verdict == _GATE_ABSENT:
            return False
        if verdict == _GATE_MIXED:
            from .folder import active_folder_id

            folder = active_folder_id()
            if not folder:
                raise MixedProvenanceError(entity_id)
            return await _upsert_entity_override(
                workspace, folder, entity_id, {}, deleted=True
            )
    elif not await entity_exists(workspace, entity_id):
        return False

    async with acquire_write_slot():
        async with get_session() as session:
            query = (
                f"MATCH (n:`{label}` {{entity_id: $eid}}) "
                "DETACH DELETE n"
            )
            try:
                result = await session.run(query, eid=entity_id)
                await result.consume()
            except Exception:
                logger.exception(
                    "graph_reader.delete_graph_entity: delete failed for %s",
                    entity_id,
                )
                return False

    # Evict any cached relations that referenced this entity so future
    # PATCH requests on stale edges return a clean 404 instead of
    # trying to write to a vanished node.
    stale_ids = [
        rid
        for rid, (ws, src, tgt) in _RELATION_ENDPOINT_CACHE.items()
        if ws == workspace and (src == entity_id or tgt == entity_id)
    ]
    for rid in stale_ids:
        _RELATION_ENDPOINT_CACHE.pop(rid, None)

    return True


async def create_graph_relation(
    workspace: str, payload: dict[str, Any]
) -> dict[str, Any] | None:
    """Manually add a relation between two existing entities.

    Returns the projected relation on success; ``None`` if either
    endpoint is missing (route maps to 422). Idempotent: re-issuing
    the same source/target pair MERGEs onto the existing edge.

    **Folder cloisonnement** (a folder is bound): both endpoints must be
    *pure-member* of the active folder. An out-of-folder endpoint → ``None``
    (422) — this is the gate that stops a relation from folder A to a B-only
    entity known by id. A *mixed* (cross-folder shared) endpoint raises
    ``MixedProvenanceError`` (→ 409): anchoring a new edge on a shared node
    would add an edge the other folder also sees, i.e. mutate the shared
    subgraph from a single folder (same doctrine as the PATCH/DELETE gate). No
    folder bound (native/legacy caller) keeps the global existence-only check.
    """
    src = _strip_node_prefix(str(payload.get("source") or ""))
    tgt = _strip_node_prefix(str(payload.get("target") or ""))
    if not src or not tgt:
        return None
    member_docs, chunk_to_doc = await _member_context(workspace)
    if member_docs is None:
        if not (
            await entity_exists(workspace, src)
            and await entity_exists(workspace, tgt)
        ):
            return None
    else:
        for endpoint in (src, tgt):
            verdict = await _entity_mutation_gate(
                workspace, endpoint, chunk_to_doc, member_docs
            )
            if verdict == _GATE_ABSENT:
                return None
            if verdict == _GATE_MIXED:
                raise MixedProvenanceError(endpoint)

    label_kw = (payload.get("label") or "").strip()
    if not label_kw:
        return None
    strength = payload.get("strength")
    try:
        weight = float(strength) if strength is not None else 0.5
    except (TypeError, ValueError):
        weight = 0.5

    props: dict[str, Any] = {"keywords": label_kw, "weight": weight}
    if payload.get("properties"):
        props["twin_props_json"] = json.dumps(dict(payload["properties"]))
    # #1a: a manual edge has no chunk provenance — stamp the active folder so
    # folder-scoped reads surface it after a refresh.
    from .folder import active_folder_id

    folder = active_folder_id()
    if folder:
        props[_REL_FOLDER_PROP] = json.dumps([folder])

    label = _sanitize_workspace(workspace)
    async with acquire_write_slot():
        async with get_session() as session:
            query = (
                f"MATCH (s:`{label}` {{entity_id: $src}}), "
                f"(t:`{label}` {{entity_id: $tgt}}) "
                "MERGE (s)-[r:DIRECTED]->(t) "
                "SET r += $props "
                "RETURN s.entity_id AS source_id"
            )
            try:
                result = await session.run(
                    query, src=src, tgt=tgt, props=props
                )
                rows = [record async for record in result]
                await result.consume()
            except Exception:
                logger.exception(
                    "graph_reader.create_graph_relation: insert failed for %s→%s",
                    src,
                    tgt,
                )
                return None
            if not rows:
                return None
    relation = await _read_one_relation(workspace, src, tgt)
    if relation is not None:
        _remember_relation(workspace, relation["id"], src, tgt)
    return relation


async def delete_graph_relation(workspace: str, rel_id: str) -> bool:
    """Remove the relation identified by ``rel_id``. Returns ``True``
    on success, ``False`` when the cache doesn't know the id (cold
    process) or no edge matches in Memgraph."""
    endpoints = lookup_relation_endpoints(rel_id)
    if endpoints is None:
        return False
    cached_workspace, src, tgt = endpoints
    if cached_workspace != workspace:
        return False

    member_docs, chunk_to_doc = await _member_context(workspace)
    # Folder cloisonnement (#1/#1b): pure-member physically deletes; mixed
    # writes a folder-local tombstone; absent stays 404.
    if member_docs is not None:
        verdict = await _relation_mutation_gate(
            workspace, src, tgt, chunk_to_doc, member_docs
        )
        if verdict == _GATE_ABSENT:
            return False
        if verdict == _GATE_MIXED:
            from .folder import active_folder_id

            folder = active_folder_id()
            if not folder:
                raise MixedProvenanceError(rel_id)
            return await _upsert_rel_override(
                workspace, folder, src, tgt, {}, deleted=True
            )

    label = _sanitize_workspace(workspace)
    async with acquire_write_slot():
        async with get_session() as session:
            query = (
                f"MATCH (s:`{label}` {{entity_id: $src}})-[r:DIRECTED]->"
                f"(t:`{label}` {{entity_id: $tgt}}) "
                "DELETE r "
                "RETURN s.entity_id AS source_id"
            )
            try:
                result = await session.run(query, src=src, tgt=tgt)
                rows = [record async for record in result]
                await result.consume()
            except Exception:
                logger.exception(
                    "graph_reader.delete_graph_relation: delete failed for %s",
                    rel_id,
                )
                return False
            if not rows:
                return False

    _RELATION_ENDPOINT_CACHE.pop(rel_id, None)
    return True


__all__ = [
    "EntityCreateBackendError",
    "EntityExistsError",
    "EntityProjectionError",
    "GraphEntityCreateError",
    "MixedProvenanceError",
    "create_graph_entity",
    "create_graph_relation",
    "delete_graph_entity",
    "delete_graph_relation",
    "entity_exists",
    "layout_position",
    "lookup_relation_endpoints",
    "map_entity_type",
    "read_graph_entities",
    "read_graph_relations",
    "update_graph_entity",
    "update_graph_relation",
]
