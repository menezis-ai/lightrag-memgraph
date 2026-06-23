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


def _node_record_to_entity(
    record: dict[str, Any],
    chunk_to_doc: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Project a Cypher entity row into the WebUI ``GraphEntity`` shape.

    ``mentions`` is the count of distinct chunks the entity appears in
    (LightRAG joins the chunk ids in ``source_id`` with ``<SEP>``).
    ``sources`` is the count of distinct parent documents — computed by
    joining each chunk id against ``chunk_to_doc`` (built once from
    ``DocStatus.chunks_list``). When the map is missing or no chunks
    can be resolved (orphan chunks, fresh KB, lookup failure) sources
    falls back to mentions so the badge never reads 0 with non-empty
    source_id.
    """
    entity_id = record.get("entity_id") or ""
    raw_type = record.get("entity_type") or ""
    mapped_type = map_entity_type(str(raw_type))
    # LightRAG joins per-chunk descriptions with `<SEP>` when an entity
    # is mentioned in multiple chunks. Replace with a visible separator
    # so the WebUI summary reads cleanly instead of leaking the marker.
    summary = (
        (record.get("description") or "")
        .replace("<SEP>", " · ")
        .strip()
    )
    source_id = record.get("source_id") or ""
    chunks = {
        c.strip() for c in str(source_id).replace("<SEP>", ",").split(",") if c.strip()
    }
    mentions = len(chunks)
    resolved_docs: set[str] = set()
    if chunk_to_doc:
        resolved_docs = {chunk_to_doc[c] for c in chunks if c in chunk_to_doc}
        sources = len(resolved_docs) if resolved_docs else mentions
    else:
        sources = mentions
    x, y = layout_position(str(entity_id), mapped_type)
    return {
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


def _edge_record_to_relation(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Project a Cypher edge row into the WebUI ``GraphRelation`` shape.

    ``index`` is accepted for backwards compatibility but ignored; the
    relation id is now derived from the endpoint pair.
    """
    del index  # ignored — id is endpoint-derived
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
    return {
        "id": _relation_id_from_endpoints(str(src), str(tgt)),
        "source": _entity_id_to_node_id(str(src)),
        "target": _entity_id_to_node_id(str(tgt)),
        "label": label,
        "strength": round(strength, 3),
        "properties": _json_str_dict(record.get("twin_props_json")),
    }


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
    query = (
        f"MATCH (n:`{label}`) "
        "RETURN n.entity_id AS entity_id, n.entity_type AS entity_type, "
        "n.description AS description, n.source_id AS source_id, "
        "n.display_name AS display_name, n.twin_tags_json AS twin_tags_json, "
        "n.twin_props_json AS twin_props_json "
        "LIMIT $max_nodes"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, max_nodes=max_nodes)
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
        return [
            _node_record_to_entity(row, chunk_to_doc)
            for row in rows
            if row.get("entity_id")
        ]
    except Exception:
        logger.exception(
            "graph_reader: failed to read entities for workspace=%s", workspace
        )
        return []


def _native_node_to_entity(
    node: Any,
    chunk_to_doc: dict[str, dict[str, str]],
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
    return _node_record_to_entity(row, chunk_to_doc)


def _native_edge_to_relation(edge: Any, index: int) -> dict[str, Any] | None:
    src = getattr(edge, "source", None)
    tgt = getattr(edge, "target", None)
    if not src or not tgt:
        return None
    eprops = getattr(edge, "properties", None) or {}
    row = {
        "source_id": src,
        "target_id": tgt,
        "keywords": eprops.get("keywords"),
        "weight": eprops.get("weight"),
        "twin_props_json": eprops.get("twin_props_json"),
    }
    return _edge_record_to_relation(row, index)


async def read_graph_native(
    rag: Any,
    workspace: str,
    *,
    node_label: str = "*",
    max_depth: int = 3,
    max_nodes: int = 1000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    the legacy reader. Returns ``([], [])`` on any failure so callers fall back
    to the seed / empty per the audit-C5 gate.
    """
    try:
        kg = await rag.get_knowledge_graph(
            node_label=node_label,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
    except Exception:
        logger.exception(
            "graph_reader: native get_knowledge_graph failed "
            "(label=%s, workspace=%s)",
            node_label,
            workspace,
        )
        return [], []

    chunk_to_doc = await _load_chunk_to_doc_index(workspace)

    entities: list[dict[str, Any]] = []
    for node in getattr(kg, "nodes", []) or []:
        entity = _native_node_to_entity(node, chunk_to_doc)
        if entity is not None:
            entities.append(entity)

    valid_ids = {e["id"] for e in entities}
    relations: list[dict[str, Any]] = []
    for i, edge in enumerate(getattr(kg, "edges", []) or []):
        rel = _native_edge_to_relation(edge, i)
        if rel is None:
            continue
        if rel["source"] not in valid_ids or rel["target"] not in valid_ids:
            continue
        _remember_relation(workspace, rel["id"], rel["source"], rel["target"])
        relations.append(rel)

    return entities, relations


async def search_graph_labels(rag: Any, q: str, *, limit: int = 50) -> list[str]:
    """Server-side fuzzy entity-label search (delegates to LightRAG's native
    ``search_labels``) so the Graph search box reaches the whole KB, not just
    the loaded subgraph."""
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
    query = (
        f"MATCH (s:`{label}`)-[r:DIRECTED]->(t:`{label}`) "
        "RETURN s.entity_id AS source_id, t.entity_id AS target_id, "
        "r.keywords AS keywords, r.weight AS weight, "
        "r.twin_props_json AS twin_props_json "
        "LIMIT $max_edges"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, max_edges=max_edges)
            rows = []
            async for record in result:
                rows.append(
                    {
                        "source_id": record["source_id"],
                        "target_id": record["target_id"],
                        "keywords": record["keywords"],
                        "weight": record["weight"],
                        "twin_props_json": record["twin_props_json"],
                    }
                )
            await result.consume()
        valid = set(valid_node_ids) if valid_node_ids is not None else None
        out: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            if not row.get("source_id") or not row.get("target_id"):
                continue
            rel = _edge_record_to_relation(row, i)
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
    props = _entity_patch_to_props(patch)
    label = _sanitize_workspace(workspace)
    if not props:
        # Nothing to write — still return the current state if the node exists.
        return await _read_one_entity(workspace, entity_id)
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
    return await _read_one_entity(workspace, entity_id)


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
    props = _relation_patch_to_props(patch)
    label = _sanitize_workspace(workspace)
    if not props:
        return await _read_one_relation(workspace, src, tgt)
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
    return await _read_one_relation(workspace, src, tgt)


async def _read_one_entity(
    workspace: str, entity_id: str
) -> dict[str, Any] | None:
    """Re-fetch a single entity to return its canonical projection."""
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (n:`{label}` {{entity_id: $eid}}) "
        "RETURN n.entity_id AS entity_id, n.entity_type AS entity_type, "
        "n.description AS description, n.source_id AS source_id, "
        "n.display_name AS display_name, n.twin_tags_json AS twin_tags_json, "
        "n.twin_props_json AS twin_props_json"
    )
    try:
        async with get_read_session() as session:
            result = await session.run(query, eid=entity_id)
            row = None
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
                break
            await result.consume()
    except Exception:
        logger.exception(
            "graph_reader._read_one_entity: read failed for %s", entity_id
        )
        return None
    if not row or not row.get("entity_id"):
        return None
    return _node_record_to_entity(row)


async def _read_one_relation(
    workspace: str, src: str, tgt: str
) -> dict[str, Any] | None:
    """Re-fetch a single relation to return its canonical projection."""
    label = _sanitize_workspace(workspace)
    query = (
        f"MATCH (s:`{label}` {{entity_id: $src}})-[r:DIRECTED]->"
        f"(t:`{label}` {{entity_id: $tgt}}) "
        "RETURN s.entity_id AS source_id, t.entity_id AS target_id, "
        "r.keywords AS keywords, r.weight AS weight, "
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
    return _edge_record_to_relation(row, 0)


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

    if not await entity_exists(workspace, entity_id):
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
    """
    src = _strip_node_prefix(str(payload.get("source") or ""))
    tgt = _strip_node_prefix(str(payload.get("target") or ""))
    if not src or not tgt:
        return None
    if not (
        await entity_exists(workspace, src)
        and await entity_exists(workspace, tgt)
    ):
        return None

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
