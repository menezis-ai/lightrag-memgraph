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

The reader is intentionally read-only. PATCH persistence (M12 batch 2)
lives in a follow-up.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any, Sequence

from .._pool import get_read_session

logger = logging.getLogger(__name__)

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


def _node_record_to_entity(record: dict[str, Any]) -> dict[str, Any]:
    """Project a Cypher entity row into the WebUI ``GraphEntity`` shape."""
    entity_id = record.get("entity_id") or ""
    raw_type = record.get("entity_type") or ""
    mapped_type = map_entity_type(str(raw_type))
    summary = (record.get("description") or "").strip()
    source_id = record.get("source_id") or ""
    # source_id is a delimited list of chunk ids (LightRAG joins with
    # `<SEP>` by default). Count distinct chunks for both the
    # "mentions" badge and the "sources" badge — LightRAG doesn't
    # encode the parent doc in the chunk id, so the WebUI uses chunk
    # count as the best available proxy for "sources" until we join
    # against DocStatus in a follow-up.
    chunks = {
        c.strip() for c in str(source_id).replace("<SEP>", ",").split(",") if c.strip()
    }
    mentions = len(chunks)
    sources = mentions
    x, y = layout_position(str(entity_id), mapped_type)
    return {
        "id": _entity_id_to_node_id(str(entity_id)),
        "name": str(entity_id),
        "type": mapped_type,
        "x": x,
        "y": y,
        "mentions": mentions,
        "sources": sources,
        "summary": summary[:600],
    }


def _edge_record_to_relation(record: dict[str, Any], index: int) -> dict[str, Any]:
    """Project a Cypher edge row into the WebUI ``GraphRelation`` shape."""
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
        "id": f"kr_{index:06d}",
        "source": _entity_id_to_node_id(str(src)),
        "target": _entity_id_to_node_id(str(tgt)),
        "label": label,
        "strength": round(strength, 3),
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
        "n.description AS description, n.source_id AS source_id "
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
                    }
                )
            await result.consume()
        return [_node_record_to_entity(row) for row in rows if row.get("entity_id")]
    except Exception:
        logger.exception(
            "graph_reader: failed to read entities for workspace=%s", workspace
        )
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
        "r.keywords AS keywords, r.weight AS weight "
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
            out.append(rel)
        return out
    except Exception:
        logger.exception(
            "graph_reader: failed to read relations for workspace=%s", workspace
        )
        return []


__all__ = [
    "layout_position",
    "map_entity_type",
    "read_graph_entities",
    "read_graph_relations",
]
