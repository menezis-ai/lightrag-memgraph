"""Graph stores — nodes with their type labels, ``DIRECTED`` edges in their
stored direction, ``GRAPH_MEMBER_OF``, folder overrides (T1.2).

Decision Q5 (2026-08-25): an edge is exported as ``startNode(r) → endNode(r)``
exactly as Memgraph stores it and re-imported with a directed ``MERGE
(s)-[r:DIRECTED]->(t)`` — never sorted, never merged with its reverse.
Ingestion writes edges with an undirected MERGE (``_buffered_graph``), the
operator with a directed one (``graph_reader``); LightRAG reads them without
direction, so preserving the stored orientation is the only choice that keeps
both writers' output identical after a round-trip.

Node type labels are the second label ``_buffered_graph`` sets
(``SET n:`Person```); the workspace label is implicit and stripped on export.
Override writes mirror ``graph_reader._upsert_entity_override`` /
``_upsert_rel_override`` byte for byte but *raise* on a missing base node
instead of returning ``False`` — an import wants the error.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from .._constants import validate_identifier
from ._io import batched, keyset, read_rows, read_scalar, write
from .stores import PortabilityError, Scope, StoreSpec, project_record, store_by_name

_DIRECTED = "DIRECTED"
_GRAPH_MEMBER_OF = "GRAPH_MEMBER_OF"
_HAS_OVERRIDE = "HAS_OVERRIDE"
_HAS_REL_OVERRIDE = "HAS_REL_OVERRIDE"


def _ws(scope: Scope) -> str:
    return validate_identifier(scope.workspace, "workspace")


def remap_relation_folder_json(value: Any, folder_map: dict[str, str]) -> str | None:
    """Return the canonical folder stamp for a portable manual relation.

    Manual relations have no chunk provenance; ``twin_folder_json`` is their
    visibility boundary.  It therefore has to follow the same folder mapping
    as memberships and overrides.  Validation calls this exact helper while
    normalising the source hash so import and proof cannot drift apart.
    """
    if value is None:
        return None
    try:
        parsed = value if isinstance(value, list) else json.loads(str(value))
    except (TypeError, ValueError) as exc:
        raise PortabilityError(
            "graph.edges: twin_folder_json must be a JSON list"
        ) from exc
    if not isinstance(parsed, list):
        raise PortabilityError("graph.edges: twin_folder_json must be a JSON list")
    mapped: list[str] = []
    for raw in parsed:
        if raw is None or not str(raw):
            continue
        source = validate_identifier(str(raw), "relation folder")
        mapped.append(
            validate_identifier(folder_map.get(source, source), "relation folder")
        )
    return json.dumps(mapped, ensure_ascii=False, separators=(",", ":"))


# ----------------------------------------------------------------- nodes


class GraphNodeStore:
    spec: StoreSpec = store_by_name("graph.nodes")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        ws = _ws(scope)

        async def fetch(after: str | None, limit: int) -> list[dict[str, Any]]:
            return await read_rows(
                f"MATCH (n:`{ws}`) WHERE n.entity_id IS NOT NULL AND ($after IS NULL OR n.entity_id > $after) "
                f"RETURN n.entity_id AS eid, labels(n) AS labels, properties(n) AS p "
                f"ORDER BY n.entity_id LIMIT {int(limit)}",
                after=after,
            )

        async for row in keyset(fetch, lambda r: r["eid"], scope.batch_size):
            props = project_record(self.spec, row["p"])
            props.pop("entity_id", None)
            labels = sorted(lbl for lbl in row["labels"] if lbl != ws)
            yield {"entity_id": row["eid"], "labels": labels, "props": props}

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        ws = _ws(scope)
        total = 0
        async for chunk in batched(records, scope.batch_size):
            rows = [
                {
                    "entity_id": r["entity_id"],
                    "props": {**r["props"], "entity_id": r["entity_id"]},
                }
                for r in chunk
            ]
            await write(
                f"portability.graph.nodes[{ws}]",
                f"UNWIND $rows AS r MERGE (n:`{ws}` {{entity_id: r.entity_id}}) SET n += r.props",
                rows=rows,
            )
            by_label: dict[str, list[str]] = {}
            for r in chunk:
                for lbl in r.get("labels") or ():
                    by_label.setdefault(
                        validate_identifier(str(lbl), "entity_type"), []
                    ).append(r["entity_id"])
            for lbl, names in by_label.items():
                await write(
                    f"portability.graph.labels[{ws}:{lbl}]",
                    f"UNWIND $names AS name MATCH (n:`{ws}` {{entity_id: name}}) SET n:`{lbl}`",
                    names=names,
                )
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}

    async def count(self, scope: Scope) -> int:
        return int(
            await read_scalar(
                f"MATCH (n:`{_ws(scope)}`) WHERE n.entity_id IS NOT NULL RETURN count(n)"
            )
            or 0
        )


# ----------------------------------------------------------------- edges


class GraphEdgeStore:
    spec: StoreSpec = store_by_name("graph.edges")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        ws = _ws(scope)

        async def fetch(
            after: tuple[str, str, int] | None, limit: int
        ) -> list[dict[str, Any]]:
            a_src, a_tgt, a_rid = after if after else (None, None, None)
            return await read_rows(
                f"MATCH (s:`{ws}`)-[r:`{_DIRECTED}`]->(t:`{ws}`) "
                "WITH s.entity_id AS src, t.entity_id AS tgt, id(r) AS rid, properties(r) AS p "
                "WHERE $a_src IS NULL OR src > $a_src OR (src = $a_src AND (tgt > $a_tgt OR (tgt = $a_tgt AND rid > $a_rid))) "
                f"RETURN src, tgt, rid, p ORDER BY src, tgt, rid LIMIT {int(limit)}",
                a_src=a_src,
                a_tgt=a_tgt,
                a_rid=a_rid,
            )

        async for row in keyset(
            fetch, lambda r: (r["src"], r["tgt"], r["rid"]), scope.batch_size
        ):
            props = project_record(
                self.spec, {**row["p"], "src": row["src"], "tgt": row["tgt"]}
            )
            props.pop("src")
            props.pop("tgt")
            yield {"src": row["src"], "tgt": row["tgt"], "props": props}

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        ws = _ws(scope)
        total = 0
        async for chunk in batched(records, scope.batch_size):
            rows = []
            for record in chunk:
                props = dict(record["props"])
                if "twin_folder_json" in props:
                    props["twin_folder_json"] = remap_relation_folder_json(
                        props["twin_folder_json"], scope.folder_map
                    )
                rows.append(
                    {"src": record["src"], "tgt": record["tgt"], "props": props}
                )
            done = await write(
                f"portability.graph.edges[{ws}]",
                f"UNWIND $rows AS r MATCH (s:`{ws}` {{entity_id: r.src}}) MATCH (t:`{ws}` {{entity_id: r.tgt}}) "
                f"MERGE (s)-[e:`{_DIRECTED}`]->(t) SET e += r.props RETURN count(e) AS c",
                rows=rows,
            )
            merged = int(done[0]["c"]) if done else 0
            if merged != len(rows):
                raise PortabilityError(
                    f"graph.edges: {len(rows) - merged} edge(s) reference a missing entity"
                )
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}

    async def count(self, scope: Scope) -> int:
        ws = _ws(scope)
        return int(
            await read_scalar(
                f"MATCH (:`{ws}`)-[r:`{_DIRECTED}`]->(:`{ws}`) RETURN count(r)"
            )
            or 0
        )


# ------------------------------------------------------- GRAPH_MEMBER_OF


class GraphMemberOfStore:
    spec: StoreSpec = store_by_name("graph.member_of")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        ws = _ws(scope)
        folder_label = store_by_name("folders").label(ws)

        async def fetch(
            after: tuple[str, str] | None, limit: int
        ) -> list[dict[str, Any]]:
            a_e, a_f = after if after else (None, None)
            return await read_rows(
                f"MATCH (n:`{ws}`)-[m:`{_GRAPH_MEMBER_OF}`]->(f:`{folder_label}`) "
                "WITH n.entity_id AS entity_id, f.id AS folder_id, properties(m) AS p "
                "WHERE $a_e IS NULL OR entity_id > $a_e OR (entity_id = $a_e AND folder_id > $a_f) "
                f"RETURN entity_id, folder_id, p ORDER BY entity_id, folder_id LIMIT {int(limit)}",
                a_e=a_e,
                a_f=a_f,
            )

        async for row in keyset(
            fetch, lambda r: (r["entity_id"], r["folder_id"]), scope.batch_size
        ):
            projected = project_record(
                self.spec,
                {
                    **row["p"],
                    "entity_id": row["entity_id"],
                    "folder_id": row["folder_id"],
                },
            )
            yield projected

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        ws = _ws(scope)
        folder_label = store_by_name("folders").label(ws)
        total = 0
        async for chunk in batched(records, scope.batch_size):
            rows = [
                {
                    "entity_id": r["entity_id"],
                    "folder_id": validate_identifier(
                        scope.mapped_folder(r["folder_id"]), "folder"
                    ),
                }
                for r in chunk
            ]
            done = await write(
                f"portability.graph.member_of[{ws}]",
                f"UNWIND $rows AS r MATCH (n:`{ws}` {{entity_id: r.entity_id}}) "
                f"MERGE (f:`{folder_label}` {{id: r.folder_id}}) MERGE (n)-[m:`{_GRAPH_MEMBER_OF}`]->(f) RETURN count(m) AS c",
                rows=rows,
            )
            merged = int(done[0]["c"]) if done else 0
            if merged != len(rows):
                raise PortabilityError(
                    f"graph.member_of: {len(rows) - merged} row(s) reference a missing entity"
                )
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}

    async def count(self, scope: Scope) -> int:
        ws = _ws(scope)
        folder_label = store_by_name("folders").label(ws)
        return int(
            await read_scalar(
                f"MATCH (:`{ws}`)-[m:`{_GRAPH_MEMBER_OF}`]->(:`{folder_label}`) RETURN count(m)"
            )
            or 0
        )


# ------------------------------------------------------------- overrides


class GraphOverrideStore:
    """Entity overrides hang off their base node (``HAS_OVERRIDE``); relation
    overrides are standalone nodes keyed ``{src, tgt, folder}``. Records:
    ``{"kind": "entity", "entity_id", "folder", "props"}`` and
    ``{"kind": "relation", "src", "tgt", "folder", "props"}`` — ``props``
    carries the override fields and the ``deleted`` tombstone."""

    spec: StoreSpec = store_by_name("graph.overrides")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        ws = _ws(scope)
        ent_label = f"GraphOverride_{ws}"
        rel_label = f"GraphRelOverride_{ws}"

        async def fetch_ent(
            after: tuple[str, str] | None, limit: int
        ) -> list[dict[str, Any]]:
            a_e, a_f = after if after else (None, None)
            return await read_rows(
                f"MATCH (b:`{ws}`)-[:`{_HAS_OVERRIDE}`]->(o:`{ent_label}`) "
                "WITH b.entity_id AS entity_id, o.folder AS folder, properties(o) AS p "
                "WHERE $a_e IS NULL OR entity_id > $a_e OR (entity_id = $a_e AND folder > $a_f) "
                f"RETURN entity_id, folder, p ORDER BY entity_id, folder LIMIT {int(limit)}",
                a_e=a_e,
                a_f=a_f,
            )

        async for row in keyset(
            fetch_ent, lambda r: (r["entity_id"], r["folder"]), scope.batch_size
        ):
            props = project_record(self.spec, row["p"])
            props.pop("folder", None)
            yield {
                "kind": "entity",
                "entity_id": row["entity_id"],
                "folder": row["folder"],
                "props": props,
            }

        async def fetch_rel(
            after: tuple[str, str, str] | None, limit: int
        ) -> list[dict[str, Any]]:
            a_s, a_t, a_f = after if after else (None, None, None)
            return await read_rows(
                f"MATCH (o:`{rel_label}`) "
                "WITH o.src AS src, o.tgt AS tgt, o.folder AS folder, properties(o) AS p "
                "WHERE $a_s IS NULL OR src > $a_s OR (src = $a_s AND (tgt > $a_t OR (tgt = $a_t AND folder > $a_f))) "
                f"RETURN src, tgt, folder, p ORDER BY src, tgt, folder LIMIT {int(limit)}",
                a_s=a_s,
                a_t=a_t,
                a_f=a_f,
            )

        async for row in keyset(
            fetch_rel, lambda r: (r["src"], r["tgt"], r["folder"]), scope.batch_size
        ):
            props = project_record(self.spec, row["p"])
            for k in ("src", "tgt", "folder"):
                props.pop(k, None)
            yield {
                "kind": "relation",
                "src": row["src"],
                "tgt": row["tgt"],
                "folder": row["folder"],
                "props": props,
            }

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        ws = _ws(scope)
        ent_label = f"GraphOverride_{ws}"
        rel_label = f"GraphRelOverride_{ws}"
        total = 0
        async for chunk in batched(records, scope.batch_size):
            ents, rels = [], []
            for r in chunk:
                folder = validate_identifier(scope.mapped_folder(r["folder"]), "folder")
                fields = dict(r["props"])
                deleted = bool(fields.pop("deleted", False))
                if r["kind"] == "entity":
                    ents.append(
                        {
                            "eid": r["entity_id"],
                            "folder": folder,
                            "fields": fields,
                            "deleted": deleted,
                        }
                    )
                elif r["kind"] == "relation":
                    rels.append(
                        {
                            "src": r["src"],
                            "tgt": r["tgt"],
                            "folder": folder,
                            "fields": fields,
                            "deleted": deleted,
                        }
                    )
                else:
                    raise PortabilityError(
                        f"graph.overrides: unknown kind {r['kind']!r}"
                    )
            if ents:
                done = await write(
                    f"portability.graph.overrides.entity[{ws}]",
                    f"UNWIND $rows AS r MATCH (n:`{ws}` {{entity_id: r.eid}}) "
                    f"MERGE (n)-[:`{_HAS_OVERRIDE}`]->(o:`{ent_label}` {{folder: r.folder}}) "
                    "SET o += r.fields SET o.deleted = r.deleted RETURN count(o) AS c",
                    rows=ents,
                )
                if (int(done[0]["c"]) if done else 0) != len(ents):
                    raise PortabilityError(
                        "graph.overrides: an entity override references a missing entity"
                    )
            if rels:
                done = await write(
                    f"portability.graph.overrides.relation[{ws}]",
                    f"UNWIND $rows AS r MATCH (s:`{ws}` {{entity_id: r.src}}) "
                    f"MERGE (o:`{rel_label}` {{src: r.src, tgt: r.tgt, folder: r.folder}}) "
                    f"MERGE (s)-[:`{_HAS_REL_OVERRIDE}`]->(o) "
                    "SET o += r.fields SET o.deleted = r.deleted RETURN count(o) AS c",
                    rows=rels,
                )
                if (int(done[0]["c"]) if done else 0) != len(rels):
                    raise PortabilityError(
                        "graph.overrides: a relation override references a missing source entity"
                    )
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}

    async def count(self, scope: Scope) -> int:
        ws = _ws(scope)
        ents = int(
            await read_scalar(
                f"MATCH (:`{ws}`)-[:`{_HAS_OVERRIDE}`]->(o:`GraphOverride_{ws}`) RETURN count(o)"
            )
            or 0
        )
        rels = int(
            await read_scalar(f"MATCH (o:`GraphRelOverride_{ws}`) RETURN count(o)") or 0
        )
        return ents + rels
