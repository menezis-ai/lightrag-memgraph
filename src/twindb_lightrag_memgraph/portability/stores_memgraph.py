"""Data-plane stores — KV, Vec, DocStatus, Folder, MEMBER_OF, TAGGED_WITH.

Design record: ``docs/adr/010-kb-portability-contract.md``.

Reads are raw Cypher by label through ``get_read_session()`` — none of the
backends exposes an enumeration API and the overlay readers hide archived
records — paged by keyset on the store key. Writes go through the store's
own ``upsert()`` (R-06 neutralisation, write slot, conflict retry, indexes)
wherever one exists; what the public API cannot restore (KV timestamps,
MEMBER_OF ``updated_at``) is set afterwards by an explicit, keyed Cypher —
never a generic ``SET n += props`` on a node the store did not create.

Vector records carry their embedding and ``vector_impl.upsert()`` keeps a
pre-computed ``embedding`` as is (``_compute_missing_embeddings``), so an
import never calls the embedding model: the store is built with an
``EmbeddingFunc`` that refuses to run.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from .._constants import storage_folder_context, validate_identifier
from ._io import batched, keyset, read_rows, read_scalar, write
from .stores import PortabilityError, Scope, StoreSpec, project_record, store_by_name


def _js(value: str, where: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError) as exc:
        raise PortabilityError(f"{where}: stored JSON is not readable ({exc})") from exc


# ------------------------------------------------------------------ KV


class KvStore:
    def __init__(self, namespace: str) -> None:
        self.spec: StoreSpec = store_by_name(f"kv.{namespace}")
        self.namespace = namespace

    def _label(self, scope: Scope) -> str:
        return self.spec.label(scope.workspace)

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        label = self._label(scope)

        async def fetch(after: str | None, limit: int) -> list[dict[str, Any]]:
            return await read_rows(
                f"MATCH (n:`{label}`) WHERE $after IS NULL OR n.id > $after "
                f"RETURN properties(n) AS p ORDER BY n.id LIMIT {int(limit)}",
                after=after,
            )

        async for row in keyset(fetch, lambda r: r["p"]["id"], scope.batch_size):
            props = project_record(self.spec, row["p"])
            yield {
                "id": props["id"],
                "value": _js(props.get("data"), f"{self.spec.name}[{props['id']}]"),
                "created_at": props.get("__created_at"),
                "updated_at": props.get("__updated_at"),
            }

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        from ..kv_impl import MemgraphKVStorage

        store = MemgraphKVStorage(
            namespace=self.namespace,
            global_config={"workspace": scope.workspace},
            embedding_func=None,
        )
        await store.initialize()
        label = self._label(scope)
        total = 0
        async for chunk in batched(records, scope.batch_size):
            await store.upsert({r["id"]: r["value"] for r in chunk})
            stamps = [
                {"id": r["id"], "c": r.get("created_at"), "u": r.get("updated_at")}
                for r in chunk
                if r.get("created_at") or r.get("updated_at")
            ]
            if stamps:
                await write(
                    f"portability.kv.stamps[{label}]",
                    f"UNWIND $rows AS r MATCH (n:`{label}` {{id: r.id}}) "
                    "SET n.__created_at = coalesce(r.c, n.__created_at), "
                    "n.__updated_at = coalesce(r.u, n.__updated_at)",
                    rows=stamps,
                )
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        label = self._label(scope)
        rows = await read_rows(
            f"MATCH (n:`{label}`) RETURN count(n) AS c, max(n.__updated_at) AS m"
        )
        return (
            {"count": rows[0]["c"], "max_updated": rows[0]["m"]}
            if rows
            else {"count": 0, "max_updated": None}
        )

    async def count(self, scope: Scope) -> int:
        return int(
            await read_scalar(f"MATCH (n:`{self._label(scope)}`) RETURN count(n)") or 0
        )


# ----------------------------------------------------------------- Vec


def _refuse_to_embed(*_args: Any, **_kwargs: Any) -> Any:
    raise PortabilityError(
        "an import never computes embeddings — the bundle must carry them"
    )


class VecStore:
    def __init__(self, namespace: str) -> None:
        self.spec: StoreSpec = store_by_name(f"vec.{namespace}")
        self.namespace = namespace

    def _label(self, scope: Scope) -> str:
        return self.spec.label(scope.workspace)

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        label = self._label(scope)

        async def fetch(after: str | None, limit: int) -> list[dict[str, Any]]:
            return await read_rows(
                f"MATCH (n:`{label}`) WHERE $after IS NULL OR n.id > $after "
                f"RETURN properties(n) AS p ORDER BY n.id LIMIT {int(limit)}",
                after=after,
            )

        async for row in keyset(fetch, lambda r: r["p"]["id"], scope.batch_size):
            props = project_record(self.spec, row["p"])
            embedding = props.pop("embedding", None)
            if not embedding:
                raise PortabilityError(
                    f"{self.spec.name}[{props['id']}]: vector without embedding"
                )
            yield {
                "id": props.pop("id"),
                "props": props,
                "embedding": [float(x) for x in embedding],
            }

    def _storage(self, scope: Scope, dim: int) -> Any:
        from lightrag.utils import EmbeddingFunc

        from ..vector_impl import MemgraphVectorDBStorage

        func = EmbeddingFunc(
            embedding_dim=dim, max_token_size=8192, func=_refuse_to_embed
        )
        return MemgraphVectorDBStorage(
            namespace=self.namespace,
            global_config={
                "workspace": scope.workspace,
                "vector_db_storage_cls_kwargs": {},
            },
            embedding_func=func,
            meta_fields=set(),
        )

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        store = None
        total = 0
        async for chunk in batched(records, scope.batch_size):
            if store is None:
                dim = len(chunk[0]["embedding"])
                if scope.embedding_dim is not None and dim != scope.embedding_dim:
                    raise PortabilityError(
                        f"{self.spec.name}: first embedding dimension {dim} "
                        f"differs from manifest dimension {scope.embedding_dim}"
                    )
                store = self._storage(scope, dim)
                await store.initialize()
            payload = {}
            for r in chunk:
                if len(r["embedding"]) != store.embedding_func.embedding_dim:
                    raise PortabilityError(
                        f"{self.spec.name}[{r['id']}]: embedding dimension mismatch"
                    )
                payload[r["id"]] = {**r["props"], "embedding": r["embedding"]}
            await store.upsert(payload)
            total += len(chunk)
        # Empty vector stores still need their canonical index.  Otherwise an
        # empty-but-valid bundle would validate differently from a freshly
        # initialised runtime and the first query would mutate the target by
        # creating the missing index lazily.
        if store is None:
            if scope.embedding_dim is None:
                raise PortabilityError(
                    f"{self.spec.name}: manifest embedding dimension is required"
                )
            store = self._storage(scope, scope.embedding_dim)
            await store.initialize()
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}

    async def count(self, scope: Scope) -> int:
        return int(
            await read_scalar(f"MATCH (n:`{self._label(scope)}`) RETURN count(n)") or 0
        )


# ------------------------------------------------------------ DocStatus

_DOCSTATUS_JSON_FIELDS = ("chunks_list", "metadata")


class DocStatusStore:
    spec: StoreSpec = store_by_name("docstatus")

    def _label(self, scope: Scope) -> str:
        return self.spec.label(scope.workspace)

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        label = self._label(scope)

        async def fetch(after: str | None, limit: int) -> list[dict[str, Any]]:
            return await read_rows(
                f"MATCH (n:`{label}`) WHERE $after IS NULL OR n.id > $after "
                f"RETURN properties(n) AS p ORDER BY n.id LIMIT {int(limit)}",
                after=after,
            )

        async for row in keyset(fetch, lambda r: r["p"]["id"], scope.batch_size):
            props = project_record(self.spec, row["p"])
            for name in _DOCSTATUS_JSON_FIELDS:
                if isinstance(props.get(name), str):
                    props[name] = _js(props[name], f"docstatus[{props['id']}].{name}")
            yield props

    @staticmethod
    def storage(scope: Scope) -> Any:
        from ..docstatus_impl import MemgraphDocStatusStorage

        return MemgraphDocStatusStorage(
            namespace="doc_status",
            global_config={"workspace": scope.workspace},
            embedding_func=None,
        )

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        """Restore documents through ``upsert()`` under the record's legacy
        ``folder`` (mapped): that first membership is created by the store;
        every other membership comes from ``member_of.jsonl``
        (:class:`MemberOfStore`), which the orchestrator imports right after."""
        store = self.storage(scope)
        await store.initialize()
        total = 0
        async for chunk in batched(records, scope.batch_size):
            by_folder: dict[str | None, dict[str, dict[str, Any]]] = {}
            for r in chunk:
                data = dict(r)
                doc_id = data.pop("id")
                folder = data.get("folder")
                if folder:
                    folder = scope.mapped_folder(folder)
                    data["folder"] = folder
                by_folder.setdefault(folder, {})[doc_id] = data
            for folder, docs in by_folder.items():
                with storage_folder_context(folder):
                    await store.upsert(docs)
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        label = self._label(scope)
        rows = await read_rows(
            f"MATCH (n:`{label}`) RETURN count(n) AS c, max(n.updated_at) AS m"
        )
        return (
            {"count": rows[0]["c"], "max_updated": rows[0]["m"]}
            if rows
            else {"count": 0, "max_updated": None}
        )

    async def count(self, scope: Scope) -> int:
        return int(
            await read_scalar(f"MATCH (n:`{self._label(scope)}`) RETURN count(n)") or 0
        )


# --------------------------------------------------------------- Folder


class FolderStore:
    spec: StoreSpec = store_by_name("folders")

    def _label(self, scope: Scope) -> str:
        return self.spec.label(scope.workspace)

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        label = self._label(scope)

        async def fetch(after: str | None, limit: int) -> list[dict[str, Any]]:
            return await read_rows(
                f"MATCH (f:`{label}`) WHERE $after IS NULL OR f.id > $after "
                f"RETURN properties(f) AS p ORDER BY f.id LIMIT {int(limit)}",
                after=after,
            )

        async for row in keyset(fetch, lambda r: r["p"]["id"], scope.batch_size):
            yield project_record(self.spec, row["p"])

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        label = self._label(scope)
        total = 0
        async for chunk in batched(records, scope.batch_size):
            rows = [
                {"id": validate_identifier(scope.mapped_folder(r["id"]), "folder")}
                for r in chunk
            ]
            await write(
                f"portability.folders[{label}]",
                f"UNWIND $rows AS r MERGE (f:`{label}` {{id: r.id}})",
                rows=rows,
            )
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}

    async def count(self, scope: Scope) -> int:
        return int(
            await read_scalar(f"MATCH (f:`{self._label(scope)}`) RETURN count(f)") or 0
        )


# ------------------------------------------------------------ MEMBER_OF


class MemberOfStore:
    spec: StoreSpec = store_by_name("member_of")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        doc_label = store_by_name("docstatus").label(scope.workspace)
        folder_label = store_by_name("folders").label(scope.workspace)

        async def fetch(
            after: tuple[str, str] | None, limit: int
        ) -> list[dict[str, Any]]:
            a_doc, a_folder = after if after else (None, None)
            return await read_rows(
                f"MATCH (d:`{doc_label}`)-[m:MEMBER_OF]->(f:`{folder_label}`) "
                "WITH d.id AS doc_id, f.id AS folder_id, properties(m) AS p "
                "WHERE $a_doc IS NULL OR doc_id > $a_doc OR (doc_id = $a_doc AND folder_id > $a_folder) "
                f"RETURN doc_id, folder_id, p ORDER BY doc_id, folder_id LIMIT {int(limit)}",
                a_doc=a_doc,
                a_folder=a_folder,
            )

        async for row in keyset(
            fetch, lambda r: (r["doc_id"], r["folder_id"]), scope.batch_size
        ):
            props = project_record(
                self.spec,
                {**row["p"], "doc_id": row["doc_id"], "folder_id": row["folder_id"]},
            )
            yield props

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        store = DocStatusStore.storage(scope)
        await store.initialize()
        doc_label = store_by_name("docstatus").label(scope.workspace)
        folder_label = store_by_name("folders").label(scope.workspace)
        total = 0
        async for chunk in batched(records, scope.batch_size):
            stamps = []
            for r in chunk:
                folder = validate_identifier(
                    scope.mapped_folder(r["folder_id"]), "folder"
                )
                if not await store.add_to_folder(r["doc_id"], folder):
                    raise PortabilityError(
                        f"member_of: document {r['doc_id']!r} is absent from the target"
                    )
                if r.get("updated_at") is not None:
                    stamps.append(
                        {
                            "doc_id": r["doc_id"],
                            "folder_id": folder,
                            "u": r["updated_at"],
                        }
                    )
            if stamps:
                await write(
                    f"portability.member_of[{doc_label}]",
                    f"UNWIND $rows AS r MATCH (d:`{doc_label}` {{id: r.doc_id}})-[m:MEMBER_OF]->"
                    f"(f:`{folder_label}` {{id: r.folder_id}}) SET m.updated_at = r.u",
                    rows=stamps,
                )
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}

    async def count(self, scope: Scope) -> int:
        doc_label = store_by_name("docstatus").label(scope.workspace)
        folder_label = store_by_name("folders").label(scope.workspace)
        return int(
            await read_scalar(
                f"MATCH (:`{doc_label}`)-[m:MEMBER_OF]->(:`{folder_label}`) RETURN count(m)"
            )
            or 0
        )


# ---------------------------------------------------------- TAGGED_WITH


class TaggedWithStore:
    """``(DocStatus_{ws})-[:TAGGED_WITH]->(WebuiTag_{folder})`` — folder-scoped."""

    spec: StoreSpec = store_by_name("tagged_with")

    async def export_records(self, scope: Scope) -> AsyncIterator[dict[str, Any]]:
        doc_label = store_by_name("docstatus").label(scope.workspace)
        tag_spec = store_by_name("tags")
        for folder_id in scope.folder_ids:
            tag_label = tag_spec.label(folder_id)

            async def fetch(
                after: tuple[str, str] | None, limit: int, *, tag_label: str = tag_label
            ) -> list[dict[str, Any]]:
                a_doc, a_tag = after if after else (None, None)
                return await read_rows(
                    f"MATCH (d:`{doc_label}`)-[r:TAGGED_WITH]->(t:`{tag_label}`) "
                    "WITH d.id AS doc_id, t.id AS tag_id, properties(r) AS p "
                    "WHERE $a_doc IS NULL OR doc_id > $a_doc OR (doc_id = $a_doc AND tag_id > $a_tag) "
                    f"RETURN doc_id, tag_id, p ORDER BY doc_id, tag_id LIMIT {int(limit)}",
                    a_doc=a_doc,
                    a_tag=a_tag,
                )

            async for row in keyset(
                fetch, lambda r: (r["doc_id"], r["tag_id"]), scope.batch_size
            ):
                yield project_record(
                    self.spec,
                    {
                        **row["p"],
                        "doc_id": row["doc_id"],
                        "folder_id": folder_id,
                        "tag_id": row["tag_id"],
                    },
                )

    async def import_records(
        self, records: AsyncIterator[dict[str, Any]], scope: Scope
    ) -> int:
        doc_label = store_by_name("docstatus").label(scope.workspace)
        tag_spec = store_by_name("tags")
        total = 0
        async for chunk in batched(records, scope.batch_size):
            by_folder: dict[str, list[dict[str, Any]]] = {}
            for r in chunk:
                folder = validate_identifier(
                    scope.mapped_folder(r["folder_id"]), "folder"
                )
                props = {
                    k: v
                    for k, v in r.items()
                    if k not in ("doc_id", "folder_id", "tag_id")
                }
                by_folder.setdefault(folder, []).append(
                    {"doc_id": r["doc_id"], "tag_id": r["tag_id"], "props": props}
                )
            for folder, rows in by_folder.items():
                tag_label = tag_spec.label(folder)
                done = await write(
                    f"portability.tagged_with[{tag_label}]",
                    f"UNWIND $rows AS r MATCH (d:`{doc_label}` {{id: r.doc_id}}) "
                    f"MATCH (t:`{tag_label}` {{id: r.tag_id}}) "
                    "MERGE (d)-[e:TAGGED_WITH]->(t) SET e += r.props RETURN count(e) AS c",
                    rows=rows,
                )
                created = int(done[0]["c"]) if done else 0
                if created != len(rows):
                    raise PortabilityError(
                        f"tagged_with[{folder}]: {len(rows) - created} edge(s) reference a missing document or tag"
                    )
            total += len(chunk)
        return total

    async def fingerprint(self, scope: Scope) -> dict[str, Any]:
        return {"count": await self.count(scope)}

    async def count(self, scope: Scope) -> int:
        doc_label = store_by_name("docstatus").label(scope.workspace)
        tag_spec = store_by_name("tags")
        total = 0
        for folder_id in scope.folder_ids:
            total += int(
                await read_scalar(
                    f"MATCH (:`{doc_label}`)-[r:TAGGED_WITH]->(:`{tag_spec.label(folder_id)}`) RETURN count(r)"
                )
                or 0
            )
        return total
