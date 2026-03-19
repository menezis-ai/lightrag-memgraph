"""
Vector Storage backend using Memgraph native vector search.

Requires: Memgraph >= 3.2 with MAGE (vector_search module).

Each vector entry is a Cypher node:
  Label: :Vec_{workspace}_{namespace}
  Properties: id, embedding (list<float>), content, + meta_fields

Vector index:
  CREATE VECTOR INDEX vec_{workspace}_{namespace}
  ON :Vec_{workspace}_{namespace}(embedding)
  WITH CONFIG {"dimension": N, "capacity": VECTOR_INDEX_CAPACITY, "metric": "cos"}

Query:
  CALL vector_search.search("vec_...", $embedding, $top_k)
  YIELD node, similarity
"""

import json
from dataclasses import dataclass
from typing import Any

from lightrag.base import BaseVectorStorage
from lightrag.utils import logger

from . import _pool
from ._constants import VECTOR_INDEX_CAPACITY, resolve_workspace, validate_identifier


@dataclass
class MemgraphVectorDBStorage(BaseVectorStorage):
    def __init__(
        self,
        namespace,
        global_config,
        embedding_func,
        meta_fields=None,
        cosine_better_than_threshold=None,
        **kwargs,
    ):
        workspace = resolve_workspace()
        validate_identifier(namespace, "namespace")
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        if hasattr(self, "_validate_embedding_func"):
            self._validate_embedding_func()

        # Extract cosine_better_than_threshold from global_config
        # (same pattern as all other LightRAG vector backends)
        vdb_kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.cosine_better_than_threshold = vdb_kwargs.get(
            "cosine_better_than_threshold", 0.2
        )

    def _label(self) -> str:
        return f"Vec_{self.workspace}_{self.namespace}"

    def _index_name(self) -> str:
        return f"vec_{self.workspace}_{self.namespace}"

    async def initialize(self):
        label = self._label()
        index_name = self._index_name()
        dim = self.embedding_func.embedding_dim

        _, database = await _pool.get_driver()
        logger.info(
            "[MemgraphVec:%s] Initializing VECTOR storage on Memgraph "
            "(db=%s, label=%s, index=%s, dim=%d, metric=cosine)",
            self.workspace,
            database,
            label,
            index_name,
            dim,
        )

        async with _pool.get_session() as session:
            # Label index on id
            try:
                await session.run(f"CREATE INDEX ON :`{label}`(id)")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug(
                        "[MemgraphVec:%s] Label index already exists", self.workspace
                    )
                else:
                    logger.warning(
                        "[MemgraphVec:%s] Label index creation failed: %s",
                        self.workspace,
                        e,
                    )

            # Vector index
            try:
                result = await session.run(
                    f"CREATE VECTOR INDEX `{index_name}` "
                    f"ON :`{label}`(embedding) "
                    f'WITH CONFIG {{"dimension": {dim}, '
                    f'"capacity": {VECTOR_INDEX_CAPACITY}, "metric": "cos"}}'
                )
                await result.consume()
                logger.info(
                    f"[MemgraphVec:{self.workspace}] "
                    f"Vector index '{index_name}' created (dim={dim})"
                )
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.debug(
                        f"[MemgraphVec:{self.workspace}] "
                        f"Vector index already exists"
                    )
                else:
                    logger.warning(
                        f"[MemgraphVec:{self.workspace}] Vector index creation: {e}"
                    )

    async def finalize(self):
        pass  # Shared driver; closed globally via _pool.close_driver()

    async def index_done_callback(self):
        pass  # Memgraph persists automatically, no flush needed

    def _parse_meta_field(self, val: Any) -> Any:
        """Deserialize a meta field value, attempting JSON parse for dicts."""
        if isinstance(val, str) and val.startswith("{"):
            try:
                return json.loads(val)
            except json.JSONDecodeError:
                pass
        return val

    def _record_to_entry(self, record) -> dict[str, Any]:
        """Convert a vector search result record to a result entry.

        All declared meta_fields are always present in the returned dict
        (set to None when absent from the node) so that callers such as
        LightRAG's operate._find_most_related_edges_from_entities can
        access result["src_id"] without KeyError.
        """
        entry = {
            "id": record["id"],
            "distance": 1.0 - record["similarity"],
            "similarity": record["similarity"],
        }
        props = record["props"]
        for field_name in self.meta_fields:
            val = props.get(field_name)
            entry[field_name] = self._parse_meta_field(val) if val is not None else None
        return entry

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] = None,
    ) -> list[dict[str, Any]]:
        if query_embedding is None:
            embedding_result = await self.embedding_func.func([query])
            query_embedding = (
                embedding_result[0].tolist()
                if hasattr(embedding_result[0], "tolist")
                else list(embedding_result[0])
            )

        index_name = self._index_name()
        async with _pool.get_read_session() as session:
            try:
                result = await session.run(
                    f"""
                    CALL vector_search.search("{index_name}", $top_k, $embedding)
                    YIELD node, similarity
                    WITH node, similarity
                    WHERE similarity >= $threshold
                    RETURN node.id AS id, similarity, properties(node) AS props
                    """,
                    embedding=query_embedding,
                    top_k=top_k,
                    threshold=self.cosine_better_than_threshold,
                )
                results = [self._record_to_entry(record) async for record in result]
                await result.consume()
            except Exception as e:
                if "does not exist" in str(e).lower():
                    logger.warning(
                        "[MemgraphVec:%s/%s] Vector index '%s' does not exist "
                        "— returning empty results. Run initialize() or index "
                        "documents to create it.",
                        self.workspace,
                        self.namespace,
                        index_name,
                    )
                    return []
                raise
            logger.debug(
                "[MemgraphVec:%s/%s] query(%r) → %d results (index=%s, "
                "threshold=%.2f, top_k=%d)",
                self.workspace,
                self.namespace,
                query[:50],
                len(results),
                index_name,
                self.cosine_better_than_threshold,
                top_k,
            )
            return results

    async def _compute_missing_embeddings(
        self, data: dict[str, dict[str, Any]]
    ) -> dict[str, list[float]]:
        """Batch-compute embeddings for items without a pre-computed one."""
        needs_embed = [
            eid
            for eid, item in data.items()
            if item.get("embedding") is None and "content" in item
        ]
        if not needs_embed:
            return {}
        contents = [data[eid]["content"] for eid in needs_embed]
        emb_results = await self.embedding_func.func(contents)
        return {
            eid: (
                emb_results[i].tolist()
                if hasattr(emb_results[i], "tolist")
                else list(emb_results[i])
            )
            for i, eid in enumerate(needs_embed)
        }

    @staticmethod
    def _build_entry(eid: str, item: dict, embedding: list[float] | None) -> dict:
        """Build a flat Cypher-compatible entry for UNWIND upsert."""
        props = {}
        for key, val in item.items():
            if key == "embedding":
                continue
            if isinstance(val, (dict, list)):
                props[key] = json.dumps(val, ensure_ascii=False, default=str)
            else:
                props[key] = val
        return {"id": eid, "props": props, "embedding": embedding}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        label = self._label()
        async with _pool.acquire_write_slot():
            computed = await self._compute_missing_embeddings(data)
            entries = [
                self._build_entry(eid, item, item.get("embedding") or computed.get(eid))
                for eid, item in data.items()
            ]
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{label}` {{id: e.id}})
                    SET n += e.props, n.embedding = e.embedding
                    """,
                    entries=entries,
                )
                await result.consume()

    async def delete_entity(self, entity_name: str) -> None:
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"MATCH (n:`{label}`) WHERE n.entity_name = $name DETACH DELETE n",
                    name=entity_name,
                )
                await result.consume()

    async def delete_entity_relation(self, entity_name: str) -> None:
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    MATCH (n:`{label}`)
                    WHERE n.src_id = $name OR n.tgt_id = $name
                    DETACH DELETE n
                    """,
                    name=entity_name,
                )
                await result.consume()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}` {{id: $id}}) RETURN properties(n) AS props",
                id=id,
            )
            record = await result.single()
            await result.consume()
            if record:
                return dict(record["props"])
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $ids AS target_id
                MATCH (n:`{label}` {{id: target_id}})
                RETURN properties(n) AS props
                """,
                ids=ids,
            )
            out = []
            async for record in result:
                out.append(dict(record["props"]))
            await result.consume()
            return out

    async def delete(self, ids: list[str]) -> None:
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $ids AS target_id
                    MATCH (n:`{label}` {{id: target_id}})
                    DETACH DELETE n
                    """,
                    ids=list(ids),
                )
                await result.consume()

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $ids AS target_id
                MATCH (n:`{label}` {{id: target_id}})
                RETURN n.id AS id, n.embedding AS embedding
                """,
                ids=ids,
            )
            out = {}
            async for record in result:
                if record["embedding"]:
                    out[record["id"]] = list(record["embedding"])
            await result.consume()
            return out

    async def drop(self) -> dict[str, str]:
        label = self._label()
        try:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(f"MATCH (n:`{label}`) DETACH DELETE n")
                    await result.consume()
                    try:
                        result = await session.run(
                            f"DROP VECTOR INDEX `{self._index_name()}`"
                        )
                        await result.consume()
                    except Exception:
                        pass
            return {"status": "success", "message": f"Vector namespace {label} dropped"}
        except Exception as e:
            logger.error("Vector drop failed for %s: %s", label, e)
            return {"status": "error", "message": "Drop operation failed"}
