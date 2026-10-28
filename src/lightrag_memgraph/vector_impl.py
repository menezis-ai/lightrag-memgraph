"""
Vector Storage backend using Memgraph native vector search.

Requires: Memgraph >= 3.2 with the vector_search module.

Each vector entry is a Cypher node:
  Label: :Vec_{workspace}_{namespace}
  Properties: id, embedding (list<float>), content, + meta_fields

Vector index:
  CREATE VECTOR INDEX vec_{workspace}_{namespace}
  ON :Vec_{workspace}_{namespace}(embedding)
  WITH CONFIG {"dimension": N, "capacity": VECTOR_INDEX_CAPACITY, "metric": "cos"}

Query:
  CALL vector_search.search("vec_...", $top_k, $embedding)
  YIELD node, similarity
"""

import asyncio
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from lightrag.base import BaseVectorStorage
from lightrag.utils import logger
from neo4j.exceptions import ClientError

from . import _pool
from ._constants import VECTOR_INDEX_CAPACITY, resolve_workspace, validate_identifier
from .capabilities import inspect_memgraph


@dataclass
class MemgraphVectorDBStorage(BaseVectorStorage):
    def __init__(
        self,
        namespace,
        global_config,
        embedding_func,
        meta_fields=None,
        cosine_better_than_threshold=None,
        workspace=None,
        **kwargs,
    ):
        workspace = resolve_workspace(workspace, global_config)
        validate_identifier(namespace, "namespace")
        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
            meta_fields=meta_fields or set(),
        )
        # This helper was absent from early LightRAG 1.4 releases.
        if self.embedding_func is None:
            raise ValueError("embedding_func is required for vector storage")

        # Extract cosine_better_than_threshold from global_config
        # (same pattern as all other LightRAG vector backends)
        vdb_kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.cosine_better_than_threshold = (
            vdb_kwargs.get("cosine_better_than_threshold", 0.2)
            if cosine_better_than_threshold is None
            else cosine_better_than_threshold
        )
        threshold = self.cosine_better_than_threshold
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(threshold)
            or not -1 <= threshold <= 1
        ):
            raise ValueError(
                "cosine_better_than_threshold must be finite and between -1 and 1"
            )
        self.cosine_better_than_threshold = float(threshold)
        self._query_mode = vdb_kwargs.get("memgraph_query_mode", "vector")
        if self._query_mode not in {"vector", "hybrid"}:
            raise ValueError("memgraph_query_mode must be 'vector' or 'hybrid'")
        self._feature_policy = vdb_kwargs.get("memgraph_feature_policy", "fallback")
        if self._feature_policy not in {"fallback", "strict"}:
            raise ValueError("memgraph_feature_policy must be 'fallback' or 'strict'")
        self._text_enabled = vdb_kwargs.get("memgraph_text_search", False)
        if not isinstance(self._text_enabled, bool):
            raise ValueError("memgraph_text_search must be a boolean")
        properties = vdb_kwargs.get("memgraph_text_properties", ["content"])
        if not isinstance(properties, (list, tuple)) or not properties:
            raise ValueError("memgraph_text_properties must be a non-empty list")
        self._text_properties = tuple(
            validate_identifier(prop, "text property") for prop in properties
        )
        self._vector_config = self._validate_index_config(
            vdb_kwargs.get("memgraph_vector_index_config", {})
        )
        self.capabilities = None
        self._text_index_ready = False

    @staticmethod
    def _validate_index_config(config: dict) -> dict:
        if not isinstance(config, dict):
            raise ValueError("memgraph_vector_index_config must be a dictionary")
        allowed = {"capacity", "resize_coefficient", "scalar_kind"}
        if set(config) - allowed:
            raise ValueError(
                f"Unsupported vector index options: {set(config) - allowed}"
            )
        config = dict(config)
        capacity = config.get("capacity", VECTOR_INDEX_CAPACITY)
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Vector capacity must be a positive integer")
        config["capacity"] = capacity
        if "resize_coefficient" in config:
            resize = config["resize_coefficient"]
            if (
                isinstance(resize, bool)
                or not isinstance(resize, (int, float))
                or not math.isfinite(resize)
                or resize <= 1
            ):
                raise ValueError("Vector resize_coefficient must be finite and > 1")
        if "scalar_kind" in config and (
            not isinstance(config["scalar_kind"], str)
            or config["scalar_kind"]
            not in {
                "f64",
                "f32",
                "f16",
                "bf16",
                "f8",
                "i8",
                "i16",
                "i32",
                "i64",
                "u8",
                "u16",
                "u32",
                "u40",
                "u64",
                "uuid",
                "b1x8",
            }
        ):
            raise ValueError("Unsupported vector scalar_kind")
        return config

    def _unsupported(self, message: str) -> None:
        if self._feature_policy == "strict":
            raise RuntimeError(message)
        logger.warning(
            "[MemgraphVec:%s] %s; using compatible defaults", self.workspace, message
        )

    def _label(self) -> str:
        return f"Vec_{self.workspace}_{self.namespace}"

    def _index_name(self) -> str:
        return f"vec_{self.workspace}_{self.namespace}"

    def _text_index_name(self) -> str:
        return f"text_{self.workspace}_{self.namespace}"

    async def _create_vector_index(self, session=None) -> None:
        """Recreate a missing index using this namespace's validated options."""
        config = dict(
            self._vector_config,
            dimension=self.embedding_func.embedding_dim,
            metric="cos",
        )
        if (
            "scalar_kind" in config
            and self.capabilities is not None
            and not self.capabilities.vector_quantization
        ):
            self._unsupported("Vector quantization requires Memgraph >= 3.4")
            config.pop("scalar_kind")
        query = (
            f"CREATE VECTOR INDEX `{self._index_name()}` "
            f"ON :`{self._label()}`(embedding) WITH CONFIG {json.dumps(config)}"
        )
        if session is not None:
            await self._create_index(session, query)
        else:
            async with _pool.get_session() as write_session:
                await self._create_index(write_session, query)

    async def initialize(self):
        label = self._label()
        index_name = self._index_name()
        dim = self.embedding_func.embedding_dim

        self._text_index_ready = False
        async with _pool.get_session() as session:
            self.capabilities = await inspect_memgraph(session)
            if not self.capabilities.vector_search:
                raise RuntimeError("Memgraph requires the vector_search.search module")
            config = dict(self._vector_config, dimension=dim, metric="cos")
            if "scalar_kind" in config and not self.capabilities.vector_quantization:
                self._unsupported("Vector quantization requires Memgraph >= 3.4")
                config.pop("scalar_kind")

            text_enabled = self._text_enabled or self._query_mode == "hybrid"
            if text_enabled and not self.capabilities.text_search:
                self._unsupported(
                    "Stable text search requires Memgraph >= 3.6 and text_search.search_all"
                )
                text_enabled = False

            result = await session.run("SHOW VECTOR INDEX INFO")
            indexes = [dict(record) async for record in result]
            await result.consume()
            existing = next(
                (row for row in indexes if row["index_name"] == index_name), None
            )
            if existing:
                self._validate_existing_index(existing, config)

            await self._create_index(session, f"CREATE INDEX ON :`{label}`(id)")
            if not existing:
                await self._create_index(
                    session,
                    f"CREATE VECTOR INDEX `{index_name}` "
                    f"ON :`{label}`(embedding) "
                    f"WITH CONFIG {json.dumps(config)}",
                )
                # Also validate after a concurrent initializer created the index.
                result = await session.run("SHOW VECTOR INDEX INFO")
                indexes = [dict(record) async for record in result]
                await result.consume()
                created = next(
                    row for row in indexes if row["index_name"] == index_name
                )
                self._validate_existing_index(created, config)

            if text_enabled:
                properties = ", ".join(f"`{p}`" for p in self._text_properties)
                await self._create_index(
                    session,
                    f"CREATE TEXT INDEX `{self._text_index_name()}` "
                    f"ON :`{label}`({properties})",
                )
                existing_text = await self._get_text_index(session)
                if (
                    existing_text is None
                    or existing_text.get("label") not in {label, f":{label}"}
                    or set(existing_text.get("property") or [])
                    != set(self._text_properties)
                ):
                    raise ValueError(
                        f"Text index {self._text_index_name()} has incompatible label or properties. "
                        "Use a new namespace or rebuild the text index explicitly."
                    )
                self._text_index_ready = True
        logger.info(
            "[MemgraphVec:%s] Memgraph %s, index=%s, dim=%d, text=%s",
            self.workspace,
            self.capabilities.version,
            index_name,
            dim,
            self._text_index_ready,
        )

    async def _get_text_index(self, session) -> dict | None:
        """Read persistent metadata, including indexes created by prior instances."""
        result = await session.run("SHOW INDEX INFO")
        expected_name = f"(name: {self._text_index_name()})"
        matches = [
            dict(record)
            async for record in result
            if expected_name in record.get("index type", "")
        ]
        await result.consume()
        return matches[0] if matches else None

    @staticmethod
    async def _create_index(session, query: str) -> None:
        try:
            result = await session.run(query)
            await result.consume()
        except ClientError as exc:
            if "already exists" not in str(exc).lower():
                raise

    def _validate_existing_index(self, existing: dict, config: dict) -> None:
        expected = {
            "dimension": config["dimension"],
            "metric": "cos",
            "property": "embedding",
        }
        # Older releases omit the colon; newer releases include it.
        label = existing.get("label")
        if label not in {self._label(), f":{self._label()}"}:
            raise ValueError(
                f"Vector index {self._index_name()} belongs to a different label: {label!r}"
            )
        if "scalar_kind" in config:
            expected["scalar_kind"] = config["scalar_kind"]
        for key, value in expected.items():
            if existing.get(key) != value:
                raise ValueError(
                    f"Vector index {self._index_name()} has {key}={existing.get(key)!r}, "
                    f"expected {value!r}. Use a new namespace or rebuild the index explicitly."
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
        """Convert a vector search result record to a result entry."""
        entry = {
            "id": record["id"],
            "distance": 1.0 - record["similarity"],
            "similarity": record["similarity"],
        }
        props = record["props"]
        for field_name in self.meta_fields:
            if field_name not in entry:
                entry[field_name] = self._parse_meta_field(props.get(field_name))
        return entry

    async def _embed_texts(
        self, texts: list[str], *, context: str
    ) -> list[list[float]]:
        # Early 1.4 wrappers lack the asymmetric-context contract. Call every
        # supported wrapper itself so dimension/token injection stays intact.
        kwargs = (
            {"context": context}
            if getattr(self.embedding_func, "supports_asymmetric", False)
            else {}
        )
        result = await self.embedding_func(texts, **kwargs)
        rows = result.tolist() if hasattr(result, "tolist") else result
        try:
            if len(rows) != len(texts):
                raise ValueError("Embedding provider returned the wrong vector count")
            vectors = [list(row) for row in rows]
        except TypeError as exc:
            raise ValueError(
                "Embedding provider must return a matrix of vectors"
            ) from exc
        dimension = self.embedding_func.embedding_dim
        for vector in vectors:
            if len(vector) != dimension:
                raise ValueError(
                    "Embedding provider returned the wrong vector dimension"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                for value in vector
            ):
                raise ValueError("Embedding coordinates must be finite numbers")
        return [[float(value) for value in vector] for vector in vectors]

    async def _embed_query(self, query: str) -> list[float]:
        return (await self._embed_texts([query], context="query"))[0]

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if getattr(self, "_query_mode", "vector") == "hybrid":
            return await self.query_hybrid(query, top_k, query_embedding)
        return await self._query_vector(query, top_k, query_embedding)

    async def _query_vector(
        self, query: str, top_k: int, query_embedding=None
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if query_embedding is None:
            query_embedding = await self._embed_query(query)
        try:
            return await self._query_vector_once(query, top_k, query_embedding)
        except Exception as exc:
            message = str(exc).lower()
            if "vector index" not in message or "does not exist" not in message:
                raise
            logger.warning(
                "Vector index %s missing; recreating and retrying", self._index_name()
            )
            try:
                await self._create_vector_index()
            except Exception:
                logger.exception(
                    "Vector index %s recreation failed", self._index_name()
                )
                raise
            return await self._query_vector_once(query, top_k, query_embedding)

    async def _query_vector_once(
        self, query: str, top_k: int, query_embedding=None
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        if query_embedding is None:
            query_embedding = await self._embed_query(query)

        index_name = self._index_name()
        async with _pool.get_read_session() as session:
            capabilities = getattr(self, "capabilities", None)
            if capabilities and capabilities.version_tuple < (3, 4, 0):
                # Early native vector modules can crash on an empty index.
                result = await session.run(
                    f"MATCH (n:`{self._label()}`) WHERE n.embedding IS NOT NULL "
                    "RETURN n.id LIMIT 1"
                )
                present = await result.single()
                await result.consume()
                if present is None:
                    return []
            # Native indexes can retain deleted hits until garbage collection.
            # Rebind IDs before property access, then expand only when stale
            # hits above the threshold would otherwise hide live neighbours.
            search_query = f"""
                CALL vector_search.search($index_name, $top_k, $embedding)
                YIELD node, similarity
                WITH id(node) AS match_id, similarity
                OPTIONAL MATCH (candidate:`{self._label()}`)
                WHERE id(candidate) = match_id
                RETURN candidate.id AS id, similarity, properties(candidate) AS props
            """
            if capabilities and capabilities.version_tuple < (3, 8, 0):
                # Keep the historical projection on pre-single-store servers.
                search_query = f"""
                    CALL vector_search.search("{index_name}", $top_k, $embedding)
                    YIELD node, similarity
                    WITH node, similarity WHERE similarity >= $threshold
                    RETURN node.id AS id, similarity, properties(node) AS props
                """
            limit = top_k
            while True:
                result = await session.run(
                    search_query,
                    index_name=index_name,
                    embedding=query_embedding,
                    top_k=limit,
                    threshold=self.cosine_better_than_threshold,
                )
                results = []
                count = 0
                stale = False
                lowest_similarity = float("inf")
                async for record in result:
                    count += 1
                    similarity = record["similarity"]
                    lowest_similarity = min(lowest_similarity, similarity)
                    if similarity < self.cosine_better_than_threshold:
                        continue
                    if record["id"] is None:
                        stale = True
                    else:
                        results.append(self._record_to_entry(record))
                await result.consume()
                if (
                    len(results) >= top_k
                    or not stale
                    or count < limit
                    or lowest_similarity < self.cosine_better_than_threshold
                ):
                    break
                limit *= 2
            results.sort(key=lambda hit: (-hit["similarity"], hit["id"]))
            results = results[:top_k]
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

    async def query_text(
        self,
        query: str,
        top_k: int,
        *,
        mode: str = "search_all",
        config: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search the namespace's text index; scores are lexical relevance.

        ``search`` and ``fuzzy_phrase_search`` accept property-qualified Tantivy
        expressions (``data.content:...``). ``search_all`` accepts plain terms.
        Older servers without score/config outputs use a constant score and
        a Cypher LIMIT. Requested fuzzy semantics are never silently discarded.
        """
        if top_k <= 0:
            return []
        if mode not in {"search_all", "search", "regex_search", "fuzzy_phrase_search"}:
            raise ValueError("Unsupported text search mode")
        if not self._text_index_ready:
            raise RuntimeError(
                "Text search is not initialized; enable memgraph_text_search"
            )
        procedure = f"text_search.{mode}"
        if not self.capabilities.has_procedure(procedure):
            raise RuntimeError(
                f"{procedure} is unavailable on Memgraph {self.capabilities.version}"
            )
        config = dict(config or {})
        if set(config) - {"fuzzy_distance", "fuzzy_prefix", "fuzzy_transpositions"}:
            raise ValueError(
                "Unsupported text search configuration (use top_k for the limit)"
            )
        if mode == "regex_search" and config:
            raise ValueError("Regex search does not support fuzzy configuration")
        fuzzy_distance = config.get("fuzzy_distance", 0)
        if type(fuzzy_distance) is not int or fuzzy_distance not in {0, 1, 2}:
            raise ValueError("fuzzy_distance must be 0, 1 or 2")
        for key in ("fuzzy_prefix", "fuzzy_transpositions"):
            if key in config and not isinstance(config[key], bool):
                raise ValueError(f"{key} must be a boolean")
        accepts_config = self.capabilities.accepts_config(procedure)
        if config and not accepts_config:
            raise RuntimeError(
                f"{procedure} does not support search configuration on this server"
            )
        params = {
            "index_name": self._text_index_name(),
            "search_query": query,
            "top_k": top_k,
        }
        arguments = "$index_name, $search_query"
        if accepts_config:
            arguments += ", $config"
            params["config"] = dict(config, limit=top_k)
        elif self.capabilities.accepts_limit(procedure):
            arguments += ", $top_k"
        has_score = self.capabilities.returns_score(procedure)
        output = "node, score" if has_score else "node"
        score = "score" if has_score else "1.0 AS score"
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"CALL {procedure}({arguments}) YIELD {output} "
                f"WITH id(node) AS match_id, {score} "
                f"MATCH (candidate:`{self._label()}`) WHERE id(candidate) = match_id "
                "RETURN candidate.id AS id, score, properties(candidate) AS props, "
                "candidate.embedding AS embedding ORDER BY score DESC, id LIMIT $top_k",
                **params,
            )
            entries = []
            async for record in result:
                entry = {
                    field: self._parse_meta_field(record["props"][field])
                    for field in self.meta_fields
                    if field in record["props"]
                }
                entry.update(
                    id=record["id"],
                    score=record["score"],
                    embedding=record["embedding"],
                )
                entries.append(entry)
            await result.consume()
            return entries

    async def query_hybrid(
        self, query: str, top_k: int, query_embedding=None
    ) -> list[dict[str, Any]]:
        """Fuse vector and text candidates using reciprocal rank fusion (k=60).

        ``score`` is the fused ranking score. ``similarity`` and ``distance``
        retain cosine semantics, including for candidates found only by text.
        An older server falls back to vector retrieval during initialize().
        """
        if top_k <= 0:
            return []
        if not self._text_index_ready:
            return await self._query_vector(query, top_k, query_embedding)
        if query_embedding is None:
            query_embedding = await self._embed_query(query)
        vector_hits, text_hits = await asyncio.gather(
            self._query_vector(query, top_k * 4, query_embedding),
            self.query_text(query, top_k * 4),
        )
        entries = {hit["id"]: dict(hit) for hit in vector_hits}
        scores: dict[str, float] = {}
        for hits in (vector_hits, text_hits):
            for rank, hit in enumerate(hits, 1):
                eid = hit["id"]
                scores[eid] = scores.get(eid, 0.0) + 1.0 / (60 + rank)
                if eid not in entries:
                    embedding = hit.get("embedding")
                    if embedding is None or len(embedding) != len(query_embedding):
                        raise ValueError(
                            f"Missing or incompatible embedding for text hit {eid!r}"
                        )
                    norm = math.hypot(*embedding) * math.hypot(*query_embedding)
                    similarity = (
                        math.fsum(a * b for a, b in zip(embedding, query_embedding))
                        / norm
                        if norm
                        else 0.0
                    )
                    entries[eid] = self._record_to_entry(
                        {"id": eid, "similarity": similarity, "props": hit}
                    )
                if hits is text_hits:
                    entries[eid]["text_score"] = hit["score"]
        for eid, entry in entries.items():
            entry["score"] = scores[eid]
        return sorted(entries.values(), key=lambda hit: (-hit["score"], hit["id"]))[
            :top_k
        ]

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
        emb_results = await self._embed_texts(contents, context="document")
        return dict(zip(needs_embed, emb_results))

    @staticmethod
    def _build_entry(eid: str, item: dict, embedding: list[float] | None) -> dict:
        """Build a flat Cypher-compatible entry for UNWIND upsert."""
        props = {}
        for key, val in item.items():
            if key in {"embedding", "id"}:
                continue
            if isinstance(val, (dict, list)):
                props[key] = json.dumps(val, ensure_ascii=False, default=str)
            else:
                props[key] = val
        return {"id": eid, "props": props, "embedding": embedding}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        if not data:
            return
        label = self._label()
        async with _pool.acquire_write_slot():
            computed = await self._compute_missing_embeddings(data)
            entries = [
                self._build_entry(
                    eid,
                    item,
                    (
                        item["embedding"]
                        if item.get("embedding") is not None
                        else computed.get(eid)
                    ),
                )
                for eid, item in data.items()
            ]
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $entries AS e
                    MERGE (n:`{label}` {{id: e.id}})
                    SET n += e.props
                    FOREACH (ignored IN CASE WHEN e.embedding IS NULL THEN [] ELSE [1] END |
                        SET n.embedding = e.embedding)
                    """,
                    entries=entries,
                )
                await result.consume()

    async def delete_entity(self, entity_name: str) -> None:
        # REMOVE label before DETACH DELETE: Memgraph 3.10+ keeps vector
        # index entries for deleted vertices around (vector_index_memory_tracked
        # is not freed on vertex delete), then strict-errors on subsequent
        # property access from vector_search results — see release notes
        # "accessing properties of a deleted node/relationship now raises an
        # error instead". Removing the indexed label first prunes the vector
        # index entry cleanly; the DETACH DELETE then removes the orphan.
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    MATCH (n:`{label}`) WHERE n.entity_name = $name
                    REMOVE n:`{label}`
                    WITH n
                    DETACH DELETE n
                    """,
                    name=entity_name,
                )
                await result.consume()

    async def delete_entity_relation(self, entity_name: str) -> None:
        # Same Memgraph 3.10 vector-index hygiene as delete_entity above.
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    MATCH (n:`{label}`)
                    WHERE n.src_id = $name OR n.tgt_id = $name
                    REMOVE n:`{label}`
                    WITH n
                    DETACH DELETE n
                    """,
                    name=entity_name,
                )
                await result.consume()

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"MATCH (n:`{label}` {{id: $id}}) RETURN properties(n) AS props, n.embedding AS embedding",
                id=id,
            )
            record = await result.single()
            await result.consume()
            if record:
                props = dict(record["props"])
                if record["embedding"] is not None:
                    props["embedding"] = record["embedding"]
                return props
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        label = self._label()
        async with _pool.get_read_session() as session:
            result = await session.run(
                f"""
                UNWIND $ids AS target_id
                MATCH (n:`{label}` {{id: target_id}})
                RETURN properties(n) AS props, n.embedding AS embedding
                """,
                ids=ids,
            )
            out = []
            async for record in result:
                props = dict(record["props"])
                if record["embedding"] is not None:
                    props["embedding"] = record["embedding"]
                out.append(props)
            await result.consume()
            return out

    async def delete(self, ids: list[str]) -> None:
        # Same Memgraph 3.10 vector-index hygiene as delete_entity above.
        label = self._label()
        async with _pool.acquire_write_slot():
            async with _pool.get_session() as session:
                result = await session.run(
                    f"""
                    UNWIND $ids AS target_id
                    MATCH (n:`{label}` {{id: target_id}})
                    REMOVE n:`{label}`
                    WITH n
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
                    text_index = await self._get_text_index(session)
                    result = await session.run("SHOW VECTOR INDEX INFO")
                    vector_indexes = [dict(record) async for record in result]
                    await result.consume()
                    vector_index = next(
                        (
                            row
                            for row in vector_indexes
                            if row["index_name"] == self._index_name()
                        ),
                        None,
                    )
                    for index in (text_index, vector_index):
                        if index is not None and index.get("label") not in {
                            label,
                            f":{label}",
                        }:
                            raise ValueError(
                                "Refusing to drop an index belonging to a different namespace"
                            )
                    result = await session.run(
                        f"MATCH (n:`{label}`) REMOVE n:`{label}` WITH n DETACH DELETE n"
                    )
                    await result.consume()
                    for index_type, name, existing in (
                        ("TEXT", self._text_index_name(), text_index),
                        ("VECTOR", self._index_name(), vector_index),
                    ):
                        if existing is None:
                            continue
                        try:
                            result = await session.run(
                                f"DROP {index_type} INDEX `{name}`"
                            )
                            await result.consume()
                        except ClientError as exc:
                            if not any(
                                text in str(exc).lower()
                                for text in (
                                    "doesn't exist",
                                    "does not exist",
                                    "not found",
                                )
                            ):
                                raise
                    self._text_index_ready = False
            return {"status": "success", "message": f"Vector namespace {label} dropped"}
        except Exception as e:
            logger.error("Vector drop failed for %s: %s", label, e)
            raise
