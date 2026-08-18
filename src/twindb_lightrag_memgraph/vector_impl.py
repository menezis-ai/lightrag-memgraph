"""
Vector Storage backend using Memgraph native vector search.

Requires: Memgraph >= 3.0 with native vector search. Vector search
(CREATE VECTOR INDEX / vector_search.search) is a CORE Memgraph feature
(stable since 3.0.0, Jan 2025) — MAGE is NOT required. The plain
``memgraph/memgraph`` image is sufficient; ``memgraph/memgraph-mage`` also
works (it is a superset that additionally bundles the graph-algorithm modules
this package does not use).

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

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from lightrag.base import BaseVectorStorage
from lightrag.utils import logger

from . import _pool
from ._constants import (
    VECTOR_INDEX_CAPACITY,
    RetrievalFilters,
    get_active_chunk_retrieval_scores,
    get_active_retrieval_filters,
    get_active_storage_folder,
    resolve_workspace,
    validate_identifier,
)
from ._retry import with_conflict_retry
from ._prompt_security import neutralize_chunk_payloads

# LightRAG's chunks vector namespace (``lightrag.namespace.NameSpace``) —
# literal on purpose, same rationale as kv_impl's _KV_TEXT_CHUNKS_NAMESPACE.
_VEC_CHUNKS_NAMESPACE = "chunks"

# Chunk-id separator LightRAG joins ``source_id`` with on entity/relation vdb
# records. Imported with a fallback so the storage layer never hard-fails on a
# LightRAG build whose constants module moved the symbol (version-skew guard,
# per feedback_lightrag_version_skew).
try:  # pragma: no cover - exercised implicitly by the import
    from lightrag.constants import GRAPH_FIELD_SEP
except Exception:  # pragma: no cover - defensive
    GRAPH_FIELD_SEP = "<SEP>"

# Cypher boolean conjunction used to glue WHERE predicates.
_CYPHER_AND = " AND "

# (workspace, namespace, folder) triples whose fail-closed scope refusal has
# already been logged at WARNING — see _log_scoped_refusal_once.
_SCOPED_REFUSAL_LOGGED: set[tuple[str, str, str]] = set()

# Folder-scoped retrieval over-fetches before the membership inner-join so that
# dropping non-member candidates still leaves ~top_k members. A folder that is a
# very thin slice of a large corpus can still under-return — a documented
# limitation of the cheap single-pass approach (no escalation loop).
#
# That cheap path is NOT used when doc/tag filters are active. A tag-filtered
# query must mean "search inside the tagged corpus", not "search globally then
# discard the first N untagged hits". For those filtered paths we do an exact
# cosine scan over the pre-filtered candidate set. It is more expensive, but it
# gives the product contract operators expect. The global cosine floor is also
# bypassed there unless the caller explicitly sends ``min_score``.
_FOLDER_SCOPE_OVERFETCH_ENV = "TWIN_QUERY_FOLDER_SCOPE_OVERFETCH"
_FOLDER_SCOPE_OVERFETCH_DEFAULT = 4
_FOLDER_SCOPE_OVERFETCH_CAP = 500
_DEFAULT_EMBEDDING_BATCH_NUM = 32


def _capture_chunk_retrieval_scores(
    namespace: str,
    results: list[dict[str, Any]],
) -> None:
    """Record measured chunk similarities for the active grounding request.

    LightRAG strips vector metrics while assembling its final
    ``aquery_llm.data.chunks`` envelope. Capturing them here preserves the
    score from the retrieval that actually grounded the answer, without
    issuing a second vector query. If the same chunk is observed more than
    once in one grounding call, retain its strongest measured similarity.
    """
    scores = get_active_chunk_retrieval_scores()
    if namespace != "chunks" or scores is None:
        return

    for result in results:
        chunk_id = result.get("id")
        similarity = result.get("similarity")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        if not isinstance(similarity, (int, float)) or isinstance(similarity, bool):
            continue
        numeric_similarity = float(similarity)
        if not math.isfinite(numeric_similarity):
            continue
        previous = scores.get(chunk_id)
        if previous is None or numeric_similarity > previous:
            scores[chunk_id] = numeric_similarity


# Ingestion-side embedding retry. A cold embedding endpoint (model still
# loading / first request of the day) can blow past LightRAG's 60s worker
# timeout on the FIRST call and fail the whole document, even though the
# endpoint is warm and fast a second later. We wrap each storage-side embedding
# call in a shorter per-attempt timeout and retry, so a cold start recovers
# automatically instead of failing prematurely. On the happy path (endpoint
# responds immediately) attempt 1 returns and behaviour is identical to before.
# Scoped to ingestion (_compute_missing_embeddings) — the query path fails fast
# by design. Tunable via env; retries re-raise the last error on exhaustion so
# LightRAG still marks the doc FAILED (graceful degradation, no silent drop).
_EMBEDDING_TIMEOUT_ENV = "TWIN_EMBEDDING_TIMEOUT"
_DEFAULT_EMBEDDING_TIMEOUT = 30.0  # seconds per attempt
_EMBEDDING_ATTEMPTS_ENV = "TWIN_EMBEDDING_ATTEMPTS"
_DEFAULT_EMBEDDING_ATTEMPTS = 3
_EMBEDDING_RETRY_BACKOFF = 1.0  # seconds between attempts (give a cold model time)


def _folder_scope_overfetch(top_k: int) -> int:
    """top_k multiplied by the configured over-fetch factor, capped."""
    try:
        factor = int(
            os.environ.get(
                _FOLDER_SCOPE_OVERFETCH_ENV, str(_FOLDER_SCOPE_OVERFETCH_DEFAULT)
            )
        )
    except (TypeError, ValueError):
        factor = _FOLDER_SCOPE_OVERFETCH_DEFAULT
    if factor < 1:
        factor = 1
    return min(max(top_k, top_k * factor), _FOLDER_SCOPE_OVERFETCH_CAP)


# ── Retrieval-filter Cypher fragments ──────────────────────────────────────
#
# These turn an active ``RetrievalFilters`` into Cypher WHERE conditions, so an
# excluded chunk/entity is dropped *inside* the vector search and never reaches
# the prompt (real grounding, not a post-hoc Sources-panel trim). Each helper
# appends only the params it actually uses and returns a list of condition
# strings — an empty filter contributes nothing, preserving strict compat.
# Tag ids are compared lower-cased (the route normalises ``$tag_*`` and the
# Cypher lower-cases the stored ids via ``toLower``).


def _doc_conditions_single(
    filters: RetrievalFilters, params: dict[str, Any], doc_expr: str
) -> list[str]:
    """Doc-filter conditions for a record with a single doc (``doc_expr``).

    ``doc_all`` is strict: ``all(x = doc)`` is true only when every requested
    doc equals the record's single doc, so ``doc_all`` with ≥2 docs excludes
    every chunk (pinned in tests) — NOT the union-as-``any`` the legacy
    post-filter conflated.
    """
    conds: list[str] = []
    if filters.doc_all:
        params["doc_all"] = sorted(filters.doc_all)
        conds.append(f"all(__x IN $doc_all WHERE __x = {doc_expr})")
    if filters.doc_any:
        params["doc_any"] = sorted(filters.doc_any)
        conds.append(f"{doc_expr} IN $doc_any")
    return conds


def _doc_conditions_set(
    filters: RetrievalFilters, params: dict[str, Any], docids_expr: str
) -> list[str]:
    """Doc-filter conditions over a *set* of docs (``docids_expr``).

    Used for entity/relation records whose ``source_id`` spans several chunks
    (hence several docs): ``doc_all`` → requested set ⊆ source docs;
    ``doc_any`` → source docs ∩ requested ≠ ∅.
    """
    conds: list[str] = []
    if filters.doc_all:
        params["doc_all"] = sorted(filters.doc_all)
        conds.append(f"all(__x IN $doc_all WHERE __x IN {docids_expr})")
    if filters.doc_any:
        params["doc_any"] = sorted(filters.doc_any)
        conds.append(f"any(__y IN {docids_expr} WHERE __y IN $doc_any)")
    return conds


def _tag_conditions(
    filters: RetrievalFilters, params: dict[str, Any], tags_expr: str
) -> list[str]:
    """Tag-filter conditions over a list of tag ids (``tags_expr``).

    ``tag_all`` → every required tag present; ``tag_any`` → ≥1 optional tag
    present. ``tag_groups`` (OR-of-groups) is emitted as ONE parenthesised
    ``(group0 OR group1 …)`` condition string so every call site — including
    the ``any(__di IN __docinfos WHERE …)`` wrapper that re-joins the returned
    list with AND — composes without change. Tag values are always ``$``-bound
    parameters, never interpolated.
    """
    conds: list[str] = []
    if filters.tag_all:
        params["tag_all"] = sorted(filters.tag_all)
        conds.append(f"all(__rt IN $tag_all WHERE __rt IN {tags_expr})")
    if filters.tag_any:
        params["tag_any"] = sorted(filters.tag_any)
        conds.append(f"any(__ot IN $tag_any WHERE __ot IN {tags_expr})")
    if filters.tag_groups:
        group_conds: list[str] = []
        for idx, (required, optional) in enumerate(filters.tag_groups):
            parts: list[str] = []
            if required:
                params[f"tag_g{idx}_all"] = sorted(required)
                parts.append(f"all(__rt IN $tag_g{idx}_all WHERE __rt IN {tags_expr})")
            if optional:
                params[f"tag_g{idx}_any"] = sorted(optional)
                parts.append(f"any(__ot IN $tag_g{idx}_any WHERE __ot IN {tags_expr})")
            if parts:
                group_conds.append("(" + _CYPHER_AND.join(parts) + ")")
        if group_conds:
            conds.append("(" + " OR ".join(group_conds) + ")")
    return conds


def _exact_cosine_projection() -> str:
    """Cypher fragment that scores the already-filtered candidate nodes.

    ``vector_search.search`` cannot receive a pre-filter, so using it before a
    tag/doc predicate makes the filter non-exhaustive. This fragment computes
    cosine similarity in Cypher over the candidate rows that survived the
    membership/doc/tag joins.
    """
    return """
            WITH DISTINCT node
            WHERE node.embedding IS NOT NULL
              AND size(node.embedding) = size($embedding)
            WITH node,
                 reduce(__dot = 0.0, __i IN range(0, size($embedding) - 1) |
                     __dot + node.embedding[__i] * $embedding[__i]
                 ) AS __dot,
                 sqrt(reduce(__n = 0.0, __v IN node.embedding |
                     __n + __v * __v
                 )) AS __node_norm,
                 $query_norm AS __query_norm
            WITH node,
                 CASE
                   WHEN __node_norm = 0.0 OR __query_norm = 0.0 THEN 0.0
                   ELSE __dot / (__node_norm * __query_norm)
                 END AS similarity
            WHERE similarity >= $threshold
            RETURN node.id AS id, similarity, properties(node) AS props
            ORDER BY similarity DESC
            LIMIT $top_k
            """


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
        workspace = validate_identifier(
            str(global_config.get("workspace") or resolve_workspace()), "workspace"
        )
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

    async def _create_vector_index(self, session=None) -> None:
        """Create the vector index. Idempotent: swallows 'already exists'.

        Called from initialize() at startup and from query() as a fallback
        when the index is missing at query time (Memgraph restart, replica
        lag, initial creation failure).
        """
        label = self._label()
        index_name = self._index_name()
        dim = self.embedding_func.embedding_dim
        query = (
            f"CREATE VECTOR INDEX `{index_name}` "
            f"ON :`{label}`(embedding) "
            f'WITH CONFIG {{"dimension": {dim}, '
            f'"capacity": {VECTOR_INDEX_CAPACITY}, "metric": "cos"}}'
        )

        async def _run(s):
            result = await s.run(query)
            await result.consume()
            logger.info(
                "[MemgraphVec:%s] Vector index '%s' created (dim=%d)",
                self.workspace,
                index_name,
                dim,
            )

        try:
            if session is not None:
                await _run(session)
            else:
                async with _pool.get_session() as s:
                    await _run(s)
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.debug(
                    "[MemgraphVec:%s] Vector index '%s' already exists",
                    self.workspace,
                    index_name,
                )
            else:
                raise

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
                result = await session.run(f"CREATE INDEX ON :`{label}`(id)")
                await result.consume()
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

            # Vector index — idempotent, reused by query() on fallback
            try:
                await self._create_vector_index(session)
            except Exception as e:
                logger.warning(
                    "[MemgraphVec:%s] Vector index creation: %s",
                    self.workspace,
                    e,
                )

    async def finalize(self):  # NOSONAR - async contract.
        pass  # Shared driver; closed globally via _pool.close_driver()

    async def index_done_callback(self):  # NOSONAR - async contract.
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

    def _log_scoped_refusal_once(self, folder: str, message: str) -> None:
        """Log a folder-scope fail-closed refusal once per (ws, ns, folder).

        The refusal is permanent by design (blended entity/relation payloads
        cannot be membership-scoped — see ``_build_search_cypher``), so it
        fires on every scoped query. At ERROR-per-query it buried real errors
        on the OVH maquette; the design decision deserves one WARNING per
        folder, then DEBUG.
        """
        key = (self.workspace, self.namespace, folder)
        if key in _SCOPED_REFUSAL_LOGGED:
            logger.debug(message, self.workspace, self.namespace, folder)
            return
        _SCOPED_REFUSAL_LOGGED.add(key)
        logger.warning(message, self.workspace, self.namespace, folder)

    def _effective_search_threshold(
        self,
        filters: RetrievalFilters | None,
    ) -> float:
        threshold = self.cosine_better_than_threshold
        if filters is None:
            return threshold
        if filters.has_doc or filters.has_tag:
            return max(0.0, filters.min_score)
        return max(threshold, filters.min_score)

    def _legacy_search_cypher(
        self,
        index_name: str,
        top_k: int,
        threshold: float,
    ) -> tuple[str, dict[str, Any]]:
        cypher = f"""
                CALL vector_search.search("{index_name}", $top_k, $embedding)
                YIELD node, similarity
                WITH node, similarity
                WHERE similarity >= $threshold
                RETURN node.id AS id, similarity, properties(node) AS props
                """
        return cypher, {
            "embedding": None,  # filled by caller
            "top_k": top_k,
            "threshold": threshold,
        }

    def _folder_search_context(
        self,
        top_k: int,
        folder: str,
        threshold: float,
    ) -> tuple[str, str, str, dict[str, Any]]:
        ws = self.workspace
        params: dict[str, Any] = {
            "embedding": None,  # filled by caller
            "top_k": top_k,
            "overfetch": _folder_scope_overfetch(top_k),
            "threshold": threshold,
            "folder": folder,
        }
        return ws, f"DocStatus_{ws}", f"Folder_{ws}", params

    def _build_exact_filtered_search(
        self,
        ws: str,
        doc_label: str,
        folder_label: str,
        folder: str,
        filters: RetrievalFilters,
        params: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        params.pop("overfetch", None)
        if "full_doc_id" in self.meta_fields:
            return self._build_exact_chunks_search(
                doc_label, folder_label, folder, filters, params
            )
        if "source_id" in self.meta_fields:
            params["sep"] = GRAPH_FIELD_SEP
            return self._build_exact_graph_search(
                ws, doc_label, folder_label, folder, filters, params
            )
        return None, params

    def _membership_join(
        self,
        ws: str,
        doc_label: str,
        folder_label: str,
        folder: str,
        filters: RetrievalFilters | None,
        params: dict[str, Any],
    ) -> str | None:
        if "full_doc_id" in self.meta_fields:
            return self._chunks_membership_join(
                doc_label, folder_label, folder, filters, params
            )
        if "source_id" in self.meta_fields:
            params["sep"] = GRAPH_FIELD_SEP
            return self._graph_membership_join(
                ws, doc_label, folder_label, folder, filters, params
            )
        return None

    def _scoped_vector_search_cypher(
        self,
        index_name: str,
        join: str,
    ) -> str:
        return f"""
            CALL vector_search.search("{index_name}", $overfetch, $embedding)
            YIELD node, similarity
            WITH node, similarity
            WHERE similarity >= $threshold
            {join}
            RETURN node.id AS id, similarity, properties(node) AS props
            ORDER BY similarity DESC
            LIMIT $top_k
            """

    def _build_search_cypher(
        self,
        top_k: int,
        folder: str | None,
        filters: RetrievalFilters | None = None,
        *,
        filtered_scan_k: int | None = None,
    ) -> tuple[str | None, dict[str, Any]]:
        """Build the vector-search Cypher + params for this query.

        When ``folder`` is None and no filters are active the query is the
        historical, un-scoped search — **byte-for-byte the legacy behaviour**, so
        the native LightRAG path and any non-folder caller are unchanged (strict
        compat).

        When a folder is active the search is constrained to records whose
        document is ``MEMBER_OF`` that folder, so no cross-folder context can
        enter the retrieval result (and therefore the prompt). The constraint is
        an *inner-join* MATCH — chunk→doc is a property equality (``full_doc_id``)
        in LightRAG, not a graph edge, so it cannot be a pattern predicate /
        ``EXISTS`` subquery (the latter is also poorly supported on Memgraph).

        ``filters`` (``RetrievalFilters``) layers the WebUI ``doc_filter`` /
        ``tag_filter`` / ``min_score`` knobs **onto the same retrieval**, so an
        excluded chunk/entity never reaches the prompt — fixing the prior "faux
        grounding" where these were only trimmed from the Sources panel after
        the LLM had already grounded on the unfiltered context. A filter
        contributes Cypher *only when non-empty*: an empty/None filter set leaves
        the folder/legacy Cypher byte-for-byte identical (required for LightRAG
        upgrades). ``min_score`` is an explicit floor; doc/tag filtered search
        disables the backend's implicit floor unless ``min_score`` is set, so a
        tagged document is not hidden by a default similarity threshold.
        Doc/tag predicates require an active folder (the Twin query path always
        binds one) — the tag label is folder-scoped (``WebuiTag_{folder}``).

        Folder-scoped graph-vector retrieval is intentionally unavailable.
        LightRAG aggregates entity and relationship payloads across all source
        documents before storing their vectors. A membership predicate could
        scope row selection, but cannot remove text contributed by documents in
        other folders. Those VDBs therefore fail closed until their payloads are
        materialized per folder or per source document.

        Chunk vectors (``full_doc_id`` in ``meta_fields``) remain available and
        are joined directly through ``DocStatus → MEMBER_OF folder``.
        """
        # Normalise: an empty/None filter set is a true no-op so the folder-only
        # and native LightRAG paths stay byte-for-byte identical.
        if filters is not None and filters.is_empty:
            filters = None
        threshold = self._effective_search_threshold(filters)

        index_name = self._index_name()
        if not folder:
            return self._legacy_search_cypher(index_name, top_k, threshold)

        if "source_id" in self.meta_fields:
            self._log_scoped_refusal_once(
                folder,
                "[MemgraphVec:%s/%s] refusing blended graph-vector retrieval "
                "under folder scope %s (by design — see _build_search_cypher "
                "docstring; logged once per folder, repeats at DEBUG)",
            )
            return None, {}

        ws, doc_label, folder_label, params = self._folder_search_context(
            top_k, folder, threshold
        )

        # Product contract: doc/tag filters define the candidate corpus. The
        # caller may provide a bounded ANN window computed as ``global_count -
        # candidate_count + top_k``. That window contains the requested allowed
        # top-k if the native search returns the true top-scan_k; Memgraph's
        # HNSW recall is approximate, so this is not a hard recall guarantee.
        # If the cheap count plan is unavailable we retain the exact-cosine
        # fallback rather than weakening the filter.
        if filters is not None and (filters.has_doc or filters.has_tag):
            if filtered_scan_k is not None:
                params["overfetch"] = max(top_k, filtered_scan_k)
                join = self._chunks_membership_join(
                    doc_label, folder_label, folder, filters, params
                )
                return self._scoped_vector_search_cypher(index_name, join), params
            return self._build_exact_filtered_search(
                ws, doc_label, folder_label, folder, filters, params
            )

        join = self._membership_join(
            ws, doc_label, folder_label, folder, filters, params
        )
        if join is None:
            # Unknown vdb category WITH an active folder: we cannot prove
            # membership, so we MUST NOT silently fall back to the global search
            # (that would leak cross-folder context the moment a future LightRAG
            # build or a custom store introduces a vdb shape we don't recognise).
            # Fail closed — signal the caller to return no results — and log loud.
            return None, params

        return self._scoped_vector_search_cypher(index_name, join), params

    def _build_exact_chunks_search(
        self,
        doc_label: str,
        folder_label: str,
        folder: str,
        filters: RetrievalFilters,
        params: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Exact cosine search over chunks after folder/doc/tag pre-filtering."""
        base, params = self._build_filtered_chunks_candidates(
            doc_label, folder_label, folder, filters, params
        )
        return base + _exact_cosine_projection(), params

    def _build_filtered_chunks_candidates(
        self,
        doc_label: str,
        folder_label: str,
        folder: str,
        filters: RetrievalFilters,
        params: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Build the shared candidate MATCH used by counts and exact fallback."""
        label = self._label()
        base = f"""
            MATCH (node:`{label}`)
            MATCH (d:`{doc_label}` {{id: node.full_doc_id}})
                  -[:MEMBER_OF]->(:`{folder_label}` {{id: $folder}})"""
        conds: list[str] = []
        if filters.doc_any and not filters.doc_all:
            params["doc_any"] = sorted(filters.doc_any)
            base += "\n            WHERE d.id IN $doc_any"
        else:
            conds = _doc_conditions_single(filters, params, "d.id")
        if filters.has_tag:
            tag_label = f"WebuiTag_{validate_identifier(folder, 'folder')}"
            base += f"""
            OPTIONAL MATCH (d)-[:TAGGED_WITH]->(__t:`{tag_label}`)
            WITH node, d, collect(DISTINCT toLower(__t.id)) AS __dtags"""
            conds += _tag_conditions(filters, params, "__dtags")
        if conds:
            base += "\n            WHERE " + _CYPHER_AND.join(conds)
        return base, params

    async def _filtered_chunks_ann_plan(
        self,
        top_k: int,
        folder: str,
        filters: RetrievalFilters,
    ) -> tuple[int, int, int]:
        """Return ``(scan_k, total, candidates)`` without scoring embeddings.

        The count queries only traverse membership/tag metadata. They replace
        the OVH hot path that interpreted a 1,536-dimension cosine reduction
        for every candidate chunk in Cypher.
        """
        ws = self.workspace
        doc_label = f"DocStatus_{ws}"
        folder_label = f"Folder_{ws}"
        candidate_query, count_params = self._build_filtered_chunks_candidates(
            doc_label, folder_label, folder, filters, {"folder": folder}
        )
        candidate_query += "\n            RETURN count(DISTINCT node) AS count"
        total_query = f"MATCH (node:`{self._label()}`) RETURN count(node) AS count"

        async with _pool.get_read_session() as session:
            total_result = await session.run(total_query)
            total_record = await total_result.single()
            candidate_result = await session.run(candidate_query, **count_params)
            candidate_record = await candidate_result.single()

        total = int(total_record["count"] if total_record is not None else 0)
        candidates = int(
            candidate_record["count"] if candidate_record is not None else 0
        )
        candidates = min(total, max(0, candidates))
        if candidates == 0:
            return 0, total, 0
        excluded = total - candidates
        scan_k = min(total, excluded + top_k)
        return max(1, scan_k), total, candidates

    def _build_exact_graph_search(
        self,
        ws: str,
        doc_label: str,
        folder_label: str,
        folder: str,
        filters: RetrievalFilters,
        params: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Exact cosine search over entity/relation rows after pre-filtering."""
        label = self._label()
        chunk_label = f"Vec_{ws}_chunks"
        base = f"""
            MATCH (node:`{label}`)
            WITH node, split(coalesce(node.source_id, ''), $sep) AS cids
            UNWIND cids AS cid
            MATCH (c:`{chunk_label}` {{id: cid}})
            MATCH (d:`{doc_label}` {{id: c.full_doc_id}})
                  -[:MEMBER_OF]->(:`{folder_label}` {{id: $folder}})"""
        if filters.doc_any and not filters.doc_all:
            params["doc_any"] = sorted(filters.doc_any)
            base += "\n            WHERE d.id IN $doc_any"
        if filters.has_tag:
            tag_label = f"WebuiTag_{validate_identifier(folder, 'folder')}"
            base += f"""
            OPTIONAL MATCH (d)-[:TAGGED_WITH]->(__t:`{tag_label}`)
            WITH node, d, collect(DISTINCT toLower(__t.id)) AS __dtags
            WITH node, collect(DISTINCT {{doc: d.id, tags: __dtags}}) AS __docinfos"""
            if filters.doc_any and not filters.doc_all:
                conds = []
            else:
                conds = _doc_conditions_set(
                    filters, params, "[__di IN __docinfos | __di.doc]"
                )
            tag_inner = _tag_conditions(filters, params, "__di.tags")
            if tag_inner:
                conds.append(
                    "any(__di IN __docinfos WHERE " + _CYPHER_AND.join(tag_inner) + ")"
                )
        else:
            base += """
                WITH node, collect(DISTINCT d.id) AS __docids"""
            conds = _doc_conditions_set(filters, params, "__docids")
        if conds:
            base += "\n            WHERE " + _CYPHER_AND.join(conds)
        return base + _exact_cosine_projection(), params

    def _chunks_membership_join(
        self,
        doc_label: str,
        folder_label: str,
        folder: str,
        filters: RetrievalFilters | None,
        params: dict[str, Any],
    ) -> str:
        """Folder MEMBER_OF join for the chunks vdb, plus doc/tag predicates.

        ``filters is None`` returns the bare membership join byte-for-byte (the
        batch-2 folder-scoping contract). With filters, doc predicates run on the
        single ``d.id`` and tag predicates run on the doc's ``TAGGED_WITH`` ids.
        """
        base = f"""
                MATCH (d:`{doc_label}` {{id: node.full_doc_id}})
                      -[:MEMBER_OF]->(:`{folder_label}` {{id: $folder}})"""
        tail = """
                WITH DISTINCT node, similarity
                """
        if filters is None:
            return base + tail
        conds = _doc_conditions_single(filters, params, "d.id")
        if filters.has_tag:
            tag_label = f"WebuiTag_{validate_identifier(folder, 'folder')}"
            base += f"""
                OPTIONAL MATCH (d)-[:TAGGED_WITH]->(__t:`{tag_label}`)
                WITH node, similarity, d, collect(DISTINCT toLower(__t.id)) AS __dtags"""
            conds += _tag_conditions(filters, params, "__dtags")
        if conds:
            base += "\n                WHERE " + _CYPHER_AND.join(conds)
        return base + tail

    def _graph_membership_join(
        self,
        ws: str,
        doc_label: str,
        folder_label: str,
        folder: str,
        filters: RetrievalFilters | None,
        params: dict[str, Any],
    ) -> str:
        """Build the legacy graph-row membership join.

        This helper is deliberately unreachable from ``_build_search_cypher``
        while a folder is active: membership can scope selection but cannot
        unblend the stored payload. It remains isolated here to ease a future
        migration to per-folder or per-source graph-vector materialization.

        ``source_id`` is split into source chunk ids → docs. ``filters is None``
        returns the bare membership join byte-for-byte. With filters, the member
        source docs (and, when a tag filter is active, their tag ids) are
        aggregated per record so set semantics apply: ``doc_all`` ⊆ source docs,
        ``doc_any`` ∩ source docs, and tag filters satisfied by ≥1 source doc.
        These predicates scope selection only; callers must not treat them as
        sufficient content isolation.
        """
        chunk_label = f"Vec_{ws}_chunks"
        base = f"""
                WITH node, similarity,
                     split(coalesce(node.source_id, ''), $sep) AS cids
                UNWIND cids AS cid
                MATCH (c:`{chunk_label}` {{id: cid}})
                MATCH (d:`{doc_label}` {{id: c.full_doc_id}})
                      -[:MEMBER_OF]->(:`{folder_label}` {{id: $folder}})"""
        tail = """
                WITH DISTINCT node, similarity
                """
        if filters is None:
            return base + tail
        if filters.has_tag:
            tag_label = f"WebuiTag_{validate_identifier(folder, 'folder')}"
            base += f"""
                OPTIONAL MATCH (d)-[:TAGGED_WITH]->(__t:`{tag_label}`)
                WITH node, similarity, d, collect(DISTINCT toLower(__t.id)) AS __dtags
                WITH node, similarity, collect(DISTINCT {{doc: d.id, tags: __dtags}}) AS __docinfos"""
            conds = _doc_conditions_set(
                filters, params, "[__di IN __docinfos | __di.doc]"
            )
            tag_inner = _tag_conditions(filters, params, "__di.tags")
            if tag_inner:
                conds.append(
                    "any(__di IN __docinfos WHERE " + _CYPHER_AND.join(tag_inner) + ")"
                )
        else:
            base += """
                WITH node, similarity, collect(DISTINCT d.id) AS __docids"""
            conds = _doc_conditions_set(filters, params, "__docids")
        if conds:
            base += "\n                WHERE " + _CYPHER_AND.join(conds)
        return base + tail

    async def query(
        self,
        query: str,
        top_k: int,
        query_embedding: list[float] = None,
    ) -> list[dict[str, Any]]:
        query_started = time.perf_counter()
        index_name = self._index_name()
        folder = get_active_storage_folder()
        filters = get_active_retrieval_filters()
        filtered_scan_k: int | None = None
        filter_total: int | None = None
        filter_candidates: int | None = None
        if (
            folder
            and filters is not None
            and (filters.has_doc or filters.has_tag)
            and "full_doc_id" in self.meta_fields
        ):
            try:
                filtered_scan_k, filter_total, filter_candidates = (
                    await self._filtered_chunks_ann_plan(top_k, folder, filters)
                )
            except Exception:
                # Correctness fallback: never silently turn a filtered query
                # into a small post-filtered ANN window.
                logger.exception(
                    "[MemgraphVec:%s/%s] filtered ANN count plan failed; "
                    "falling back to exact filtered cosine",
                    self.workspace,
                    self.namespace,
                )
                filtered_scan_k = None
            if filter_candidates == 0:
                logger.info(
                    "[MemgraphVec:%s/%s] filtered corpus empty folder=%s total=%d",
                    self.workspace,
                    self.namespace,
                    folder,
                    filter_total or 0,
                )
                return []

        cypher, params = self._build_search_cypher(
            top_k, folder, filters, filtered_scan_k=filtered_scan_k
        )
        if cypher is None:
            # Fail-closed: the active folder cannot safely scope this VDB shape
            # (including globally blended entity/relation payloads). Never leak
            # a global result set into a folder-scoped retrieval.
            self._log_scoped_refusal_once(
                folder,
                "[MemgraphVec:%s/%s] folder=%s active but this vdb category "
                "cannot be safely scoped — returning empty (fail-closed, by "
                "design) rather than leaking a global result set. Logged once "
                "per folder, repeats at DEBUG.",
            )
            return []

        embedding_ms = 0
        if query_embedding is None:
            embedding_started = time.perf_counter()
            embedding_result = await self.embedding_func.func([query])
            embedding_ms = int((time.perf_counter() - embedding_started) * 1000)
            query_embedding = (
                embedding_result[0].tolist()
                if hasattr(embedding_result[0], "tolist")
                else list(embedding_result[0])
            )
        params["embedding"] = query_embedding
        params["query_norm"] = math.sqrt(
            sum(float(v) * float(v) for v in query_embedding)
        )

        search_started = time.perf_counter()
        async with _pool.get_read_session() as session:
            try:
                result = await session.run(cypher, **params)
                results = [self._record_to_entry(record) async for record in result]
                await result.consume()
            except Exception as e:
                if "does not exist" not in str(e).lower():
                    raise
                logger.warning(
                    "[MemgraphVec:%s/%s] Vector index '%s' missing "
                    "— auto-creating and retrying query.",
                    self.workspace,
                    self.namespace,
                    index_name,
                )
                try:
                    await self._create_vector_index()
                except Exception as create_err:
                    logger.error(
                        "[MemgraphVec:%s/%s] Auto-create failed: %s — "
                        "returning empty results.",
                        self.workspace,
                        self.namespace,
                        create_err,
                    )
                    return []
                result = await session.run(cypher, **params)
                results = [self._record_to_entry(record) async for record in result]
                await result.consume()
            _capture_chunk_retrieval_scores(self.namespace, results)
            search_ms = int((time.perf_counter() - search_started) * 1000)
            total_ms = int((time.perf_counter() - query_started) * 1000)
            timing_log = (
                logger.warning
                if filters is not None and total_ms >= 30_000
                else logger.info if filters is not None else logger.debug
            )
            timing_log(
                "[MemgraphVec:%s/%s] timings embedding=%dms search=%dms "
                "total=%dms results=%d folder=%s filtered=%s "
                "filter_total=%s filter_candidates=%s ann_window=%s",
                self.workspace,
                self.namespace,
                embedding_ms,
                search_ms,
                total_ms,
                len(results),
                folder or "-",
                filters is not None,
                filter_total if filter_total is not None else "-",
                filter_candidates if filter_candidates is not None else "-",
                filtered_scan_k if filtered_scan_k is not None else "exact-or-default",
            )
            logger.debug(
                "[MemgraphVec:%s/%s] query(%r) → %d results (index=%s, "
                "threshold=%.2f, top_k=%d, folder=%s)",
                self.workspace,
                self.namespace,
                query[:50],
                len(results),
                index_name,
                self.cosine_better_than_threshold,
                top_k,
                folder or "-",
            )
            return results

    async def _compute_missing_embeddings(
        self, data: dict[str, dict[str, Any]]
    ) -> dict[str, list[float]]:
        """Batch-compute embeddings for items without a pre-computed one."""
        needs_embed = [
            (eid, item["content"])
            for eid, item in data.items()
            if item.get("embedding") is None and "content" in item
        ]
        if not needs_embed:
            return {}
        batch_size = self._embedding_batch_size()
        out: dict[str, list[float]] = {}
        for start in range(0, len(needs_embed), batch_size):
            batch = needs_embed[start : start + batch_size]
            eids = [eid for eid, _ in batch]
            contents = [content for _, content in batch]
            emb_results = await self._embed_with_retry(contents)
            rows = list(emb_results)
            if len(rows) != len(eids):
                raise ValueError(
                    "Embedding function returned "
                    f"{len(rows)} rows for {len(eids)} input texts"
                )
            for eid, embedding in zip(eids, rows):
                out[eid] = (
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                )
        return out

    async def _embed_with_retry(self, contents: list[str]):
        """Call the embedding function with a per-attempt timeout and retry.

        Recovers from a cold embedding endpoint: a first attempt that times out
        wakes the model; a later, now-warm attempt succeeds. The happy path
        (endpoint responds within the timeout) returns on attempt 1 with no
        added latency. Once attempts are exhausted the last error is re-raised
        so LightRAG still marks the document FAILED — never a silent drop.

        Tradeoff: a legitimately slow (not cold) endpoint that always exceeds
        the per-attempt timeout will fail slower than a single native call.
        Raise ``TWIN_EMBEDDING_TIMEOUT`` for such endpoints.
        """
        timeout = self._embedding_timeout()
        attempts = self._embedding_attempts()
        last_exc: BaseException = RuntimeError("no embedding attempt ran")
        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.wait_for(
                    self.embedding_func.func(contents), timeout=timeout
                )
            except (asyncio.TimeoutError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                logger.warning(
                    "[MemgraphVec:%s] embedding attempt %d/%d failed after "
                    "~%.0fs (cold start?) for %d texts: %s",
                    self.workspace,
                    attempt,
                    attempts,
                    timeout,
                    len(contents),
                    type(exc).__name__,
                )
                if attempt < attempts:
                    await asyncio.sleep(_EMBEDDING_RETRY_BACKOFF)
        raise last_exc

    def _embedding_timeout(self) -> float:
        """Per-attempt embedding timeout (seconds). Env-tunable, default 30s."""
        raw = os.environ.get(_EMBEDDING_TIMEOUT_ENV, "")
        try:
            val = float(raw)
            if val <= 0:
                raise ValueError
            return val
        except (TypeError, ValueError):
            return _DEFAULT_EMBEDDING_TIMEOUT

    def _embedding_attempts(self) -> int:
        """Number of embedding attempts. Env-tunable, default 3 (>= 1)."""
        raw = os.environ.get(_EMBEDDING_ATTEMPTS_ENV, "")
        try:
            val = int(raw)
            if val < 1:
                raise ValueError
            return val
        except (TypeError, ValueError):
            return _DEFAULT_EMBEDDING_ATTEMPTS

    def _embedding_batch_size(self) -> int:
        """Batch size for storage-side embedding calls.

        LightRAG passes ``embedding_batch_num`` in ``global_config``. Respect it
        here so ingestion does not send hundreds of chunks through a single
        60-second-wrapped embedding call.
        """
        raw = (self.global_config or {}).get(
            "embedding_batch_num", _DEFAULT_EMBEDDING_BATCH_NUM
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_EMBEDDING_BATCH_NUM
        return max(1, value)

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
        if self.namespace == _VEC_CHUNKS_NAMESPACE:
            # Audit 2026-08-06, R-06: neutralize reserved prompt delimiters
            # at the storage boundary, BEFORE embedding, so the embedding
            # matches the stored (neutralized) text.
            data = neutralize_chunk_payloads(data)
        computed = await self._compute_missing_embeddings(data)
        entries = [
            self._build_entry(eid, item, item.get("embedding") or computed.get(eid))
            for eid, item in data.items()
        ]

        async def _write() -> None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(
                        f"""
                        UNWIND $entries AS e
                        MERGE (n:`{label}` {{id: e.id}})
                        SET n += e.props,
                            n.embedding = coalesce(e.embedding, n.embedding)
                        """,
                        entries=entries,
                    )
                    await result.consume()

        # Re-runnable: MERGE + SET with the same already-computed entries. The
        # embeddings were resolved above, so a retry never re-bills the model.
        await with_conflict_retry(f"MemgraphVec.upsert[{label}]", _write)

    async def delete_entity(self, entity_name: str) -> None:
        # REMOVE label before DETACH DELETE: Memgraph 3.10+ keeps vector
        # index entries for deleted vertices around (vector_index_memory_tracked
        # is not freed on vertex delete), then strict-errors on subsequent
        # property access from vector_search results — see release notes
        # "accessing properties of a deleted node/relationship now raises an
        # error instead". Removing the indexed label first prunes the vector
        # index entry cleanly; the DETACH DELETE then removes the orphan.
        label = self._label()

        async def _write() -> None:
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

        # Re-runnable: the label is already gone, so a retry matches nothing.
        await with_conflict_retry(f"MemgraphVec.delete_entity[{label}]", _write)

    async def delete_entity_relation(self, entity_name: str) -> None:
        # Same Memgraph 3.10 vector-index hygiene as delete_entity above.
        label = self._label()

        async def _write() -> None:
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

        # Re-runnable: the label is already gone, so a retry matches nothing.
        await with_conflict_retry(
            f"MemgraphVec.delete_entity_relation[{label}]", _write
        )

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
        # Same Memgraph 3.10 vector-index hygiene as delete_entity above.
        label = self._label()

        async def _write() -> None:
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

        # Re-runnable: the label is already gone, so a retry matches nothing.
        await with_conflict_retry(f"MemgraphVec.delete[{label}]", _write)

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
        # Same Memgraph 3.10 vector-index hygiene as delete_entity above:
        # REMOVE label before DETACH DELETE so the vector index entry is
        # pruned cleanly, otherwise the subsequent DROP VECTOR INDEX can
        # leave stale refs that break the next test or the next ingest.
        label = self._label()

        async def _write() -> None:
            async with _pool.acquire_write_slot():
                async with _pool.get_session() as session:
                    result = await session.run(f"""
                        MATCH (n:`{label}`)
                        REMOVE n:`{label}`
                        WITH n
                        DETACH DELETE n
                        """)
                    await result.consume()
                    try:
                        result = await session.run(
                            f"DROP VECTOR INDEX `{self._index_name()}`"
                        )
                        await result.consume()
                    except Exception as e:
                        # Swallow ONLY the idempotent "index already gone" case.
                        # A bare `except Exception: pass` here previously buried
                        # EVERY failure — connection resets, auth errors, permission
                        # denials — while drop() still returned {"status": "success"}.
                        # That false-success hid a half-completed drop (the nodes
                        # were deleted but the vector index was left behind, the
                        # exact stale state the REMOVE-before-DELETE dance above
                        # tries to avoid) and lied to the caller. Match the "does not
                        # exist" message like the query() auto-create path does, and
                        # re-raise anything else so real failures surface instead of
                        # being reported as success.
                        msg = str(e).lower()
                        if "does not exist" not in msg and "doesn't exist" not in msg:
                            raise
                        logger.debug(
                            "[MemgraphVec:%s] Vector index '%s' already absent on "
                            "drop",
                            self.workspace,
                            self._index_name(),
                        )

        # Re-runnable: an emptied label matches nothing and the index drop
        # already tolerates "already absent" — which is exactly the state a
        # retried attempt finds.
        await with_conflict_retry(f"MemgraphVec.drop[{label}]", _write)
        return {"status": "success", "message": f"Vector namespace {label} dropped"}
