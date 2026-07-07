# BNP Twin KMS Retrieval, Metrics, and Prompt FAQ

Audience: BNP Twin KMS operators, support engineers, platform owners, and technical reviewers.

Last updated: 2026-07-03.

Scope: this FAQ describes the Twin KMS overlay behavior implemented in this repository. LightRAG native internals are described only where the Twin code calls or constrains them. The upstream `lightrag` package is not vendored in this repository, so this document does not reproduce upstream prompt text.

## 1. Executive Summary

### What does Twin KMS add on top of LightRAG?

Twin KMS is a Memgraph-backed LightRAG runtime and WebUI overlay. It keeps LightRAG as the retrieval and generation engine, then adds:

- folder-based scoping on top of one physical LightRAG/Memgraph workspace;
- source projection for the React WebUI;
- explicit `answer_status` values for grounded, no-context, no-retrieval, projection-failure, and query-failure states;
- document and tag filters enforced inside Memgraph retrieval, before the LLM prompt is built;
- operator controls such as query mode, `top_k`, `chunk_top_k`, `min_score`, history depth, optional user prompt, and rerank toggle;
- optional Twin intelligence and ontology prompts for intent detection, query reasoning, reranking, synthesis, ontology extraction, clustering, enrichment, and validation.

### What is the most important grounding rule?

For the nominal `/twin/api/query` path, sources are projected from LightRAG's `aquery_llm` envelope, specifically `data.references` and `data.chunks`. Twin KMS does not run a second vector search to reconstruct sources after the answer has been produced. This avoids showing sources that were not actually used by LightRAG to ground the answer.

Implementation reference:

- `src/twindb_lightrag_memgraph/server/query/router.py`
- `src/twindb_lightrag_memgraph/server/_lightrag_compat.py`

## 2. Retrieval Entry Points

### Which API endpoint does the WebUI use for normal answers?

The normal answer path is:

1. WebUI calls `POST /twin/api/query`.
2. Backend builds a LightRAG `QueryParam` from the request.
3. Backend binds the active Twin folder and retrieval filters.
4. Backend calls `rag.aquery_llm(query, param=param)`.
5. Backend classifies the answer envelope.
6. Backend projects `data.references` into the WebUI `sources[]` contract.
7. WebUI renders answer text, citations, and source cards.

Returned shape:

```json
{
  "response": "answer text",
  "sources": [
    {
      "n": 1,
      "type": "file",
      "name": "runbook.pdf",
      "meta": "1 chunk",
      "score": 0.91,
      "doc_id": "doc-...",
      "chunk_id": "chunk-..."
    }
  ],
  "answer_status": "grounded"
}
```

### What is `/twin/api/query/data` for?

`POST /twin/api/query/data` returns structured retrieval data rather than a final natural-language answer. It calls `rag.aquery_data(query, param=param)` and returns the structured blocks LightRAG exposes, such as entities, relationships, chunks, references, and metadata.

This endpoint is useful for diagnostics, audits, and technical inspection because it exposes retrieval material before final synthesis.

### When does Twin use `rag.aquery()` instead of `aquery_llm()`?

Twin uses `rag.aquery()` for `only_need_context` and `only_need_prompt`, because those modes intentionally return a raw context or prompt body rather than the structured `aquery_llm` envelope. They are marked as `answer_status = "no_retrieval"` in the Twin response because no sourced final answer is being claimed.

### What is bypass mode?

`bypass` mode calls the LLM directly without retrieval grounding. Twin returns the answer with an empty source list and `answer_status = "no_retrieval"`.

Operator meaning: bypass is useful for comparing raw model behavior against retrieval-grounded behavior, but it should not be used as a sourced KMS answer mode.

## 3. Ingestion and Indexing Priority

### If a user uploads documents, then uploads more documents, which ones are processed first?

For ingestion and indexing, Twin KMS should be understood as a processing queue, not as a retrieval-ranking system.

When a user uploads a first batch of documents, those documents are submitted to the ingestion/indexing pipeline. If the same user uploads a second batch before the first batch has fully completed, the new files are added to the processing workload. In principle, processing follows task arrival order, usually FIFO by batch or document, unless an explicit retry, resume, worker scheduling, or parallel-processing mechanism changes the execution order.

A more recent upload does not automatically jump ahead of a document that is already `PROCESSING` or already waiting in the queue.

### When does an uploaded document become searchable?

A document should be considered searchable only after it has reached `PROCESSED` and its chunks/index data are available to the retrieval layer.

If the pipeline runs multiple workers or processes files in parallel, a document from a later upload can become searchable before every document from an earlier batch has completed. This is an execution effect, not a business-priority rule.

Operationally, the important states are:

- `PENDING`: accepted but not yet processed;
- `PROCESSING`: ingestion/indexing is running;
- `PROCESSED`: available for retrieval, assuming chunks/index data are present;
- `FAILED`: ingestion/indexing failed and requires retry, cleanup, or operator inspection.

### Is indexing priority the same as search-result priority?

No. Indexing priority controls when a document becomes available. Search-result priority controls which already indexed sources are selected for an answer.

Once several documents are `PROCESSED`, newer documents do not rank higher just because they were uploaded later. Retrieval ranking is driven by query similarity, filters, graph mode, source scores, and optional reranking.

## 4. Search Modes

### Which query modes are exposed in the WebUI?

The WebUI exposes:

- `naive`
- `local`
- `global`
- `hybrid`
- `mix`
- `bypass`

These are sent as the LightRAG `mode` value through `QueryParam`.

### What does each mode mean operationally?

| Mode | Operational meaning | Source behavior |
| --- | --- | --- |
| `naive` | Chunk/vector-oriented retrieval without the graph-centric path. | Usually best for direct text similarity and precise runbook fragments. |
| `local` | Graph-oriented local retrieval around closely related entities. | Useful for entity-neighborhood questions. |
| `global` | Graph-oriented broader retrieval. | Useful for high-level or cross-document questions. |
| `hybrid` | Combines graph retrieval strategies in LightRAG's KG path. | Useful when both local and broader KG context matter. |
| `mix` | Chunk-inclusive mode used by Twin as the operator-friendly default. | Preserves KG data when present and still lets chunks surface when graph rows are sparse. |
| `bypass` | Direct LLM call with no retrieval. | No sources by design. |

### Why does Twin sometimes fall back to `mix` for structured retrieval?

For `POST /twin/api/query/data`, upstream LightRAG's `hybrid`, `local`, and `global` structured retrieval can report `no_results` when the graph side is empty, even if filtered chunks exist. If document or tag filters are active and `fallback_to_mix` is enabled, Twin retries as `mix`.

The fallback is annotated in metadata:

```json
{
  "requested_mode": "hybrid",
  "fallback_mode": "mix",
  "fallback_reason": "filtered_graph_mode_no_results"
}
```

This is not an answer-quality fallback; it is a structured-data completeness fallback for filtered corpora.

## 5. Source Scores and Metrics

### Is the source score available?

Yes. Each returned source has a numeric `score`.

Backend contract:

```python
class TwinRetrievalSource(BaseModel):
    n: int
    type: str = "file"
    name: str
    meta: str | None = None
    score: float = 0.0
    doc_id: str | None = None
    chunk_id: str | None = None
```

Frontend contract:

```ts
export interface RetrievalSource {
  n: number;
  type: SourceType;
  name: string;
  meta?: string | null;
  score: number;
  doc_id?: string | null;
  chunk_id?: string | null;
}
```

The WebUI renders it with two decimals.

### How is the source score calculated?

Twin reads the first available numeric metric from each LightRAG chunk/reference in this order:

1. `score`
2. `similarity`
3. `cosine_similarity`
4. nested `__metrics__.score`
5. nested `__metrics__.similarity`
6. nested `__metrics__.cosine_similarity`

If no metric is present, Twin synthesizes a rank-based fallback:

```text
0.95 for the first item down to 0.50 for the last item
```

The fallback exists to keep the UI sortable and stable. It should not be interpreted as an actual vector similarity.

### Is the score cosine similarity?

When the Memgraph vector backend returns the metric, yes. The Memgraph vector storage projects:

- `similarity`: cosine similarity returned or computed by the vector search path;
- `distance`: `1.0 - similarity`.

The vector index is configured with cosine metric semantics.

### Is the displayed source score a reranker score?

No. The WebUI source score is the `sources[].score` field, normally cosine/hybrid similarity on a `0..1` scale.

The optional cognitive reranker uses a separate LLM score on a `0..10` scale. That score is used to filter or reorder chunks inside the intelligence layer when enabled, but it is not the displayed source-card score in the current WebUI contract.

### Is the displayed score generated before or after reranking?

The displayed source score is a retrieval metric generated before any reranker-driven final ordering. It reflects the initial retrieval signal, usually vector cosine similarity or a LightRAG-provided source/chunk score.

If reranking is enabled, the final source order may be changed by a separate relevance assessment. Therefore the visible source list is not guaranteed to be sorted by the displayed score. A lower displayed score can appear above a higher one when the reranker considered it more useful for answering the question.

### Is the score a confidence that the final answer is correct?

No. The score estimates retrieval closeness for a source. It is not:

- an answer correctness probability;
- a factuality score;
- a legal or operational validation score;
- a guarantee that the answer used every sentence in the source.

For answer trust, operators should evaluate:

- `answer_status`;
- source presence;
- citation alignment;
- retrieved source content;
- whether filters and folder scoping were applied;
- whether the answer says the available information is partial.

### What does `min_score` do?

`min_score` is an explicit source-score floor in the request. It is applied at retrieval time where possible and also enforced during source projection.

Important details:

- `min_score` range is `0.0..1.0`.
- When no document/tag filters are active, the backend keeps the configured LightRAG/Memgraph cosine threshold and applies `max(default_threshold, min_score)`.
- When document/tag filters are active, Twin treats the filter as the candidate corpus and uses exact cosine scoring over that filtered corpus. The default global cosine floor is not allowed to hide a tagged or selected document unless the user explicitly sets `min_score`.

## 6. Filtering and Folder Scoping

### Are filters applied before or after answer generation?

They are applied before prompt construction in the storage layer.

Twin binds the active folder and retrieval filters through context variables around `aquery_llm`, `aquery_data`, and `aquery`. The Memgraph vector backend reads that context and constrains the retrieval query itself. This means excluded chunks/entities do not enter the LLM prompt in the nominal path.

### What filters are supported?

Twin supports:

- active folder scoping;
- document filters;
- tag filters;
- `min_score`.

Document and tag filters support:

- `any`: at least one selected value must match;
- `all`: all selected values must match.

### What is special about `doc_all`?

For chunks, a chunk belongs to one full document. Therefore `doc_all` with two or more document ids is intentionally strict and normally empty. It is not silently converted into `any`.

### How are tag filters evaluated?

Tag ids are compared case-insensitively by lower-casing tag ids at the route boundary and at the Memgraph query level.

### How are filters implemented in Memgraph?

There are two retrieval shapes:

1. Chunk vector DB records include `full_doc_id`. Twin joins `node.full_doc_id` to `DocStatus_{workspace}` and then to `Folder_{workspace}` membership.
2. Entity/relation vector DB records include `source_id`, a separator-joined list of chunk ids. Twin splits `source_id`, joins those chunks back to documents and folder membership, then keeps the entity/relation if the source chunk set satisfies the filter.

When document or tag filters are active, Twin computes exact cosine similarity over the pre-filtered candidate set. This avoids approximate-nearest-neighbor overfetch issues where a relevant tagged document could be missed because untagged neighbors occupied the initial vector window.

### Are graph entity and relationship contents fully folder-isolated?

Selection is folder-scoped. The textual payload of a kept graph entity or relation can still be blended from multiple source documents because LightRAG aggregates entity/relation content during extraction. If a kept graph record was originally aggregated from multiple documents, its text may encode context from outside the active folder.

This is a known residual of graph-level aggregation. Fully unblending it would require per-folder graph extraction.

## 7. Answer Status

### What values can `answer_status` have?

| Status | Meaning |
| --- | --- |
| `grounded` | Retrieval ran and the answer is considered grounded by the returned context. Sources should be shown when projection succeeds. |
| `insufficient_information` | Retrieval ran but LightRAG signaled no usable context, either through `failure_reason = "no_results"` or the `[no-context]` marker. |
| `source_projection_failed` | LightRAG produced a grounded answer, but Twin could not project the envelope into `sources[]`. The answer is shown with an explicit sources-unavailable cue. |
| `no_retrieval` | No sourced retrieval was attempted by design, for example `bypass`, `only_need_context`, or `only_need_prompt`. |
| `query_failed` | A backend failure occurred during streaming after HTTP status had already been committed. |

### Why does Twin distinguish `insufficient_information` from `no_retrieval`?

Because they mean different things:

- `insufficient_information`: retrieval was attempted, but no usable evidence was found.
- `no_retrieval`: retrieval was intentionally not used.

The distinction prevents operators from confusing a direct LLM answer with a grounded KMS answer.

### What is the `[no-context]` marker?

It is a LightRAG no-context signal that Twin treats as defense-in-depth. If it appears in answer content, Twin strips it from the operator-facing text and sets `answer_status = "insufficient_information"`.

## 8. Reranking

### What does the WebUI `Enable rerank` switch control?

The WebUI sends `enable_rerank` into `QueryParam` when the installed LightRAG version supports or consumes it. Twin keeps the field resilient to upstream version changes: if a field is not accepted by the `QueryParam` constructor, Twin sets it as a runtime attribute.

### What is Twin cognitive reranking?

The optional Twin intelligence reranker scores each passage for answer relevance using an LLM prompt. It asks for compact JSON:

```json
{"s": [{"p": 0, "v": 8}, {"p": 1, "v": 2}]}
```

Where:

- `p` is the passage index;
- `v` is a relevance score from `0` to `10`.

Default interpretation:

- `10`: directly contains the complete technical answer;
- `7-9`: highly relevant;
- `4-6`: partial context;
- `1-3`: weakly related;
- `0`: irrelevant.

The default reranking threshold is `7.0`.

### What happens if reranking is too strict or fails?

Twin falls back to deterministic top-K ordering:

- by raw retrieval score when no rerank scores are usable;
- by rerank score when partial rerank scores are available and preferred.

This is designed to degrade retrieval quality gracefully instead of failing the user query.

## 9. Query Parameters

### Which retrieval parameters are available?

The backend request model supports:

| Parameter | Meaning |
| --- | --- |
| `query` | User question. |
| `mode` | LightRAG query mode: `naive`, `local`, `global`, `hybrid`, `mix`, or `bypass`. |
| `top_k` | High-level result limit. |
| `chunk_top_k` | Chunk-specific result limit where supported by LightRAG. |
| `max_entity_tokens` | Entity context budget. |
| `max_relation_tokens` | Relationship context budget. |
| `max_total_tokens` | Overall context budget. |
| `only_need_context` | Return context body instead of final answer. |
| `only_need_prompt` | Return assembled prompt instead of final answer. |
| `hl_keywords` | High-level keywords passed to LightRAG. |
| `ll_keywords` | Low-level keywords passed to LightRAG. |
| `conversation_history` | Conversation turns sent with the query. |
| `history_turns` | Number of history turns to retain. |
| `user_prompt` | Optional operator instruction appended to retrieval behavior where supported. |
| `enable_rerank` | Rerank toggle. |
| `min_score` | Minimum source score, `0..1`. |
| `tag_filter` | Tag filter with `all`/`any` keys. |
| `doc_filter` | Document filter with `all`/`any` keys. |
| `fallback_to_mix` | Allows structured retrieval fallback from KG modes to `mix`. |

### Why does Twin tolerate unknown `QueryParam` fields?

LightRAG changes `QueryParam` fields between versions. Twin introspects constructor fields, passes known fields into the constructor, and applies unknown-but-useful fields as runtime attributes. This avoids turning a minor LightRAG version change into a query failure.

## 10. Prompt Families

### Which prompt families are relevant?

There are two categories:

1. LightRAG native prompts used by the upstream package for ingestion, entity extraction, keyword extraction, KG retrieval, and answer generation. These are not vendored here.
2. Twin intelligence and ontology prompts versioned in this repository.

This FAQ documents the second category precisely and describes the first category only through the Twin integration points.

### Does Twin replace LightRAG's native ingestion prompts?

Not by default. The storage overlay registers Memgraph implementations and server/WebUI behavior. It does not rewrite upstream LightRAG prompt templates in this repository.

Twin has a separate optional ontology pipeline. That pipeline has its own prompts for extraction, clustering, enrichment, and validation.

### Where are Twin prompts stored?

Prompt files live under:

```text
src/twindb_lightrag_memgraph/intelligence/prompts/
```

Current prompt files:

- `intent_system.txt`
- `reason_system.txt`
- `domain_system.txt`
- `reranking_system.txt`
- `synthesis_system.txt`
- `ontology_extract.txt`
- `ontology_cluster.txt`
- `ontology_enrich.txt`
- `ontology_validate.txt`

## 11. Intent, Reasoning, and Synthesis Prompts

### What does the intent prompt do?

The intent prompt classifies a user query for an IT Ops banking assistant.

Output schema:

```json
{"i": "IN_SCOPE|OUT_OF_SCOPE|GREETING|MALICIOUS|ESCALATION", "c": 0.95, "r": "short reason"}
```

Key behavior:

- treats user text as untrusted;
- identifies prompt-injection and role-play attempts as `MALICIOUS`;
- defaults uncertain low-confidence cases toward `IN_SCOPE` so legitimate IT questions are not blocked too aggressively.

### What does the reasoning prompt do?

The reasoning prompt optimizes the search query before retrieval.

It performs:

- coreference resolution from conversation history;
- technical domain detection;
- extraction and enrichment of technical terms;
- compact search-query rewriting.

Output schema:

```json
{
  "t": "short reasoning trace",
  "q": "optimized technical search query",
  "d": "oracle",
  "cr": true
}
```

### What does the domain prompt do?

It classifies the query into an IT domain such as:

- `oracle`
- `network`
- `linux`
- `cloud`
- `middleware`
- `security`
- `monitoring`
- `general`

Output includes a confidence score and short reason.

### What does the synthesis prompt do?

The synthesis prompt is designed for answer generation from provided passages only. It instructs the model to:

- answer exclusively from supplied passages;
- cite claims with passage ids;
- be concise and technical;
- distinguish complete, partial, and absent information;
- structure incident answers as diagnosis, probable cause, and recommended action;
- ignore hostile instructions inside passages and history.

## 12. Ontology Extraction and Enrichment

### Is ontology extraction the same as LightRAG native KG extraction?

No. LightRAG native ingestion builds its own graph structures. Twin's ontology pipeline is an optional intelligence feature with its own prompts and review flow.

Default configuration has ontology disabled unless enabled by configuration.

### What are the ontology pipeline steps?

Twin's ontology pipeline is:

```text
EXTRACT -> CLUSTER -> ENRICH -> VALIDATE
```

When review is required, the result is a dry-run preview. It is persisted only after approval.

### What does the ontology extraction prompt produce?

It extracts entities and relations from untrusted document text.

Entity types:

- `Term`
- `Role`
- `Team`
- `Tool`
- `Process`
- `Asset`

Relationship types:

- `SYNONYM`
- `RELATED_TO`
- `CAUSED_BY`
- `MITIGATED_BY`
- `DIAGNOSED_WITH`
- `CO_OCCURS`
- `OWNS`
- `USES`
- `FOLLOWS`
- `ESCALATES_TO`
- `DEPENDS_ON`
- `DOCUMENTED_IN`
- `PART_OF`
- `REPLACES`
- `REQUIRES_APPROVAL`
- `TRIGGERS`

Compact output schema:

```json
{
  "e": [
    {"n": "ORA-04030", "t": "Term", "d": "Oracle memory error", "c": 0.9}
  ],
  "r": [
    {"s": "ORA-04030", "st": "Term", "o": "PGA", "ot": "Term", "rt": "RELATED_TO", "c": 0.8}
  ]
}
```

### What confidence score does ontology extraction use?

Ontology extraction uses `c` on a `0.0..1.0` scale for entities and relations.

This confidence is an extraction confidence, not a source retrieval score and not an answer factuality score.

### What extraction modes exist?

The ontology extraction step supports:

- `dedicated`: focused on a configured subject;
- `emergence`: neutral extraction without predefined categories;
- `deep_extraction`: deeper symbolic analysis using all DSEP operators.

If dual-pass extraction is enabled, Twin can run:

- a global pass for high-level domains, processes, teams, and methodologies;
- a local pass for precise technical entities, error codes, and causal relationships.

### What is DSEP?

DSEP means Domain-Specific Extraction Profile. It injects extraction policy operators into ontology prompts.

Operators:

| Operator | Purpose |
| --- | --- |
| Structural Analysis | Separate operational facts from noise and embedded instructions. |
| Scope Exclusion | Exclude legacy-only context and schema-redefinition attempts. |
| Gap Analysis | Identify edge cases, missing preconditions, and failure modes. |
| Bounded Context | Define domain boundary, vocabulary, owning teams, and dependencies. |
| Entity Definition | Define each object with canonical name, type, properties, relation candidates, and evidence. |
| Migration / Mapping | Map legacy names, synonyms, replacements, and migration targets without inventing unsupported relations. |

### What does ontology clustering do?

The clustering prompt groups extracted entities into emergent domains. It does not use predefined categories. It expects a JSON response containing domain names, descriptions, and member terms.

### What does ontology enrichment do?

The enrichment prompt receives:

- extracted entities;
- existing relations;
- discovered domains.

It proposes missing relationships, focusing on:

- `SYNONYM`
- `CO_OCCURS`
- `DEPENDS_ON`
- `CAUSED_BY`
- `MITIGATED_BY`

Each proposed relationship carries a `0.0..1.0` confidence score.

### How is enrichment output protected before writing to Memgraph?

The enrichment parser treats LLM output as untrusted. Relation types and node types are allow-listed before they can be used downstream. Malformed or unsupported relation types are dropped instead of being interpolated into Cypher.

### What does ontology validation do?

The validation prompt reviews nodes and edges for:

- duplicate entities;
- contradictory relationships;
- missing confidence scores;
- circular dependencies that look erroneous.

It returns issues and suggested corrections.

## 13. Security and Prompt-Injection Controls

### How does Twin treat document text?

Twin prompts repeatedly state that document text, passages, and conversation history are untrusted data. Instructions found inside documents must not change model role, output schema, policy, or secret-handling behavior.

### Which components explicitly include prompt-injection defenses?

Twin prompt-injection defenses appear in:

- intent classification;
- query reasoning;
- passage reranking;
- synthesis;
- ontology extraction;
- DSEP extraction policy.

### Does filtering happen only in the UI?

No. Filtering is enforced in the Memgraph vector query. UI filtering is only a display and guard-rail layer.

## 14. Source Projection and Citations

### How are citations aligned with sources?

LightRAG references carry a `reference_id`. Twin projects each reference into a WebUI source where:

```text
source.n = reference_id
```

The frontend parser recognizes citation markers such as `[N]` and maps them to `sources[n]`.

Twin intentionally does not deduplicate references by `file_path`, because doing so could make a citation marker point to the wrong source.

### What does `meta` mean on a source?

`meta` is informational. It usually carries the number of chunks attached to a reference, for example:

- `1 chunk`
- `3 chunks`

In legacy fallback paths it can also carry a chunk order or chunk id suffix.

### What happens if source projection fails?

Twin returns the answer with:

```json
"answer_status": "source_projection_failed"
```

and an empty sources list. This tells the UI to show a sources-unavailable cue instead of pretending the answer had no sources.

## 15. Common Operator Questions

### Why did I get an answer with no sources?

Check `answer_status`:

- `no_retrieval`: the selected mode does not produce sources by design, for example `bypass`, `only_need_context`, or `only_need_prompt`.
- `insufficient_information`: retrieval found no usable context.
- `source_projection_failed`: an answer was produced, but sources could not be projected.
- `query_failed`: backend failure during streaming.

### Why are scores not always strictly decreasing?

Scores come from source/chunk metrics when present, before final reranker ordering. When a reference contains multiple chunks, Twin uses the first available metric from matching chunks. In other paths, graph rows may propagate the maximum available source score for a chunk. Reranking, graph aggregation, and fallback scoring can all affect visible ordering.

### Why can a lower score still answer better than a higher score?

Vector similarity measures closeness to the query embedding, not completeness of answer. A lower-scored source may contain a decisive procedure step, while a higher-scored source may contain broader context.

### Why did a filtered query return fewer sources than `top_k`?

Possible reasons:

- fewer documents/chunks satisfy the folder, document, tag, and score filters;
- exact filtered cosine search found fewer candidates above `min_score`;
- source projection removed references that could not be validated;
- LightRAG retrieval selected fewer references than the requested UI limit.

### Why can a graph result mention context outside a folder?

Entity and relationship selection is folder-scoped, but LightRAG graph text can be aggregated from multiple documents at extraction time. If the same entity was built from multiple sources, the selected entity payload may contain blended descriptions. Chunk-level sources remain the stronger evidence for strict folder/document inspection.

### Should operators use `hybrid` or `mix`?

For most WebUI use, `mix` is the safest default because it preserves graph retrieval when available and still allows chunk evidence to surface when graph rows are sparse. `hybrid`, `local`, and `global` are useful for targeted graph-retrieval behavior. `naive` is useful for direct text similarity. `bypass` is not a sourced KMS mode.

## 16. Implementation Reference Map

| Topic | Reference |
| --- | --- |
| Twin query route and answer/source contract | `src/twindb_lightrag_memgraph/server/query/router.py` |
| LightRAG envelope adapter and source score projection | `src/twindb_lightrag_memgraph/server/_lightrag_compat.py` |
| Memgraph vector search, cosine scoring, folder/filter scoping | `src/twindb_lightrag_memgraph/vector_impl.py` |
| Retrieval filters context model | `src/twindb_lightrag_memgraph/_constants.py` |
| WebUI retrieval source type and query modes | `lightrag_webui_twin/src/types/retrieval.ts` |
| WebUI parameter panel | `lightrag_webui_twin/src/components/RetrievalTab.tsx` |
| Twin intelligence config and thresholds | `src/twindb_lightrag_memgraph/intelligence/config.py` |
| Cognitive reranker | `src/twindb_lightrag_memgraph/intelligence/features/cognitive_reranker.py` |
| Ontology pipeline | `src/twindb_lightrag_memgraph/intelligence/ontology/pipeline.py` |
| Ontology extraction step | `src/twindb_lightrag_memgraph/intelligence/ontology/steps/extract.py` |
| Ontology enrichment step | `src/twindb_lightrag_memgraph/intelligence/ontology/steps/enrich.py` |
| DSEP operators | `src/twindb_lightrag_memgraph/intelligence/ontology/dsep.py` |
| Twin prompt files | `src/twindb_lightrag_memgraph/intelligence/prompts/` |

## 17. Short Glossary

| Term | Meaning |
| --- | --- |
| LightRAG workspace | Physical Memgraph namespace used in labels such as `Vec_{workspace}_chunks`. |
| Twin folder | User-facing logical scope enforced through document membership. |
| Source | A projected WebUI citation entry derived from LightRAG references. |
| Chunk | Text unit stored in the chunk vector DB. |
| Entity/relation vector DB | Vector storage for LightRAG graph records. |
| `reference_id` | LightRAG reference number used for citations. |
| `score` | Source retrieval score, normally cosine similarity on `0..1`. |
| `rerank_score` | Optional LLM relevance score on `0..10` inside Twin intelligence. |
| `min_score` | User-configured retrieval/source score floor on `0..1`. |
| DSEP | Domain-Specific Extraction Profile for ontology extraction prompts. |
