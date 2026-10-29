# Storage integrity qualification

The package supplies storage backends. It includes no intelligence engine,
ontology, retrieval/synthesis overlay, authorization layer or answer-quality
guarantee. Distribution checks verify that no such runtime modules are packaged.

## Workspace migration

`register()` makes graph, KV, vector and document-status constructors resolve
workspaces identically: explicit argument, global configuration, Memgraph-specific
environment alias, LightRAG environment variable, then `base`. No constructor
temporarily changes process environment variables. Invalid explicit values raise
instead of falling back to a shared namespace.

Version 0.7 additionally replaces physical identities for all four stores with
length-framed workspace/namespace components and distinct role prefixes. Graph
and document-status namespaces now participate in isolation. A persistent registry
verifies ownership; initialization refuses legacy data/indexes and unclaimed v2
targets before application I/O. Follow the [migration guide](migration-0.7.0.md)
for the read-only inventory, backup, ownership review and explicit re-ingestion.
There is no automatic in-place migration or legacy opt-out.

## Label collision requalification

The historical main revision `22f1057e02cd21acc2cd0841abdd5ade1de74196` was
requalified on 2026-10-29 against disposable Memgraph MAGE 3.14.0. KV
`(a, b_c)` / `(a_b, c)` cross-read and overwrote records. A graph workspace
matching `KV_<workspace>_text_chunks` could drop a KV record. That reproduction
reopened [issue #21](https://github.com/menezis-ai/lightrag-memgraph/issues/21).

See the pinned [0.7 qualification evidence](storage-identity-qualification.md).
The regressions in `tests/test_storage_identity_integration.py` use actual
storage classes and identical IDs on a real server. They cover underscore
boundaries, namespaces in all four stores, graph workspaces matching legacy/new
storage labels and the registry name, graph type aliases, cross-read/overwrite/
delete/drop, vector retrieval after another scope is dropped, and 24 concurrent
initializers retaining one owner. Legacy data and empty indexes refuse startup
without mutation; legacy ANN remains searchable. Unknown v2 data/indexes and
invalid registry records are refused. Explicit fresh-workspace re-ingestion and
a read-only inventory are also exercised. Pure tests cover the reserved entity
prefix, 400 distinct kind/workspace/namespace combinations and invalidation when
configuration or pool lifetime changes. These checks do not provide authorization
against other Cypher clients or concurrent legacy writers.

## Document deletion

Document status stores `chunks_list` and `metadata` as JSON properties. All raw
record read paths decode them to their declared list/dict types. Point reads used
by LightRAG deletion fail on invalid JSON or a wrong decoded type, preserving the
record for diagnosis instead of treating unreadable evidence as an empty chunk set.
UI/status hydration has its separate documented relaxed-read semantics.

The integration regression exercises actual `ainsert`, `aquery` and
`adelete_by_doc_id` with native Memgraph persistence and deterministic LLM/embedding
fixtures. It verifies document/status
absence, missing KV chunks, missing chunk/entity/relationship vectors, empty graph
data and absence of native chunk-search hits after deletion. A second tenant with
identical IDs remains intact and searchable. Another test corrupts the stored
chunk-list JSON: deletion must fail while the full document, chunks, vectors and
status record remain present.

The native zero-hit assertion runs on Memgraph 3.4 and later. Memgraph 3.2 is
checked for physical deletion but retains the documented ANN deletion/churn
limitation described in the [deployment notes](../README.md#upgrading-an-existing-memgraph-deployment).

LightRAG 1.4.9 uses a process-wide ingestion busy flag. Concurrent `ainsert`
calls for separate tenants can leave the second tenant pending; its queue must
be processed afterward with `apipeline_process_enqueue_documents()`. The minimum
dependency test explicitly drains these queues before checking persisted data.
It does not demonstrate independent concurrent ingestion in that upstream
release. Early upstream deletion handlers can also raise an exception while
reporting a corrupt status read; current releases return a failure result.
Both paths are checked for preservation of the stored data.

For a historical deletion incident, inspect the affected workspace on a snapshot
or with ingestion paused. For example, read derived vector ownership without
deleting anything (replace the example workspace explicitly):

```cypher
MATCH (v:`Vec_example_chunks`)
OPTIONAL MATCH (s:`DocStatus_example` {id: v.full_doc_id})
WITH v, s
WHERE s IS NULL
RETURN v.id AS vector_id, v.full_doc_id AS claimed_document_id;
```

Treat these as candidates for investigation, not an automatic deletion list:
in-flight ingestion, older schemas or missing provenance may need separate
interpretation. Compare KV chunk JSON and the original document manifest with
each candidate. No production data scan, repair or deletion was performed by this
qualification, and no original deployment/source-document evidence was supplied.

## Concurrency, identifiers and failure reporting

Document-level buffered graph writes remain removed. Real graph persistence stays
inside LightRAG's entity/relation locks; the existing shared-entity and
shared-relationship regressions still exercise actual upstream merges. This does
not provide a distributed lock spanning independent deployments sharing one DB.

Legacy LightRAG versions interpolate `entity_type` into quoted Cypher labels.
The adapter rejects empty values, backticks, NUL and the reserved `LRMG2_` prefix before that write path while
preserving spaces, punctuation and Unicode that are safe inside quoted labels.
Values in node properties are not rewritten. Graph workspace identifiers use the
same stricter validation as the other stores.

KV writes and graph initialization/probes explicitly consume their results.
Deferred KV/probe failures propagate instead of being left to best-effort session
closure. Graph index-creation failures retain their existing warning behavior;
they are now observed during result consumption too.
Read-pool sessions request `READ`, including when using a routing URI. Routing
parameters and failure paths are unit-tested; live Enterprise authorization,
replica failover and HA behavior remain outside the Community matrix.
