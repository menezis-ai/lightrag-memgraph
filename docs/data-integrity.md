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

This corrects a legacy inconsistency: LightRAG's standard `workspace=` could
separate its graph while the extension's other three stores still wrote to
`base` or an environment-selected namespace. It also removes the old graph
constructor's priority for the global `MEMGRAPH_WORKSPACE` over an explicit
argument. Check the intended workspace configuration before upgrading a deployment
that relied on that override. Single-workspace environment-only setups retain
their existing label names.

The change does not move previously written data. Before an affected deployment
resumes ingestion, inventory labels and indexes with read-only Cypher:

```cypher
MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label;
SHOW INDEX INFO;
```

Compare the graph's workspace labels with `KV_<workspace>_<namespace>`,
`Vec_<workspace>_<namespace>` and `DocStatus_<workspace>`. Historical records in a
shared `base` namespace cannot be attributed safely to tenants solely from their
label or document ID: IDs may already have collided. Use original source manifests
and backups to determine ownership. Migration or re-ingestion requires a separate,
reviewed plan; changing constructor priority does not repair historical collisions.

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
The adapter rejects empty values, backticks and NUL before that write path while
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
