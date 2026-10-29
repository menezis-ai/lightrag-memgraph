# Upgrade to lightrag-memgraph 0.7

0.7.0 is an unreleased candidate that supersedes the unpublished 0.6.2
preparation. The published 0.6.1 format has ambiguous storage identities. This
upgrade changes physical labels and index names and refuses unsafe startup.

## New identity and ownership contract

Every backend encodes `(kind, workspace, namespace)` with ASCII component
lengths: `LRMG2_<KIND>_w<length>_<workspace>_n<length>_<namespace>`.
Kinds are `KV`, `VECTOR`, `STATUS` and `GRAPH`. Vector/text index roles are
`VECTOR_INDEX` and `TEXT_INDEX`. Length framing distinguishes `(a, b_c)` from
`(a_b, c)` even when a component contains apparent length markers. Namespace
now scopes document status and graph data too. Logical workspace/namespace
attributes and document IDs remain application-facing values.

`LRMG2_REGISTRY` stores `{id, format: 2, kind, workspace, namespace}` for each
physical label, with a unique `id` constraint. Back it up with application data.
It remains after `drop()` so an empty store can reopen safely. Do not manually
edit/delete ownership records or add application labels beginning `LRMG2_`.
Entity types with that prefix (ignoring case/leading whitespace) are rejected
before graph I/O. Reserved-looking logical workspace/namespace names are valid:
they are encoded and cannot address the registry or another physical store.

Before registering a new identity, initialization inspects legacy labels and
vector/text index names, including empty indexes. Legacy nodes already carrying
a verified v2 ownership label are recognized as v2 entity-type aliases. Other
legacy data/indexes raise `LegacyStorageError`; unknown data/indexes at a v2
target raise `StorageIdentityError`. Malformed registry records also fail closed.
The checks run before application DDL or data access. No legacy data or index
is renamed, rewritten, dropped or automatically attributed to a tenant.

## Migration procedure

1. Stop ingestion and all old-version writers. Keep the source database and a
   tested backup/snapshot, including schema/index definitions. A rolling upgrade
   with old writers sharing the target database is unsupported.
2. Against a snapshot or the paused source database, run the read-only inventory:

   ```bash
   export MEMGRAPH_URI=bolt://source-host:7687
   python -m lightrag_memgraph.migration --output inventory.json
   ```

   This reads labels/counts, schema and ownership records; it creates no registry
   or index and writes only the requested local JSON file. A label marked
   `legacy-or-application` is evidence to investigate, not a tenant assignment.
3. Resolve ownership from original source manifests and backups. Concatenated
   labels and document IDs cannot prove which tenant wrote a record. Previously
   overwritten values cannot be reconstructed by this adapter. If attribution
   remains uncertain, keep those records isolated for investigation.
4. Create a fresh Memgraph database/deployment (recommended), or explicitly choose
   a fresh workspace after verifying that none of its legacy or v2 labels/indexes
   exists. Install the reviewed 0.7 wheel or Git revision in a clean environment:

   ```bash
   pip install "git+https://github.com/menezis-ai/lightrag-memgraph.git"
   ```

   The same-named PyPI project is a separate integration. Use
   `from lightrag_memgraph import register; register()` before creating LightRAG.
   Re-ingest each tenant's trusted source documents into the intended workspace.
   Do not copy ambiguous legacy nodes into v2 labels or fabricate registry rows.
5. Compare source manifests, document/status counts, graph relations, persisted
   vectors, retrieval and full document deletion. Include identical IDs in two
   workspaces and verify the other workspace survives deletion. Retain the old
   database and backup for rollback. Switch application traffic only after this
   qualification; do not point the older adapter at v2 data.

There is no unsafe legacy mode and no in-place migration command. A fresh
workspace deliberately exposes only its newly ingested data; it does not claim
that historical data moved. Operators retain the old source for comparison.
Workspace scoping is not authorization; database credentials/access controls
must still protect independent tenants. Enterprise permissions and HA failover
remain outside the Community matrix.

## Initialization and connection lifecycle

`await rag.initialize_storages()` verifies every backend. Direct storage-class
users must call `await storage.initialize()` before reading, writing or dropping
data. Failed initialization leaves access disabled. `finalize()`, shared pool
closure and workspace/namespace/connection changes invalidate cached verification.
To change connection settings, finalize instances and close shared drivers first,
then create/initialize the storages against the new configuration. Never mutate
connection environment variables while operations are running. The registry is
provenance for cooperating adapter clients, not protection against arbitrary
Cypher writers or administrators.

Concurrent initialization of the same identity uses a uniqueness constraint and
bounded retries for transaction conflicts. This does not resolve the separate
shared-driver ownership issue across simultaneously active event loops (#27).

## Retrieval behavior changes

Document and query embeddings now call the LightRAG `EmbeddingFunc` wrapper,
preserving dimension/token injection and concurrency wrappers. Asymmetric models
receive `context="query"` or `context="document"` when their wrapper advertises
support. Query embeddings enter LightRAG's queue at `DEFAULT_QUERY_PRIORITY`
(5 on supported 1.4 versions); ingestion retains the queue's normal priority.
This prioritizes waiting queries without preempting running provider calls.
Provider results must have one finite numeric vector of the declared
dimension per input text. Invalid output raises before writes rather than
persisting malformed vectors; adapter errors include actual counts, dimensions
or the invalid coordinate.

LightRAG installs its priority/concurrency scheduler around the provider when
constructing a `LightRAG` instance. If constructing a storage class directly,
use the same assembly: `EmbeddingFunc(func=priority_limit_async_func_call(...)(provider),
embedding_dim=...)`, and shut down the scheduled function on application exit.
The scheduler consumes `_priority`; a bare, unscheduled `EmbeddingFunc` forwards
it to the provider, which must accept it. Supplying `query_embedding` avoids
provider calls entirely, including during index recovery.

Document preparation now respects `embedding_batch_num` (default 10), with at
most `embedding_func_max_async` batches (default 8) outstanding per upsert.
Both settings must be positive integers. The LightRAG queue still enforces the
shared provider concurrency limit across upserts. All batches are prepared and
validated before acquiring a Memgraph write slot; failures or cancellation
prevent that upsert from writing partially prepared data. Slow model calls do
not occupy the slots shared by KV, status, vector and graph writes.

If a query finds a missing vector index, it attempts recreation and retries
once, reusing the query embedding. Recreation failures and retry failures now
raise `RuntimeError("Vector index <name> recovery failed")` to vector and hybrid
storage callers, with the original failure in `exception.__cause__`. This
changes the exception type for recovery failures: applications previously
catching `neo4j.exceptions.ClientError`, `AuthError`, `Forbidden` or
`ServiceUnavailable` need to inspect `__cause__` inside that `RuntimeError`.
Unrelated initial search failures retain their original type. The adapter does
not log a second exception traceback before raising. Hybrid failures and caller
cancellation cancel and await the sibling search before returning.

Qualification with real storage and naive retrieval on LightRAG 1.5.7 checks
three distinct boundaries: the storage call raises with its index name and
chained Neo4j cause; `aquery_llm()` catches that exception and returns
`status="failure"` with the index in `message`; its legacy `aquery()` wrapper
returns only the response content, which is `None` on this failure. Applications
needing structured error reporting should inspect `aquery_llm()` status/message.
The Python exception/cause is not retained in that serialized response.

LightRAG 1.4.x may catch adapter errors earlier inside `_get_vector_context` and
return empty context; the adapter's propagation does not change that upstream
behavior. Applications retaining those versions must qualify their own query/
error path. Do not interpret an empty upstream response as proof of a healthy
backend.

