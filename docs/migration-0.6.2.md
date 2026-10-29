# Upgrade preparation for lightrag-memgraph 0.6.2

0.6.2 is a release candidate under preparation, not a published release. The
latest published release remains 0.6.1. Publication is held while the P1
[workspace label collision](https://github.com/menezis-ai/lightrag-memgraph/issues/21)
remains unresolved.

## Package and import

Use a clean virtual environment when upgrading from a differently named
distribution; avoid loading two registrations into the same Python process.
Install this repository from GitHub or a reviewed GitHub release wheel. The
same-named PyPI distribution is a separate integration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install "git+https://github.com/menezis-ai/lightrag-memgraph.git"
```

Update application imports and any module paths in configuration:

```python
from lightrag_memgraph import register

register()
```

Restart long-running processes after upgrading. `register()` changes process-wide
LightRAG registries and patches, and is not a hot-reload/unregister mechanism.

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

## Existing data

No database labels, IDs, vector index names or embeddings are migrated or
rewritten by this release. Back up the database and check workspace ownership
before resuming ingestion. The constructor precedence fix from 0.6.0 does not
resolve ambiguous label encoding: `(a, b_c)` and `(a_b, c)` can still collide,
and raw graph workspace labels can overlap storage labels. See the
[requalification evidence](data-integrity.md#label-collision-requalification).

Use the [fixture lifecycle](tutorial.md) on a disposable server to check the
installation, then qualify your model configuration and application error
handling against a copy of your data. Enterprise authorization and HA failover
remain outside the Community test matrix.
