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
support. Provider results must have one finite numeric vector of the declared
dimension per input text. Invalid output raises before writes rather than
persisting malformed vectors.

If a query finds a missing vector index, it attempts recreation and retries
once, reusing the query embedding. Recreation failures and retry failures now
propagate to both vector and hybrid callers. Applications should handle these
as errors and expose an appropriate retry/failure path. They must not rely on
an empty hit list to hide unavailable indexes.

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
