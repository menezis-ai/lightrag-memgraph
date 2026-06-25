# Test doctrine — Graph = contract, not screen

> The Knowledge Graph is the most fragile **and** the most central surface of the Twin product. Treat it as a backend contract, not as a UI screen.

## Why this doctrine exists

The graph wires together more components than any other surface:

- vector retrieval (`chunks_vdb`)
- doc status (`MemgraphDocStatusStorage`)
- citations + parent-doc resolution (`source_docs`)
- folder request binding (the operator-facing guard)
- MSW fixtures **and** real backend (dual mode, both must work)
- TanStack Query cache invalidation across cascade deletes

That density of integration is exactly where contract drift compounds. UI-only testing misses backend regressions; backend-only testing misses cache invalidation regressions. Neither is sufficient on its own.

## The four sensitive axes

Each axis has already produced at least one regression. They are the contract surface that tests must cover.

### 1. Front cache

TanStack Query cache keys must:

- Refetch when the active **folder** changes. The graph is stored in one LightRAG/Memgraph workspace for the deployed KB, but graph reads are folder-scoped through document membership.
- Survive **cascade deletes** — when a doc is deleted, the graph nodes/edges tied to it disappear; the cache must invalidate the right entries without orphaning others.

Regression class fixed by: `505b5a1 fix(delete): cascade graph cache + DELETING UI overlay + status normalize`.

### 2. Seed fallback

`GRAPH_ENTITY_FIXTURES` (MSW) must still hydrate when the real backend is absent — this is what local dev, fixture-only review builds, and frontend e2e use. But the seed must **not** leak into prod-with-backend mode: `resolveRuntimeConfig()` decides at boot, and that decision must remain testable.

### 3. Folder binding

The Twin request contract is `X-Twin-Folder`. The graph remains stored under the single LightRAG/Memgraph workspace configured for the deployed KB (`MEMGRAPH_WORKSPACE` / `WORKSPACE`), but graph reads are folder-scoped by membership: a graph entity/relation is visible only when at least one `source_docs` chunk belongs to a document `MEMBER_OF` the active folder. Do not invent a per-folder graph label until LightRAG storage isolation actually exists.

**Test the negative case explicitly:** unknown or unauthorized folders must be rejected at the API boundary, switching folders must not reuse stale graph cache state, and querying folder A must return zero nodes/edges sourced only from folder B. A physically isolated folder tier would be a separate storage model, not an extension of the current membership relation.

### 4. `source_docs`

Graph nodes carry `source_docs` joins back to `DocStatus` for parent-doc resolution. The semantic must be **distinct parent documents**, not chunk count — confused for the chunk count once already.

Regression class fixed by: `9179e74 fix(graph): sources = distinct parent docs, not chunk count`.

Guard with deterministic fixtures (one parent doc, multiple chunks → assert `sources.length === 1`).

## The rule

Every PR touching:

- `src/twindb_lightrag_memgraph/server/graph_reader.py`
- `lightrag_webui_twin/src/components/GraphTab.*`
- `chunks_vdb` query path
- graph cache keys (`useGraph*`, `['graph-entities']`, `['graph-relations']`)

…must add **at least one end-to-end contract test** that exercises the full chain:

```
Cypher → API response shape → cache state → UI invalidation
```

### What does **not** count as a contract test

- A screenshot.
- A Cypher unit test that asserts only the query string is built correctly.
- A React component test mounted against `GRAPH_ENTITY_FIXTURES` only.
- A backend test that mocks the Memgraph driver.

### What **does** count

- A test that drives the real `graph_reader.py` against an in-memory Memgraph (or the integration-test container), hits the FastAPI route, asserts the response shape matches the TypeScript `GraphEntity` / `GraphRelation` contract, then asserts the WebUI cache is updated.
- A test that switches folder and asserts graph requests/refetches do not reuse stale cache from the previous folder.
- A test that deletes a doc and asserts the cascade removes the graph nodes **and** invalidates the corresponding cache key.

## Related

- `CLAUDE.md` § "Folders", "Folder membership model", and "Vrai Graph" — LightRAG `workspace` remains the physical storage namespace; Twin `folder` is the logical membership relation used for read cloisonnement.
- `WEBUI-WIRING-PLAN.md` — graph read/mutation status.
- `docs/test-doctrine-lightrag-compat.md` — the sibling doctrine on regression discipline.
