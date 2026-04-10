# Changelog — twindb-lightrag-memgraph (0.3.x LTS branch)

Stable LTS branch. L1 storage + L3 intelligence, **sans** les features 0.4.0
(memory budget, TTL, lazy full_docs, batched deletes, vector f16).

---

## v0.3.2 — 2026-04-08

LTS release cherry-picked from main.

### Fix
- **Bulk indexation silencieuse** (from `c70327e`) : ajout de `await result.consume()` sur tous les `session.run()` — les erreurs étaient silencieusement ignorées. Write throttle via `_pool.get_session()` + `acquire_write_slot()` au lieu d'un accès direct au driver. Error propagation dans `_BufferedGraphProxy.flush()`.
- **query_embedding forwarding** (from `6285a80`) : LightRAG >= 1.4.11 pré-calcule les embeddings en batch et passe `query_embedding=` à `_get_node_data`. Le patch `_fused_get_node_data` n'acceptait pas ce kwarg. Ajout de `query_embedding=None` avec forwarding vers `entities_vdb.query()`. Rétro-compatible.

### NOT included (vs main 0.4.x)
- Memory budget enforcement (`_memory.py`)
- Batched deletes (`_batched_ops.py`)
- TTL natif (`_ttl.py`)
- Lazy full_docs (`_lazy_full_docs.py`)
- Vector quantization f16 par défaut
- TransientError retry (`_retry.py`)
- Auto-create vector index on query
- Replica-aware retry profile

### Purpose
Branche de repli pour environnements BNP où les features 0.4.0 causent des régressions.
Zéro env var `MEMGRAPH_MEMORY_LIMIT`, `MEMGRAPH_BUDGET_ENFORCE`, `MEMGRAPH_TTL_*`,
`MEMGRAPH_PURGE_FULL_DOCS`, `MEMGRAPH_VECTOR_SCALAR_KIND` requise — comportement
identique à 0.3.1 avec les fixes critiques de production appliqués.

### Tests
- 190 tests unitaires, 0 régression.
