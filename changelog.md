# Changelog — twindb-lightrag-memgraph (0.5.x stable)

Stable branch : LTS 0.3.2 + auto-create vector index on query.
**Zéro import, zéro ligne du 0.4.x**.

---

## v0.5.1 — 2026-04-16

### Fix
- **Auto-create vector index on query** : si le vector index est absent au moment d'une requête (restart Memgraph, réplicat en retard, `initialize()` a échoué silencieusement), `query()` crée l'index à la volée et retente la recherche une fois. Plus de `[no-context]` silencieux sur index manquant.
- **`_create_vector_index()` extrait** : méthode idempotente réutilisée par `initialize()` et `query()`. Si l'index existe déjà, log DEBUG et continue ; si autre erreur, propage.
- **`get_docs_paginated` perf — probe 502 fix** : le tri `ORDER BY n.updated_at` forçait un full scan (pas d'index sur les champs sortables) → query > 60s → nginx timeout → 502 Bad Gateway sur `/documents/paginated` qui faisait crasher le front LightRAG. Deux fixes : (1) indexes `updated_at` et `created_at` créés par `initialize()` → sort O(log n). (2) Count et fetch tournent en parallèle via `asyncio.gather` sur deux read sessions — divise le temps par ~2 sur grosses bases.

### Tests
- 191 tests unitaires (1 nouveau sur l'auto-create, tests ajustés pour les 6 indexes DocStatus et la parallélisation paginated), 0 régression.
- **Probe e2e regression tests** (`tests/test_probe_e2e.py`, 7 nouveaux tests) — seed 10/100/500 docs contre un Memgraph réel, vérifie les 6 indexes via `SHOW INDEX INFO`, asserte les budgets de latence (150ms / 300ms / 1s), valide sort DESC, pagination sans overlap, filtre par status. Si un dev futur casse l'index `updated_at` ou re-sérialise count+fetch, CI rouge immédiat. Picked up automatiquement par le job `integration-tests` existant.
- **260 tests au total** contre Memgraph réel (unit + integration + probe e2e), 0 régression.

---

## v0.3.2 — 2026-04-08 (LTS)

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
- Replica-aware retry profile

### Tests
- 190 tests unitaires, 0 régression.
