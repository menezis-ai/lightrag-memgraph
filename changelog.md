# Changelog — twindb-lightrag-memgraph (version BNPP)

Distribution complète incluant L1 (storage), L2 (server) et L3 (intelligence).

---

## v0.4.5 — 2026-04-08

### Features
- **Replica-aware retry profile** : détection automatique des erreurs SYNC replica ("At least one SYNC replica has not confirmed") dans `retry_transient()`. Escalade dynamique vers un profil dédié : 20 tentatives, 2s base delay, 30s cap (vs 6/50ms/2s pour MVCC). Configurable via `MEMGRAPH_REPLICA_RETRIES` (défaut 20) et `MEMGRAPH_REPLICA_RETRY_DELAY_MS` (défaut 2000ms).

### Tests
- 460 tests (11 nouveaux — détection réplicat, escalade profil, config readers), 0 régression.

---

## v0.4.4 — 2026-04-08

### Fix
- **Auto-create vector index on query** : si le vector index est absent au moment d'une requête (restart Memgraph, réplicat en retard, `initialize()` échoué), `query()` le recrée automatiquement via `_ensure_vector_index()` et retente la recherche. Plus de `[no-context]` silencieux sur index manquant.
- **Retry sur création vector index** : `initialize()` utilise `retry_transient()` pour la création du vector index (TransientError possible sur grosses bases lors de l'indexation initiale).
- **`_ensure_vector_index()` réutilisable** : méthode extraite, utilisable depuis `initialize()` et `query()`, avec gestion idempotente ("already exists" silencieux, erreurs réelles en ERROR + raise).

### Tests
- 449 tests (3 nouveaux — auto-create, fallback, initialize), 0 régression.

---

## v0.4.3 — 2026-04-07

### Fix
- **MVCC TransientError retry** : Memgraph en SNAPSHOT_ISOLATION provoque des `TransientError` (GQL 50N42) lors de writes concurrents sur le même label index. Nouveau module `_retry.py` avec `retry_transient()` — exponential backoff (50ms base, 6 attempts, 2s cap, full jitter). Appliqué aux **11 write paths** : KV/Vector/DocStatus upsert+delete, `_BufferedGraphProxy` flush nodes+edges, `batched_delete`+`batched_delete_by_ids`. Corrige le bug où 100% des docs d'un batch parallèle échouaient et rien ne persistait en base.

### Features
- **Retry configurable** : `MEMGRAPH_RETRY_MAX_ATTEMPTS` (défaut 6) et `MEMGRAPH_RETRY_BASE_DELAY_MS` (défaut 50ms) pour adapter la stratégie de retry selon le profil de charge.

### Tests
- 446 tests (24 nouveaux — helper retry + config + wiring par backend), 0 régression.

---

## v0.4.2 — 2026-03-30

### Fix
- **query_embedding forwarding** : LightRAG >= 1.4.11 pré-calcule les embeddings en batch et passe `query_embedding=` à `_get_node_data`. Le patch `_fused_get_node_data` n'acceptait pas ce kwarg, provoquant des échecs silencieux ("No relevant context found"). Ajout de `query_embedding=None` avec forwarding vers `entities_vdb.query()`. Rétro-compatible avec les versions antérieures.

### CI
- Matrice étendue à LightRAG 1.4.11 et 1.4.12.
- Ajout du test de régression `TestQueryEmbeddingForwarding`.

---

## v0.4.1 — 2026-03-27

### Fix
- **Budget per-database** : `SHOW STORAGE INFO` est instance-wide mais `MEMGRAPH_MEMORY_LIMIT` est per-database. Nouveau `estimate_database_usage()` interroge les nodes/edges réels de la base courante au lieu des métriques globales. Corrige le blocage d'inserts sur bases vides.

### Features
- **Dual LLM config** : variables `TWIN_RAG_INDEXING_*` pour l'indexation documents (GPU prod), séparées de `TWIN_RAG_LLM_*` pour le chat (GPU dev). Fallback automatique sur la config chat si non défini.
- **FeedbackStore wiring** : ajout de `trace_id` dans `QueryTrace`, `record_feedback()` sur le moteur.
- **NODE_TYPES dans le prompt** : les types sont injectés dynamiquement dans le prompt d'extraction (plus de hardcode). Validation des types dans l'étape ontology validate.

### Cleanup
- Suppression de code mort : import `QueryParam` inutilisé, constante `RELATION_PROPERTIES`, fixtures de test orphelines.
- Ajout de `.env.example` à l'export ZIP BNP.

### Tests
- 421 tests, 86% coverage, SonarQube quality gate passed (90.1% new code coverage, 0 new violations).

---

## v0.4.0 — 2026-03-26

### Features
- **Memory budget enforcement** : `MEMGRAPH_MEMORY_LIMIT` + `MEMGRAPH_BUDGET_ENFORCE` (off/warn/reject). Vérification pré-insert via `SHOW STORAGE INFO` pour les déploiements à budget contraint (10 GB / 75 GB).
- **Batched deletes** : toutes les méthodes `drop()` utilisent une boucle `LIMIT` pour éviter les OOM (`_batched_ops.py`).
- **TTL natif** : support TTL Memgraph Enterprise via label `:TTL` + propriété `ttl` (`_ttl.py`).
- **Lazy full_docs** : write-then-purge + reconstruction depuis les chunks (`_lazy_full_docs.py`). ~35-45% d'économie stockage avec `MEMGRAPH_PURGE_FULL_DOCS=on`.
- **Vector quantization** : `scalar_kind=f16` par défaut (50% d'économie mémoire vecteurs).
- **Memory introspection** : `get_memory_usage()`, `estimate_insert_cost()`, `check_memory_budget()`.

### Tests
- 341 tests (51 nouveaux), 92% coverage L1, SonarQube triple-A, 0 new smells.

---

## v0.3.1 — 2026-03-19

### Fix
- **Bulk indexation silencieuse** : ajout de `await result.consume()` sur tous les `session.run()` — les erreurs étaient silencieusement ignorées.
- **Write throttle** : le flush passe par `_pool.get_session()` + `acquire_write_slot()` au lieu d'un accès direct au driver (600 flushes concurrents saturaient le pool Bolt de 50 connexions).
- **Error propagation** : ajout de try/except avec logging dans `flush()` — les échecs remontent au lieu d'être avalés.
- **result.consume()** sur KV upsert (chemin d'écriture le plus fréquent), tous les `CREATE INDEX`, les appels `USE DATABASE`, et les `drop()`.
- **Vector query crash** : catch "index does not exist" dans `query()`, retourne des résultats vides au lieu de crasher (corrige l'erreur `vec_base_relationships`).
- **Backward compat LightRAG 1.4.9–1.4.10** : guards `getattr`/`hasattr` pour `multimodal_processed` et `_validate_embedding_func`.

### CI
- Matrice de compatibilité complète : Python 3.10–3.13 × LightRAG {1.4.9, 1.4.9.11, 1.4.10} × Memgraph MAGE {3.7.2, 3.8.0, latest}.

### Tests
- 19 tests de régression (`test_buffered_writes.py` + `test_consume_and_drop.py`).
- `test_vector_missing_index.py` pour le crash sur index absent.

---

## v0.3.0 — 2026-03-17

### Features (L2 — Server)
- **FastAPI server layer** : couche HTTP complète pour exposer LightRAG + Memgraph en tant qu'API.
  - Routes : `/health`, `/query`, `/insert`.
  - **Chunk expansion** : `/chunks/*/context`, `/chunks/*/document`, `/documents/*/chunks`.
  - **Dual auth** : API key statique + JWT (compatible agent CFT).
  - **LangSmith distributed tracing** : spans LLM, embedding et rerank.
  - `full_doc_id` extraction dans la réponse `/query`.
- **135 tests serveur** : HTTP-level tests pour toutes les routes, auth edge cases, tracing, chunk routes, lifespan.

---

## v0.2.5 — 2026-03-12

### Fix
- **meta_fields KeyError** : correction du crash lors de l'accès à des champs manquants dans les métadonnées vectorielles.
- **DocStatus JSON deserialization** : correction de la désérialisation des statuts de documents.
- **Vector upsert** : ajout de `result.consume()` manquant.
- **test_query_e2e** : correction des collisions de workspace et du mismatch de dimensions en CI, seuil cosine à 0, délai de sync de l'index vectoriel.

---

## v0.2.0 — 2026-02-09

### Features (L3 — Intelligence)

#### Intelligence Engine
- **ReAct agent** : pipeline Observe → Reason → Act avec boucle cognitive pour les requêtes complexes.
- **Intent classifier** : classification automatique des requêtes (factual, analytical, exploratory, procedural).
- **Cognitive reranker** : re-scoring LLM des résultats RAG avec prise en compte du contexte de la requête.
- **Query expander v1** : expansion des requêtes via thésaurus JSON (IT Ops, 314 entrées).
- **Feedback store** : collecte et stockage des retours utilisateur sur les réponses.
- **Prompts** : 5 fichiers de prompts système (domain, intent, reason, reranking, synthesis).
- 42 tests, installable via `pip install .[intelligence]`.

#### Ontology Pipeline
- **Pipeline 4 étapes** : Extract → Cluster → Enrich → Validate. Construction d'un knowledge graph dans Memgraph depuis les documents.
- **3 modes** : dedicated, emergence, deep_extraction.
- **DSEP** (Domain-Specific Extraction Profiles) : 6 opérateurs injectés comme directives de prompt pour l'extraction contrôlée d'entités.
- **OntologyStorage** : persistance Memgraph (labels `Onto_{ws}`), traversée de graphe pour l'expansion de requêtes, données normatives de seed.
- **QueryExpander v2** : expansion graph-based via Cypher, check `has_data()` léger dans le hot path, fallback v1 thésaurus JSON.
- **pipeline.approve(result, workspace)** : dry-run par défaut, revue humaine avant MERGE.
- **Config JSON** : pas de dépendance PyYAML, feature désactivée quand `ontology.json` absent (zero behavior change).
- **Dual-pass extraction** : opt-in via `dual_pass: true` dans `ontology.json`. Pass global (structure haut-niveau) + pass local (entités précises depuis les chunks). Merge avec déduplication (highest confidence wins).

#### Workspace Router (F06)
- **Nexus Router** : résolution de workspaces Memgraph via waterfall cascade : L4 override > TopologyContext > Keyword match > Default.
- Regex pré-compilées pour un hot-path < 1ms.
- 8 règles MVP : Oracle, Linux, Windows, Network, Middleware, Containers, ITSM, Monitoring.
- Rétro-compatible : `workspace=` explicite bypass F06.

### Tests
- 104 tests (62 nouveaux ontologie + 42 intelligence existants).
- Tests d'intégration ontologie (`test_ontology_integration.py`, 529 lignes).
- 93% coverage, 0 SonarQube issues.

---

## v0.1.1 — 2026-02-20

### Features
- **Memgraph Enterprise multi-database** : support `USE DATABASE` via `_pool.get_session()`. Memgraph Enterprise ne supporte pas le paramètre Bolt `database=` (GQL 50N42).
- **TLS** : `MEMGRAPH_ENCRYPTED=true/false` pour activer TLS sur le driver Bolt. `MEMGRAPH_TRUST=TRUST_ALL` (défaut) ou `TRUST_SYSTEM_CA`.
- **_SafeDriverWrapper** : wraps le driver MemgraphStorage built-in, strip `database=` des appels `session()` (évite GQL 50N42 sur Coordinator) et injecte `USE DATABASE`.
- **Version print** : affichage de la version au `register()` pour visibilité DevOps.
- **Logging explicite** : logs des rôles de chaque backend storage à l'enregistrement.

---

## v0.1.0 — 2026-02-08

### Initial Release
- **3 backends storage** enregistrés via monkey-patching des registries LightRAG :
  - `MemgraphKVStorage` : paires clé-valeur comme nœuds Cypher.
  - `MemgraphVectorDBStorage` : recherche vectorielle native Memgraph (MAGE >= 3.2).
  - `MemgraphDocStatusStorage` : suivi du statut de traitement des documents.
- **Pool de connexions async** : gestion de driver event-loop-aware.
- **Buffered graph proxy** : écriture en batch avec flush configurable.
- **Skip USE DATABASE** pour la base `memgraph` par défaut (compatibilité Community).
- **SonarQube** : 0 issues (fix de 11 code smells initiaux).
- **Production checklist** : documentation de la limitation du shared driver pool.
- Licence MIT.

### Tests
- 33 tests fonctionnels + 29 benchmarks de performance.
- Tests prod checklist et pré-prod (threshold default, status resilience, batch upserts).
