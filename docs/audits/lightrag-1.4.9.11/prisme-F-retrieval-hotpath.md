# Prisme F — Hot path retrieval LightRAG 1.4.9.11

Audit du chemin retrieval de LightRAG **tel qu'installé dans le venv local** (`/Users/julien/twindb-lightrag-memgraph/.venv/lib/python3.14/site-packages/lightrag/`) et confrontation avec le code du wrapper `twindb-lightrag-memgraph` (branche `feat/maquette-source-revalidation`). Toutes les références `$LRAG/...` ci-dessous désignent l'arborescence du venv ; les références sans préfixe (`src/...`) désignent ce repo.

## Avertissement de version

Le venv expose `lightrag.__version__ = "v1.4.10"` (cf. `$LRAG/__init__.py:3`). La cible BNP prod tourne en `1.4.9.11`. Pour ce prisme :

- la version du venv (1.4.10) sert d'oracle pour les signatures et l'arbre d'appels exact ;
- la chaîne `kg_query` / `naive_query` / `_build_query_context` / `_perform_kg_search` / `_apply_token_truncation` / `_merge_all_chunks` / `_build_context_str` / `process_chunks_unified` / `apply_rerank_if_enabled` existe à l'identique sur 1.4.9.11 (la matrice CI Forgejo `1.4.9` / `1.4.9.11` / `1.4.11` / `1.4.12` passe avec les mêmes patches `_fused_get_node_data` / `_fused_find_edges`, cf. `src/twindb_lightrag_memgraph/__init__.py:520-628`) ;
- le diff `1.4.9.11 -> 1.4.10` concerne principalement le pipeline d'indexation (entity extraction, source_ids limit) et n'a **pas modifié** la signature des dispatchers de retrieval ni l'ordre des appels dans le hot path. Aucun `_find_related_text_unit_from_*` ni `kg_query` n'a vu son ABI changer ;
- tout numéro de ligne `$LRAG/...:LLLL` ci-dessous est donc le numéro **1.4.10** ; en 1.4.9.11 la fonction est la même, à un offset de quelques lignes près lié aux changements en amont du fichier. Les patches du wrapper référencent les fonctions par **nom**, pas par ligne, donc l'asymétrie ne casse rien.

Lecteur visé : ingénieur LightRAG/Twin chargé de cabler l'intelligence layer L3 (`TwinRAGEngine`) sur le retrieval natif sans patcher le code source de LightRAG, et son relecteur architecte.

---

## 1. Trace exhaustive d'une query `mode="hybrid"`

### 1.1 Entrée publique

Le caller fait :

```python
result = await rag.aquery("Pourquoi ORA-04030 ?", param=QueryParam(mode="hybrid"))
```

`QueryParam` est défini en `$LRAG/base.py:84-169` :

- `mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"` (`$LRAG/base.py:88`) — la valeur par défaut est `mix`, pas `hybrid`.
- `top_k`, `chunk_top_k` (défaut env `TOP_K` / `CHUNK_TOP_K`).
- `max_entity_tokens`, `max_relation_tokens`, `max_total_tokens` — budgets dynamiques par sous-contexte (`$LRAG/base.py:117-130`).
- `hl_keywords`, `ll_keywords` — listes pré-fournies qui bypass l'extraction LLM si non vides (`$LRAG/base.py:132-136`).
- `conversation_history` (forme `[{"role": ..., "content": ...}]`) — passé tel quel au LLM final, **n'est pas** utilisé pour le retrieval (`$LRAG/base.py:138-142`).
- `model_func` — override LLM par requête, pour permettre des modèles différents par mode.
- `user_prompt` — instruction utilisateur injectée dans `PROMPTS["rag_response"]` ou `PROMPTS["naive_rag_response"]`.
- `enable_rerank: bool = (RERANK_BY_DEFAULT == "true")` — défaut `True` (`$LRAG/base.py:160-163`).

### 1.2 `aquery` → `aquery_llm` → dispatch

`LightRAG.aquery` (`$LRAG/lightrag.py:2459-2491`) est devenu un **simple wrapper de compat**. Il délègue à `aquery_llm` puis extrait `llm_response.content` ou l'iterator de stream :

```python
# $LRAG/lightrag.py:2459-2491
async def aquery(self, query, param=QueryParam(), system_prompt=None):
    result = await self.aquery_llm(query, param, system_prompt)
    llm_response = result.get("llm_response", {})
    if llm_response.get("is_streaming"):
        return llm_response.get("response_iterator")
    else:
        return llm_response.get("content", "")
```

`aquery_llm` (`$LRAG/lightrag.py:2721-2857`) est l'**entry point réel**. Il :

1. construit `global_config = asdict(self)` (`$LRAG/lightrag.py:2743`) — c'est ce dict qui transporte `tokenizer`, `llm_model_func`, `rerank_model_func`, `max_total_tokens`, `kg_chunk_pick_method`, etc. jusqu'aux helpers `operate.py` ;
2. dispatche sur `param.mode` :

   | `param.mode`               | Fonction appelée (`$LRAG/operate.py`)                     |
   |----------------------------|-----------------------------------------------------------|
   | `local`, `global`, `hybrid`, `mix` | `kg_query` (`$LRAG/operate.py:3084-3291`)        |
   | `naive`                    | `naive_query` (`$LRAG/operate.py:4827-5071`)              |
   | `bypass`                   | LLM direct sans retrieval (`$LRAG/lightrag.py:2770-2807`) |

3. appelle `await self._query_done()` (`$LRAG/lightrag.py:2811`, défini ligne 2882 : `await self.llm_response_cache.index_done_callback()`).
4. enveloppe le `QueryResult` en dict applatissable côté caller (`$LRAG/lightrag.py:2830-2842`).

Tout `try` qui échoue retourne un dict `status: "failure"` (`$LRAG/lightrag.py:2844-2857`) — pas d'exception remontée à `aquery`.

### 1.3 `kg_query` — pipeline mode `hybrid`

`$LRAG/operate.py:3084-3291` :

1. **Garde** (`:3126-3127`) : query vide → `QueryResult(content=PROMPTS["fail_response"])`.
2. **Sélection du LLM** (`:3129-3134`) : `param.model_func` sinon `global_config["llm_model_func"]` wrappé avec `partial(..., _priority=5)`.
3. **Extraction de mots-clés** (`:3136-3138`) : `get_keywords_from_query(query, param, global_config, hashing_kv)`.
   - Définie en `$LRAG/operate.py:3294-3323` ; si `param.hl_keywords` ou `param.ll_keywords` est déjà rempli, court-circuite l'appel LLM.
   - Sinon délègue à `extract_keywords_only` (`$LRAG/operate.py:3326-3433`).
4. **`extract_keywords_only`** (`:3326-3433`) :
   - examples concaténés depuis `PROMPTS["keywords_extraction_examples"]` (`$LRAG/prompt.py:398-431`) ;
   - `args_hash = compute_args_hash(param.mode, text, language)` puis cache lookup via `handle_cache(hashing_kv, args_hash, text, param.mode, cache_type="keywords")` (`:3344-3362`) ;
   - **cache hit** : `keywords_data = json_repair.loads(cached_response)` → retour direct ;
   - **cache miss** : `PROMPTS["keywords_extraction"]` (`$LRAG/prompt.py:374-396`) formaté avec `query=text, examples=examples, language=language` puis `use_model_func(kw_prompt, keyword_extraction=True)` (`:3385`) ;
   - parsing JSON via `json_repair.loads` (`:3390`) ;
   - cache write conditionnel (`:3403-3431`).
5. **Garde keywords vides** (`:3144-3153`) :
   - log warning si `ll_keywords == []` en mode `local|hybrid|mix` ;
   - log warning si `hl_keywords == []` en mode `global|hybrid|mix` ;
   - si les deux sont vides et `len(query) < 50` → `ll_keywords = [query]` (fallback brutal), sinon retourne `fail_response`.
6. **Construction du contexte** (`:3158-3169`) : `_build_query_context(query, ll_str, hl_str, kg_inst, ent_vdb, rel_vdb, text_kv, param, chunks_vdb)`.
7. **Short-circuit `only_need_context`** (`:3176-3179`) : si flag positionné, retour direct du contexte sans LLM final.
8. **Construction du system prompt** (`:3189-3194`) :
   - template = `system_prompt` paramètre ou `PROMPTS["rag_response"]` (`$LRAG/prompt.py:224-276`) ;
   - `.format(response_type=..., user_prompt=..., context_data=context_result.context)`.
9. **Short-circuit `only_need_prompt`** (`:3198-3200`) : retour du prompt assemblé sans appel LLM.
10. **Cache de la réponse** (`:3209-3234`) : hash sur 12 champs de `query_param`, lookup `handle_cache(..., cache_type="query")`.
11. **Appel LLM final** (`:3236-3242`) : `await use_model_func(user_query, system_prompt=sys_prompt, history_messages=param.conversation_history, enable_cot=True, stream=param.stream)`.
12. **Cache write** (`:3244-3268`).
13. **Post-processing string** (`:3271-3284`) : si réponse non-streaming, supprime les occurrences résiduelles du system prompt, des balises `<system>...</system>`, des marqueurs `user` / `model`.
14. **Retour** : `QueryResult(content=response, raw_data=context_result.raw_data)` ou streaming variant.

### 1.4 `_build_query_context` — la chaîne 4-stages

`$LRAG/operate.py:4118-4235`. Pipeline :

```text
Stage 1 : _perform_kg_search       → search_result {final_entities, final_relations, vector_chunks, chunk_tracking, query_embedding}
Stage 2 : _apply_token_truncation  → truncation_result {entities_context, relations_context, filtered_entities, filtered_relations, id_maps}
Stage 3 : _merge_all_chunks        → merged_chunks (round-robin vector ⨁ entity ⨁ relation)
Stage 4 : _build_context_str       → (context_string, raw_data dict)
```

#### Stage 1 — `_perform_kg_search` (`$LRAG/operate.py:3493-3659`)

- Pré-calcul de l'embedding query (`:3522-3538`) — un seul `embedding_func([query])` réutilisé par toutes les recherches VECTOR aval. `query_embedding=None` si la query est vide ou si ni `chunks_vdb` ni `kg_chunk_pick_method == "VECTOR"`.
- **Branchement mode** (`:3540-3591`) :

  | Mode     | Appels                                                                                          |
  |----------|-------------------------------------------------------------------------------------------------|
  | `local`  | `_get_node_data(ll_keywords, kg_inst, ent_vdb, param)` (`:3541-3547`)                           |
  | `global` | `_get_edge_data(hl_keywords, kg_inst, rel_vdb, param)` (`:3549-3555`)                           |
  | `hybrid` | `_get_node_data` (si ll) + `_get_edge_data` (si hl), pas de vector_chunks (`:3557-3571`)        |
  | `mix`    | identique `hybrid` + `_get_vector_context(query, chunks_vdb, param, query_embedding)` (`:3574-3591`) |

- **`_get_node_data`** (`$LRAG/operate.py:4238-4293`) :
  1. `entities_vdb.query(query=ll_str, top_k=top_k)` → liste de `{entity_name, ...}` (`:4249`).
  2. `node_ids = [r["entity_name"] for r in results]` puis `asyncio.gather(get_nodes_batch(ids), node_degrees_batch(ids))` (`:4258-4261`).
  3. fusion par zip → `node_datas` enrichi avec `rank, created_at`.
  4. `_find_most_related_edges_from_entities(node_datas, param, kg_inst)` (`:4281-4285`).
  5. retour `(node_datas, use_relations)` — entités triées par cosine, relations triées par `(rank, weight)`.

  **PATCH twin actif ici** — `src/twindb_lightrag_memgraph/__init__.py:520-575` (`_fused_get_node_data`) remplace cette fonction si `kg_inst.get_nodes_with_degrees_batch` existe (cas Memgraph), réduisant 2 sessions Bolt en 1.

- **`_find_most_related_edges_from_entities`** (`$LRAG/operate.py:4296-4349`) :
  1. `get_nodes_edges_batch(node_names)` (`:4302`) — pour chaque nœud, liste des `(src, tgt)` voisins.
  2. dédup `seen_edges` (`:4304-4313`).
  3. `asyncio.gather(get_edges_batch(pairs_dicts), edge_degrees_batch(pairs_tuples))` (`:4322-4325`).
  4. reconstruction `all_edges_data` avec `rank, weight`, tri descendant.

  **PATCH twin actif ici** — `_fused_find_edges` (`src/twindb_lightrag_memgraph/__init__.py:577-625`) remplace l'`asyncio.gather` par `get_edges_with_degrees_batch` (1 session Bolt).

- **`_get_edge_data`** (`$LRAG/operate.py:4511-4564`) :
  1. `relationships_vdb.query(hl_keywords, top_k=param.top_k)` (`:4521`).
  2. `get_edges_batch(edge_pairs_dicts)` (`:4529`).
  3. `_find_most_related_entities_from_relationships(edge_datas, param, kg_inst)` (`:4554`) → `get_nodes_batch(entity_names)` (`:4584`).
  4. retour `(edge_datas, use_entities)` — relations en ordre vector, entités dans l'ordre d'apparition.

- **`_get_vector_context`** (`$LRAG/operate.py:3436-3490`) :
  - `search_top_k = param.chunk_top_k or param.top_k` (`:3459`).
  - `chunks_vdb.query(query, top_k=search_top_k, query_embedding=query_embedding)` (`:3462-3464`).
  - mapping vers `{content, created_at, file_path, source_type: "vector", chunk_id}` (`:3471-3481`).

- **Round-robin entités** (`:3593-3612`) : entrelace `local_entities` et `global_entities` par index, dédup sur `entity_name`.
- **Round-robin relations** (`:3614-3647`) : idem, clé `tuple(sorted([src_id, tgt_id]))` ou `tuple(sorted(src_tgt))`.

#### Stage 2 — `_apply_token_truncation` (`$LRAG/operate.py:3662-3830`)

- Lit `tokenizer = global_config["tokenizer"]` (`:3670`) ; warning + retour minimal si manquant.
- Récupère `max_entity_tokens`, `max_relation_tokens` via `getattr(param, ..., global_config.get(...))` (`:3683-3692`).
- Construit `entities_context = [{entity, type, description, created_at, file_path}, ...]` (`:3702-3720`).
- Construit `relations_context = [{entity1, entity2, description, created_at, file_path}, ...]` (`:3723-3747`).
- **Troncature** : pour entities et relations, retire `file_path` + `created_at` du dict, appelle `truncate_list_by_token_size` (`:3753-3788`).
- Reconstruit `filtered_entities` / `filtered_relations` (les dicts complets, pas les contextes minimisés) — utilisés en Stage 3 pour récupérer les `source_id` (`:3795-3821`).
- Retourne aussi `entity_id_to_original`, `relation_id_to_original` — mappings pour `convert_to_user_format` en Stage 4.

#### Stage 3 — `_merge_all_chunks` (`$LRAG/operate.py:3833-3932`)

- `_find_related_text_unit_from_entities(filtered_entities, param, text_chunks_db, kg_inst, query, chunks_vdb, chunk_tracking, query_embedding)` (`$LRAG/operate.py:4352-4508`).
  - Stratégie de sélection selon `kg_chunk_pick_method` (`WEIGHT` ou `VECTOR`).
  - `WEIGHT` : `pick_by_weighted_polling` (`$LRAG/utils.py`, polling pondéré par occurrence count).
  - `VECTOR` : `pick_by_vector_similarity` (`$LRAG/utils.py:2438-2558`) — cosine query/chunk-embedding via `chunks_vdb.get_vectors_by_ids`.
  - retour `result_chunks` avec `source_type="entity"`, `chunk_id` injecté pour dédup ; `chunk_tracking[chunk_id] = {"source": "E", "frequency": ..., "order": ...}`.
- `_find_related_text_unit_from_relations(filtered_relations, param, text_chunks_db, entity_chunks, query, chunks_vdb, chunk_tracking, query_embedding)` (`$LRAG/operate.py:4600-4800`).
  - Identique en logique mais dédoublonne contre `entity_chunks` déjà sélectionnés.
  - `chunk_tracking[chunk_id] = {"source": "R", ...}`.
- **Round-robin merge** vector_chunks ⨁ entity_chunks ⨁ relation_chunks (`:3882-3927`) avec dédup sur `chunk_id`.

#### Stage 4 — `_build_context_str` (`$LRAG/operate.py:3935-4114`)

- **Allocation dynamique de tokens** (`:3992-4017`) :
  ```
  pre_kg_context = kg_context_template.format(entities_str, relations_str, "", "")  # taille du squelette JSON
  pre_sys_prompt = sys_prompt_template.format(context_data="", response_type, user_prompt)
  available_chunk_tokens = max_total_tokens - (sys_prompt_tokens + kg_context_tokens + query_tokens + 200)
  ```
- **`process_chunks_unified(query, merged_chunks, param, global_config, source_type=param.mode, chunk_token_limit=available_chunk_tokens)`** (`$LRAG/utils.py:2702-2808`) :
  1. **Rerank natif** (`:2730-2738`) : si `param.enable_rerank and query and unique_chunks` → `apply_rerank_if_enabled(query, unique_chunks, global_config, enable_rerank, top_n=param.chunk_top_k or len(unique_chunks))`.
  2. **Filtre min_rerank_score** (`:2741-2763`) : si `min_rerank_score > 0.0`, retire les chunks `rerank_score < seuil`.
  3. **Cap chunk_top_k** (`:2766-2771`) : tronc head.
  4. **Token truncation finale** (`:2774-2799`) : `truncate_list_by_token_size(unique_chunks, key=..., max_token_size=chunk_token_limit, tokenizer=...)`.
  5. **Injection `id` `DC1`, `DC2`, ...** (`:2802-2806`).
- **`apply_rerank_if_enabled`** (`$LRAG/utils.py:2618-2699`) :
  - retourne docs inchangés si `enable_rerank=False` ou `retrieved_docs` vide ;
  - lit `rerank_func = global_config.get("rerank_model_func")` (`:2641`) — **c'est le slot officiel d'injection d'un reranker** ;
  - extrait `doc["content"] or doc["text"] or doc["chunk_content"] or doc["document"] or str(doc)` (`:2651-2660`) ;
  - appelle `await rerank_func(query=..., documents=..., top_n=top_n)` (`:2663-2667`) ;
  - attend `[{"index": int, "relevance_score": float}, ...]` (`:2672-2688`) ; format legacy `[doc_dict, ...]` toléré (`:2689-2692`).
- `generate_reference_list_from_chunks(truncated_chunks)` (`:4031-4033`) : produit `reference_list = [{"reference_id": str, "file_path": str}, ...]`.
- **Assemblage final** (`:4091-4096`) :
  ```python
  result = PROMPTS["kg_query_context"].format(
      entities_str=...,
      relations_str=...,
      text_chunks_str=...,
      reference_list_str=...,
  )
  ```
  `PROMPTS["kg_query_context"]` est défini en `$LRAG/prompt.py:332-357`.
- `convert_to_user_format(entities_context, relations_context, truncated_chunks, reference_list, param.mode, entity_id_to_original, relation_id_to_original)` (`:4102-4110`) produit le `raw_data` retourné via `QueryContextResult(context=result, raw_data=raw_data)`.

### 1.5 Diagramme ASCII — `hybrid` end-to-end

```
caller
  │  await rag.aquery("Pourquoi ORA-04030 ?", QueryParam(mode="hybrid"))
  ▼
LightRAG.aquery                                                  $LRAG/lightrag.py:2459
  │  delegate to aquery_llm
  ▼
LightRAG.aquery_llm                                              $LRAG/lightrag.py:2721
  │  global_config = asdict(self)
  │  switch on param.mode
  │  ├── "local"|"global"|"hybrid"|"mix" → kg_query(...)         :2748-2760
  │  ├── "naive"                          → naive_query(...)     :2761-2769
  │  └── "bypass"                         → LLM direct           :2770-2807
  ▼                       (mode="hybrid")
kg_query(query, kg, ent_vdb, rel_vdb, text_kv, param, gc, hashing_kv, sys_prompt, chunks_vdb)
                                                                 $LRAG/operate.py:3084
  │  step 1) keyword extraction
  │           get_keywords_from_query                            :3136
  │             └─ extract_keywords_only                         :3326
  │                  ├─ cache lookup (handle_cache "keywords")
  │                  ├─ LLM call PROMPTS["keywords_extraction"]  :3385  (LLM #1)
  │                  └─ json_repair.loads
  │  step 2) _build_query_context                                :3159
  │           ├─ Stage 1: _perform_kg_search                     :3493
  │           │     ├─ pre-embed query (1x)                      :3522
  │           │     ├─ _get_node_data(ll_keywords)               :3541
  │           │     │    ├─ entities_vdb.query(top_k)            :4249
  │           │     │    ├─ [TWIN PATCH] _fused_get_node_data
  │           │     │    │    └─ kg.get_nodes_with_degrees_batch :__init__.py:539
  │           │     │    └─ _find_most_related_edges_from_entities :4281
  │           │     │         └─ [TWIN PATCH] _fused_find_edges
  │           │     │              └─ kg.get_edges_with_degrees_batch
  │           │     ├─ _get_edge_data(hl_keywords)               :3549
  │           │     │    ├─ relationships_vdb.query(top_k)       :4521
  │           │     │    ├─ kg.get_edges_batch                   :4529
  │           │     │    └─ _find_most_related_entities          :4554
  │           │     │         └─ kg.get_nodes_batch              :4584
  │           │     ├─ (mix only) _get_vector_context            :3574
  │           │     │    └─ chunks_vdb.query(top_k)              :3462
  │           │     └─ round-robin merge entities/relations      :3593-3647
  │           ├─ Stage 2: _apply_token_truncation                :3662
  │           │     ├─ entities_context list build               :3702
  │           │     ├─ relations_context list build              :3722
  │           │     └─ truncate_list_by_token_size (each)        :3763,3781
  │           ├─ Stage 3: _merge_all_chunks                      :3833
  │           │     ├─ _find_related_text_unit_from_entities     :4352
  │           │     │    └─ pick_by_weighted_polling | pick_by_vector_similarity
  │           │     ├─ _find_related_text_unit_from_relations    :4600
  │           │     │    └─ (idem, dedup vs entity chunks)
  │           │     └─ round-robin merge vector⨁entity⨁relation  :3879
  │           └─ Stage 4: _build_context_str                     :3935
  │                 ├─ available_chunk_tokens calc               :4012
  │                 ├─ process_chunks_unified                    :4021  → utils.py:2702
  │                 │    ├─ apply_rerank_if_enabled              :2730  → utils.py:2618
  │                 │    │    └─ rerank_model_func(query, docs)         (LLM/HTTP #2 ?)
  │                 │    ├─ min_rerank_score filter              :2741
  │                 │    ├─ chunk_top_k cap                      :2766
  │                 │    └─ token truncation final               :2787
  │                 ├─ generate_reference_list_from_chunks       :4031
  │                 ├─ format PROMPTS["kg_query_context"]        :4091
  │                 └─ convert_to_user_format                    :4102
  │  step 3) cache lookup (handle_cache "query")                 :3225
  │  step 4) sys_prompt = PROMPTS["rag_response"].format(...)    :3189
  │  step 5) await use_model_func(query, system_prompt, ...)     :3236   (LLM #3, final)
  │  step 6) post-strip artifacts                                :3271
  │  return QueryResult(content=response, raw_data=...)
  ▼
LightRAG.aquery_llm
  │  await self._query_done()                                    :2811
  │  enrich raw_data["llm_response"]                             :2832
  ▼
LightRAG.aquery
  │  unwrap llm_response.content
  ▼
caller
```

LLM calls par query `hybrid` non cachée : **2** (keyword extraction + réponse finale), **3** si reranker LLM-based actif via `rerank_model_func`.

---

## 2. Surface d'extensibilité native LightRAG

### 2.1 Hooks officiels par `QueryParam`

`QueryParam` est le **seul** mécanisme d'override que LightRAG documente pour le retrieval (`$LRAG/base.py:84-169`) :

| Champ                | Effet                                                                              | Niveau d'invasion |
|----------------------|------------------------------------------------------------------------------------|-------------------|
| `model_func`         | Override LLM par requête, court-circuite `global_config["llm_model_func"]`         | nul (API publique)|
| `enable_rerank`      | Active/désactive `apply_rerank_if_enabled` (`$LRAG/utils.py:2618`)                 | nul               |
| `hl_keywords` / `ll_keywords` | Bypass `extract_keywords_only` si pré-remplis (`$LRAG/operate.py:3316-3317`) | nul               |
| `user_prompt`        | Injecté dans `PROMPTS["rag_response"]` (`$LRAG/prompt.py:270`)                     | nul               |
| `only_need_context`  | Court-circuite l'appel LLM final, retourne le context (`$LRAG/operate.py:3176`)    | nul               |
| `only_need_prompt`   | Court-circuite la génération, retourne le prompt assemblé (`$LRAG/operate.py:3198`)| nul               |
| `mode`               | Sélectionne `kg_query` vs `naive_query` vs `bypass` (`$LRAG/lightrag.py:2748-2807`)| nul               |

### 2.2 Hook officiel reranker

`LightRAG.__init__` accepte `rerank_model_func: Callable[..., object] | None` (`$LRAG/lightrag.py:359-360`). Cette callable est appelée par `apply_rerank_if_enabled` (`$LRAG/utils.py:2641-2667`) avec la signature :

```python
async def rerank_model_func(query: str, documents: list[str], top_n: int | None) -> list[dict]:
    # retour attendu : [{"index": int, "relevance_score": float}, ...]
```

C'est **le seul** point d'injection cognitive officiel sur le hot path retrieval, et il porte sur les **chunks**, pas sur les entities/relations. Côté Twin, `CognitiveReranker.rerank(question, chunks)` (`src/twindb_lightrag_memgraph/intelligence/features/cognitive_reranker.py:68`) retourne des `ChunkResult`, pas `[{"index","relevance_score"}]` — il y a un **adaptateur à écrire** (voir §5).

### 2.3 Hooks officiels indexation

Hors scope retrieval mais à signaler pour contexte :

- `chunking_func: Callable[..., Union[list, Awaitable[list]]]` (`$LRAG/lightrag.py:249-282`) — chunking custom.
- `addon_params: dict` (`$LRAG/lightrag.py:427-434`) — entity_types, language.
- Le repo expose `register_post_index_hook` (`src/twindb_lightrag_memgraph/_hooks.py`), patché via `_patch_insert_done` (`src/twindb_lightrag_memgraph/__init__.py:689-713`) — pour la phase **indexation**, **pas retrieval**.

### 2.4 Pas de callback retrieval

LightRAG n'expose **aucun** de ces patterns sur le retrieval :

- pas de `on_before_query` / `on_after_query` ;
- pas de plugin registry (`STORAGES` ne couvre que les backends de stockage) ;
- pas de middleware FastAPI sur le retrieval applicatif (les middlewares serveur sont HTTP-only, cf. Prisme A) ;
- pas de signal/event bus ;
- pas de classe abstraite avec slots `protected` ou `_hooks` ;
- les fonctions `kg_query`, `naive_query`, `_build_query_context`, `_perform_kg_search`, `_apply_token_truncation`, `_merge_all_chunks`, `_build_context_str`, `process_chunks_unified`, `apply_rerank_if_enabled` sont toutes des **fonctions module-level**, ce qui les rend monkey-patchables mais sans contrat de stabilité.

### 2.5 Niveau d'invasion requis

Pour injecter L3 sans modifier le code LightRAG, **trois niveaux** d'invasivité existent (détaillés en §7) :

- **monkey-patch d'`aquery` / `aquery_llm`** sur l'instance ou la classe (effort minimal) ;
- **sous-classe `TwinLightRAG(LightRAG)`** (effort modéré, mais `@final` sur la classe `LightRAG` à `$LRAG/lightrag.py:129` — voir warning §7.B) ;
- **patches granulaires d'`operate.py`** sur le modèle de `_patch_operate_hot_paths` existant (effort élevé, déjà la pratique du repo pour les fused queries).

---

## 3. Phase Intent (F05) — où injecter ?

### 3.1 Effet recherché

`IntentClassifier.classify(question)` (`src/twindb_lightrag_memgraph/intelligence/features/intent_classifier.py:78-121`) doit s'exécuter **avant le retrieval** pour court-circuiter sur :

- `OUT_OF_SCOPE` → réponse scriptée, **0 appel retrieval** ;
- `GREETING` → réponse scriptée ;
- `MALICIOUS` → réponse scriptée + log warning ;
- `ESCALATION` → réponse scriptée + lien humain ;
- `IN_SCOPE` → laisser passer la pipeline.

C'est un cas typique de gate **avant** `kg_query`/`naive_query`.

### 3.2 Point d'injection

La ligne exacte est juste après `global_config = asdict(self)` et avant le `switch` sur `param.mode` dans `aquery_llm` (`$LRAG/lightrag.py:2743-2748`). En patch monkey-style on intercepte plus haut, dans `aquery` directement, pour éviter le wrapping `aquery → aquery_llm` :

```python
# Pseudo-patch (cf. §7.A pour la version complète)
_orig_aquery = LightRAG.aquery
async def _twin_aquery(self, query, param=QueryParam(), system_prompt=None):
    intent_result = await _twin_intent.classify(query)
    if intent_result.intent in {IntentType.OUT_OF_SCOPE, IntentType.GREETING, IntentType.MALICIOUS}:
        return _twin_scripted_response(intent_result.intent)
    return await _orig_aquery(self, query, param, system_prompt)
LightRAG.aquery = _twin_aquery
```

### 3.3 Faut-il modifier `QueryParam.mode` ?

Oui — c'est une optimisation orthogonale à la fonction principale du F05. Le `IntentClassifier` actuel renvoie un `IntentType` mais pas un mode-hint. Une évolution possible (v2 du F05) :

- intent `INCIDENT_DIAGNOSTIC` → forcer `param.mode = "mix"` (KG + vector pour couvrir runbooks + procédures) ;
- intent `LOOKUP_FACTUAL` → forcer `param.mode = "naive"` (pure vector, plus rapide) ;
- intent `EXPLAINER` → forcer `param.mode = "hybrid"` (sans vector_chunks).

Cette logique n'existe pas encore dans `intent_classifier.py` — il faudra étendre `IntentResult` avec un champ `mode_hint: Literal["local", "global", "hybrid", "mix", "naive"]`.

### 3.4 Effets de bord acceptables

- **Latence** : +1 LLM call (`llm_effort_intent="low"`, `max_tokens=100`, cf. `config.py:30, intent_classifier.py:96`). En pratique 200-500 ms sur `gpt-oss-120b@low`. Acceptable pour BNP IT Ops où la latence baseline est dominée par le retrieval graph (1-3 s).
- **Coût** : ~150 tokens prompt + ~50 tokens réponse = négligeable vs un cycle hybrid complet (~3-5 k tokens).
- **Fallback** : `IntentClassifier.classify` retourne déjà `IntentType.IN_SCOPE` avec `confidence=0.0` sur exception (`intent_classifier.py:115-121`). Le wrapper doit traiter ce cas comme "let through" — pas de blocage si l'OOS-detector tombe.
- **Cache** : LightRAG n'aura **pas** de cache pour cet appel (il n'est plus dans `extract_keywords_only` qui cache via `hashing_kv`). Si nécessaire, ajouter un cache LRU dans `IntentClassifier` (clé : hash de `question`).

---

## 4. Phase REASON (coref + F03 expansion) — où injecter ?

### 4.1 Deux stratégies, deux endroits

#### Stratégie A — Enrichir la query AVANT `kg_query`

```python
# Avant kg_query : on remplace la query par sa version "reasoned + expanded"
reasoning_result = await _twin_reasoning.analyze(query, param.conversation_history)
expanded = await _twin_expander.expand_v2(reasoning_result.search_query, workspace, reasoning_result.domain_hint)
query = expanded.expanded_query   # remplacement direct
return await _orig_aquery(self, query, param, system_prompt)
```

- **Avantage** : un seul retrieval. Logique idiomatique pour LightRAG (le `extract_keywords_only` natif va voir une query déjà enrichie → meilleurs keywords).
- **Inconvénient** : la résolution de coréférence ré-écrit la query → l'utilisateur ne saura plus exactement quelle query a été retrieved. Tracker via `QueryTrace.resolved_query` (déjà fait par `engine.py:154`).

#### Stratégie B — N retrievals (un par variante) puis fusion

```python
for variant in expanded.added_terms:
    result_n = await _orig_aquery(self, f"{base_query} {variant}", param, sp)
    ...
# fusion manuelle des contextes
```

- **Avantage** : recall potentiellement plus large.
- **Inconvénient** : coût explosif (N fois plus de retrieval), pas de fusion native LightRAG (`kg_query` retourne un blob LLM, pas une liste de chunks fusionnables proprement). Le `TwinRAGEngine.aquery` actuel ne fait pas ça non plus (`engine.py:164-170` lance N retrievals **par workspace**, pas par variante de query).

### 4.2 Recommandation

**Stratégie A**. Le pipeline natif a déjà un keyword extractor (`extract_keywords_only`, `$LRAG/operate.py:3326-3433`) qui produit `hl_keywords` + `ll_keywords` via LLM. Trois sous-options :

1. **Pre-fill `param.hl_keywords` / `param.ll_keywords`** depuis `reasoning_result` — bypass complet de l'extraction LLM native. C'est le pattern le plus propre (utilise le slot officiel `$LRAG/operate.py:3316-3317`) :

   ```python
   reasoning_result = await _twin_reasoning.analyze(query, history)
   param.hl_keywords = reasoning_result.high_level_terms
   param.ll_keywords = reasoning_result.low_level_terms
   ```

   À l'heure actuelle `ReasoningResult` (`reason.py:68-76`) ne sépare pas `hl` / `ll`. Évolution mineure côté schéma.

2. **Réécrire la query** + laisser `extract_keywords_only` faire son boulot dessus. Plus simple, plus coûteux (LLM call supplémentaire).

3. **Enrichir le prompt `PROMPTS["keywords_extraction"]`** (`$LRAG/prompt.py:374-396`) avec un appendix domain-specific (IT Ops). Hack le plus invasif — il faut écraser `PROMPTS["keywords_extraction"]` dans `register()`, ce qui est globalement risqué (cassure silencieuse en upgrade).

### 4.3 Compatibilité avec l'extraction native

Si on pre-fill `hl_keywords` / `ll_keywords` :

- `get_keywords_from_query` (`$LRAG/operate.py:3294-3323`) court-circuite l'appel LLM → -1 LLM call ;
- gain de latence : ~500-800 ms par query ;
- gain de coût : ~200-500 tokens par query.

**Ne pas** modifier `PROMPTS["keywords_extraction"]` — cassure silencieuse en upgrade LightRAG.

### 4.4 Coréférence

La coréférence n'est pas gérée nativement par LightRAG — la query passée à `aquery_llm` est utilisée brute par `extract_keywords_only` et par le LLM final. Le `ReasoningEngine.analyze` (`reason.py:95-153`) résout la coréférence avec `conversation_history`, c'est donc **lui** qui doit produire la query résolue avant `aquery`. Pas de compétition possible avec un mécanisme natif.

---

## 5. Phase ACT (retrieve + F04 cognitive rerank) — où injecter ?

### 5.1 Ranking interne LightRAG

| Niveau                | Mécanisme natif                                                                                  | Source                                              |
|-----------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| Entities              | cosine similarity (`entities_vdb.query(query, top_k)` retourne déjà ordonné)                     | `$LRAG/operate.py:4249`                             |
| Relations from entities| `sorted(..., key=lambda x: (x["rank"], x["weight"]), reverse=True)`                              | `$LRAG/operate.py:4345-4347` (= `_fused_find_edges` ligne 622) |
| Relations from kwds   | ordre vector search préservé (pas de re-tri)                                                     | `$LRAG/operate.py:4552`                             |
| Entities from relations| ordre d'apparition dans `edge_datas` (pas de re-tri)                                            | `$LRAG/operate.py:4584-4595`                        |
| Chunks (vector)       | cosine similarity (`chunks_vdb.query`)                                                           | `$LRAG/operate.py:3462`                             |
| Chunks (entity-related)| `pick_by_weighted_polling` (occurrence count) ou `pick_by_vector_similarity` (cosine query/chunk)| `$LRAG/operate.py:4435-4485`                        |
| Chunks (relation-related)| idem, dédup vs entity_chunks                                                                  | `$LRAG/operate.py:4724-4763`                        |
| Chunks finaux         | `apply_rerank_if_enabled` (slot `rerank_model_func`)                                             | `$LRAG/utils.py:2618-2699`                          |

Le ranking natif est **multi-étapes** et hétérogène (cosine + weight + occurrence). Le seul point officiel où injecter du LLM-rerank est `rerank_model_func`.

### 5.2 Option 1 — Brancher `CognitiveReranker` comme `rerank_model_func`

Adaptateur :

```python
# Pseudo-code
async def _twin_rerank_model_func(query: str, documents: list[str], top_n: int | None) -> list[dict]:
    pseudo_chunks = [
        ChunkResult(chunk_id=str(i), text=d, score=0.5, source_workspace="lightrag")
        for i, d in enumerate(documents)
    ]
    reranked = await _twin_reranker.rerank(query, pseudo_chunks)
    return [
        {"index": int(c.chunk_id), "relevance_score": c.rerank_score or c.score}
        for c in reranked
    ]
```

- **Pros** : zéro patch sur LightRAG, le slot existe ;
- **Cons** :
  - `CognitiveReranker.rerank` (`cognitive_reranker.py:68-150`) filtre par `reranking_score_threshold >= 7.0` et tronque à `final_limit=8` — le retour à LightRAG sera **artificiellement réduit**, alors que LightRAG attend top_n chunks et fera ensuite sa propre troncature tokens (`process_chunks_unified` step 4) ;
  - L'effort `low` du rerank Twin (`llm_effort_reranker`) tourne sur **tous** les chunks à chaque query ;
  - Si Twin renvoie moins de docs que `top_n`, LightRAG s'en accommode mais le `min_rerank_score` natif (`$LRAG/utils.py:2742`) va re-filtrer encore — il faut **désactiver** `min_rerank_score` côté `LightRAG(...)` (ou exposer `MIN_RERANK_SCORE=0` env).
- **Position pipeline** : appel via `process_chunks_unified` (`$LRAG/utils.py:2732`), donc **après** le KG retrieval et **avant** la troncature finale tokens. Latence ajoutée : 1 LLM call sur N=top_n chunks (config `chunk_top_k` ou défaut `DEFAULT_CHUNK_TOP_K`, généralement 10-20).

### 5.3 Option 2 — Rerank entre `_merge_all_chunks` et `_build_context_str`

Plus invasif (patch granulaire d'`_build_query_context`) mais permet d'avoir accès aux entités + relations + chunks dans un seul rerank — ce que le slot `rerank_model_func` ne permet pas (il ne reçoit que les chunks textuels).

Position dans `_build_query_context` (`$LRAG/operate.py:4118-4235`) :

```
Stage 3 _merge_all_chunks → merged_chunks
  ▼ INJECT ICI : await twin_reranker.rerank(query, merged_chunks_as_ChunkResult)
Stage 4 _build_context_str(merged_chunks=reranked, ...)
```

Effort de patch : remplacer `_build_query_context` entier ou intercepter avec un wrapper. Risque d'API drift : la signature de `_build_query_context` a changé entre 1.4.9 et 1.4.10 (ajout de `chunks_vdb=None` selon le diff observé sur les autres prismes).

### 5.4 Recommandation

**Option 1** (rerank_model_func adapter) pour v1, **Option 2** pour v2 si on observe qu'entities + relations devraient être rerankées (typiquement pour des requêtes `local` ou `global` pures où il n'y a pas de chunks à reranker).

Latence acceptable : `CognitiveReranker` envoie tous les chunks dans un seul prompt LLM, donc **1 LLM call** total, pas K (`cognitive_reranker.py:101-107`). Avec `llm_effort_reranker="low"` et `max_tokens=500`, ~500-800 ms. Acceptable.

---

## 6. Phase OBSERVE (synthesis) — où injecter ?

### 6.1 Synthesis natif LightRAG

LightRAG construit la réponse via `use_model_func(user_query, system_prompt=PROMPTS["rag_response"].format(...))` en `$LRAG/operate.py:3236-3242` (mode KG) et `:5019-5025` (mode naive). Le prompt est `PROMPTS["rag_response"]` (`$LRAG/prompt.py:224-276`), qui :

- enjoint l'usage exclusif du contexte ;
- demande des citations par `[reference_id]` ;
- génère une section `### References` ;
- accepte `{user_prompt}` (champ `QueryParam.user_prompt`) pour customisation.

Le contexte (`context_data`) est le `kg_query_context` ou `naive_query_context` formaté en Stage 4 (`$LRAG/prompt.py:332-372`).

### 6.2 `SynthesisEngine` Twin

`SynthesisEngine.synthesize(question, chunks, conversation_history)` (`observe.py:85-164`) :

- prompt `_DEFAULT_SYSTEM_PROMPT` IT/Ops spécifique BNP (`observe.py:35-63`) ;
- format `[Passage X]` (pas `[reference_id]`) ;
- `llm_effort_synthesis="high"` ;
- retourne `SynthesisResult(answer, citations: list[Citation], tokens_used)`.

### 6.3 Trois options de branchement

#### Option A — Remplacer le LLM call final

Le plus invasif. On bypass `PROMPTS["rag_response"]` :

```python
# Dans wrapper aquery
if intent_result.intent == IntentType.IN_SCOPE:
    # 1. Récupérer le contexte sans LLM (only_need_context=True)
    param.only_need_context = True
    ctx_result = await _orig_aquery(self, query, param, system_prompt)
    # 2. Parser ctx_result en chunks (cf. PROMPTS["kg_query_context"] format)
    chunks = _parse_kg_query_context(ctx_result)
    # 3. Synthesis Twin
    return await _twin_synthesis.synthesize(query, chunks, history)
```

- **Pros** : prompt système Twin BNP-spécifique, format citations `[Passage X]`, `llm_effort` séparé ;
- **Cons** :
  - parsing inverse du format `PROMPTS["kg_query_context"]` (JSON blocks pour entities/relations/chunks) — fragile en upgrade ;
  - on perd le cache de réponse natif (`handle_cache(..., cache_type="query")` dans `kg_query`) ;
  - on ne profite plus des optimisations natives du `rag_response` (sections Markdown, gestion `response_type`).

#### Option B — Modifier le prompt natif

Remplacer `PROMPTS["rag_response"]` dans `register()` :

```python
from lightrag.prompt import PROMPTS
PROMPTS["rag_response"] = _TWIN_RAG_RESPONSE_PROMPT
```

- **Pros** : zéro changement de pipeline, format `[reference_id]` natif conservé ;
- **Cons** : cassure silencieuse en upgrade LightRAG si la signature des placeholders change (actuellement `{response_type}`, `{user_prompt}`, `{context_data}`). À ne **pas** faire — c'est exactement le pattern interdit selon la doctrine du repo (`CLAUDE.md` Intelligence — Prompt files use `.format()`, JSON braces `{{ }}` escaped — un override silencieux est piégé).

#### Option C — Wrapper post-processing après réponse

```python
response = await _orig_aquery(self, query, param, system_prompt)
return await _twin_postprocess(response, query, history)
```

- **Pros** : non-invasif, transparent ;
- **Cons** : le post-processing reçoit déjà une réponse formatée — on ne peut plus changer le ton ou la structure ; au mieux on peut ajouter un disclaimer ou normaliser les citations.

### 6.4 Recommandation

**Option A** pour la fidélité au prompt Twin BNP (`observe.py:_DEFAULT_SYSTEM_PROMPT`), mais en passant par `aquery_data(query, param)` (`$LRAG/lightrag.py:2514-2719`) plutôt que par `aquery(... only_need_context=True)` :

- `aquery_data` retourne directement `raw_data` structuré (`{entities, relationships, chunks, references}`) — pas de parsing inverse ;
- bypass `apply_rerank_if_enabled` ? Non — il est dans `process_chunks_unified` qui est appelé par `_build_context_str`, lui-même appelé par `_build_query_context`, qui est appelé par `kg_query` via `aquery_data`. Donc `aquery_data` exécute bien tout le pipeline retrieval + rerank, juste sans le LLM final.

Blast radius : faible. Stable entre 1.4.9 et 1.4.10 (la signature de `aquery_data` n'a pas changé). Acceptable.

Si on adopte Option A, on perd le **cache LLM final** (`handle_cache(..., cache_type="query")`). Compensation : le `LLMCache` Twin n'existe pas encore — à prévoir en roadmap.

---

## 7. Plan d'injection minimal viable

### 7.A — Plug-in ponctuel (monkey-patch `aquery`)

**Cible** : un seul wrap sur `LightRAG.aquery`, branchement L3 en pre/post.

**Fonction à patcher** : `LightRAG.aquery` (`$LRAG/lightrag.py:2459-2491`).

**Patch** :

```python
# src/twindb_lightrag_memgraph/__init__.py — nouvelle fonction
def _patch_intelligence_layer(intelligence_config: TwinRAGConfig):
    """Wrap LightRAG.aquery with L3 pre/post hooks.

    Pre-hooks (avant retrieval): F05 intent, REASON+F03 expansion.
    Post-hooks (apres retrieval): aucune pour v1 (rerank et synthesis
    sont injectes via rerank_model_func et aquery_data Option A).
    Idempotent via _intelligence_registered flag.
    """
    from lightrag.lightrag import LightRAG
    from .intelligence.engine import TwinRAGEngine
    from .intelligence.models.schemas import IntentType

    if getattr(LightRAG, "_twin_intelligence_patched", False):
        return

    _orig_aquery = LightRAG.aquery
    _twin_engine = TwinRAGEngine(intelligence_config)

    async def _twin_aquery(self, query, param=None, system_prompt=None):
        from lightrag import QueryParam
        param = param or QueryParam()

        # 1. F05 Intent
        if _twin_engine.config.enable_oos_detection:
            intent = await _twin_engine.intent_classifier.classify(query)
            if intent.intent in {IntentType.OUT_OF_SCOPE,
                                 IntentType.GREETING,
                                 IntentType.MALICIOUS}:
                return _twin_engine._scripted_response(intent.intent)

        # 2. REASON + F03
        history = param.conversation_history or []
        reasoning = await _twin_engine.reasoning.analyze(query, history)
        expanded = await _twin_engine._expand_query(
            reasoning.search_query,
            workspace=getattr(self, "workspace", "commons"),
            domain_hint=reasoning.domain_hint,
        )
        resolved_query = expanded.expanded_query

        # 3. Delegate to native retrieval (LightRAG cache + rerank + LLM final)
        return await _orig_aquery(self, resolved_query, param, system_prompt)

    LightRAG.aquery = _twin_aquery
    LightRAG._twin_intelligence_patched = True
```

**Effort LOC** : ~50 LOC dans `__init__.py` + un nouveau paramètre `register(intelligence=False)` (~10 LOC). Total : ~60 LOC.

**Risque cassure upstream** : nul (on patche `aquery`, dont la signature est stable).

**Limites** :

- Pas d'override du reranker — pour ça il faut passer `rerank_model_func` au `LightRAG(...)` côté caller, ou patcher `__post_init__` pour l'injecter automatiquement.
- Pas d'override du synthesis — pour ça il faut Option A de §6.

### 7.B — Sous-classe `TwinLightRAG(LightRAG)`

**Cible** : créer `TwinLightRAG(LightRAG)` qui override `aquery`, `aquery_llm`, `aquery_data`.

**WARNING — `@final`** : la classe `LightRAG` est décorée `@final` (`$LRAG/lightrag.py:129-131`). `typing.final` n'est qu'une **annotation** (pas une enforcement runtime), donc l'héritage techniquement fonctionne, mais :

- les linters/type checkers vont remonter une erreur ;
- aucun engagement de l'upstream à ne pas casser cette structure ;
- conflit doctrinal avec la politique du repo "ne pas modifier le code LightRAG sans le savoir" — un caller qui voit `TwinLightRAG` doit comprendre que tout `aquery` est intercepté, ce que `register()` rend déjà clair par son nom.

**Code** :

```python
# src/twindb_lightrag_memgraph/twin_lightrag.py
from lightrag import LightRAG, QueryParam
from .intelligence.engine import TwinRAGEngine

class TwinLightRAG(LightRAG):
    """LightRAG enriched with L3 intelligence (F05 + REASON + F03)."""
    intelligence_config: object = None

    def __post_init__(self):
        super().__post_init__()
        self._twin_engine = TwinRAGEngine(self.intelligence_config)

    async def aquery(self, query, param=QueryParam(), system_prompt=None):
        # Same body as 7.A but on self._twin_engine
        ...
```

**Effort LOC** : ~100 LOC (incluant tests).

**Risque cassure upstream** : modéré. Toute modification de `LightRAG.__post_init__` (très probable en upgrade) demande une re-synchro. Le diff `1.4.9 → 1.4.10` n'a pas touché `__post_init__` mais c'est un risque structurel.

**Doctrine du repo** : à éviter sauf si la sous-classe apporte aussi d'autres comportements qui justifient un type séparé. Pour juste F05/REASON/F03, le wrap §7.A est plus économe.

### 7.C — Remplacement profond (patches granulaires `operate.py`)

**Cible** : patches isolés par phase, comme `_patch_operate_hot_paths` existant.

**Fonctions à patcher** :

| Phase L3        | Fonction patchée                                                    | Source                              |
|-----------------|---------------------------------------------------------------------|-------------------------------------|
| F05 Intent      | `lightrag.lightrag.LightRAG.aquery_llm` (gate au début)             | `$LRAG/lightrag.py:2721`            |
| REASON+F03      | `lightrag.operate.get_keywords_from_query`                          | `$LRAG/operate.py:3294`             |
| ACT (search)    | `lightrag.operate._perform_kg_search` (multi-workspace)             | `$LRAG/operate.py:3493`             |
| ACT (rerank)    | `lightrag.utils.apply_rerank_if_enabled` (Twin-aware)               | `$LRAG/utils.py:2618`               |
| OBSERVE         | `lightrag.operate.kg_query` (remplace le LLM call final)            | `$LRAG/operate.py:3236`             |

**Effort LOC** : ~400-600 LOC (un patch par phase, fallbacks, tests).

**Risque cassure upstream** : élevé. Chaque upgrade demande de re-vérifier 5 signatures. Modèle déjà éprouvé pour `_fused_get_node_data` / `_fused_find_edges` (3 versions LightRAG dans la matrice CI), mais ces patches restent comportementaux-identiques (juste perf). Les patches L3 changent la sémantique → audit obligatoire par version.

**Snippet exemple — REASON+F03 via `get_keywords_from_query`** :

```python
def _patch_keyword_extraction_with_reason(engine: TwinRAGEngine):
    """Replace LightRAG's keyword extraction with Twin REASON+F03.

    Twin's reasoning produces hl + ll keywords directly (no separate LLM
    call for keyword extraction). Net -1 LLM call per query.
    """
    import lightrag.operate as operate

    _orig = operate.get_keywords_from_query

    async def _twin_get_keywords(query, query_param, global_config, hashing_kv=None):
        if query_param.hl_keywords or query_param.ll_keywords:
            return query_param.hl_keywords, query_param.ll_keywords

        history = query_param.conversation_history or []
        reasoning = await engine.reasoning.analyze(query, history)
        expanded = await engine._expand_query(
            reasoning.search_query, "commons", reasoning.domain_hint
        )
        # ReasoningResult doit etre etendu pour separer hl/ll keywords.
        # Fallback : tout en ll_keywords pour ne pas perdre de recall.
        return [], expanded.added_terms + reasoning.search_query.split()

    operate.get_keywords_from_query = _twin_get_keywords
```

### 7.D — Comparatif

| Stratégie | Effort LOC | Risque upstream | Couverture L3      | Réversibilité | Recommandation v1 |
|-----------|------------|-----------------|--------------------|---------------|-------------------|
| A — Plug-in `aquery` | ~60 | nul         | F05 + REASON + F03 | totale        | **OUI**           |
| B — Sous-classe      | ~100| modéré      | Tout, propre       | facile        | non (doctrine)    |
| C — Granulaire op.py | ~500| élevé       | Tout, fin          | difficile     | v3 si besoin      |

---

## 8. Compatibilité avec patches existants

### 8.1 Le pattern double-patch

`_patch_merge_write_path` (`src/twindb_lightrag_memgraph/__init__.py:631-686`) fait explicitement :

```python
operate.merge_nodes_and_edges = _buffered_merge_nodes_and_edges
_lr_mod.merge_nodes_and_edges = _buffered_merge_nodes_and_edges
```

Raison : `lightrag.lightrag` importe `from lightrag.operate import merge_nodes_and_edges` (`$LRAG/lightrag.py:89-95`), créant une copie locale `lightrag.lightrag.merge_nodes_and_edges` qui ne suit **pas** le patch de `operate.merge_nodes_and_edges`. D'où le besoin de patcher les deux.

**Vérification pour notre périmètre** :

`$LRAG/lightrag.py:89-95` importe **explicitement** `kg_query, naive_query, merge_nodes_and_edges, extract_entities, chunking_by_token_size, rebuild_knowledge_from_chunks` :

```python
from lightrag.operate import (
    chunking_by_token_size,
    extract_entities,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    rebuild_knowledge_from_chunks,
)
```

→ **`kg_query` et `naive_query` sont importés dans `lightrag.lightrag`** comme symboles locaux. Si on patche `operate.kg_query`, le patch ne s'applique **pas** à `lightrag.lightrag.kg_query`, qui est utilisé dans `aquery_llm` et `aquery_data`.

**Conséquence pour 7.C** : tout patch sur `operate.kg_query` ou `operate.naive_query` doit être doublé sur `lightrag.lightrag.kg_query` / `lightrag.lightrag.naive_query`. Pattern :

```python
import lightrag.operate as operate
import lightrag.lightrag as _lr_mod

_orig_kg_query = operate.kg_query
async def _twin_kg_query(*args, **kwargs):
    ...
operate.kg_query = _twin_kg_query
_lr_mod.kg_query = _twin_kg_query   # double-patch obligatoire
```

`_perform_kg_search`, `_build_query_context`, `_build_context_str`, `_apply_token_truncation`, `_merge_all_chunks`, `_get_node_data`, `_find_most_related_edges_from_entities` : utilisées **uniquement** depuis `operate.py` (mêmes-module calls), donc **pas** de double-patch. C'est ce que fait déjà `_patch_operate_hot_paths` (`src/twindb_lightrag_memgraph/__init__.py:509-628`) : il patche `operate._get_node_data` et `operate._find_most_related_edges_from_entities` sans double-patch, ce qui est correct.

### 8.2 Risque de collision avec `_fused_*`

`_fused_get_node_data` patche `operate._get_node_data` (`__init__.py:627`). Si on ajoute un patch L3 sur `_get_node_data` (rerank des entities, par exemple), il faut **wrapper le `_fused_*` plutôt que le natif**, sinon on perd le gain perf des fused queries. Modèle :

```python
_lightrag_get_node_data = operate._get_node_data   # déjà = _fused_get_node_data après _patch_operate_hot_paths
async def _twin_get_node_data(*args, **kwargs):
    node_datas, use_relations = await _lightrag_get_node_data(*args, **kwargs)
    # Post-process Twin
    return node_datas, use_relations
operate._get_node_data = _twin_get_node_data
```

### 8.3 Ordre d'appel recommandé dans `register()`

`register()` actuel (`__init__.py:39-105`) exécute :

```
1. STORAGE_IMPLEMENTATIONS / ENV_REQUIREMENTS / STORAGES update
2. _patch_builtin_memgraph_storage()        ← contient _patch_operate_hot_paths()
3. _patch_merge_write_path()
4. _patch_insert_done()
5. _patch_version_string()
```

Ordre proposé après ajout de l'intelligence layer :

```
1. STORAGE_IMPLEMENTATIONS / ENV_REQUIREMENTS / STORAGES update
2. _patch_builtin_memgraph_storage()        ← contient _patch_operate_hot_paths()
3. _patch_merge_write_path()
4. _patch_insert_done()
5. _patch_version_string()
6. if intelligence: _patch_intelligence_layer(config)   ← NOUVEAU, dernier
```

**Justification** :

- `_patch_intelligence_layer` doit voir `operate._get_node_data` déjà remplacé par `_fused_get_node_data` (sinon les patches L3 sur `_get_node_data` partent sur la version non-optimisée) ;
- `_patch_intelligence_layer` est le **plus haut niveau** (intercepte `aquery`), donc l'invariant "patches bas-niveau d'abord" tient ;
- placer L3 **après** `_patch_version_string` est neutre — le marker version est statique, indépendant du runtime.

### 8.4 Pas de double-patch nécessaire pour `aquery`

`LightRAG.aquery` est une **méthode d'instance** appelée via `self.aquery(...)`. Patcher `LightRAG.aquery` au niveau classe propage à toutes les instances existantes et futures. Pas de copie locale dans un autre module — c'est différent des fonctions module-level comme `kg_query`. La stratégie 7.A est donc sûre du point de vue patching.

---

## 9. Risques et points d'attention

### 9.1 Latence cumulative

Comparaison `hybrid` baseline vs Twin-augmenté (estimation, hors cache) :

| Phase            | Baseline (s)     | Twin (s)         | Delta        |
|------------------|------------------|------------------|--------------|
| F05 Intent       | -                | 0.2 - 0.5        | +0.3 s       |
| REASON (coref + domain) | -         | 0.5 - 1.0        | +0.7 s       |
| F03 Expansion (v1 thesaurus) | -    | 0.01 - 0.05      | +0.03 s      |
| F03 Expansion (v2 graph)     | -    | 0.1 - 0.3        | +0.2 s       |
| Keyword extraction (économisé si pre-fill) | 0.5 - 0.8 | 0 (bypass) | -0.6 s     |
| _perform_kg_search (Memgraph + fused) | 0.8 - 2.0 | 0.8 - 2.0 | 0          |
| _apply_token_truncation       | 0.01 - 0.05 | 0.01 - 0.05 | 0       |
| _merge_all_chunks (WEIGHT/VECTOR) | 0.1 - 0.5 | 0.1 - 0.5 | 0          |
| F04 Cognitive rerank (1 LLM call, all chunks) | 0 (cosine) | 0.5 - 0.8 | +0.6 s |
| LLM réponse finale (rag_response) | 2.0 - 4.0 | 2.0 - 4.0   | 0          |
| **Total**        | **~4 - 8 s**     | **~5 - 10 s**    | **+1 - 2 s** |

L'overhead n'est **pas** linéaire dans les LLM calls car REASON + F03 fonctionnent à `effort=medium/low`, et F05 / rerank à `effort=low`. La synthesis (effort=high) domine en baseline et en Twin.

### 9.2 Coût (LLM calls par query)

| Configuration        | LLM calls non-cachés | Tokens approx |
|----------------------|----------------------|---------------|
| Baseline `hybrid` (no rerank LLM, no cache) | 2 (keywords + answer) | ~3-5 k        |
| Baseline `hybrid` (rerank LLM activé) | 3 (keywords + rerank + answer) | ~4-6 k     |
| Twin v1 (7.A : F05 + REASON + F03, pre-fill keywords, rerank slot) | 4 (intent + reason + rerank + answer) | ~5-7 k |
| Twin v2 (Option A synthesis remplacé) | 4 (intent + reason + rerank + twin-synthesis) | ~5-7 k |

**Séparation chat_llm / indexing_llm** (`config.py:35-44`) : tous les LLM calls retrieval (intent, reason, expansion v2 si LLM, rerank, synthesis) utilisent **`llm_api_key` + `llm_api_base` + `llm_model`** (chat LLM). `indexing_*` n'est pas touché par le hot path retrieval — c'est ce qu'on veut.

### 9.3 Concurrence

- LightRAG retrieval est natif `async`, multiples concurrent queries sont déjà supportées via `asyncio.gather` (cf. `_perform_kg_search`, `_get_node_data`).
- `TwinRAGEngine.aquery` (`engine.py:81`) est `async`. `IntentClassifier.classify`, `ReasoningEngine.analyze`, `CognitiveReranker.rerank` créent chacun un `AsyncOpenAI` à chaque appel (`intent_classifier.py:81`, `reason.py:120`, `cognitive_reranker.py:96`). **Anti-pattern** : créer un client OpenAI par call casse les pools HTTP/2 et redonne `keepalive` à zéro. À refactorer côté Twin pour réutiliser un client unique par engine — c'est orthogonal au câblage L1/L3 mais déjà observé par le code (rien dans LightRAG ne l'oblige).
- Concurrence-safe : oui, à condition que `register(intelligence=True)` soit appelé **avant** la première instanciation `LightRAG(...)`, comme `register()` actuel. Si on appelle après, les instances créées avant ne bénéficient pas du patch `aquery` (la classe est patchée, mais une instance déjà créée peut avoir un `__dict__` qui shadow ?). En pratique non — `aquery` est une méthode définie sur la classe, pas sur l'instance, donc le patch propage. À tester unitaire-ment.

### 9.4 Fallback / résilience

Comportement actuel `TwinRAGEngine` :

- `IntentClassifier.classify` : exception → return `IntentType.IN_SCOPE, confidence=0.0, reason="Fallback (error): ..."` (`intent_classifier.py:115-121`). Donc échec OOS = laisse passer. **OK**.
- `ReasoningEngine.analyze` : exception → `search_query=question` (la query brute) (`reason.py:147-153`). **OK**.
- `QueryExpander.expand_v2` : exception → fallback `expand` v1 thesaurus (`query_expander.py:136-139`). **OK**.
- `CognitiveReranker.rerank` : exception → tri par `chunks.score` baseline, retour `chunks[:final_limit]` (`cognitive_reranker.py:147-150`). **OK**.

Côté wrapper `aquery` (7.A) — il faut **try/except** autour de chaque step L3 pour ne **pas** faire crasher le retrieval baseline si Twin a un bug :

```python
try:
    intent = await engine.intent_classifier.classify(query)
except Exception as e:
    logger.warning("Intent classification failed: %s — proceeding with retrieval", e)
    intent = None
```

C'est le pattern que `register()` adopte déjà pour `_use DATABASE`, `index already exists`, etc. Cohérent.

### 9.5 Feature flag : `register(intelligence=False)` (default)

Signature proposée :

```python
def register(intelligence: bool = False,
             intelligence_config: TwinRAGConfig | None = None) -> None:
    """Monkey-patch LightRAG's storage registries (+ optionally intelligence layer)."""
```

Garanties si `intelligence=False` :

- `_patch_intelligence_layer` n'est **pas** appelé → `LightRAG.aquery` reste la version native ;
- aucun import de `intelligence.engine`, `intelligence.features.*` n'est déclenché (sauf si on les importe au top du module — actuellement ils sont **lazy** via l'import à l'intérieur de `_patch_intelligence_layer`). Vérifier que `engine.py:17` ne fait pas d'import side-effect lourd : `from lightrag import LightRAG, QueryParam` est neutre, OK ;
- coût mémoire additionnel : 0 octet (rien n'est chargé) ;
- coût latence : 0 (pas de wrapper Python additionnel sur `aquery`).

Le test `pytest -k "not intelligence"` doit passer sans `OPENAI_API_KEY` configuré, car aucune des classes Twin n'est instanciée. À documenter en ligne dans le docstring de `register()`.

### 9.6 Cache

LightRAG cache deux types :

1. **keywords** (`cache_type="keywords"`) — `extract_keywords_only` (`$LRAG/operate.py:3344-3362, 3402-3431`). Si on **bypass** via `param.hl_keywords/ll_keywords` (option REASON A), ce cache n'est pas alimenté ni consulté.
2. **query** (`cache_type="query"`) — la réponse finale (`$LRAG/operate.py:3225-3268`). Conservé tant qu'on garde le LLM call natif (i.e. Twin 7.A sans Option A synthesis).

Si on choisit Option A synthesis (§6.4), on perd **les deux** caches LLM. À reconstruire côté Twin (LRU local sur `(query, mode, workspace)` → réponse). Pas critique pour v1.

### 9.7 Compatibilité avec `aquery_data` / `aquery_llm`

Le patch 7.A wrap **uniquement `aquery`**. Les callers internes du wrapper L3 qui passent par `aquery_data` (ex : la maquette `/api/retrieval`, le `webui_router.py` du Twin server, `aquery_llm` direct) ne sont **pas** interceptés. Si l'on veut couvrir aussi ces routes :

- patcher aussi `aquery_data` (`$LRAG/lightrag.py:2514-2719`) ;
- patcher aussi `aquery_llm` (`$LRAG/lightrag.py:2721-2857`).

Pour v1 : `aquery_data` est destiné aux callers qui veulent **les données brutes** (maquette, WebUI) — il serait contre-productif d'y injecter Intent/REASON qui transforment la question. Garder `aquery_data` natif.

`aquery_llm` est appelé par `aquery`, donc patcher `aquery_llm` revient à patcher tout. Préférer patcher au niveau `aquery` (consommateurs publics) pour ne pas affecter `aquery_data` qui appelle aussi `kg_query` mais sans passer par `aquery_llm`.

### 9.8 Streaming

`aquery(stream=True)` retourne un `AsyncIterator[str]` (`$LRAG/lightrag.py:2488-2491`). Le wrapper L3 doit **propager** ce mode sans bufferiser :

```python
async def _twin_aquery(self, query, param=QueryParam(), system_prompt=None):
    # ... intent + reason + f03 ...
    return await _orig_aquery(self, resolved_query, param, system_prompt)
    # async iterator passe transparent
```

Pas de risque si `_orig_aquery` retourne déjà l'iterator (pas un `await` inutile dessus).

### 9.9 Multi-workspace

`TwinRAGEngine.aquery` (`engine.py:164-170`) lance N retrievals en parallèle (un par workspace). Le wrapper 7.A sur `LightRAG.aquery` se fait sur **une** instance LightRAG, donc un workspace. Pour multi-workspace, deux options :

- bypass `LightRAG.aquery` et utiliser directement `TwinRAGEngine.aquery` qui gère lui-même les instances LightRAG ;
- patcher au niveau `LightRAG.aquery` pour le single-workspace, et garder `TwinRAGEngine` comme orchestrateur multi-workspace par-dessus.

Pour v1, choix recommandé : **le second**. `TwinRAGEngine.aquery` reste l'entry-point caller (`engine.py:81-198`), et chacun de ses `self.search.hybrid_search(rag, query, config)` (`engine.py:169`) appelle un `rag.aquery(query, ...)` qui passe par le wrapper L3 single-instance — mais ça boucle ! Le wrapper L3 re-applique intent/reason/f03 sur une query déjà reason+expanded.

**Résolution** : le wrapper L3 doit savoir si on est dans un appel **direct caller → LightRAG.aquery** (cas FastAPI ou maquette) ou **TwinRAGEngine.search → LightRAG.aquery** (cas L3 déjà fait son boulot). Solution :

- ajouter un flag `param._twin_intelligence_skip: bool = False` (champ privé de `QueryParam`) ;
- `TwinRAGEngine.search.hybrid_search` met `_twin_intelligence_skip = True` ;
- le wrapper L3 le respecte → pass-through direct vers `_orig_aquery`.

Code :

```python
async def _twin_aquery(self, query, param=QueryParam(), system_prompt=None):
    if getattr(param, "_twin_intelligence_skip", False):
        return await _orig_aquery(self, query, param, system_prompt)
    # ... full L3 pipeline ...
```

C'est un hack léger mais qui évite la double-application et garde la séparation de responsabilités. Variante propre : sous-classer `QueryParam` en `TwinQueryParam(QueryParam)` avec champ `bypass_intelligence`. Plus rigoureux, +20 LOC.

---

## 10. Recommandation finale

### 10.1 Choix de stratégie pour v1

**Stratégie A (Plug-in `aquery`)**, raison : effort minimal (~60 LOC), risque upstream nul, réversibilité totale, couverture L3 complète des trois phases pre-retrieval (F05, REASON, F03), et la phase rerank passe par le slot officiel `rerank_model_func`.

OBSERVE/synthesis reste en mode natif pour v1 — l'engagement `[Passage X]` vs `[reference_id]` est moins critique que la lecture du sub-graphe filtré par tag (priorité maquette).

### 10.2 Roadmap progressive

#### v1 — `register(intelligence=True)` minimaliste

- Patch `LightRAG.aquery` (Stratégie A) avec :
  - F05 Intent (gate OOS / GREETING / MALICIOUS) ;
  - REASON (coref + domain detect) → query résolue ;
  - F03 Expansion v1/v2 → query enrichie ;
  - délégation à `_orig_aquery` (LightRAG natif).
- Adapter `CognitiveReranker` → `rerank_model_func` signature, l'injecter via `LightRAG(rerank_model_func=...)` côté caller (responsabilité du caller, pas de `register()`).
- Flag `_twin_intelligence_skip` sur `QueryParam` pour éviter la double-application en multi-workspace orchestré par `TwinRAGEngine`.
- Tests :
  - `register(intelligence=False)` doit conserver `LightRAG.aquery` natif (pas de wrapper) ;
  - `register(intelligence=True)` doit court-circuiter sur intent OOS ;
  - `register(intelligence=True)` doit pre-fill `param.hl_keywords/ll_keywords` (option B-1 de §4.2) pour éviter le LLM call keyword extraction natif ;
  - fallback `intelligence` désactivé en local-only via env `TWIN_RAG_ENABLE_OOS_DETECTION=false`.

#### v2 — Synthesis Twin via `aquery_data`

- Option A de §6.4 : wrapper `aquery` qui appelle `aquery_data` puis `SynthesisEngine.synthesize` au lieu du `rag_response` natif.
- Garder le mode natif activable via `param.user_prompt = "USE_LIGHTRAG_NATIVE_SYNTHESIS"` (escape hatch).
- Cache LRU réponse côté Twin (clé : hash de query + mode + workspace).
- Tests sur le format `[Passage X]` vs `[reference_id]`.

#### v3 — Patches granulaires (Stratégie C, si v1+v2 insuffisants)

- Multi-step ReAct loop (cf. `reason → act → observe → critique → re-act → observe-final`).
- Rerank cross-modal (entities + relations + chunks dans un seul LLM call).
- Patches sur `_perform_kg_search` pour multi-workspace en interne LightRAG (vs. `TwinRAGEngine` orchestrant N instances).

### 10.3 Risque global

Niveau **faible** pour v1. Les trois fonctions L3 (`IntentClassifier`, `ReasoningEngine`, `QueryExpander`) ont chacune un fallback documenté (cf. §9.4). Le seul risque non-fallbackable est :

- Si `OPENAI_API_KEY` (chat LLM) n'est pas configuré et `register(intelligence=True)` est appelé, le premier `IntentClassifier.classify` crashe en `OpenAIError`. Le fallback retourne `IN_SCOPE, confidence=0.0`, donc la pipeline continue, **mais** ce n'est pas explicit ; à logger en `ERROR` plutôt qu'en `WARNING` pour visibilité.

### 10.4 Acceptance criteria v1

- `pytest tests/ --ignore=tests/test_bench.py` toujours vert ;
- nouveau test unitaire `tests/test_intelligence_register.py` couvrant les 4 scénarios (F05 gates + REASON + F03 + délégation native) ;
- benchmark `tests/test_bench.py` avec `register(intelligence=False)` → identique au baseline (régression nulle) ;
- benchmark avec `register(intelligence=True)` → +1 à 2 s latence max sur 95 percentile, documenté dans `changelog.md`.

---

## Cross-prismes notes

- **Prisme A (HTTP UI mount)** — pas de contradiction. Le wrapper L3 vit sous l'API `lightrag-server`, pas dans la route HTTP elle-même.
- **Prisme B (Boot lifecycle)** — point d'attention : si `register(intelligence=True)` est appelé **avant** `lightrag-server` import, le patch s'applique. Si l'appel est différé (lazy), les premières requêtes peuvent passer en natif. Prisme B documente probablement les hooks de boot — vérifier que `register()` est appelé en `pre_init` ou `__post_init__` de `LightRAG`.
- **Prisme C (API contracts)** — `aquery_data` retourne un schéma fixé (`{entities, relationships, chunks, references}`, cf. `$LRAG/lightrag.py:2538-2603`). Option A v2 (§6.4) **doit** respecter ce schéma si la maquette ou la WebUI lisent `aquery_data` plutôt que `aquery` — vérifier dans Prisme C que les endpoints `/api/query/retrieval` ou équivalent passent par `aquery_data`. Si oui, ne pas patcher `aquery_data`.
- **Prisme D (Auth/Security)** — pas de contradiction. F05 Intent classification = filtre sémantique, pas un filtre auth. Une question MALICIOUS ne dispense **pas** des contrôles JWT / RBAC documentés en Prisme D.
- **Prisme E (Observability)** — `engine.py:73` (`self.trace.start()` / `trace.stop()`) produit un `QueryTrace` qui n'est **pas** logué via le `lightrag` logger natif. À aligner sur le logger `lightrag` (cf. Prisme E) ou exposer en tracing LangSmith (extra `[tracing]` du repo, `tracing.py` du serveur). Le wrapper 7.A doit, dans son fallback `except`, logger via `logger = logging.getLogger("twindb_lightrag_memgraph")` (déjà créé dans `__init__.py:25`).
- **Prisme G (Vulnerabilities)** — `IntentType.MALICIOUS` court-circuite la pipeline avec une réponse scriptée. Vérifier que la classification MALICIOUS ne devient pas un **side-channel** : un attaquant qui sonde le système peut deviner que sa requête a été classée MALICIOUS si la latence est sensiblement plus courte (~300 ms vs ~5 s). Ajouter une latence artificielle minimale (`asyncio.sleep(random.uniform(1.0, 2.0))`) avant retour scripté si Prisme G recommande cette pratique.

---

## Annexe — Tableau hook-points × phase L3

| Phase L3        | Hook-point natif (`$LRAG/...`)                                   | Slot officiel ?           | Effort patch (Strategie A) |
|-----------------|------------------------------------------------------------------|---------------------------|----------------------------|
| F05 Intent      | `LightRAG.aquery` `lightrag.py:2459`                              | non (wrap)                | trivial (5 LOC)            |
| REASON coref    | `LightRAG.aquery` `lightrag.py:2459` (avant `aquery_llm`)         | non (wrap)                | trivial (5 LOC)            |
| F03 Expansion (pre-fill keywords) | `param.hl_keywords/ll_keywords` lus en `operate.py:3316-3317` | **OUI** (QueryParam) | trivial (2 LOC)            |
| F03 Expansion (rewrite query)     | `LightRAG.aquery` `lightrag.py:2459`                  | non (wrap)                | trivial (3 LOC)            |
| F04 Rerank chunks                 | `LightRAG.rerank_model_func` `lightrag.py:359-360`    | **OUI** (callable slot)   | adapter (15 LOC)           |
| F04 Rerank entities/relations     | `_apply_token_truncation` `operate.py:3662`           | non (patch granulaire)    | élevé (v3)                 |
| OBSERVE (replace synthesis)       | `LightRAG.aquery` + `aquery_data` `lightrag.py:2514` | non (wrap +bypass)        | modéré (20 LOC)            |
| OBSERVE (modify prompt)           | `PROMPTS["rag_response"]` `prompt.py:224`             | non (interdit doctrine)   | trivial mais dangereux     |
| OBSERVE (post-process)            | wrap `LightRAG.aquery` post-return                    | non                       | trivial (5 LOC)            |
| Multi-workspace orchestration     | `TwinRAGEngine.aquery` external                       | externe (par-dessus)      | déjà fait (`engine.py:81`)|

---

## Annexe — Imports critiques à connaître

Les symboles **réimportés** dans `lightrag.lightrag` (donc nécessitant double-patch si modifiés au niveau `lightrag.operate`) :

```python
# $LRAG/lightrag.py:89-95
from lightrag.operate import (
    chunking_by_token_size,
    extract_entities,
    merge_nodes_and_edges,
    kg_query,           # CRITIQUE : double-patch obligatoire si modifié
    naive_query,        # CRITIQUE : double-patch obligatoire si modifié
    rebuild_knowledge_from_chunks,
)
```

Les symboles utilisés **uniquement intra-`operate.py`** (pas de double-patch) :

- `_perform_kg_search`, `_build_query_context`, `_apply_token_truncation`, `_merge_all_chunks`, `_build_context_str` ;
- `_get_node_data`, `_get_edge_data`, `_find_most_related_edges_from_entities`, `_find_most_related_entities_from_relationships` ;
- `_find_related_text_unit_from_entities`, `_find_related_text_unit_from_relations` ;
- `get_keywords_from_query`, `extract_keywords_only` ;
- `_get_vector_context`.

Les utilities **inter-modules** (potentiel double-patch à vérifier) :

- `process_chunks_unified` (`$LRAG/utils.py:2702`) — utilisé par `operate.kg_query` (via `_build_context_str`) ET par `operate.naive_query` ;
- `apply_rerank_if_enabled` (`$LRAG/utils.py:2618`) — utilisé par `process_chunks_unified` ;
- les deux sont importés via `from lightrag.utils import ...` au top d'`operate.py:15-42` — donc `operate.py` a des **copies locales** de ces symboles, et patcher `lightrag.utils.process_chunks_unified` ne propage pas à `lightrag.operate.process_chunks_unified`. Si on patche, double-patch sur les deux modules.

Cette annexe est la source de vérité pour tout futur `_patch_intelligence_layer` plus profond.
