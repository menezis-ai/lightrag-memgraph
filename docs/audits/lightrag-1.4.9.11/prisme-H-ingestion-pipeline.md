# Prisme H — Pipeline d'ingestion LightRAG 1.4.9.11

Audit du chemin d'**indexation** (POST `/documents/upload` → DocStatus + KV + Vector + Graph + extraction entités/relations) de LightRAG **tel qu'installé dans le venv local** (`/Users/julien/twindb-lightrag-memgraph/.venv/lib/python3.14/site-packages/lightrag/`) et confrontation avec le code du wrapper `twindb-lightrag-memgraph` (branche `feat/maquette-source-revalidation`). Complément structurel au Prisme F (retrieval). Toutes les références `$LRAG/...` ci-dessous désignent le venv ; les références sans préfixe (`src/...`) désignent ce repo.

## Avertissement de version

Le venv expose `lightrag.__version__ = "v1.4.10"` (cf. `$LRAG/__init__.py:3`). La cible BNP prod tourne en `1.4.9.11`. Pour ce prisme :

- la version du venv (1.4.10) sert d'oracle pour les signatures et l'arbre d'appels exact ;
- le diff `1.4.9.11 → 1.4.10` concerne **précisément** ce périmètre (pipeline d'ingestion, source-ids limit, gleaning) ;
- la matrice CI Forgejo (`1.4.9 / 1.4.9.11 / 1.4.11 / 1.4.12`) passe avec le même `_patch_merge_write_path` et le même `_patch_insert_done`, ce qui valide que les **noms** `merge_nodes_and_edges`, `_insert_done`, `ainsert`, `apipeline_enqueue_documents`, `apipeline_process_enqueue_documents`, `_process_extract_entities` ainsi que leurs signatures sont stables sur l'intervalle ;
- tout numéro de ligne `$LRAG/...:LLLL` ci-dessous est le numéro **1.4.10** ; en 1.4.9.11 la fonction est la même, à un offset de quelques lignes près lié aux changements en amont du fichier. Le code du wrapper (`src/twindb_lightrag_memgraph/__init__.py:649-685`) prend volontairement les deux signatures de `merge_nodes_and_edges` (ancien `(entity_map, edge_map, knowledge_graph_inst, ...)` et nouveau `(chunk_results, knowledge_graph_inst, entity_vdb, ...)`) en charge, donc le patch reste compatible 1.4.9.x et 1.4.10+.

Lecteur visé : ingénieur LightRAG/Twin chargé de (a) tenir la doctrine compliance Eric "chunks + vecteurs only, jamais le doc entier", (b) étendre les hooks `_patch_merge_write_path` / `_patch_insert_done` pour un audit trail BCE/DORA, (c) brancher l'intelligence layer L3 (DSEP) côté indexation, et son relecteur architecte.

---

## 1. Trace exhaustive d'un upload PDF

### 1.1 Diagramme ASCII complet

```text
HTTP POST /documents/upload (multipart/form-data, file=UploadFile)
│
│   $LRAG/api/routers/document_routes.py:2073-2231
│   upload_to_input_dir(background_tasks, file)
│
├─ sanitize_filename(file.filename, doc_manager.input_dir)            :2132
│  │ Path-traversal guard (`..`, `/`, `\`, control chars stripped).
│  │ Verifies final_path.is_relative_to(input_dir.resolve()).
│
├─ doc_manager.is_supported_file(safe_filename)                       :2134
│  │ Whitelist of 36 extensions (.txt .md .pdf .docx .pptx .xlsx
│  │ .json .py .cpp …) defined in DocumentManager.__init__
│  │ ($LRAG/api/routers/document_routes.py:768-808).
│
├─ rag.doc_status.get_doc_by_file_path(safe_filename)                 :2162
│  │ Synchronous filename-duplicate check → if hit, return
│  │ InsertResponse(status="duplicated", track_id=existing).
│
├─ file_path.exists()  (filesystem duplicate)                          :2176
│  │ → status="duplicated", track_id="".
│
├─ Async streaming write to input_dir                                  :2188-2207
│  │ aiofiles.open(file_path, "wb")
│  │ loop: chunk = await file.read(1MiB)
│  │       check bytes_written <= MAX_UPLOAD_SIZE (default 100MB)
│  │       await out_file.write(chunk)
│  │ ── At this point the *full PDF bytes* live on disk in
│  │    `${INPUT_DIR}/${workspace}/{safe_filename}`.
│
├─ track_id = generate_track_id("upload")                             :2222
│
└─ background_tasks.add_task(pipeline_index_file, rag, file_path,     :2225
                              track_id)
   │  HTTP returns 200 InsertResponse(status="success", track_id=…).
   │
   ▼ (FastAPI fires the background coroutine after the response is sent)

pipeline_index_file(rag, file_path, track_id)
│   $LRAG/api/routers/document_routes.py:1651-1668
│
└─ await pipeline_enqueue_file(rag, file_path, track_id)              :1660
   │   $LRAG/api/routers/document_routes.py:1194-1648
   │
   ├─ aiofiles.open(file_path, "rb").read()  → `file` bytes in RAM     :1224
   │
   ├─ match ext:                                                       :1268-1530
   │  │  case ".pdf":  content = await asyncio.to_thread(
   │  │                  _extract_pdf_pypdf, file, pdf_decrypt_password)
   │  │       _extract_pdf_pypdf:1: PdfReader(BytesIO(file_bytes))
   │  │       loop: content += page.extract_text() + "\n"
   │  │  case ".docx": _extract_docx(file)
   │  │  case ".pptx": _extract_pptx(file)
   │  │  case ".xlsx": _extract_xlsx(file)
   │  │  case .txt|.md|.json|.py|.css|...:
   │  │       content = file.decode("utf-8")
   │  │  case _: → apipeline_enqueue_error_documents([{...}])
   │
   ├─ if not content.strip(): → apipeline_enqueue_error_documents      :1550
   │
   ├─ await rag.apipeline_enqueue_documents(                           :1566
   │      content, file_paths=file_path.name, track_id)
   │   $LRAG/lightrag.py:1265-1442
   │   │
   │   ├─ sanitize_text_for_encoding(doc)                              :1324
   │   ├─ doc_id = compute_mdhash_id(content, prefix="doc-")           :1343
   │   ├─ Build new_docs[doc_id] = {status: PENDING, content_summary,
   │   │    content_length, created_at, updated_at, file_path,
   │   │    track_id}  (no `content` field here)                       :1351-1364
   │   ├─ unique_new_doc_ids = doc_status.filter_keys(all_new)         :1370
   │   ├─ Handle filename collisions → DocStatus(FAILED, dup-…)        :1373-1412
   │   ├─ full_docs_data[doc_id] = {content, file_path}                :1427
   │   ├─ await self.full_docs.upsert(full_docs_data)                  :1434
   │   │   → MemgraphKV node :KV_{ws}_full_docs {id, data=JSON(content
   │   │     + file_path)} via UNWIND+MERGE (kv_impl.py:123-145).
   │   ├─ await self.full_docs.index_done_callback()                   :1436
   │   │   (no-op on Memgraph backend)
   │   └─ await self.doc_status.upsert(new_docs)                       :1439
   │       → MemgraphDocStatus node :DocStatus_{ws} {id, status:
   │         "PENDING", track_id, file_path, content_summary,
   │         content_length, ...} (docstatus_impl.py).
   │
   └─ if success:
      await rag.apipeline_process_enqueue_documents(                  :1664
            split_by_character=None, split_by_character_only=False)
       │   $LRAG/lightrag.py:1642-2203
       │
       ├─ pipeline_status = get_namespace_data("pipeline_status", ws)  :1660
       │  pipeline_status_lock = get_namespace_lock("pipeline_status",
       │  workspace=ws)
       │
       ├─ if pipeline_status["busy"]: set request_pending=True, return :1701-1706
       │
       ├─ Gather PROCESSING + FAILED + PENDING docs from doc_status    :1671-1680
       │
       ├─ pipeline_status.update({busy: True, job_name: "Default Job",
       │   docs, batchs, cur_batch, request_pending: False,
       │   cancellation_requested: False, ...})                         :1686-1700
       │
       ├─ _validate_and_fix_document_consistency()                     :1736
       │   (purge PROCESSING/FAILED entries without full_docs payload,
       │    reset PROCESSING+FAILED → PENDING)
       │
       ├─ semaphore = asyncio.Semaphore(max_parallel_insert)            :1778
       │   default MAX_PARALLEL_INSERT = 4 ($LRAG/constants.py)
       │
       ├─ doc_tasks = [process_document(doc_id, status_doc, ...)        :2136-2148
       │               for each PENDING doc]
       │   await asyncio.gather(*doc_tasks)                             :2152
       │
       │   For each doc, process_document(...) at :1780-2133:
       │   │
       │   ├─ async with semaphore: (≤ MAX_PARALLEL_INSERT in flight)
       │   │
       │   ├─ content_data = await self.full_docs.get_by_id(doc_id)    :1840
       │   │  content = content_data["content"]
       │   │  ── Full PDF text loaded back into RAM ──
       │   │
       │   ├─ chunking_result = self.chunking_func(                    :1848
       │   │      self.tokenizer, content, split_by_character,
       │   │      split_by_character_only, chunk_overlap_token_size,
       │   │      chunk_token_size)
       │   │   → chunking_by_token_size (operate.py:99-162) by default.
       │   │
       │   ├─ chunks = {compute_mdhash_id(dp["content"], "chunk-"):    :1869
       │   │              {**dp, full_doc_id, file_path,
       │   │               llm_cache_list: []}
       │   │            for dp in chunking_result}
       │   │
       │   ├─ Stage 1 — fire 3 parallel tasks via asyncio.gather:      :1892-1932
       │   │   ├─ doc_status.upsert({doc_id: {status: PROCESSING,
       │   │   │     chunks_count, chunks_list, ...}})
       │   │   ├─ chunks_vdb.upsert(chunks)
       │   │   │   → embedding_func(chunk.content) for each chunk
       │   │   │   → MemgraphVec node :Vec_{ws}_chunks
       │   │   │     {id=chunk_id, embedding, content, full_doc_id,
       │   │   │      file_path} (vector_impl.py).
       │   │   └─ text_chunks.upsert(chunks)
       │   │       → MemgraphKV node :KV_{ws}_text_chunks
       │   │         {id, data=JSON(content+tokens+full_doc_id+
       │   │          chunk_order_index+file_path+llm_cache_list)}
       │   │
       │   ├─ Stage 2 — entity_relation_task =                          :1935
       │   │   await self._process_extract_entities(chunks, ...)
       │   │
       │   │   _process_extract_entities → extract_entities             :2205-2224, $LRAG/operate.py:2813-3081
       │   │   ├─ semaphore = asyncio.Semaphore(llm_model_max_async)   :3017-3018
       │   │   │   default MAX_ASYNC = 4
       │   │   ├─ For each chunk in chunks:                             :3037
       │   │   │   _process_single_content(chunk):
       │   │   │   ├─ PROMPTS["entity_extraction_system_prompt"]         operate.py:2881
       │   │   │   │  + PROMPTS["entity_extraction_user_prompt"]         operate.py:2885
       │   │   │   ├─ final_result = await use_llm_func_with_cache(      operate.py:2892
       │   │   │   │     user_prompt, system_prompt, cache_type="extract",
       │   │   │   │     chunk_id, cache_keys_collector)
       │   │   │   ├─ (maybe_nodes, maybe_edges) = _process_extraction_result(
       │   │   │   │       final_result, chunk_key, timestamp, file_path,
       │   │   │   │       tuple_delimiter, completion_delimiter)        operate.py:2907
       │   │   │   ├─ if entity_extract_max_gleaning > 0:                operate.py:2917
       │   │   │   │   glean_result = use_llm_func_with_cache(           operate.py:2938
       │   │   │   │       continue_extraction_user_prompt, ...)
       │   │   │   │   merge glean_nodes/glean_edges into maybe_nodes/edges
       │   │   │   ├─ update_chunk_cache_list(chunk_key, text_chunks,    operate.py:2995
       │   │   │   │     cache_keys_collector, "entity_extraction")
       │   │   │   └─ return (maybe_nodes, maybe_edges)
       │   │   └─ chunk_results = list of (maybe_nodes, maybe_edges)
       │   │
       │   └─ Stage 3 — merge_nodes_and_edges(                          :2023-2040
       │           chunk_results, knowledge_graph_inst=
       │           self.chunk_entity_relation_graph,
       │           entity_vdb=self.entities_vdb,
       │           relationships_vdb=self.relationships_vdb,
       │           full_entities_storage, full_relations_storage,
       │           doc_id, pipeline_status, pipeline_status_lock,
       │           llm_response_cache, entity_chunks_storage,
       │           relation_chunks_storage, current_file_number,
       │           total_files, file_path)
       │
       │       ──> intercepted by `_buffered_merge_nodes_and_edges`
       │           in src/twindb_lightrag_memgraph/__init__.py:649-680
       │           which substitutes the graph instance with a
       │           _BufferedGraphProxy (buffered_graph.py:17-78)
       │           and calls proxy.flush() at the end (2 UNWIND queries).
       │
       │       merge_nodes_and_edges body ($LRAG/operate.py:2443-2810):
       │       ├─ Aggregate all_nodes / all_edges across chunks         :2493-2505
       │       ├─ Phase 1: entities (concurrent, sem(llm_max_async*2))  :2520-2619
       │       │   For each (entity_name, entities):
       │       │     get_storage_keyed_lock([entity_name], ns=ws+":GraphDB")
       │       │     _merge_nodes_then_upsert(...)
       │       │       ├─ knowledge_graph_inst.get_node(entity_name)
       │       │       ├─ entity_chunks_storage.get_by_id(entity_name)
       │       │       ├─ _handle_entity_relation_summary(...)  ← LLM if
       │       │       │     ≥ force_llm_summary_on_merge descriptions
       │       │       │     OR if joined > summary_context_size tokens
       │       │       ├─ entity_chunks_storage.upsert({entity: {chunk_ids,
       │       │       │     count}})
       │       │       ├─ knowledge_graph_inst.upsert_node(entity_name,
       │       │       │     node_data)    ← BUFFERED via proxy
       │       │       └─ entity_vdb.upsert({mdhash("ent-"): {content,
       │       │             entity_name, entity_type, source_id, file_path}})
       │       ├─ Phase 2: relations (concurrent)                       :2621-2739
       │       │   For each (edge_key, edges):
       │       │     get_storage_keyed_lock(sorted(edge_key), ws+":GraphDB")
       │       │     _merge_edges_then_upsert(...)
       │       │       ├─ knowledge_graph_inst.has_edge / get_edge
       │       │       ├─ relation_chunks_storage.get_by_id(storage_key)
       │       │       ├─ _handle_entity_relation_summary(...)  ← maybe LLM
       │       │       ├─ relation_chunks_storage.upsert({key: {chunk_ids,
       │       │       │     count}})
       │       │       ├─ For src,tgt: if !has_node → upsert_node(...) ← BUFFERED
       │       │       ├─ knowledge_graph_inst.upsert_edge(src, tgt,    ← BUFFERED
       │       │       │     edge_data)
       │       │       └─ relationships_vdb.upsert({mdhash("rel-"): ...})
       │       ├─ Phase 3: full_entities_storage.upsert({doc_id:        :2741-2794
       │       │     {entity_names, count}})  /
       │       │   full_relations_storage.upsert({doc_id:
       │       │     {relation_pairs, count}})
       │       │
       │       └─ proxy.flush()  (added by wrapper)
       │           _BufferedGraphProxy._flush_nodes  → 1 UNWIND MERGE   buffered_graph.py:108-142
       │             + 1 query per distinct entity_type for SET n:`type`
       │           _BufferedGraphProxy._flush_edges  → 1 UNWIND MATCH   buffered_graph.py:144-168
       │             + MERGE [r:DIRECTED] + SET r += props
       │
       ├─ doc_status.upsert({doc_id: {status: PROCESSED,                :2045-2065
       │     chunks_count, chunks_list, ...,
       │     metadata: {processing_start_time, processing_end_time}}})
       │
       ├─ await self._insert_done()                                    :2068
       │   ──> intercepted by `_hooked_insert_done` in
       │       src/twindb_lightrag_memgraph/__init__.py:705-712
       │
       │   _insert_done body ($LRAG/lightrag.py:2226-2255):
       │   asyncio.gather(*[storage.index_done_callback()
       │                    for storage in 12 storages])
       │     full_docs, doc_status, text_chunks, full_entities,
       │     full_relations, entity_chunks, relation_chunks,
       │     llm_response_cache, entities_vdb, relationships_vdb,
       │     chunks_vdb, chunk_entity_relation_graph.
       │
       │   On Memgraph backends index_done_callback() is a no-op
       │   (kv_impl.py:68, vector_impl.py search, docstatus_impl.py:80,
       │   memgraph_impl.py:119-121). Memgraph persists automatically.
       │
       │   Then `_run_post_index_hooks(self)` fires every callback
       │   registered via register_post_index_hook(...)
       │   (_hooks.py:36-42).
       │
       └─ if success: file moved to ${input_dir}/__enqueued__/...      :1574-1589
          (renamed `pipeline_enqueue_file` end). The original PDF
          stays on disk in __enqueued__ until the operator runs
          `delete_file=True` on the document.
```

### 1.2 Résumé : 6 transitions principales

| # | Fichier:Ligne | Acteur | Effet réseau / disque |
|---|---|---|---|
| 1 | `document_routes.py:2188-2207` | FastAPI worker | Écriture streaming du PDF sur disque (`input_dir/`) |
| 2 | `document_routes.py:1224` | Background task | Relecture des bytes du PDF en mémoire |
| 3 | `lightrag.py:1434-1439` | `apipeline_enqueue_documents` | UPSERT KV `full_docs` (1) + UPSERT DocStatus PENDING (1) |
| 4 | `lightrag.py:1916-1932` | `process_document` Stage 1 | UPSERT DocStatus PROCESSING + UPSERT Vec `chunks` (N chunks, embedding) + UPSERT KV `text_chunks` (N chunks), en parallèle |
| 5 | `lightrag.py:1935-1940` | `_process_extract_entities` | N appels LLM (chunk → entities/relations), max `llm_model_max_async` en parallèle |
| 6 | `lightrag.py:2023-2040` | `merge_nodes_and_edges` | E UPSERT graph nodes + R UPSERT graph edges (intercepté par notre proxy → 2-3 UNWIND), + UPSERT entity/relation VDB, + UPSERT `entity_chunks`/`relation_chunks` KV, + UPSERT `full_entities`/`full_relations` KV |
| 7 | `lightrag.py:2045-2068` | `_insert_done` | UPSERT DocStatus PROCESSED + flush index_done_callback() (no-op sur Memgraph) + `_run_post_index_hooks(self)` |

Pour un PDF qui produit 50 chunks × 5 entités et 5 relations par chunk en moyenne (≈ 250 entités déduplicables + 250 relations) :
- **5 + 250 + 250 = ~505 appels LLM** au maximum (1 extraction par chunk + 1 gleaning si activé, + 1 résumé par entité/relation au-delà de `force_llm_summary_on_merge` descriptions cumulées, défaut 6) ;
- en pratique le LLM cache (`llm_response_cache`) court-circuite la plupart des résumés sur réingestion ;
- **2 UNWIND graph queries** (au lieu de ~500 round-trips) grâce à notre `_BufferedGraphProxy` (cf. §10.3 ci-dessous).

---

## 2. Où le document brut existe-t-il en mémoire / sur disk

### 2.1 Inventaire exhaustif des copies du contenu

| # | Localisation | Type | Source | Durée de vie |
|---|---|---|---|---|
| **D1** | `${INPUT_DIR}/${workspace}/{safe_filename}` | Disque (PDF binaire) | Écriture streaming par `upload_to_input_dir` (`document_routes.py:2188-2207`) | **Permanente jusqu'à `delete_by_doc_id(delete_file=True)`**, ensuite déplacée vers `__enqueued__/` après `pipeline_enqueue_file` (`document_routes.py:1586`) |
| **D2** | `${INPUT_DIR}/${workspace}/__enqueued__/{filename}` | Disque (PDF binaire) | `file_path.rename(target_path)` (`document_routes.py:1586`) | **Permanente** jusqu'à `delete_file=True` sur ce doc, sinon **infinie** |
| **M1** | RAM, `file: bytes` | Mémoire (PDF binaire) | `await f.read()` dans `pipeline_enqueue_file` (`document_routes.py:1224`) | Le scope de la coroutine `pipeline_enqueue_file` (libéré au `return`) |
| **M2** | RAM, `content: str` | Mémoire (texte extrait UTF-8) | `_extract_pdf_pypdf(file)` / `_extract_docx(file)` etc. (`document_routes.py:1372-1499`) | Survit jusqu'au `await rag.apipeline_enqueue_documents(content, ...)` puis libéré |
| **K1** | Memgraph `:KV_{ws}_full_docs` node, propriété `data` (JSON) | Memgraph (texte extrait) | `full_docs.upsert({doc_id: {content, file_path}})` (`lightrag.py:1434`) | **Permanente** jusqu'à `adelete_by_doc_id` (`lightrag.py:3624`) |
| **M3** | RAM, `content_data["content"]` (relu depuis K1) | Mémoire (texte extrait) | `await self.full_docs.get_by_id(doc_id)` (`lightrag.py:1840`) | Scope de `process_document` (libéré en sortie) |
| **M4** | RAM, `chunks[chunk_id]["content"]` (par chunk) | Mémoire (chunks) | `chunking_by_token_size(...)` (`operate.py:99-162`) | Vit pendant `process_document` |
| **K2** | Memgraph `:KV_{ws}_text_chunks` node, propriété `data` | Memgraph (texte par chunk) | `text_chunks.upsert(chunks)` (`lightrag.py:1920`) | **Permanente** jusqu'à `adelete_by_doc_id` (`lightrag.py:3440`) |
| **V1** | Memgraph `:Vec_{ws}_chunks` node, propriétés `content` + `embedding` | Memgraph (texte par chunk + vecteur) | `chunks_vdb.upsert(chunks)` (`lightrag.py:1917`) | **Permanente** jusqu'à `adelete_by_doc_id` (`lightrag.py:3439`) |
| **M5** | RAM, prompt `entity_extraction_user_prompt` formaté (contient `content` du chunk) | Mémoire (chunk text) | `PROMPTS["entity_extraction_user_prompt"].format(**{..., "input_text": content})` (`operate.py:2885`) | Scope du call LLM ; envoyé via `use_llm_func_with_cache` à OpenAI/Ollama/etc. |
| **K3** | Memgraph `:KV_{ws}_llm_response_cache` (si `enable_llm_cache_for_entity_extract=True`) | Memgraph (texte chunk + JSON entités/relations) | `save_to_cache(...)` dans `use_llm_func_with_cache` ($LRAG/utils.py) | **Permanente** sauf `aclear_cache` ou `delete_llm_cache=True` sur le doc (`lightrag.py:3630`) |

### 2.2 Doctrine Eric "chunks + vecteurs only, jamais le doc entier"

**La doctrine n'est PAS tenue par construction dans LightRAG 1.4.9.11.**

Trois fuites avérées :

1. **K1 — `full_docs`** : `apipeline_enqueue_documents` persiste le **texte complet** du document (post-extraction, après pypdf/docx) dans la KV namespace `full_docs` (`lightrag.py:1434`). Notre `MemgraphKVStorage` le stocke en JSON sur la propriété `data` du node `:KV_{ws}_full_docs`. **Persiste tant que le document n'est pas explicitement supprimé.**

2. **K3 — `llm_response_cache`** : par défaut `enable_llm_cache_for_entity_extract=True` (`lightrag.py:376`). Chaque appel LLM d'extraction d'entités stocke le prompt complet (qui contient le `content` du chunk via `entity_extraction_user_prompt`, `operate.py:2885`) + la réponse LLM dans la KV `llm_response_cache`. Sur 50 chunks ce sont 50 copies texte du chunk, + 50 si gleaning activé.

3. **D1/D2 — input_dir** : le PDF binaire **reste sur disque** dans `${INPUT_DIR}/${workspace}/` (ou `__enqueued__/` après enqueue). Aucune purge automatique — seul `adelete_by_doc_id(delete_file=True)` côté `background_delete_documents` (`document_routes.py:1882-1985`) supprime physiquement le fichier.

Les chunks (K2 `text_chunks`, V1 `chunks` vector) sont des copies **par chunk**, conformes à la doctrine. Les entités/relations stockent `source_id` (concat des `chunk-{hash}`) et `file_path` mais pas le texte intégral.

### 2.3 Mesures pour tenir la doctrine

| Fuite | Remediation possible | Coût |
|---|---|---|
| K1 `full_docs` | Court-circuiter `full_docs.upsert` après ingestion : à la fin de `process_document` (avant `_insert_done`), purger l'entrée `full_docs[doc_id]`. **Risque** : casse `_validate_and_fix_document_consistency` (`lightrag.py:1529`) qui relit `full_docs.get_by_id(doc_id)` pour vérifier la cohérence DocStatus↔contenu lors d'un redémarrage. **Workaround** : remplacer le `content` par un placeholder `"[purged after indexation]"` plutôt que delete. | Faible, à patcher dans un nouveau `_patch_full_docs_purge_after_index()` |
| K3 `llm_response_cache` | (a) `enable_llm_cache_for_entity_extract=False` à l'instanciation `LightRAG(...)` ; (b) ou hook `_insert_done` qui purge les entrées de cache liées aux chunks du doc (utilise déjà `delete_llm_cache=True` côté `adelete_by_doc_id`, `lightrag.py:3177-3226`) | Si désactivé, on perd la réindexation gratuite via `rebuild_knowledge_from_chunks` ($LRAG/operate.py:553-1066) ; mauvaise idée pour BNP. Mieux : laisser le cache, mais offrir un mode "compliance" qui hash le prompt avec un sel par doc (la clef devient opaque). |
| D1/D2 `input_dir` | Définir `DELETE_AFTER_INGEST=true` côté worker : à la fin de `pipeline_enqueue_file`, plutôt que `rename` vers `__enqueued__/`, faire `file_path.unlink()`. Conséquence : on perd la possibilité de re-traitement du PDF sans réupload, mais on tient la doctrine. | Faible, à brancher via override de `pipeline_enqueue_file` ou wrapper FastAPI middleware |

---

## 3. Chunking strategy

### 3.1 `chunking_by_token_size` — la fonction par défaut

`$LRAG/operate.py:99-162` :

```python
def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
) -> list[dict[str, Any]]:
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        ...
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + chunk_token_size])
            results.append({
                "tokens": min(chunk_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            })
    return results
```

Paramètres et défauts :
- `chunk_token_size = int(os.getenv("CHUNK_SIZE", 1200))` (`lightrag.py:231`) — **1200 tokens par chunk**.
- `chunk_overlap_token_size = int(os.getenv("CHUNK_OVERLAP_SIZE", 100))` (`lightrag.py:234-236`) — **100 tokens d'overlap**.
- `split_by_character` / `split_by_character_only` : passés depuis `apipeline_process_enqueue_documents` (`lightrag.py:1644-1645`), tous deux `None`/`False` par défaut (pas d'usage côté HTTP — seul un caller Python peut les forcer via `rag.insert(..., split_by_character=...)`).
- `split_by_character` non vide : split d'abord sur le caractère, puis re-split par token-size si > `chunk_token_size`. Avec `split_by_character_only=True` : pas de re-split, lève `ChunkTokenLimitExceededError` (`operate.py:121-125`).

Le frontière est purement token-based — **aucun respect des frontières sémantiques** (paragraphes, sections, phrases). Pour les PDFs structurés (rapports BNP, normes ISO), c'est sous-optimal : un chunk peut couper au milieu d'une définition.

### 3.2 Custom chunkers : `chunking_func`

`LightRAG` accepte un `chunking_func` configurable (`lightrag.py:249-282`) :

```python
chunking_func: Callable[
    [Tokenizer, str, Optional[str], bool, int, int],
    Union[List[Dict[str, Any]], Awaitable[List[Dict[str, Any]]]],
] = field(default_factory=lambda: chunking_by_token_size)
```

- Signature attendue : `(tokenizer, content, split_by_character, split_by_character_only, chunk_overlap_token_size, chunk_token_size)`.
- Retour : `list[{"tokens": int, "content": str, "chunk_order_index": int}]` ou awaitable de cette liste — `inspect.isawaitable(chunking_result)` est vérifié (`lightrag.py:1858`).
- L'override se fait à l'instanciation : `LightRAG(..., chunking_func=my_chunker)`.

Hookpoint propre pour brancher un chunker semantic-aware (cf. LangChain RecursiveCharacterTextSplitter + headers detection) sans patcher LightRAG. **Aucun usage actuel dans Twin** — l'appel `LightRAG(...)` dans `server/app.py` n'override pas `chunking_func`, donc on est sur `chunking_by_token_size` par défaut.

### 3.3 Tokenizer — `TiktokenTokenizer`

`$LRAG/utils.py` instancie un `TiktokenTokenizer(self.tiktoken_model_name)` (`lightrag.py:507`). Le `tiktoken_model_name` par défaut est `"gpt-4o-mini"` (`lightrag.py:246`).

- **Library** : `tiktoken 0.12.0` (cf. Prisme G §2).
- **Encoding** pour `gpt-4o-mini` : `o200k_base` (50k+ vocabulary, BPE).
- **CVE** : aucune advisory connue par `pip-audit` à la date du Prisme G (2026-05-28).
- **Risque d'attaque sur tokenizer** : `tokenizer.encode()` charge un fichier d'encoding depuis l'IM `tiktoken/encodings/` à l'install ; au runtime aucun fichier externe n'est lu, donc pas de path-traversal exploitable via un PDF malveillant. Bonne posture.

### 3.4 Override CHUNK_SIZE pour la doctrine compliance

Pour réduire la surface du `text_chunks` KV (chaque chunk = 1 entrée), Eric pourrait pousser `CHUNK_SIZE=512` (vs 1200) — 2.3× plus d'entrées, mais chunks plus petits, alignement courant des embeddings (BGE/E5 family). Tradeoff embedding qualité vs nombre de round-trips graph. À benchmarker.

---

## 4. Extraction entités/relations

### 4.1 Pipeline `extract_entities`

`$LRAG/operate.py:2813-3081`. Pour chaque chunk :

1. **Construction prompt** (`operate.py:2881-2890`) :
   - `system_prompt` = `PROMPTS["entity_extraction_system_prompt"].format(tuple_delimiter, completion_delimiter, entity_types, language)` (`$LRAG/prompt.py:11-61`).
   - `user_prompt` = `PROMPTS["entity_extraction_user_prompt"].format(... , input_text=content)` (`$LRAG/prompt.py:63-82`).

2. **Appel LLM principal** (`operate.py:2892-2900`) :
   ```python
   final_result, timestamp = await use_llm_func_with_cache(
       entity_extraction_user_prompt,
       use_llm_func,
       system_prompt=entity_extraction_system_prompt,
       llm_response_cache=llm_response_cache,
       cache_type="extract",
       chunk_id=chunk_key,
       cache_keys_collector=cache_keys_collector,
   )
   ```
   - `use_llm_func = global_config["llm_model_func"]` (rappel : déjà wrappé `priority_limit_async_func_call(llm_model_max_async)` dans `lightrag.py:664-674`).
   - Cache key = hash(system+user+history+model). Hit → retour direct sans LLM call.

3. **Parsing format délimiteur custom** (`operate.py:2907-2914`) :
   - `_process_extraction_result(text, ...)` parse les lignes du format :
     ```
     entity<|#|>Paris<|#|>location<|#|>Paris is the capital of France.
     relation<|#|>Paris<|#|>France<|#|>capital, geography<|#|>Paris is the capital of France.
     <|COMPLETE|>
     ```
   - Délimiteurs : `<|#|>` (`PROMPTS["DEFAULT_TUPLE_DELIMITER"]`, `prompt.py:8`), `<|COMPLETE|>` (`prompt.py:9`).
   - Helpers : `_handle_single_entity_extraction` (`operate.py:379-463`) — attend 4 champs, valide `entity_type` (interdit `'`, `(`, `)`, `<`, `>`, `|`, `/`, `\`), sanitize, returns dict ou None.
   - `_handle_single_relationship_extraction` (`operate.py:466-550`) — attend 5 champs, weight parsing avec `is_float_regex` fallback à 1.0.
   - **Gestion erreur JSON malformé** : un parse partiel renvoie `None` silencieusement (juste un `logger.warning`), la ligne est skippée. Aucune exception ne remonte.

4. **Gleaning** (`operate.py:2917-2992`), gated par `entity_extract_max_gleaning > 0` (env `MAX_GLEANING`, défaut 1) :
   - Calcul du token count du prochain prompt : `len(tokenizer.encode(system + history_json + continue_user_prompt))`. Si dépasse `max_extract_input_tokens` (env `MAX_EXTRACT_INPUT_TOKENS`, défaut élevé), gleaning skippé.
   - Sinon : `use_llm_func_with_cache(entity_continue_extraction_user_prompt, ...)` avec `history_messages=history`.
   - Fusion entité-par-entité : on garde la version avec la description la plus longue (`operate.py:2960-2975`).

5. **Concurrency** (`operate.py:3016-3070`) :
   - `chunk_max_async = global_config.get("llm_model_max_async", 4)` (`operate.py:3017`) — soit `LightRAG.llm_model_max_async` (env `MAX_ASYNC`, défaut `DEFAULT_MAX_ASYNC=4` dans `$LRAG/constants.py`).
   - `semaphore = asyncio.Semaphore(chunk_max_async)`.
   - **MAIS** le `use_llm_func` est *déjà* wrappé par `priority_limit_async_func_call(self.llm_model_max_async, ...)` en `lightrag.py:664-674`. Donc le sémaphore extract + le sémaphore LLM s'empilent — la limite effective reste `llm_model_max_async` (le plus restrictif).

6. **Erreur LLM** :
   - Timeout : géré par `priority_limit_async_func_call` qui propage l'erreur après timeout.
   - Réponse vide / JSON corrompu : `_process_extraction_result` retourne `(maybe_nodes={}, maybe_edges={})` ; le chunk est compté comme "0 Ent + 0 Rel" mais ne fait pas planter le pipeline. Logguée en `INFO` (`operate.py:3006`).
   - Toute autre exception dans `_process_single_content` → `create_prefixed_exception(e, chunk_id)` (`operate.py:3034`) ; remonte vers `extract_entities` qui `wait(tasks, return_when=FIRST_EXCEPTION)` (`operate.py:3044`), cancel les pending, raise la première exception → bubble up vers `process_document` → DocStatus FAILED.

### 4.2 Retour de `extract_entities`

```python
chunk_results: list[tuple[
    dict[entity_name, list[entity_dict]],   # maybe_nodes
    dict[(src,tgt), list[edge_dict]],       # maybe_edges
]]
```

`entity_dict` = `{entity_name, entity_type, description, source_id (=chunk_key), file_path, timestamp}`.
`edge_dict` = `{src_id, tgt_id, weight, description, keywords, source_id, file_path, timestamp}`.

---

## 5. Write path consolidé

### 5.1 Orchestration KV + Vector + Graph + DocStatus

#### Étape pré-extraction (parallèle via `asyncio.gather`, `lightrag.py:1892-1932`)
- DocStatus.upsert(PROCESSING) — 1 round-trip Memgraph (`docstatus_impl.py`).
- chunks_vdb.upsert(chunks) — 1 embedding batch + 1 UNWIND insert vectoriel (`vector_impl.py`).
- text_chunks.upsert(chunks) — 1 UNWIND insert KV (`kv_impl.py:123-145`).

#### Étape extraction (sérielle dans le flow, mais parallèle inter-chunks)
- N appels LLM (cf. §4).

#### Étape merge (intercepté `_patch_merge_write_path`)
- Phase 1 entités (concurrent, locked par entity_name) :
  - Reads en passthrough : `get_node`, `entity_chunks_storage.get_by_id`. **Read-your-own-writes** via `_BufferedGraphProxy.get_node` (`_buffered_graph.py:54-58`) qui check le buffer avant de déléguer.
  - Éventuel LLM call (`_handle_entity_relation_summary`) si descriptions agrégées dépassent un seuil.
  - Writes graph : `upsert_node` → **bufferisé** dans `_node_buffer` (`_buffered_graph.py:33-40`).
  - Writes VDB : `entity_vdb.upsert({mdhash("ent-"): {...}})` — **PAS bufferisé**, 1 round-trip par entité (avec embedding).
  - Writes KV : `entity_chunks_storage.upsert(...)` — **PAS bufferisé**, 1 round-trip par entité.
- Phase 2 relations (concurrent, locked par sorted-edge-pair) :
  - Reads/writes graph idem (relations bufferisées dans `_edge_buffer`).
  - Writes VDB : `relationships_vdb.upsert` — 1 round-trip par relation.
- Phase 3 (sérielle après gather) :
  - `full_entities.upsert({doc_id: {entity_names, count}})` — 1 round-trip.
  - `full_relations.upsert({doc_id: {relation_pairs, count}})` — 1 round-trip.
- `proxy.flush()` :
  - 1 UNWIND MERGE pour TOUS les nodes du buffer.
  - 1 UNWIND query par `entity_type` distinct pour `SET n:`type``.
  - 1 UNWIND MATCH+MERGE pour TOUS les edges.

#### Étape finalisation (`lightrag.py:2045-2068`)
- DocStatus.upsert(PROCESSED) — 1 round-trip.
- `_insert_done()` → `gather([index_done_callback() for storage in 12])` — tous no-op sur Memgraph. + `_run_post_index_hooks(self)`.

### 5.2 Pas de transaction Memgraph globale

**Aucune transaction Memgraph n'embrasse l'ensemble des writes d'un document.** Chaque `session.run(...)` est sa propre transaction implicite (`execute_write` côté `memgraph_impl.py:528`). Conséquences :

- Si le merge edge échoue à mi-parcours, **les nodes déjà upsertés restent dans le graph**. La rebuild logic (`rebuild_knowledge_from_chunks`, `$LRAG/operate.py:553-1066`) peut récupérer à partir des chunks lors du redémarrage, mais on a un état transitoire incohérent.
- Le proxy buffer **n'est PAS transactionnel** : `_flush_nodes` puis `_flush_edges` sont 2 sessions séparées. Si l'edge flush plante, les nodes restent avec leur properties mais sans relations.

### 5.3 Race conditions sur entités partagées

Deux docs ingérés concurremment (`asyncio.gather` dans `process_document` doc tasks, `lightrag.py:2152`) qui partagent l'entité "Paris" :

- Le `get_storage_keyed_lock([entity_name], namespace="ws:GraphDB")` (`operate.py:2539`) **sérialise** les writes sur "Paris" : seul un task à la fois passe par `_merge_nodes_then_upsert("Paris", ...)`.
- Lock implémenté dans `$LRAG/kg/shared_storage.py` (par-clef in-process, fonctionne **uniquement intra-process**). Si deux workers uvicorn ingèrent en parallèle, ce lock est inopérant — la cohérence retombe sur Memgraph MVCC (cf. §10).
- À l'intérieur du proxy, le buffer est partagé par `process_document` du même doc, donc pas de race intra-doc. Inter-doc : chaque doc a son propre `_BufferedGraphProxy` instance.

---

## 6. Idempotence et redémarrage

### 6.1 État persisté dans DocStatus

DocStatus est l'**unique** source de vérité pour la reprise. Cinq statuts possibles (`$LRAG/base.py`) : `PENDING`, `PROCESSING`, `PREPROCESSED`, `PROCESSED`, `FAILED`.

Transitions :
- Upload → PENDING (`lightrag.py:1351-1364`)
- Début de `process_document` → PROCESSING (`lightrag.py:1893-1914`)
- Fin OK → PROCESSED (`lightrag.py:2045-2065`)
- Erreur extraction → FAILED + `error_msg` + `metadata.processing_start_time/end_time` (`lightrag.py:1989-2008`)
- Erreur merge → FAILED idem (`lightrag.py:2115-2133`)

### 6.2 Crash à mi-chemin

**Cas A — crash entre `apipeline_enqueue_documents` et `apipeline_process_enqueue_documents`** :
- DocStatus reste à `PENDING`, full_docs contient le content, aucun chunk/entity/edge en base.
- Au redémarrage : `apipeline_process_enqueue_documents` lit `get_docs_by_status(PENDING)` (`lightrag.py:1674`), repart de zéro.
- **Idempotent** car `compute_mdhash_id(content)` produit le même doc_id, et la déduplication via `filter_keys` en amont aurait skip si on retentait via HTTP.

**Cas B — crash pendant `process_document` (chunks insérés, extraction en cours)** :
- DocStatus à `PROCESSING`, `chunks_list` rempli, chunks dans KV+Vec.
- Au redémarrage : `apipeline_process_enqueue_documents` lit `get_docs_by_status(PROCESSING)` (`lightrag.py:1672`) + `_validate_and_fix_document_consistency` (`lightrag.py:1736`) :
  - check `full_docs.get_by_id(doc_id)` existe → si oui, **reset PROCESSING → PENDING** (`lightrag.py:1607-1628`), avec `error_msg=""`, `metadata={}`.
  - Le doc repart à zéro depuis le chunking.
- **Effet de bord** : les chunks déjà en KV/Vec sont écrasés par `MERGE` (idempotent). Les entités/edges partiellement insérés **ne sont PAS purgés** avant le retry. Ils seront fusionnés par `_merge_nodes_then_upsert` (qui agrège descriptions et déduplique source_ids via `merge_source_ids`, `$LRAG/utils.py`) — donc pas de duplication, mais on garde des descriptions qui peuvent ne plus refléter le contenu final si la deuxième passe diffère du premier (mais avec LLM cache, généralement deterministe).

**Cas C — crash pendant `merge_nodes_and_edges` après proxy flush** :
- DocStatus à `PROCESSING`. Merge peut être partiellement fini (nodes OK, edges KO).
- Au redémarrage idem cas B : on repasse depuis le chunking, mais le LLM cache (chunks ↔ entités extraites) hit, donc le ré-extraction est gratuit. Merge ré-écrase via `MERGE`.

**Cas D — crash après `proxy.flush()` mais avant `doc_status.upsert(PROCESSED)`** :
- Graph + VDB + KV cohérents, mais DocStatus à `PROCESSING`. Au restart : reset → PENDING → tout re-fait. **Sur-coût mais pas de corruption.**

### 6.3 `reprocess_failed_documents` route

`$LRAG/api/routers/document_routes.py` (route POST `/documents/reprocess_failed`) — non visualisée directement, mais sa logique standard :
- Lit `get_docs_by_status(DocStatus.FAILED)`.
- Pour chaque doc FAILED : réinjecte un task `pipeline_index_file` ou équivalent.
- Le doc passe par `_validate_and_fix_document_consistency` qui **reset FAILED → PENDING** si `full_docs.get_by_id(doc_id)` existe (`lightrag.py:1607-1628`).
- Sinon (full_docs vide), le doc est skip avec un message "Preserving failed document entries for manual review" (`lightrag.py:1542-1546`).

**Implication compliance** : un FAILED dont le `full_docs` a été purgé manuellement reste FAILED. C'est le placeholder visible dans le maquette "Inbox" pour validation humaine (cf. Prisme C §4).

---

## 7. Hooks existants `_patch_insert_done`

### 7.1 Mécanique du hook

`src/twindb_lightrag_memgraph/__init__.py:689-713` :

```python
def _patch_insert_done():
    from lightrag.lightrag import LightRAG
    from ._hooks import _run_post_index_hooks
    _original = LightRAG._insert_done
    async def _hooked_insert_done(self, pipeline_status=None, pipeline_status_lock=None):
        await _original(self, pipeline_status, pipeline_status_lock)
        await _run_post_index_hooks(self)
    _hooked_insert_done.__name__ = "hooked_insert_done"
    LightRAG._insert_done = _hooked_insert_done
```

- Single-patch suffit car `_insert_done` est appelé via `self._insert_done(...)` (méthode, pas import).
- Le hook s'exécute **après** que toutes les `index_done_callback()` des 12 storages aient run (`lightrag.py:2229-2247`).
- Le timing : à chaque fin de document (`lightrag.py:2068`) ET à la fin de chaque `adelete_by_doc_id` réussi (`lightrag.py:3588, 3672`).

### 7.2 Registre `register_post_index_hook`

`src/twindb_lightrag_memgraph/_hooks.py` :

```python
_post_index_hooks: list = []

def register_post_index_hook(callback) -> None:
    _post_index_hooks.append(callback)

async def _run_post_index_hooks(lightrag_instance) -> None:
    for hook in _post_index_hooks:
        try:
            await hook(lightrag_instance)
        except Exception:
            logger.exception("Post-index hook %s failed", hook.__name__)
```

Contrat callback :
- **Signature** : `async def my_hook(lightrag_instance: LightRAG) -> None`. Pas de `doc_id`, pas de `chunks`. Le callback doit récupérer le contexte autrement (e.g., via `pipeline_status`, `doc_status.get_docs_by_status(PROCESSED)`, ou un sidecar registry).
- **Ordre** : séquentiel, dans l'ordre d'enregistrement.
- **Erreurs** : loggées via `logger.exception`, **jamais propagées**. Un hook KO ne casse pas la pipeline (bon pour la robustesse, mauvais pour un audit trail qui doit *garantir* l'écriture). Voir §8.
- **Asynchrone** : tous les hooks `await`és, donc bloquants. Un hook lent retarde le `process_document` suivant (le worker semaphore n'est libéré qu'après le hook).

### 7.3 Cas d'usage actuels

Grep `register_post_index_hook` dans le repo :
- défini dans `_hooks.py:15` ;
- exporté dans `__init__.py:23` ;
- **aucun call site** dans le code de production (server/, intelligence/). Seuls les tests (`tests/`) l'invoquent — le hook est **un point d'extension publié mais pas encore consommé en prod**.

C'est exactement la boucle "infra prête, branchement à faire" attendue. Le hook est prêt pour brancher (a) un middleware audit (BCE/DORA), (b) un trigger DSEP côté indexation, (c) une notif WebUI.

### 7.4 Limitations actuelles à corriger pour audit BCE

Le contrat actuel est insuffisant pour un audit trail compliant :

1. **Pas de `doc_id` exposé** : le hook reçoit `LightRAG`, mais doit re-deviner quel document vient d'être indexé. Workaround : lire `pipeline_status["latest_message"]` (fragile, parse log string), ou diff `doc_status.get_docs_by_status(PROCESSED)` avant/après (race condition si parallèle).
2. **Pas de granularité event-type** : un seul callback fires pour upload-completed, delete-completed, rebuild-completed indistinctement.
3. **Pas de "MUST-LOG" guarantee** : exception swallowée. Pour BCE, une non-écriture audit doit faire échouer l'opération en amont.

Remediation proposée pour un Prisme H' / Phase 2 :

```python
# Proposition d'extension du registry
register_pre_index_hook(callback)        # before _process_extract_entities
register_post_extract_hook(callback)     # after extract_entities, before merge
register_post_index_hook(callback)       # current (rename to post_doc_hook)
register_post_delete_hook(callback)      # after adelete_by_doc_id

# Callback signature étendue
class IndexEventCtx:
    event_type: Literal["upload.accepted", "extract.completed", "index.completed", "index.failed", "delete.requested", "delete.completed"]
    doc_id: str
    workspace: str
    track_id: str
    file_path: str
    chunks_count: int
    error_msg: str | None
    timestamp: int
    trace_id: str   # propagated from FastAPI middleware via contextvars
    actor: str | None  # JWT sub claim if available

# MUST-LOG guarantee
register_post_index_hook(my_hook, must_log=True)
# If must_log=True and hook raises → DocStatus stays at PROCESSING + alarm raised.
```

---

## 8. Hookpoints pour audit trail compliance

Les `MUST-LOG` events du Prisme E (référence : §3 du Prisme E) à brancher :

| Event MUST-LOG | Hookpoint recommandé | Fichier:Ligne | Notes |
|---|---|---|---|
| `document.upload.accepted` | FastAPI middleware sur `POST /documents/upload`, avant le return `InsertResponse` | `document_routes.py:2225` (juste avant `background_tasks.add_task`) | Capture `doc_id` post-hash, `track_id`, `file_path`, `content_length`, JWT actor. Pas de hook LightRAG ici — pur FastAPI. |
| `document.index.started` | Wrapper sur `process_document`, début du `async with semaphore:` | `lightrag.py:1798` (pas patchable proprement sans `register_pre_index_hook` à créer) | Alternative : hook sur `_validate_and_fix_document_consistency` qui voit tous les docs candidats. |
| `document.index.completed` | **Hook existant `register_post_index_hook`** (à condition d'enrichir le contrat avec `doc_id`) | Wrapper `_hooked_insert_done`, déjà en place, `__init__.py:705-712` | Le hook actuel fires aussi sur `adelete_by_doc_id._insert_done()` (`lightrag.py:3588`) — distinguer en lisant `pipeline_status["job_name"]` (commence par "Deleting" pour delete) OU créer un registry séparé. |
| `document.index.failed` | Wrapper sur les `except` blocks `lightrag.py:1943-2008` (extract failure) et `:2078-2133` (merge failure) | Patch `LightRAG._process_extract_entities` (déjà wrappable) ou patch `process_document` lui-même | Pas patchable proprement sans monkey-patcher la closure `process_document` interne — alternative : un hook sur `doc_status.upsert` qui détecte `status=FAILED` (cf. ligne ci-dessous). |
| `document.delete.requested` | FastAPI middleware sur `DELETE /documents/{doc_id}` ou patch `LightRAG.adelete_by_doc_id` début | `lightrag.py:2989` (entry) | Capture `doc_id`, JWT actor, `delete_llm_cache`, `delete_file`. |
| `document.delete.completed` | Hook actuel `register_post_index_hook` (fires via `_insert_done` à la fin de `adelete_by_doc_id`, `lightrag.py:3588`) | `__init__.py:705-712` | Distinguer "completed-deletion" vs "completed-index" via `pipeline_status["job_name"]`. |
| `storage.write` | Patch sur chaque `upsert(...)` des 4 backends Memgraph (`kv_impl.py:123`, `vector_impl.py`, `docstatus_impl.py`, et `MemgraphStorage.upsert_node/upsert_edge` post `_BufferedGraphProxy.flush`). | Voir §8.2 plus bas | High-volume — sample ou batch via la bounded queue. |
| `storage.delete` | Patch sur chaque `delete(...)` des backends | `kv_impl.py:147-159`, `vector_impl.py`, `docstatus_impl.py`, `memgraph_impl.py:706-717` | Idem. |
| `cache.clear` | Patch sur `LightRAG.aclear_cache` | `lightrag.py:2885-2909` | Single point — patch trivial. |

### 8.1 Tableau hookpoints × usage

| Hookpoint | Audit BCE/DORA | DSEP L3 indexation | Notif WebUI | Trace contextvars |
|---|---|---|---|---|
| FastAPI middleware sur `/documents/*` | OK (point d'entrée acteur+JWT) | non | non | OK (init trace_id) |
| `LightRAG.apipeline_enqueue_documents` entry | redondant avec middleware FastAPI | non | non | propage trace |
| `LightRAG._process_extract_entities` post-call | **`document.extract.completed`** | **OK** (nodes/edges extraits = signal DSEP) | non | propage |
| `merge_nodes_and_edges` post-Phase 3 (avant proxy flush) | non | **OK** (entités/relations agrégées par doc) | non | propage |
| `_BufferedGraphProxy.flush` post-success | **`storage.write` agrégé** (N nodes, E edges) | non | non | propage |
| `_hooked_insert_done` (registry `register_post_index_hook`) | **`document.index.completed` / `document.delete.completed`** | trigger `dsep_pipeline.run(workspace)` async | **notif "doc X indexed"** | propage |
| `adelete_by_doc_id` entry | **`document.delete.requested`** | non | **notif "delete started"** | OK |
| `LightRAG.aclear_cache` | **`cache.clear`** | non | non | OK |

### 8.2 Snippets de code recommandés

**Snippet 1 — Extension du registry pour event-typed hooks**

```python
# Proposition pour src/twindb_lightrag_memgraph/_hooks.py
from dataclasses import dataclass
from typing import Callable, Awaitable, Literal

EventType = Literal[
    "document.upload.accepted",
    "document.extract.completed",
    "document.index.completed",
    "document.index.failed",
    "document.delete.requested",
    "document.delete.completed",
    "storage.write",
    "storage.delete",
    "cache.clear",
]

@dataclass
class IndexEvent:
    event_type: EventType
    doc_id: str | None
    workspace: str
    track_id: str | None
    file_path: str | None
    chunks_count: int | None
    error_msg: str | None
    timestamp: int
    trace_id: str | None
    actor: str | None
    extra: dict

_event_hooks: dict[EventType, list[Callable[[IndexEvent], Awaitable[None]]]] = {}

def register_event_hook(event_type: EventType, callback, must_log: bool = False):
    _event_hooks.setdefault(event_type, []).append((callback, must_log))

async def _fire(event: IndexEvent):
    for callback, must_log in _event_hooks.get(event.event_type, []):
        try:
            await callback(event)
        except Exception as e:
            if must_log:
                raise  # MUST-LOG: propagate to fail the operation
            logger.exception("Hook %s failed for %s", callback.__name__, event.event_type)
```

**Snippet 2 — Patch `_BufferedGraphProxy.flush` pour fire `storage.write` aggregated**

```python
# Add to src/twindb_lightrag_memgraph/_buffered_graph.py, après _flush_edges()
async def flush(self):
    node_count = len(self._node_buffer)
    edge_count = len(self._edge_buffer)
    try:
        if self._node_buffer:
            await self._flush_nodes()
        if self._edge_buffer:
            await self._flush_edges()
    except Exception:
        await _fire(IndexEvent(
            event_type="document.index.failed",
            workspace=self._real.workspace,
            error_msg=f"buffered_flush failed: {node_count} nodes, {edge_count} edges",
            ...
        ))
        raise
    # Success — fire one aggregated storage.write event
    from ._hooks import _fire, IndexEvent
    await _fire(IndexEvent(
        event_type="storage.write",
        workspace=self._real.workspace,
        chunks_count=None,
        extra={"target": "graph", "nodes": node_count, "edges": edge_count},
        ...
    ))
```

**Snippet 3 — FastAPI middleware pour `document.upload.accepted`**

```python
# server/middleware.py (à créer)
from fastapi import Request
from twindb_lightrag_memgraph._hooks import _fire, IndexEvent

async def audit_upload_middleware(request: Request, call_next):
    response = await call_next(request)
    if (
        request.url.path == "/documents/upload"
        and request.method == "POST"
        and response.status_code == 200
    ):
        # Parse response body (idempotent: response already sent to client)
        body = response.body
        data = json.loads(body)
        actor = getattr(request.state, "actor", None)  # set by auth dep
        await _fire(IndexEvent(
            event_type="document.upload.accepted",
            doc_id=None,  # not known yet at this stage (computed in enqueue)
            workspace=os.environ.get("MEMGRAPH_WORKSPACE", ""),
            track_id=data.get("track_id"),
            file_path=None,
            chunks_count=None,
            error_msg=None,
            timestamp=int(time.time()),
            trace_id=getattr(request.state, "trace_id", None),
            actor=actor,
            extra={"status": data.get("status"), "client_ip": request.client.host},
        ))
    return response
```

**Snippet 4 — Patch `_hooked_insert_done` enrichi avec `doc_id`**

```python
# src/twindb_lightrag_memgraph/__init__.py, remplacement de _patch_insert_done
def _patch_insert_done():
    from lightrag.lightrag import LightRAG
    from ._hooks import _run_post_index_hooks, _fire, IndexEvent

    _original = LightRAG._insert_done

    async def _hooked_insert_done(self, pipeline_status=None, pipeline_status_lock=None):
        await _original(self, pipeline_status, pipeline_status_lock)

        # Extract doc context from pipeline_status (best effort)
        doc_id = None
        job_name = ""
        if pipeline_status:
            job_name = pipeline_status.get("job_name", "")
            # latest_message format: "Completed processing file X/Y: file_path"
            # Not reliable; ideally LightRAG should expose the current doc_id via contextvars.
            # Workaround: pull from doc_status PROCESSED with most recent updated_at.
            try:
                recent = await self.doc_status.get_docs_by_status(DocStatus.PROCESSED)
                if recent:
                    doc_id = max(recent.items(), key=lambda kv: kv[1].updated_at)[0]
            except Exception:
                pass

        event_type = (
            "document.delete.completed"
            if job_name.lower().startswith("deleting") or "deletion" in job_name.lower()
            else "document.index.completed"
        )

        await _fire(IndexEvent(
            event_type=event_type,
            doc_id=doc_id,
            workspace=self.workspace,
            track_id=None,
            file_path=None,
            chunks_count=None,
            error_msg=None,
            timestamp=int(time.time()),
            trace_id=None,
            actor=None,
            extra={"job_name": job_name},
        ))

        # Backward compat: legacy register_post_index_hook still fires
        await _run_post_index_hooks(self)

    _hooked_insert_done.__name__ = "hooked_insert_done"
    LightRAG._insert_done = _hooked_insert_done
```

---

## 9. Hookpoints pour intelligence layer L3 indexation

### 9.1 État actuel

`src/twindb_lightrag_memgraph/intelligence/ontology/pipeline.py` (la DSEP pipeline) est conçue pour **retrieval-time expansion** (cf. Prisme F §4). Côté **indexation**, le branchement naturel est :

- **Trigger** : à la fin du `_insert_done` d'un doc, lancer une étape DSEP `extract → cluster → enrich → validate` qui :
  - Lit les entités/relations du doc (via `full_entities.get_by_id(doc_id)` puis `chunk_entity_relation_graph.get_nodes_batch(...)`).
  - Détecte des candidats d'ontologie (types récurrents, relations sémantiques fortes).
  - Stocke dans `ontology.json` ou un side-cart Memgraph les candidats en `dry-run` (pas de persistence sans `pipeline.approve()`).
- **Hookpoint propre** : `register_post_index_hook(dsep_index_hook)` — exactement le contrat existant. Pas besoin de patcher LightRAG.

### 9.2 Hookpoints alternatifs (plus profonds dans la pipeline)

| Hookpoint | Avantage | Inconvénient |
|---|---|---|
| Après `_process_extract_entities` (`lightrag.py:1940`) | DSEP voit les entités **avant** merge ; peut influencer le merge (rename, dedupe) | Pas de hook officiel ; nécessite un patch sur `LightRAG._process_extract_entities` ou un wrapper monkey-patch sur `extract_entities` |
| Pendant `merge_nodes_and_edges` Phase 3 | DSEP voit les entités finales agrégées | Idem, pas de hook officiel |
| Après `proxy.flush()` (notre `_BufferedGraphProxy`) | Graph est dans son état final pour ce doc | Doit être un override dans `_buffered_graph.py` (acceptable, c'est notre code) |
| Après `_insert_done` complet (**hook actuel**) | Tout est persisté, lecture safe | DSEP voit le graph entier pas seulement ce doc — il faut filtrer par `source_id` |

### 9.3 Recommandation

Le hook `register_post_index_hook` est suffisant pour le **MVP DSEP côté indexation**. La DSEP pipeline interne se charge de la sélection (lire `full_entities[doc_id]` puis aller chercher les nodes correspondants dans le graph). Pas besoin de monkey-patch supplémentaire.

Pour aller plus loin (si DSEP doit *modifier* les nodes pendant l'indexation, e.g. ajouter un label `:DSEP_validated`) : créer un wrapper sur `_BufferedGraphProxy.flush` qui appelle un `register_pre_flush_hook(buffer) -> None` permettant à DSEP d'ajouter ses propres mutations dans le buffer **avant** que le flush parte.

---

## 10. Performance et MVCC

### 10.1 Sémaphore `acquire_write_slot`

`src/twindb_lightrag_memgraph/_pool.py:341-386` :

- Configuration : env `MEMGRAPH_WRITE_CONCURRENCY` (`_constants.py`), défaut `DEFAULT_WRITE_CONCURRENCY = 10`.
- C'est un `asyncio.Semaphore(10)` global au process, bound au loop courant.
- Acquis **uniquement** dans nos backends Memgraph côté write : `kv_impl.py:134`, `kv_impl.py:149`, `kv_impl.py:173`, `vector_impl.py` (upsert/delete/drop), `docstatus_impl.py`, et dans `_BufferedGraphProxy._flush_nodes/_flush_edges` (`_buffered_graph.py:116, 156`).
- **PAS acquis** par les writes graph de LightRAG (i.e. `MemgraphStorage.upsert_node/upsert_edge` non-bufferisés, `$LRAG/kg/memgraph_impl.py:484-573`), car notre proxy intercepte avant qu'ils ne s'exécutent. Donc en pratique, le seul moment où `MemgraphStorage.upsert_node` court-circuite le buffer = appels en dehors du `merge_nodes_and_edges` (e.g. `ainsert_custom_kg` à `lightrag.py:2332`, ou les rebuild paths).

**Mesure d'usage** : aucune métrique exposée actuellement. Pour vérifier saturation, instrumenter via :
```python
# dans _pool.py
_pending_write_count = 0
async def acquire_write_slot():
    global _pending_write_count
    _pending_write_count += 1
    if _pending_write_count > 8:
        logger.warning("Write semaphore near saturation: %d in flight", _pending_write_count)
    sem = _get_write_semaphore()
    async with sem:
        try:
            yield
        finally:
            _pending_write_count -= 1
```

Hypothèse honnête : avec `MAX_PARALLEL_INSERT=4` (default) × jusqu'à 2-3 writes simultanés par doc (`doc_status.upsert` + `chunks_vdb.upsert` + `text_chunks.upsert` en parallèle, puis `entity_chunks` + `entity_vdb` + buffered flush), on a un floor de ~10-12 in flight au peak. La sémaphore défaut **est probablement saturée** pendant les pics, à instrumenter pour confirmer.

### 10.2 MVCC contention sur entités partagées

#### 10.2.1 Hypothèse Tony

Confirmée par lecture du Cypher généré :

- **`MemgraphStorage.upsert_node`** (`$LRAG/kg/memgraph_impl.py:484-571`) :
  ```cypher
  MERGE (n:`{workspace}` {entity_id: $entity_id})
  SET n += $properties
  SET n:`{entity_type}`
  ```
  Memgraph MVCC pose un lock write sur le node trouvé par MERGE. Deux transactions simultanées qui MERGE la même `entity_id` se sérialisent — la deuxième attend.

- **`_BufferedGraphProxy._flush_nodes`** (`_buffered_graph.py:117-126`) :
  ```cypher
  UNWIND $entries AS e
  MERGE (n:`{workspace}` {entity_id: e.entity_id})
  SET n += e.properties
  ```
  Une seule transaction qui MERGE plusieurs entities. Si deux docs (= 2 process_document tasks) flushent en parallèle et partagent des entities, la deuxième transaction attend que la première relâche les locks.

- **Retry logic** : `MemgraphStorage.upsert_node` a un retry exponentiel sur `TransientError` ("Cannot resolve conflicting transactions", `memgraph_impl.py:541`), 100 retries × 0.2s × 1.1^attempt. **Notre buffered flush n'a PAS ce retry** — si Memgraph throw transient, on raise.

#### 10.2.2 Vérification empirique

`scripts/test_mvcc_parallelization.py` est conçu exactement pour discriminer :
- Scénario A : 2 docs partageant "Paris", "Louvre", "Napoleon" → mesure seq vs par.
- Scénario B : 2 docs disjoints "Paris" vs "Tokyo" → mesure seq vs par.

Interprétation (`scripts/test_mvcc_parallelization.py:246-269`) :
- `speedup_A ≈ 1.0` ET `speedup_B > 1.5` → MVCC contention confirmé.
- Speedup non mesuré ici (pas de run récent) — à exécuter pour valider la confirmation, mais le code Cypher confirme déjà mécaniquement la possibilité.

**Mitigation Application-level** :
- Notre buffer rend la contention plus visible (un gros flush par doc plutôt que ~130 petits writes), mais ne la résout pas si deux flushs partagent des entities.
- Solution propre : pré-partitionner les docs par cluster d'entities avant `asyncio.gather`. Heuristique : un pass préliminaire d'extraction (cheap) pour estimer les entities, puis batch les docs disjoints ensemble.

### 10.3 Gain de `_BufferedGraphProxy`

**Théorique** (sans buffer) :
- 50 entités × `upsert_node` (1 round-trip cherche-merge + 1 ronde SET label) → ~100 round-trips.
- 80 relations × `upsert_edge` (1 round-trip cherche-merge) → ~80 round-trips.
- Soit ~180 round-trips par doc (matches l'estimation "~130" du CLAUDE.md, ordre de grandeur).

**Avec buffer** :
- 1 UNWIND MERGE nodes (toutes entités).
- ~K UNWIND queries pour `SET n:`type`` où K = nombre de types distincts (typiquement 5-10).
- 1 UNWIND MATCH+MERGE edges.
- Total : **2 + K** round-trips = ~7-12 round-trips par doc.

**Réduction : ~15-25× round-trips.** Pour un RTT Memgraph local (1ms) c'est ~150ms vs ~10ms ; pour un Memgraph remote (10ms) c'est ~1.8s vs ~70ms. Différence cruciale en ingestion bulk.

**Bémol** : la concurrency Phase 1/Phase 2 dans `merge_nodes_and_edges` (`operate.py:2520-2739`) est **annulée** par le buffer puisque toutes les writes sont accumulées en mémoire et flushées en un seul site post-Phase 3. Le gain RTT > la perte de parallélisme inter-entity.

---

## 11. Risques compliance

### 11.1 RawDocModal vs pipeline natif

Le `RawDocModal` côté maquette (`maquette-deploy/source/components/RawDocModal/`) montre le **document entier** dans un overlay pour validation. Ça matche K1 (`full_docs` KV) et M3 (texte rechargé en RAM pendant le processing) — donc, oui, **le pipeline LightRAG natif fait la même chose** : le document complet existe simultanément en :
- D1/D2 disque (PDF binaire)
- K1 Memgraph (`full_docs` JSON)
- M2/M3/M4 RAM transitoire
- K3 Memgraph (`llm_response_cache`) si flag activé — duplicaté en N copies par chunk

**Conclusion** : la maquette n'invente pas la fuite, elle l'expose. Si la fuite est compliance-bloquante, c'est tout le pipeline qu'il faut hardener, pas juste cacher le modal.

### 11.2 Logs en clair contenant du contenu document

Grep dans `$LRAG/lightrag.py` et `$LRAG/operate.py` :

- **`logger.info(f"Processing d-id: {doc_id}")`** (`lightrag.py:1825`) — pas de contenu, OK.
- **`logger.info(f"Inserting {len(new_docs)} docs")`** (`lightrag.py:1230`) — pas de contenu, OK.
- **`logger.info(f"Extracting stage {current_file_number}/{total_files}: {file_path}")`** (`lightrag.py:1822`) — `file_path` (peut être nom de fichier confidentiel).
- **`logger.info(f"Successfully extracted and enqueued file: {file_path.name}")`** (`document_routes.py:1570`) — nom de fichier.
- **`logger.warning(f"Duplicate document detected: {doc_id} ({file_path})")`** (`lightrag.py:1378`) — nom de fichier.
- **`logger.warning(f"...: '{record_attributes[1]}'")`** dans `_handle_single_entity_extraction` (`operate.py:386-401`) — **contient le name d'entité extraite du document**.
- **`verbose_debug(msg, *args)`** (`utils.py:240-264`) — truncate à 150 chars ; si `VERBOSE=true`, full message → potentiellement contenu chunk.
- **Le contenu des chunks** lui-même est *jamais* loggé en clair, mais le prompt LLM final (contient le chunk) est envoyé en HTTP outbound vers l'inference endpoint. Si l'inference endpoint n'est pas TLS ou est tiers (OpenAI, Anthropic), c'est une exfiltration de fait.

**Doctrine Eric** : logs purement structurés (doc_id, chunk_id, status, durée) — **interdire** noms d'entités, noms de fichier en clair, payloads. À configurer via un `LightragPathFilter`-like custom filter sur le logger root.

### 11.3 Classification niveau confidentiel (Louis 2026-05-28)

Si la feature "classification niveau confidentiel" doit être branchée :

**Phase A — pré-chunking** :
- Hookpoint : `pipeline_enqueue_file` après `content` extraction, avant `apipeline_enqueue_documents(content)` (`document_routes.py:1566`).
- Faire un classifier LLM léger qui renvoie `{level: "public" | "internal" | "confidential" | "secret"}` sur le doc complet (ou un sample).
- Stocker dans DocStatus.metadata pour audit.
- Si `level >= confidential` : router vers un workspace dédié (label séparé), ou refuser l'ingestion si workspace courant pas habilité.

**Phase B — pré-write graph** :
- Hookpoint : avant `_BufferedGraphProxy.flush` (à wrapper avec un `register_pre_flush_hook`).
- Pour chaque entity/edge, vérifier que le `source_id` chunk ne provient pas d'un doc `confidential` qui ne devrait pas alimenter le graph public.
- Si conflit → écraser source_id par opaque token ou refuser le write.

**Phase C — pré-LLM** :
- Hookpoint : dans `extract_entities._process_single_content` avant `use_llm_func_with_cache` (`operate.py:2892`).
- Si chunk `confidential` ET LLM est tiers (pas on-prem) → refuser l'appel, mettre DocStatus à FAILED avec error_msg "confidential content cannot be sent to external LLM".

La Phase A est la plus simple et la plus défensible pour BCE. Les phases B et C sont des belt-and-suspenders.

---

## Cross-prismes notes

- **Prisme B (boot lifecycle)** : confirme que `register()` est idempotent et patche `merge_nodes_and_edges` + `_insert_done` à l'import du wrapper. Cet audit reprend ces deux patches sans réviser leur correctness.
- **Prisme C (api contracts)** : les routes `/documents/upload`, `/documents/text`, `/documents/{doc_id}` (DELETE) auditées en Prisme C — cohérent avec les trace flows ici.
- **Prisme D (auth security)** : la doctrine compliance §11 dépend du middleware auth pour extraire l'`actor` (JWT sub). Le `combined_auth` dependency (`document_routes.py:2046`) est le bon hookpoint.
- **Prisme E (observability)** : les MUST-LOG events documentés dans Prisme E §3 sont mappés vers les hookpoints concrets dans ce prisme §8. Aucune contradiction.
- **Prisme F (retrieval)** : aucune intersection structurelle (Prisme F = read, Prisme H = write). La seule connection : les KV `text_chunks` et VDB `chunks` produits ici sont les sources lues en `naive_query` et `_build_chunks_for_specific_relationships` côté retrieval.
- **Prisme G (vulns)** : les CVE LightRAG (`CVE-2026-30762`, `CVE-2026-39413`) concernent l'auth JWT et non le pipeline d'ingestion. Pas de vuln spécifique ingestion remontée par pip-audit, mais l'absence de transaction globale (§5.2) est une faiblesse design plus qu'une CVE.

## Signaux faibles à surveiller (non bloquants mais à noter)

- **`MemgraphStorage.initialize` du package natif** (`$LRAG/kg/memgraph_impl.py:65-109`) ouvre **son propre driver** indépendant du `_pool.py`. C'est bien noté dans CLAUDE.md ("production has 3 pools by design"), mais ça veut dire qu'au cas où Memgraph saturé sur connexion pool, les 3 pools s'ignorent — impossible de fixer un cap global au process.

- **`get_storage_keyed_lock`** (`$LRAG/kg/shared_storage.py`) est **in-process only**. Sur multi-worker uvicorn, deux workers peuvent écrire la même `entity_name` simultanément. La seule sérialisation reste alors le MVCC + retry de `MemgraphStorage.upsert_node`. Pour un multi-worker BNP, soit forcer `--workers 1`, soit ajouter un lock distribué (Redis SETNX, Memgraph natif via `LOCK` Cypher non standard).

- **Logs `[no-context]`** issue (mentionné dans CLAUDE.md storage idioms) : déjà corrigé en 0.5.1 sur le vector backend via auto-create vector index + retry une fois (`vector_impl.py:_create_vector_index`). Pas d'équivalent côté KV/Graph — un drop accidentel de label ne se rebuild pas tout seul.

- **`max_extract_input_tokens` non bound à `embedding_token_limit`** : si l'embedding model a une window plus petite que le tokenizer LLM (e.g. BGE-M3 8K vs gpt-4o 128K), un chunk LLM-conformément taillé peut dépasser pour l'embedding. `embedding_token_limit` est lu (`lightrag.py:538`) mais juste loggué en warning si dépassé sur les résumés (`operate.py:370-374`) — **pas de truncation hard**. Sur les chunks bruts (chunks_vdb.upsert), pas de check du tout. Risque silencieux : embedding API refuse → vector entry manquante mais chunk présent en KV.

- **`pipmaster.install("neo4j")` au runtime** dans `$LRAG/kg/memgraph_impl.py:14-15` — déjà flag-é Prisme G §1. Côté ingestion ça ne change rien, mais en image immuable BCE, ça doit être désactivé.

Aucun secret en clair trouvé dans le pipeline (les credentials Memgraph viennent de `MEMGRAPH_USERNAME`/`MEMGRAPH_PASSWORD` env, et les credentials LLM via `LLM_BINDING_API_KEY` env également). Pas de bug critique inédit — les risques sont des choix d'architecture documentés (no transaction globale, in-process locks, full_docs persistant) plus que des défauts de code.
