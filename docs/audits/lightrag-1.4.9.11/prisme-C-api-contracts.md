# Prisme C - API routes & Pydantic contracts LightRAG 1.4.9.11

Audit realise sur le wheel exact `lightrag-hku==1.4.9.11` extrait dans `/private/tmp/lrag14911`, plus le fork `lightrag_webui_twin/` et le serveur Twin local. Les references `$LRAG/api/...` designent ce wheel 1.4.9.11.

## 1. Inventaire LightRAG natif

Auth : les routes marquees "Oui" utilisent `Depends(combined_auth)` cree par `get_combined_auth_dependency(api_key)` (`$LRAG/api/lightrag_server.py:459`, `$LRAG/api/utils_api.py:80`). Les routes inline `/docs`, `/`, `/auth-status`, `/login` et le WebUI statique ne sont pas protegees par cette dependance.

| Methode | Path | Handler | Request model | Response model | Auth | Ref |
|---|---|---|---|---|---|---|
| GET | `/docs` | `custom_swagger_ui_html` | - | Swagger HTML | Non | `$LRAG/api/lightrag_server.py:1106` |
| GET | `/docs/oauth2-redirect` | `swagger_ui_redirect` | - | OAuth2 redirect HTML | Non | `$LRAG/api/lightrag_server.py:1119` |
| GET | `/` | `redirect_to_webui` | - | `RedirectResponse(/webui|/docs)` | Non | `$LRAG/api/lightrag_server.py:1124` |
| GET | `/auth-status` | `get_auth_status` | - | dict `{auth_configured, auth_mode, core_version, api_version, ...}` | Non | `$LRAG/api/lightrag_server.py:1132` |
| POST | `/login` | `login` | `OAuth2PasswordRequestForm` | dict `{access_token, token_type, auth_mode, ...}` | Non | `$LRAG/api/lightrag_server.py:1162` |
| GET | `/health` | `get_status` | header `LIGHTRAG-WORKSPACE` optionnel | dict health/config/pipeline | Oui | `$LRAG/api/lightrag_server.py:1197` |
| MOUNT | `/static/swagger-ui` | `StaticFiles` | - | static assets | Non | `$LRAG/api/lightrag_server.py:1335` |
| MOUNT | `/webui` | `SmartStaticFiles` | - | native SPA files | Non | `$LRAG/api/lightrag_server.py:1345` |
| GET | `/webui`, `/webui/` | `webui_redirect_to_docs` | - | `RedirectResponse(/docs)` | Non | fallback only, `$LRAG/api/lightrag_server.py:1357` |
| POST | `/documents/scan` | `scan_for_new_documents` | - | `ScanResponse` | Oui | `$LRAG/api/routers/document_routes.py:2045` |
| POST | `/documents/upload` | `upload_to_input_dir` | multipart `UploadFile` | `InsertResponse` | Oui | `$LRAG/api/routers/document_routes.py:2070` |
| POST | `/documents/text` | `insert_text` | `InsertTextRequest` | `InsertResponse` | Oui | `$LRAG/api/routers/document_routes.py:2145` |
| POST | `/documents/texts` | `insert_texts` | `InsertTextsRequest` | `InsertResponse` | Oui | `$LRAG/api/routers/document_routes.py:2223` |
| DELETE | `/documents` | `clear_documents` | - | `ClearDocumentsResponse` | Oui | `$LRAG/api/routers/document_routes.py:2306` |
| GET | `/documents/pipeline_status` | `get_pipeline_status` | - | `PipelineStatusResponse` | Oui | `$LRAG/api/routers/document_routes.py:2500` |
| GET | `/documents` | `documents` | - | `DocsStatusesResponse` | Oui | `$LRAG/api/routers/document_routes.py:2601` |
| DELETE | `/documents/delete_document` | `delete_document` | `DeleteDocRequest` | local `DeleteDocByIdResponse` | Oui | `$LRAG/api/routers/document_routes.py:2711` |
| POST | `/documents/clear_cache` | `clear_cache` | `ClearCacheRequest` | `ClearCacheResponse` | Oui | `$LRAG/api/routers/document_routes.py:2790` |
| DELETE | `/documents/delete_entity` | `delete_entity` | `DeleteEntityRequest` | `DeletionResult` | Oui | `$LRAG/api/routers/document_routes.py:2824` |
| DELETE | `/documents/delete_relation` | `delete_relation` | `DeleteRelationRequest` | `DeletionResult` | Oui | `$LRAG/api/routers/document_routes.py:2859` |
| GET | `/documents/track_status/{track_id}` | `get_track_status` | path `track_id` | `TrackStatusResponse` | Oui | `$LRAG/api/routers/document_routes.py:2897` |
| POST | `/documents/paginated` | `get_documents_paginated` | `DocumentsRequest` | `PaginatedDocsResponse` | Oui | `$LRAG/api/routers/document_routes.py:2971` |
| GET | `/documents/status_counts` | `get_document_status_counts` | - | `StatusCountsResponse` | Oui | `$LRAG/api/routers/document_routes.py:3058` |
| POST | `/documents/reprocess_failed` | `reprocess_failed_documents` | - | `ReprocessResponse` | Oui | `$LRAG/api/routers/document_routes.py:3085` |
| POST | `/documents/cancel_pipeline` | `cancel_pipeline` | - | `CancelPipelineResponse` | Oui | `$LRAG/api/routers/document_routes.py:3131` |
| POST | `/query` | `query_text` | `QueryRequest` | `QueryResponse` | Oui | `$LRAG/api/routers/query_routes.py:196` |
| POST | `/query/stream` | `query_text_stream` | `QueryRequest` | NDJSON chunks shaped like `StreamChunkResponse` | Oui | `$LRAG/api/routers/query_routes.py:456` |
| POST | `/query/data` | `query_data` | `QueryRequest` | `QueryDataResponse` | Oui | `$LRAG/api/routers/query_routes.py:742` |
| GET | `/graph/label/list` | `get_graph_labels` | - | `list[str]` runtime | Oui | `$LRAG/api/routers/graph_routes.py:92` |
| GET | `/graph/label/popular` | `get_popular_labels` | query `limit:int=300` | `list[str]` runtime | Oui | `$LRAG/api/routers/graph_routes.py:109` |
| GET | `/graph/label/search` | `search_labels` | query `q`, `limit:int=50` | `list[str]` runtime | Oui | `$LRAG/api/routers/graph_routes.py:133` |
| GET | `/graphs` | `get_knowledge_graph` | query `label`, `max_depth`, `max_nodes` | graph dict runtime | Oui | `$LRAG/api/routers/graph_routes.py:159` |
| GET | `/graph/entity/exists` | `check_entity_exists` | query `name` | `{exists: bool}` | Oui | `$LRAG/api/routers/graph_routes.py:197` |
| POST | `/graph/entity/edit` | `update_entity` | `EntityUpdateRequest` | dict status/message/data | Oui | `$LRAG/api/routers/graph_routes.py:220` |
| POST | `/graph/relation/edit` | `update_relation` | `RelationUpdateRequest` | dict status/message/data | Oui | `$LRAG/api/routers/graph_routes.py:410` |
| POST | `/graph/entity/create` | `create_entity` | `EntityCreateRequest` | dict status/message/data | Oui | `$LRAG/api/routers/graph_routes.py:445` |
| POST | `/graph/relation/create` | `create_relation` | `RelationCreateRequest` | dict status/message/data | Oui | `$LRAG/api/routers/graph_routes.py:518` |
| POST | `/graph/entities/merge` | `merge_entities` | `EntityMergeRequest` | dict status/message/data | Oui | `$LRAG/api/routers/graph_routes.py:607` |
| GET | `/api/version` | `OllamaAPI.get_version` | - | `OllamaVersionResponse` runtime | Oui | `$LRAG/api/routers/ollama_api.py:233`; mounted `$LRAG/api/lightrag_server.py:1103` |
| GET | `/api/tags` | `OllamaAPI.get_tags` | - | `OllamaTagResponse` runtime | Oui | `$LRAG/api/routers/ollama_api.py:238`; mounted `$LRAG/api/lightrag_server.py:1103` |
| GET | `/api/ps` | `OllamaAPI.get_running_models` | - | `OllamaPsResponse` runtime | Oui | `$LRAG/api/routers/ollama_api.py:261`; mounted `$LRAG/api/lightrag_server.py:1103` |
| POST | `/api/generate` | `OllamaAPI.generate` | `OllamaGenerateRequest` | `OllamaGenerateResponse`/stream dicts | Oui | `$LRAG/api/routers/ollama_api.py:285`; mounted `$LRAG/api/lightrag_server.py:1103` |
| POST | `/api/chat` | `OllamaAPI.chat` | `OllamaChatRequest` | `OllamaChatResponse`/stream dicts | Oui | `$LRAG/api/routers/ollama_api.py:462`; mounted `$LRAG/api/lightrag_server.py:1103` |

Modeles Pydantic natifs principaux :

- Documents : `ScanResponse` `$LRAG/api/routers/document_routes.py:133`, `ReprocessResponse` `:160`, `CancelPipelineResponse` `:188`, `InsertTextRequest` `:210`, `InsertTextsRequest` `:243`, `InsertResponse` `:283`, `ClearDocumentsResponse` `:308`, `ClearCacheRequest` `:330`, `ClearCacheResponse` `:341`, `DeleteDocRequest` `:379`, `DeleteEntityRequest` `:409`, `DeleteRelationRequest` `:420`, `DocStatusResponse` `:432`, `DocsStatusesResponse` `:471`, `TrackStatusResponse` `:537`, `DocumentsRequest` `:579`, `PaginatedDocsResponse` `:648`, `StatusCountsResponse` `:702`, `PipelineStatusResponse` `:725`.
- Query : `QueryRequest` `$LRAG/api/routers/query_routes.py:16`, `QueryResponse` `:157`, `QueryDataResponse` `:167`, `StreamChunkResponse` `:178`.
- Graph : `EntityUpdateRequest` `$LRAG/api/routers/graph_routes.py:16`, `RelationUpdateRequest` `:23`, `EntityMergeRequest` `:29`, `EntityCreateRequest` `:44`, `RelationCreateRequest` `:63`.
- Ollama : `OllamaChatRequest` `$LRAG/api/routers/ollama_api.py:34`, `OllamaGenerateRequest` `:49`, `OllamaVersionResponse` `:71`, `OllamaTagResponse` `:93`, `OllamaPsResponse` `:116`.
- `DeletionResult` est une dataclass, pas un `BaseModel` : `$LRAG/base.py:835`.

## 2. Inventaire WebUI fork TS

`apiFetch<T>()` est le seul wrapper HTTP. Il construit `BASE_URL + path`, ajoute `Accept: application/json`, JSON-stringify les bodies non-GET, et ajoute `Authorization: Bearer` si `VITE_AUTH_TOKEN` existe (`lightrag_webui_twin/src/api/client.ts:15`, `:63`, `:76`).

Appels reels dans `lightrag_webui_twin/src/api/resources.ts` :

| Fonction TS | Methode | Path | Query/body | Retour attendu | Ref |
|---|---|---|---|---|---|
| `listDocuments` | GET | `/documents` | query `{status,q,tag,cursor?}` | `ListEnvelope<Document>` = `{items,total}` | `lightrag_webui_twin/src/api/resources.ts:33` |
| `listWorkspaces` | GET | `/workspaces` | - | `Workspace[]` | `lightrag_webui_twin/src/api/resources.ts:37` |
| `listNotifications` | GET | `/notifications` | - | `Notification[]` | `lightrag_webui_twin/src/api/resources.ts:39` |
| `markAllNotificationsRead` | POST | `/notifications/read-all` | - | `{ok:true}` | `lightrag_webui_twin/src/api/resources.ts:41` |
| `clearNotifications` | DELETE | `/notifications` | - | `{ok:true}` | `lightrag_webui_twin/src/api/resources.ts:43` |
| `listThesaurus` | GET | `/thesaurus` | - | `ThesaurusEntry[]` | `lightrag_webui_twin/src/api/resources.ts:47` |
| `listTags` | GET | `/tags` | - | `TagEntry[]` | `lightrag_webui_twin/src/api/resources.ts:49` |
| `listTagCategories` | GET | `/tags/categories` | - | `TagCategory[]` | `lightrag_webui_twin/src/api/resources.ts:50` |
| `requestTag` | POST | `/tags` | `{tag, def, category, aliases?, justification?, actor?}` | `TagEntry` | `lightrag_webui_twin/src/api/resources.ts:54` |
| `approveTag` | POST | `/tags/{name}/approve` | `{actor?}` | `TagEntry` | `lightrag_webui_twin/src/api/resources.ts:65` |
| `rejectTag` | POST | `/tags/{name}/reject` | `{reason, actor?}` | `TagEntry` | `lightrag_webui_twin/src/api/resources.ts:71` |
| `editTag` | PATCH | `/tags/{name}` | `{def?, category?, aliases?, deprecates?, actor?}` | `TagEntry` | `lightrag_webui_twin/src/api/resources.ts:81` |
| `deprecateTag` | POST | `/tags/{name}/deprecate` | `{reason?, actor?}` | `TagEntry` | `lightrag_webui_twin/src/api/resources.ts:97` |
| `updateTagSynonyms` | POST | `/tags/{name}/synonyms` | `{aliases, actor?}` | `TagEntry` | `lightrag_webui_twin/src/api/resources.ts:107` |
| `deleteTag` | DELETE | `/tags/{name}` | `{strategy?, to?, actor?}` | `{ok:boolean}` | `lightrag_webui_twin/src/api/resources.ts:117` |
| `listActivity` | GET | `/activity` | query `{range?,kind?,sev?,actor?,q?}` | `ListEnvelope<ActivityEvent> & {nowMs?}` | `lightrag_webui_twin/src/api/resources.ts:129` |
| `getOpenApi` | GET | `/openapi` | - | `{groups: OpenApiGroup[], version}` | `lightrag_webui_twin/src/api/resources.ts:139` |
| `listGraphEntities` | GET | `/graph/entities` | query `{workspace?,type?}` | `GraphEntity[]` | `lightrag_webui_twin/src/api/resources.ts:143` |
| `listGraphRelations` | GET | `/graph/relations` | query `{workspace?}` | `GraphRelation[]` | `lightrag_webui_twin/src/api/resources.ts:148` |

Contrats TS/fixtures qui precisent les shapes :

- `Document` attend `{id,type,source,summary,tags,status,chunks,updated,visibility,workspace}` (`lightrag_webui_twin/src/types/document.ts:20`).
- `Workspace` et `Notification` sont les contrats topbar (`lightrag_webui_twin/src/types/topbar.ts:15`, `:33`).
- `TagEntry`, `TagCategory` et mutations tag sont dans `lightrag_webui_twin/src/types/tag.ts:39`, `:60`.
- `ActivityEvent` et son envelope `{items,total}` sont documentes dans `lightrag_webui_twin/src/types/activity.ts:5`, `:43`.
- `GraphEntity`/`GraphRelation` sont des teasers pre-layout, pas le graph natif LightRAG (`lightrag_webui_twin/src/types/graph.ts:7`, `:22`, `:37`).
- `OpenApiGroup` est une vue curatee `{m,p,s}`, pas le JSON OpenAPI FastAPI brut (`lightrag_webui_twin/src/types/api.ts:12`, `:21`).
- Le catalogue fixture affiche des routes natives LightRAG (`lightrag_webui_twin/src/fixtures/api.ts:12` a `:77`) mais n'est pas la liste des fetchs reels de `resources.ts`.

Contrats TS pas encore consommes par `src/api/resources.ts` :

- Retrieval : `POST /retrieval`, `GET /threads`, `POST /threads`, `DELETE /threads/{id}` sont declares comme template de phase 1 dans `lightrag_webui_twin/src/types/retrieval.ts:4` a `:9`, mais aucun fetch reel correspondant n'existe dans `resources.ts`.
- Document tag CRUD historique : `POST /documents`, `PATCH /documents/{id}/tags`, `DELETE /documents/{id}/tags` sont mentionnes dans `lightrag_webui_twin/src/types/document.ts:4` a `:10`, mais pas consommes par `resources.ts`.

## 3. Inventaire `server/webui_router.py`

Le routeur Twin est root-level (`router = APIRouter(tags=["webui"])`, `src/twindb_lightrag_memgraph/server/webui_router.py:329`) et est inclus dans l'app principale avec auth dans `src/twindb_lightrag_memgraph/server/app.py:293`.

| Methode | Path | Handler | Request model | Response model | Ref |
|---|---|---|---|---|---|
| GET | `/documents` | `list_documents` | query `status,q,tag` | `ListEnvelope[Document]` | `src/twindb_lightrag_memgraph/server/webui_router.py:335` |
| GET | `/workspaces` | `list_workspaces` | - | `list[Workspace]` | `src/twindb_lightrag_memgraph/server/webui_router.py:345` |
| GET | `/notifications` | `list_notifications` | - | `list[Notification]` | `src/twindb_lightrag_memgraph/server/webui_router.py:350` |
| POST | `/notifications/read-all` | `mark_all_notifications_read` | - | `AckResponse` | `src/twindb_lightrag_memgraph/server/webui_router.py:355` |
| DELETE | `/notifications` | `clear_notifications` | - | `AckResponse` | `src/twindb_lightrag_memgraph/server/webui_router.py:361` |
| GET | `/thesaurus` | `list_thesaurus` | - | `list[ThesaurusEntry]` | `src/twindb_lightrag_memgraph/server/webui_router.py:367` |
| GET | `/tags` | `list_tags` | - | `list[TagEntry]` | `src/twindb_lightrag_memgraph/server/webui_router.py:372` |
| GET | `/tags/categories` | `list_tag_categories` | - | `list[TagCategory]` | `src/twindb_lightrag_memgraph/server/webui_router.py:377` |
| GET | `/activity` | `list_activity` | query `kind,sev,actor,q` | `ActivityEnvelope` | `src/twindb_lightrag_memgraph/server/webui_router.py:382` |
| GET | `/openapi` | `get_openapi_groups` | - | `OpenApiEnvelope` | `src/twindb_lightrag_memgraph/server/webui_router.py:395` |
| GET | `/graph/entities` | `list_graph_entities` | - | `list[GraphEntity]` | `src/twindb_lightrag_memgraph/server/webui_router.py:401` |
| GET | `/graph/relations` | `list_graph_relations` | - | `list[GraphRelation]` | `src/twindb_lightrag_memgraph/server/webui_router.py:406` |
| POST | `/tags` | `request_tag` | `TagRequestBody` | `TagEntry` | `src/twindb_lightrag_memgraph/server/webui_router.py:439` |
| POST | `/tags/{name}/approve` | `approve_tag` | `TagApproveBody` | `TagEntry` | `src/twindb_lightrag_memgraph/server/webui_router.py:486` |
| POST | `/tags/{name}/reject` | `reject_tag` | `TagRejectBody` | `TagEntry` | `src/twindb_lightrag_memgraph/server/webui_router.py:519` |
| PATCH | `/tags/{name}` | `edit_tag` | `TagEditBody` | `TagEntry` | `src/twindb_lightrag_memgraph/server/webui_router.py:549` |
| POST | `/tags/{name}/deprecate` | `deprecate_tag` | `TagDeprecateBody` | `TagEntry` | `src/twindb_lightrag_memgraph/server/webui_router.py:590` |
| POST | `/tags/{name}/synonyms` | `update_synonyms` | `TagSynonymsBody` | `TagEntry` | `src/twindb_lightrag_memgraph/server/webui_router.py:622` |
| DELETE | `/tags/{name}` | `delete_tag` | `TagDeleteBody \| None` | `AckResponse` | `src/twindb_lightrag_memgraph/server/webui_router.py:651` |

Pydantic Twin :

- Base tolerant/camel alias : `src/twindb_lightrag_memgraph/server/webui_models.py:26`.
- `ListEnvelope` `:37`, `Document` `:47`, `Workspace` `:70`, `Notification` `:79`, `ThesaurusEntry` `:95`, `TagEntry` `:118`, `TagCategory` `:142`, `ActivityEvent` `:164`, `ActivityEnvelope` `:186`, `OpenApiEnvelope` `:210`, `GraphEntity` `:220`, `GraphRelation` `:231`, `AckResponse` `:244`.
- Tag mutation bodies : `TagRequestBody` `:253`, `TagEditBody` `:265`, `TagApproveBody` `:275`, `TagRejectBody` `:281`, `TagDeprecateBody` `:288`, `TagSynonymsBody` `:295`, `TagDeleteBody` `:302`.

## 4. Matrice de couverture

Legende : ✅ existe et contrat compatible ; 🚧 route existe mais contrat different ou partiel ; <span style="color:red">❌ missing</span> absent.

| Route attendue WebUI | Fournie LightRAG natif | Fournie server Twin | Couverture / action |
|---|---|---|---|
| GET `/documents` -> `{items: Document[], total}` | 🚧 Oui, mais retourne `DocsStatusesResponse {statuses: ...}` (`$LRAG/api/routers/document_routes.py:2601`) | ✅ Oui `ListEnvelope[Document]` (`webui_router.py:335`) | Utiliser Twin pour le fork. |
| GET `/documents?status&q&tag&cursor` | 🚧 Natif ignore `status/q/tag/cursor`; le filtrage comparable est POST `/documents/paginated` avec autre body (`$LRAG/api/routers/document_routes.py:2971`) | 🚧 `status/q/tag` oui, `cursor` ignore (`webui_router.py:335`) | Ajouter pagination/cursor si necessaire. |
| GET `/workspaces` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:345`) | Twin only. |
| GET `/notifications` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:350`) | Twin only. |
| POST `/notifications/read-all` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:355`) | Twin only. |
| DELETE `/notifications` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:361`) | Twin only. |
| GET `/thesaurus` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:367`) | Twin only. |
| GET `/tags` -> `TagEntry[]` | 🚧 Natif `/api/tags` existe mais c'est Ollama model tags, pas governance tags (`$LRAG/api/routers/ollama_api.py:238`) | ✅ Oui (`webui_router.py:372`) | Ne pas mapper vers `/api/tags`. |
| GET `/tags/categories` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:377`) | Twin only. |
| POST `/tags` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:439`) | Twin only. |
| POST `/tags/{name}/approve` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:486`) | Twin only. |
| POST `/tags/{name}/reject` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:519`) | Twin only. |
| PATCH `/tags/{name}` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:549`) | Twin only. |
| POST `/tags/{name}/deprecate` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:590`) | Twin only. |
| POST `/tags/{name}/synonyms` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:622`) | Twin only. |
| DELETE `/tags/{name}` | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:651`) | Twin only. |
| GET `/activity?range&kind&sev&actor&q` -> `{items,total,nowMs?}` | <span style="color:red">❌ missing</span> | 🚧 `kind/sev/actor/q` oui, `range` ignore (`webui_router.py:382`) | Ajouter filtre `range` cote Twin si l'UI l'utilise strictement. |
| GET `/openapi` -> `{groups,version}` curatee | 🚧 Natif expose `/openapi.json`, pas `/openapi`, et schema FastAPI brut | ✅ Oui (`webui_router.py:395`) | Twin fournit le format UI. |
| GET `/graph/entities` -> `GraphEntity[]` teaser | <span style="color:red">❌ missing</span>; natif a `/graphs` et `/graph/label/*` | ✅ Oui (`webui_router.py:401`) | Twin only, ou adaptateur sur graph LightRAG. |
| GET `/graph/relations` -> `GraphRelation[]` teaser | <span style="color:red">❌ missing</span> | ✅ Oui (`webui_router.py:406`) | Twin only, ou adaptateur sur graph LightRAG. |
| POST `/retrieval` -> `{tokens,sources}` (template TS) | 🚧 Natif `/query` peut fournir reponse + references, mais pas tokens UI (`query_routes.py:196`) | <span style="color:red">❌ missing</span> | A backporter si Retrieval tab devient live. |
| GET `/threads` -> `RetrievalThread[]` (template TS) | <span style="color:red">❌ missing</span> | <span style="color:red">❌ missing</span> | A backporter si historique conversation live. |
| POST `/threads` | <span style="color:red">❌ missing</span> | <span style="color:red">❌ missing</span> | A backporter. |
| DELETE `/threads/{id}` | <span style="color:red">❌ missing</span> | <span style="color:red">❌ missing</span> | A backporter. |
| POST `/documents` -> `Document` (template TS) | <span style="color:red">❌ missing</span>; natif a `/documents/upload|text|texts` | <span style="color:red">❌ missing</span> | Decider si AddSource utilise natif upload/text ou route Twin. |
| PATCH `/documents/{id}/tags` | <span style="color:red">❌ missing</span> | <span style="color:red">❌ missing</span> | A backporter pour retag. |
| DELETE `/documents/{id}/tags` | <span style="color:red">❌ missing</span> | <span style="color:red">❌ missing</span> | A backporter pour untag. |

Routes du catalogue API fixture (`lightrag_webui_twin/src/fixtures/api.ts`) :

| Route affichee dans l'onglet API | LightRAG natif | Twin server | Note |
|---|---|---|---|
| POST `/documents/upload` | ✅ (`document_routes.py:2070`) | <span style="color:red">❌ missing</span> | Peut etre appelee directement contre LightRAG natif. |
| POST `/documents/text` | ✅ (`document_routes.py:2145`) | <span style="color:red">❌ missing</span> | Peut etre appelee directement contre LightRAG natif. |
| POST `/documents/texts` | ✅ (`document_routes.py:2223`) | <span style="color:red">❌ missing</span> | Peut etre appelee directement contre LightRAG natif. |
| POST `/documents/scan` | ✅ (`document_routes.py:2045`) | <span style="color:red">❌ missing</span> | Peut etre appelee directement contre LightRAG natif. |
| GET `/documents` | 🚧 contrat natif different | ✅ contrat WebUI | Collision si Twin est root-mounted. |
| GET `/documents/pipeline_status` | ✅ (`document_routes.py:2500`) | <span style="color:red">❌ missing</span> | Peut rester natif. |
| DELETE `/documents` | ✅ (`document_routes.py:2306`) | <span style="color:red">❌ missing</span> | Collision methode/path avec Twin GET seulement non bloquante par methode, mais meme path. |
| DELETE `/documents/delete_document` | ✅ (`document_routes.py:2711`) | <span style="color:red">❌ missing</span> | Peut rester natif. |
| POST `/documents/clear_cache` | ✅ (`document_routes.py:2790`) | <span style="color:red">❌ missing</span> | Peut rester natif. |
| POST `/query` | ✅ (`query_routes.py:196`) | <span style="color:red">❌ missing</span> | Reutilisable tel quel pour simple retrieval. |
| POST `/query/stream` | ✅ (`query_routes.py:456`) | <span style="color:red">❌ missing</span> | Format NDJSON natif, pas tokens UI. |
| GET `/graph/label/*`, `/graphs`, `/graph/entity/exists` | ✅ | <span style="color:red">❌ missing</span> | Natif utile pour explorateur graph upstream. |
| POST `/graph/entity|relation/*` | ✅ | <span style="color:red">❌ missing</span> | Natif utile pour graph CRUD LightRAG. |
| GET/POST `/api/*` Ollama | ✅ | <span style="color:red">❌ missing</span> | Attention conflit avec prefix `/api/twin` si mal route par proxy. |
| `/`, `/auth-status`, `/login`, `/health` | ✅ | `/health` existe dans Twin app, autres non | Auth modes differents. |

## 5. Conflits potentiels de route

Si on monte `server/` sur `/api/twin` :

- Aucun conflit direct avec LightRAG natif : LightRAG ne sert que `/api/version`, `/api/tags`, `/api/ps`, `/api/generate`, `/api/chat` (`$LRAG/api/lightrag_server.py:1103`, `$LRAG/api/routers/ollama_api.py:233` a `:462`). Le prefix `/api/twin/...` ne matche pas ces routes exactes.
- Risque proxy/client : le fork TS appelle aujourd'hui des paths root-level (`/documents`, `/tags`, etc.) depuis `BASE_URL` (`client.ts:41`). Si Twin est sous `/api/twin`, `VITE_API_BASE_URL` doit pointer vers l'origine + `/api/twin`, ou le client doit prefixer les appels.
- Risque sémantique : `/api/tags` existe deja en natif mais signifie "Ollama model tags". Ne jamais monter la gouvernance tag Twin sous `/api/tags`.

Si on monte `server/` a la racine de l'app LightRAG :

- Conflit dur sur `GET /documents` : LightRAG natif retourne `DocsStatusesResponse`, Twin retourne `ListEnvelope[Document]`. L'ordre d'enregistrement decide qui gagne.
- Conflit potentiel sur `GET /health` si on monte toute l'app Twin, car `src/twindb_lightrag_memgraph/server/app.py` declare `/health` (`src/twindb_lightrag_memgraph/server/app.py:229`) et LightRAG aussi (`$LRAG/api/lightrag_server.py:1197`).
- Pas de conflit methode identique pour `DELETE /documents` si Twin ne declare que `GET /documents`, mais un mount root-level de sous-app est impraticable parce qu'il capture tout le prefix.

Si on monte sous `/twin/api` :

- Aucun conflit avec `/api/*` Ollama natif.
- Les URLs deviennent clairement separees de la compat Ollama et de l'API LightRAG native.
- Il faut configurer `VITE_API_BASE_URL=/twin/api` pour le fork TS.

## 6. Recommandations

Routes WebUI a backporter/completer dans `server/` :

- `GET /activity` : ajouter le filtre `range` attendu par le client (`resources.ts:129`) ; actuellement ignore (`webui_router.py:382`).
- `GET /documents` : remplacer progressivement les seed docs par une adaptation reelle depuis `rag.doc_status`, en conservant le contrat `ListEnvelope[Document]`.
- `GET /graph/entities` et `GET /graph/relations` : brancher sur le graph LightRAG reel au lieu des fixtures, en gardant les champs pre-layout `x/y`.
- Retrieval live si le tab devient dynamique : `POST /retrieval`, `GET /threads`, `POST /threads`, `DELETE /threads/{id}` sont absents des deux surfaces mais declares par les types (`types/retrieval.ts:4`).
- Tagging document : `PATCH /documents/{id}/tags` et `DELETE /documents/{id}/tags` sont absents des deux surfaces mais declares par `types/document.ts:7`.

Routes LightRAG natives que le WebUI fork peut consommer telles quelles :

- Ingestion : `/documents/upload`, `/documents/text`, `/documents/texts`, `/documents/scan`.
- Pipeline ops : `/documents/pipeline_status`, `/documents/track_status/{track_id}`, `/documents/status_counts`, `/documents/reprocess_failed`, `/documents/cancel_pipeline`.
- Retrieval brut : `/query`, `/query/data`; `/query/stream` si le client accepte NDJSON et pas le token stream custom.
- Graph CRUD natif : `/graph/label/list`, `/graph/label/popular`, `/graph/label/search`, `/graphs`, `/graph/entity/exists`, `/graph/entity/edit`, `/graph/relation/edit`, `/graph/entity/create`, `/graph/relation/create`, `/graph/entities/merge`.
- Auth/status natif : `/auth-status`, `/login`, `/health`, si on conserve le mode auth LightRAG.

Prefix de mount recommande : `/twin/api`.

Justification : `/api/*` est deja une surface Ollama-compatible native LightRAG, donc `/api/twin` est techniquement possible mais brouille la frontiere semantique et augmente le risque de proxy rules trop larges. `/twin/api` isole clairement la sous-app Twin, evite les collisions avec `/documents`, `/query`, `/graph` natifs, et se configure proprement via `VITE_API_BASE_URL=/twin/api` sans modifier tous les appels de `resources.ts`. Garder le WebUI statique sous `/webui` et pointer son `BASE_URL` vers `/twin/api` donne une separation nette : assets UI a `/webui`, API operateur Twin a `/twin/api`, API LightRAG native intacte aux chemins historiques.
