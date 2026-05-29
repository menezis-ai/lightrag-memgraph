# Prisme E — Logging, audit trail, observability LightRAG 1.4.9.11

Audit réalisé sur LightRAG exact `1.4.9.11` extrait sous `/private/tmp/lrag14911/lightrag` ; les références `$LRAG/...` ci-dessous pointent vers cette arborescence. Fichiers twin lus : `src/twindb_lightrag_memgraph/server/webui_activitystore.py` et `src/twindb_lightrag_memgraph/server/webui_notificationstore.py`.

## 1. Loggers natifs

LightRAG expose un logger applicatif global `lightrag`, créé dans `$LRAG/utils.py:75-89`. Par défaut, avant configuration serveur complète, il a :

- nom : `lightrag` ;
- niveau : `INFO` ;
- propagation : `False` ;
- handler : `SafeStreamHandler` console ;
- formatter : `%(levelname)s: %(message)s` ;
- `httpx` forcé à `WARNING`.

```python
# $LRAG/utils.py:75-89
logger = logging.getLogger("lightrag")
logger.propagate = False  # prevent log message send to root logger
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = SafeStreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)
```

Quand le serveur est lancé via `lightrag-server`, `main()` appelle explicitement `configure_logging()` avant `create_app(global_args)` et `uvicorn.run(..., log_config=None)` (`$LRAG/api/lightrag_server.py:1494-1513`). Cette configuration remet à zéro les handlers/filters de `uvicorn`, `uvicorn.access`, `uvicorn.error`, `lightrag`, puis installe un handler console et un `RotatingFileHandler`.

```python
# $LRAG/api/lightrag_server.py:1373-1451
def configure_logging():
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []

    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, DEFAULT_LOG_FILENAME))
    log_max_bytes = get_env_value("LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES, int)
    log_backup_count = get_env_value("LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT, int)

    logging.config.dictConfig(
        {
            "formatters": {
                "default": {"format": "%(levelname)s: %(message)s"},
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "uvicorn": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
                "uvicorn.access": {"handlers": ["console", "file"], "level": "INFO", "propagate": False, "filters": ["path_filter"]},
                "uvicorn.error": {"handlers": ["console", "file"], "level": "INFO", "propagate": False},
                "lightrag": {"handlers": ["console", "file"], "level": "INFO", "propagate": False, "filters": ["path_filter"]},
            },
            "filters": {"path_filter": {"()": "lightrag.utils.LightragPathFilter"}},
        }
    )
```

Constantes de rotation : `DEFAULT_LOG_FILENAME = "lightrag.log"`, `DEFAULT_LOG_MAX_BYTES = 10485760`, `DEFAULT_LOG_BACKUP_COUNT = 5` (`$LRAG/constants.py:102-104`). La rétention effective par défaut est donc environ 60 MB localement, pas un archivage DORA 5 ans.

`setup_logger()` dans `$LRAG/utils.py:318-385` duplique la même logique pour d'autres usages : console simple, fichier détaillé, rotation, option `add_filter`. Le formatter reste texte, pas JSON, sans champs structurés (`trace_id`, `user`, `workspace`, `event_type`, `resource_id`).

Un filtre important supprime volontairement des access logs de chemins très fréquents :

```python
# $LRAG/utils.py:276-310
class LightragPathFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.filtered_paths = [
            "/documents",
            "/documents/paginated",
            "/health",
            "/webui/",
            "/documents/pipeline_status",
        ]

    def filter(self, record):
        method = record.args[1]
        path = record.args[2]
        status = record.args[4]
        if (
            (method == "GET" or method == "POST")
            and (status == 200 or status == 304)
            and path in self.filtered_paths
        ):
            return False
```

Le grep Python `logger|logging\.|getLogger` confirme que les logs natifs couvrent largement `operate.py`, `lightrag.py`, `kg/*`, `api/routers/*`, mais via messages libres. Côté twin, `webui_activitystore.py:27` et `webui_notificationstore.py:23` créent des loggers `logging.getLogger(__name__)`, donc `twindb_lightrag_memgraph.server.webui_activitystore` et `twindb_lightrag_memgraph.server.webui_notificationstore`.

## 2. Couverture native vs silencieux

### Auth events

`/login` ne log ni succès, ni échec, ni login guest. L'échec credentials lève directement `HTTPException(401)` (`$LRAG/api/lightrag_server.py:1162-1184`).

```python
# $LRAG/api/lightrag_server.py:1162-1184
@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not auth_handler.accounts:
        guest_token = auth_handler.create_token(
            username="guest", role="guest", metadata={"auth_mode": "disabled"}
        )
        return {"access_token": guest_token, "token_type": "bearer", ...}
    username = form_data.username
    if auth_handler.accounts.get(username) != form_data.password:
        raise HTTPException(status_code=401, detail="Incorrect credentials")

    user_token = auth_handler.create_token(
        username=username, role="user", metadata={"auth_mode": "enabled"}
    )
```

La dépendance d'auth native log seulement le renouvellement automatique de token (`$LRAG/api/utils_api.py:186-200`). Les refus auth/API key lèvent `HTTPException` sans log (`$LRAG/api/utils_api.py:236-259`).

```python
# $LRAG/api/utils_api.py:186-200
logger.info(
    f"Token auto-renewed for user {username} "
    f"(role: {role}, remaining: {remaining_seconds:.0f}s)"
)
...
logger.warning(f"Token auto-renew failed: {e}")
```

### Retrieval / query

Les handlers de query logguent les erreurs, pas l'événement métier "query soumise / terminée / documents consultés". Exemple non-streaming : succès silencieux, erreur avec stack (`$LRAG/api/routers/query_routes.py:447-454`).

```python
# $LRAG/api/routers/query_routes.py:447-454
if request.include_references:
    return QueryResponse(response=response_content, references=references)
else:
    return QueryResponse(response=response_content, references=None)
except Exception as e:
    logger.error(f"Error processing query: {str(e)}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))
```

Streaming idem : erreurs du générateur et erreurs handler sont logguées (`$LRAG/api/routers/query_routes.py:712-739`), mais pas le volume de chunks, le mode RAG, l'acteur, la durée, ni les `doc_id` accédés.

Dans le core retrieval, `operate.py` log beaucoup de progression et d'anomalies, parfois des mots-clés ou la requête brute en debug, par exemple `logger.warning(f"Forced low_level_keywords to origin query: {query}")` (`$LRAG/operate.py:3081`). Ce n'est pas un audit trail : pas structuré, pas corrélé, pas exhaustif.

### Mutations storage / documents

Upload document : succès HTTP silencieux côté logger, erreur logguée avec nom de fichier (`$LRAG/api/routers/document_routes.py:2126-2143`).

```python
# $LRAG/api/routers/document_routes.py:2126-2143
with open(file_path, "wb") as buffer:
    shutil.copyfileobj(file.file, buffer)
track_id = generate_track_id("upload")
background_tasks.add_task(pipeline_index_file, rag, file_path, track_id)
return InsertResponse(status="success", ...)
...
logger.error(f"Error /documents/upload: {file.filename}: {str(e)}")
logger.error(traceback.format_exc())
```

Drop storage loggue les composants supprimés (`$LRAG/api/routers/document_routes.py:2399-2416`), mais sans acteur, source IP, raison, approbation, ni identifiant d'opération.

```python
# $LRAG/api/routers/document_routes.py:2399-2416
if isinstance(result, Exception):
    error_msg = f"Error dropping {storage_name}: {str(result)}"
    logger.error(error_msg)
else:
    namespace = storages[i].namespace
    workspace = storages[i].workspace
    logger.info(f"Successfully dropped {storage_name}: {workspace}/{namespace}")
```

Le core LightRAG expose des points importants mais ne les transforme pas en audit events : `ainsert()` (`$LRAG/lightrag.py:1150-1178`), `aquery()` (`$LRAG/lightrag.py:2422-2454`), `adelete_by_doc_id()` (`$LRAG/lightrag.py:2952-2988`). La suppression document log un début de processus et `not_found` (`$LRAG/lightrag.py:3045-3057`), mais pas un événement auditable complet.

### Graph mutations

Les routes graph logguent validation/erreurs ; les succès HTTP sont majoritairement silencieux au niveau route. Exemple entity edit (`$LRAG/api/routers/graph_routes.py:386-405`) et entity create (`$LRAG/api/routers/graph_routes.py:496-513`). `utils_graph.py` ajoute des `logger.info` pour certaines opérations internes (`$LRAG/utils_graph.py:306`, `$LRAG/utils_graph.py:1000`, `$LRAG/utils_graph.py:1266-1269`), mais là aussi sans acteur/corrélation.

### Exceptions / validation

Le handler FastAPI `RequestValidationError` renvoie une réponse structurée pour `/query/data` mais ne log rien (`$LRAG/api/lightrag_server.py:409-436`). Les `HTTPException` d'auth ne sont pas interceptées par un logger dédié.

## 3. Trace ID / corrélation

LightRAG natif ne propage pas de `correlation_id` applicatif entre handlers. Le seul usage de `ContextVar` trouvé dans LightRAG est interne aux locks de storage (`$LRAG/kg/shared_storage.py:1491-1517`), pas pour la corrélation HTTP. La recherche `trace_id|request_id|correlation|ContextVar` ne trouve pas de middleware de requête natif ; `exceptions.py` peut lire `x-request-id` d'une réponse externe (`$LRAG/exceptions.py:12-20`), mais ce n'est pas injecté dans les logs serveur.

Point de branchement recommandé : middleware ASGI ajouté après `create_app(args)`, dans notre wrapper `_patch_lightrag_server_ui()`, avant exposition de l'app. Le middleware doit :

- extraire `traceparent`, `x-request-id`, `x-trace-id`, `langsmith-trace` ;
- générer un `request_id` si absent ;
- stocker `trace_id`, `request_id`, `actor`, `workspace`, `source_ip` dans des `contextvars` ;
- ajouter `x-request-id` / `traceparent` en réponse ;
- émettre un event `http.request.completed` vers une bounded queue, en filtrant statiques `/webui`, `/static/swagger-ui` ;
- enrichir les logs texte via un `logging.Filter` ou `LogRecordFactory`.

Côté twin, `server/app.py` extrait déjà un contexte trace sur `/query`, mais seulement localement dans le handler et en debug (`src/twindb_lightrag_memgraph/server/app.py:252-258`). Les helpers acceptent `langsmith-trace`, `traceparent`, `x-trace-id` (`src/twindb_lightrag_memgraph/server/tracing.py:175-220`). Il faut remonter cette logique au middleware pour couvrir toutes les routes, pas uniquement `/query`.

```python
# src/twindb_lightrag_memgraph/server/tracing.py:175-220
def extract_trace_parent(headers: dict[str, str]) -> dict[str, Any] | None:
    langsmith_trace = headers.get("langsmith-trace") or headers.get("x-langsmith-trace")
    if langsmith_trace:
        return {"langsmith_trace_id": langsmith_trace}
    traceparent = headers.get("traceparent")
    if traceparent:
        parts = traceparent.split("-")
        if len(parts) >= 3:
            return {"trace_id": parts[1], "parent_span_id": parts[2], "traceparent": traceparent}
    trace_id = headers.get("x-trace-id")
    if trace_id:
        return {"trace_id": trace_id}
    return None
```

## 4. Points d'extension pour `@traced`

1. **Middleware HTTP global, robustesse haute.** Patch sur `lightrag.api.lightrag_server.create_app` : créer l'app native, ajouter `EventAuditMiddleware` sur l'app parent. Couverture : auth, query, documents, graph, twin sub-app, erreurs 4xx/5xx, latence. Limite : ne voit pas les opérations internes asynchrones après retour HTTP, sauf si on propage le contexte dans les background tasks.

2. **Wrapper de routes FastAPI après création, robustesse moyenne.** Parcourir `app.routes`, sélectionner `APIRoute`, wrapper `route.endpoint`. Utile pour classifier précisément `/login`, `/query`, `/documents/*`, `/graph/*` et capturer request/response models. Blast radius plus fort car dépend des internals FastAPI/Starlette et des noms de fonctions.

3. **Patch méthodes core LightRAG, robustesse moyenne.** Wrapper `LightRAG.aquery`, `ainsert`, `apipeline_enqueue_documents`, `apipeline_process_enqueue_documents`, `adelete_by_doc_id`, `adelete_by_entity`, `adelete_by_relation` (`$LRAG/lightrag.py:1150`, `1257`, `1605`, `2422`, `2952`, `3669`, `3699`). Couverture des opérations déclenchées hors HTTP aussi. Limite : nécessite extraire acteur/workspace via contextvars et ne remplace pas l'audit auth.

4. **Patch storage writes, robustesse basse à moyenne.** Wrapper les storages déjà attachés au `rag` après `initialize_storages()` dans le lifespan (`$LRAG/api/lightrag_server.py:349-365`) ou patcher les classes de storage (`kg/*`). Couverture très fine des writes KV/vector/graph/docstatus, mais surface large et dépendante backend.

5. **Extension twin activity feed, robustesse haute sur UI governance.** `WebuiStore.record_activity()` appelle `ActivityStore.append()` (`src/.../webui_router.py:286-287`) et `_emit_tag_audit()` centralise tag activity + notification (`src/.../webui_router.py:414-436`). C'est le meilleur point pour brancher l'audit trail métier du fork TS, en envoyant le même event vers la bounded queue centrale.

```python
# src/twindb_lightrag_memgraph/server/webui_router.py:414-436
async def _emit_tag_audit(...):
    event = _make_event(
        kind=kind,
        sev=sev,
        actor=actor,
        target_label=target_label,
        summary=summary,
        meta=meta,
    )
    await store.record_activity(event)
    if notification is not None:
        await store.push_notification(notification)
```

Les stores twin sont déjà append-like et Memgraph-backed, mais ce ne sont pas encore des logs d'audit BCE : `MemgraphActivityStore.append()` écrit un JSON libre dans `n.data` sur un label workspace (`src/.../webui_activitystore.py:184-200`) ; `MemgraphNotificationStore.push()` fait pareil pour les notifications (`src/.../webui_notificationstore.py:138-153`).

```python
# src/twindb_lightrag_memgraph/server/webui_activitystore.py:184-200
async def append(self, event: dict[str, Any]) -> dict[str, Any]:
    if "id" not in event:
        raise ValueError("append requires event['id']")
    payload = json.dumps(event, sort_keys=True)
    async with _pool.acquire_write_slot():
        async with _pool.get_session() as session:
            result = await session.run(
                f"""
                MERGE (n:`{self._label}` {{id: $id}})
                ON CREATE SET n.`__created_at` = timestamp()
                SET n.data = $data, n.`__updated_at` = timestamp()
                """,
                id=str(event["id"]),
                data=payload,
            )
            await result.consume()
    return copy.deepcopy(event)
```

## 5. Compatibilité Log-as-a-Service BNP

Format LightRAG actuel :

- console : texte court `LEVEL: message` ;
- fichier : texte `timestamp - logger - level - message` ;
- rotation locale `lightrag.log`, 10 MB x 5 backups par défaut ;
- pas de JSON ;
- pas de schema stable ;
- pas de `event_type`, `request_id`, `trace_id`, `actor`, `workspace`, `resource_id`, `outcome`, `duration_ms`, `data_classification` ;
- filtre qui supprime des succès HTTP fréquents, dont `/documents` et `/health` (`$LRAG/utils.py:282-310`).

Format attendu pour ingestion Splunk/ELK/Datadog/Log-as-a-Service BNP : JSON line ou OTEL/ECS compatible, immuable et horodaté UTC, avec au minimum :

| Champ | Statut actuel | Cible BNP / audit |
|---|---:|---|
| `ts` UTC ISO-8601 | partiel dans fichier texte | obligatoire |
| `service`, `version`, `env`, `host`, `pid` | absent | obligatoire pour routage LaaS |
| `event_type` | absent | obligatoire, enum stable ~60 types |
| `event_id`, `request_id`, `trace_id`, `span_id` | absent | obligatoire pour corrélation |
| `actor.id`, `actor.role`, `auth.method` | absent | obligatoire pour accès sensibles |
| `source.ip`, `user_agent` | absent | obligatoire sécurité |
| `http.method`, `http.path`, `status`, `duration_ms` | seulement uvicorn texte | structuré |
| `workspace`, `tenant`, `resource.type`, `resource.id` | absent | obligatoire métier |
| `action`, `outcome`, `severity` | message libre | structuré |
| `data_classification`, `sensitive_access` | absent | nécessaire EBA/GL/2019/04 |
| `error.type`, `error.message`, `stack_hash` | message libre | structuré, stack complète seulement côté incident |
| `retention_class`, `legal_hold` | absent | nécessaire DORA/BCE |

Conclusion : on peut continuer à exporter stdout/fichier LightRAG vers Splunk/ELK/Datadog pour observabilité technique, mais ce flux ne suffit pas pour l'audit trail réglementaire. Il faut un second flux `audit_event` structuré, bounded queue + durable sink, qui ne dépend pas de la rotation locale LightRAG.

## 6. Audit trail data path

Table centrale des événements critiques :

| Event critique | Origine LightRAG ou twin | Mode actuel | Manquant pour audit BCE/DORA/EBA | Action requise |
|---|---|---|---|---|
| Login succès | LightRAG `/login` `$LRAG/api/lightrag_server.py:1162-1188` | Aucun log ; token retourné | `actor`, `auth_method`, `source_ip`, `request_id`, `outcome=success`, horodatage durable | Middleware + wrapper `/login` émet `auth.login.success` |
| Login échec | LightRAG `/login` `$LRAG/api/lightrag_server.py:1179-1181` | `HTTPException(401)` sans log | tentative, user hash, IP, user-agent, raison, anti-bruteforce signal | Wrapper `/login` ou exception middleware émet `auth.login.failure` |
| Guest token / auth disabled | LightRAG `/auth-status` et `/login` `$LRAG/api/lightrag_server.py:1132-1151`, `1164-1178` | Aucun log | accès guest traçable, mode auth disabled, durée token | `auth.guest_token.issued` avec `actor=guest`, `auth_mode=disabled` |
| Token/API key refusé | LightRAG auth dependency `$LRAG/api/utils_api.py:236-259` | `HTTPException` sans log | statut 401/403, auth method, IP, path, cause | Middleware 4xx + patch auth dependency émet `auth.access_denied` / `auth.api_key.invalid` |
| Token auto-renew | LightRAG auth dependency `$LRAG/api/utils_api.py:186-200` | `logger.info/warning` texte | event_id, trace_id, expiration, actor structuré | Convertir en `auth.token.renewed` / `auth.token.renew_failed` |
| Query retrieval soumise | LightRAG `/query`, `/query/stream`, `/query/data`; core `aquery` `$LRAG/lightrag.py:2422-2454` | Succès silencieux ; erreurs `logger.error` `$LRAG/api/routers/query_routes.py:452-454`, `738-739` | prompt hash, mode, workspace, actor, durée, nb chunks/docs, sensitive_access | Middleware + wrapper `LightRAG.aquery/aquery_data` émet `retrieval.query.submitted/completed/failed` |
| Accès aux données sensibles | LightRAG document list / graph / query refs | Access logs partiels, `/documents` succès filtrés `$LRAG/utils.py:282-310` | ressource consultée, classification, finalité, acteur | `data.access.document`, `data.access.graph`, `data.access.chunk` avec contenu hashé seulement |
| Upload document | LightRAG `/documents/upload` `$LRAG/api/routers/document_routes.py:2126-2143` | Succès silencieux ; erreur log texte avec filename | actor, file hash, size, MIME, track_id, workspace, malware/result, status indexation | `document.upload.accepted`, puis `document.index.completed/failed` depuis background task |
| Insert text/texts | LightRAG `/documents/text`, `/documents/texts` `$LRAG/api/routers/document_routes.py:2145-2303` | Erreurs log texte ; succès silencieux | doc_id, input hash, volume, workspace, actor | `document.insert.accepted/completed/failed` |
| Delete document | LightRAG `/documents/delete_document`; core `adelete_by_doc_id` `$LRAG/lightrag.py:2952-3057` | logs opérationnels start/not_found ; pas d'actor | approbation, actor, doc_id, cascade graph/vector/KV, outcome | `document.delete.requested/completed/failed`, avec sous-events storage |
| Clear cache / drop storage | LightRAG document routes `$LRAG/api/routers/document_routes.py:2385-2416`, cache endpoints around `2820-2856` | logs succès/erreur texte | actor, justification, scope, approbateur, blast radius | `storage.drop.requested/completed/failed`, `cache.clear.*` |
| Graph entity/relation CRUD | LightRAG graph routes `$LRAG/api/routers/graph_routes.py:380-410`, `490-516`; utils graph `$LRAG/utils_graph.py:306`, `1000`, `1266-1269` | erreurs texte + quelques infos internes | actor, old/new values hash, relation endpoints, workspace, outcome | `graph.entity.create/edit/delete/merge`, `graph.relation.create/edit/delete` |
| Tag CRUD / governance | Twin `_emit_tag_audit()` `src/.../webui_router.py:414-436` | Activity feed + notification ; Memgraph possible `src/.../webui_activitystore.py:184-200` | pas de retention_class, trace_id, IP, immutabilité, schema central | Brancher `_emit_tag_audit()` vers `audit_event_queue`; garder ActivityFeed comme projection UI |
| Notification read/clear | Twin `MemgraphNotificationStore.mark_all_read/clear/push` `src/.../webui_notificationstore.py:116-153` | mutation store sans audit séparé | actor, notification ids, raison, outcome | `notification.read_all`, `notification.clear`, `notification.push` |
| Incident 5xx / timeout / queue full | LightRAG logs erreurs et warnings divers ; queue utilitaire `$LRAG/utils.py:616-940` | texte technique dispersé | incident_id, severity, service impact, DORA classification, lifecycle | Middleware 5xx + queue bounded `incident.detected/escalated/resolved` |

### MUST-LOG manquants aujourd'hui

1. `auth.login.success` et `auth.login.failure` pour `/login`.
2. `auth.access_denied`, `auth.api_key.invalid`, `auth.token.invalid`.
3. `retrieval.query.submitted`, `retrieval.query.completed`, `retrieval.query.failed`.
4. `data.access.document`, `data.access.chunk`, `data.access.graph`.
5. `document.upload.accepted`, `document.index.completed`, `document.index.failed`.
6. `document.delete.requested`, `document.delete.completed`, `document.delete.failed`.
7. `storage.write`, `storage.delete`, `storage.drop`, `cache.clear`.
8. `graph.entity.*` et `graph.relation.*` mutations.
9. `tag.create/request/approve/reject/deprecate/delete/synonym.*` vers sink central, pas seulement ActivityFeed.
10. `incident.5xx`, `incident.timeout`, `incident.queue_full`, `incident.storage_unavailable`.

## Patch recommandé

Stratégie recommandée : **A = middleware HTTP + contextvars + queue centrale**, complétée par **E = branchement direct du twin ActivityFeed**. Le middleware est le point le plus robuste face aux upgrades LightRAG 1.4.10/1.4.11/1.4.12 : il ne dépend que de `create_app(args)` qui existe déjà pour le patch UI, couvre toutes les routes natives et la sous-app twin, et peut enrichir les logs via `LogRecordFactory` sans modifier les handlers LightRAG.

Pour les événements métier longs ou asynchrones, ajouter ensuite des wrappers ciblés autour de `LightRAG.aquery`, `ainsert`, `adelete_by_doc_id` et des tag mutations. Ne pas essayer de transformer les logs texte LightRAG en audit trail : ils sont utiles pour debug, mais non structurés, filtrés, locaux et sans rétention. Le flux BCE/DORA doit être un `audit_event` JSON stable, poussé dans une bounded queue avec backpressure explicite, stockage central, et projection UI optionnelle via `/activity`.
