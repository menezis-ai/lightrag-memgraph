# Prisme A - HTTP & UI mount LightRAG 1.4.9.11

Audit realise sur le wheel exact `lightrag-hku==1.4.9.11` extrait dans `/private/tmp/lrag14911`. Le repo local a actuellement `1.4.10` dans `.venv`, donc les references ci-dessous utilisent le chemin logique `$LRAG/api/...` pour la version cible `1.4.9.11`.

## 1. Factory FastAPI

La factory principale est `$LRAG/api/lightrag_server.py:287` :

```python
def create_app(args):
```

Elle instancie FastAPI dans la meme fonction, a `$LRAG/api/lightrag_server.py:390` et `$LRAG/api/lightrag_server.py:407` :

```python
    app_kwargs = {
        "title": "LightRAG Server API",
        "description": swagger_description,
        "version": __api_version__,
        "openapi_url": "/openapi.json",  # Explicitly set OpenAPI schema URL
        "docs_url": None,  # Disable default docs, we'll create custom endpoint
        "redoc_url": "/redoc",  # Explicitly set redoc URL
        "lifespan": lifespan,
    }

    app = FastAPI(**app_kwargs)
```

Wrapper secondaire : `$LRAG/api/lightrag_server.py:1366`.

```python
def get_application(args=None):
    """Factory function for creating the FastAPI application"""
    if args is None:
        args = global_args
    return create_app(args)
```

Chemins d'appel :

- Uvicorn single-process : `$LRAG/api/lightrag_server.py:1504` cree l'app avec `create_app(global_args)`, puis `$LRAG/api/lightrag_server.py:1526` lance `uvicorn.run(**uvicorn_config)` avec l'instance ASGI deja construite.
- Gunicorn : `$LRAG/api/run_with_gunicorn.py:258` charge `get_application`, puis `$LRAG/api/run_with_gunicorn.py:262` retourne `get_application(global_args)`.
- `if __name__ == "__main__"` existe dans les deux entrypoints : `$LRAG/api/lightrag_server.py:1529` et `$LRAG/api/run_with_gunicorn.py:281`.

`create_app(args)` est stateful, pas idempotente :

- Elle lit l'etat environnement/module : `.env` charge au module import `$LRAG/api/lightrag_server.py:70`, `webui_title`/`webui_description` `$LRAG/api/lightrag_server.py:73`, `auth_configured` `$LRAG/api/lightrag_server.py:81`.
- Elle mute `args.llm_binding_host` et `args.embedding_binding_host` si absents : `$LRAG/api/lightrag_server.py:325` et `$LRAG/api/lightrag_server.py:329`.
- Elle instancie un nouveau `DocumentManager` et un nouveau `LightRAG` a chaque appel : `$LRAG/api/lightrag_server.py:347`, `$LRAG/api/lightrag_server.py:1052`.
- Les routers des modules sont globaux (`router = APIRouter(...)`) et les fonctions `create_*_routes()` ajoutent des routes sur ces objets globaux. Appeler deux fois `create_app()` dans le meme process peut donc dupliquer les routes dans les routers globaux. References : `$LRAG/api/routers/document_routes.py:79`, `$LRAG/api/routers/query_routes.py:13`, `$LRAG/api/routers/graph_routes.py:13`.

## 2. WebUI mount

Le WebUI natif est servi sous `/webui`, pas sous `/` directement. `/` est seulement une redirection.

Detection des assets : `$LRAG/api/lightrag_server.py:158`.

```python
def check_frontend_build():
    """Check if frontend is built and optionally check if source is up-to-date
```

Le chemin natif est `$LRAG/api/webui/index.html`, determine a `$LRAG/api/lightrag_server.py:166`.

```python
    webui_dir = Path(__file__).parent / "webui"
    index_html = webui_dir / "index.html"
```

Route racine : `$LRAG/api/lightrag_server.py:1124`.

```python
    @app.get("/")
    async def redirect_to_webui():
        """Redirect root path based on WebUI availability"""
        if webui_assets_exist:
            return RedirectResponse(url="/webui")
        else:
            return RedirectResponse(url="/docs")
```

Mount Swagger local : `$LRAG/api/lightrag_server.py:1332`.

```python
    # Mount Swagger UI static files for offline support
    swagger_static_dir = Path(__file__).parent / "static" / "swagger-ui"
    if swagger_static_dir.exists():
        app.mount(
            "/static/swagger-ui",
            StaticFiles(directory=swagger_static_dir),
            name="swagger-ui-static",
        )
```

Mount WebUI natif : `$LRAG/api/lightrag_server.py:1341`.

```python
    # Conditionally mount WebUI only if assets exist
    if webui_assets_exist:
        static_dir = Path(__file__).parent / "webui"
        static_dir.mkdir(exist_ok=True)
        app.mount(
            "/webui",
            SmartStaticFiles(
                directory=static_dir, html=True, check_dir=True
            ),  # Use SmartStaticFiles
            name="webui",
        )
        logger.info("WebUI assets mounted at /webui")
    else:
        logger.info("WebUI assets not available, /webui route not mounted")
```

Fallback quand les assets n'existent pas : `$LRAG/api/lightrag_server.py:1356`.

```python
        # Add redirect for /webui when assets are not available
        @app.get("/webui")
        @app.get("/webui/")
        async def webui_redirect_to_docs():
            """Redirect /webui to /docs when WebUI is not available"""
            return RedirectResponse(url="/docs")
```

La classe statique custom est locale a `create_app` : `$LRAG/api/lightrag_server.py:1303`.

```python
    # Custom StaticFiles class for smart caching
    class SmartStaticFiles(StaticFiles):  # Renamed from NoCacheStaticFiles
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
```

Routes exactes exposees par ce bloc :

- `/` -> redirect `/webui` si assets presents, sinon `/docs`.
- `/webui` et `/webui/...` -> `SmartStaticFiles(directory=$LRAG/api/webui, html=True)`.
- `/webui/assets/...` -> assets Vite natifs.
- `/static/swagger-ui/...` -> assets Swagger locaux.
- Pas de mount generique `/static`.

## 3. Ordre d'enregistrement des routers/mounts

Ordre dans `create_app(args)` :

1. `FastAPI(**app_kwargs)` : `$LRAG/api/lightrag_server.py:407`.
2. Exception handler validation : `$LRAG/api/lightrag_server.py:410`.
3. Middleware CORS : `$LRAG/api/lightrag_server.py:448`.
4. Creation `LightRAG(...)` : `$LRAG/api/lightrag_server.py:1052`.
5. Router documents : `$LRAG/api/lightrag_server.py:1091`.
6. Router query : `$LRAG/api/lightrag_server.py:1098`.
7. Router graph : `$LRAG/api/lightrag_server.py:1099`.
8. Router Ollama sous prefix `/api` : `$LRAG/api/lightrag_server.py:1102`.
9. Routes locales `/docs`, `/docs/oauth2-redirect`, `/`, `/auth-status`, `/login`, `/health` : `$LRAG/api/lightrag_server.py:1106`, `$LRAG/api/lightrag_server.py:1119`, `$LRAG/api/lightrag_server.py:1124`, `$LRAG/api/lightrag_server.py:1132`, `$LRAG/api/lightrag_server.py:1162`, `$LRAG/api/lightrag_server.py:1197`.
10. Mount `/static/swagger-ui` : `$LRAG/api/lightrag_server.py:1335`.
11. Mount `/webui` ou fallback GET `/webui` : `$LRAG/api/lightrag_server.py:1345` ou `$LRAG/api/lightrag_server.py:1357`.

Precedence : Starlette/FastAPI parcourt les routes dans l'ordre d'insertion. Le premier match complet gagne ; les mounts capturent leur prefix. Donc :

- Les endpoints API ajoutes avant `/webui` prennent precedence sur leurs chemins exacts.
- Une route/mount ajoutee apres le mount natif `/webui` sur le meme prefix ne remplacera pas le WebUI natif. Il faut remplacer la route existante `name="webui"` ou l'inserer au meme index.
- Monter notre sous-app sur un prefix non conflictuel (`/twin`, `/twin-api`, etc.) peut etre fait apres creation de l'app. Monter sous `/webui/...` doit etre fait avant ou en remplacant le mount `/webui`.

Routers globaux :

- `$LRAG/api/routers/document_routes.py:79` : `APIRouter(prefix="/documents", tags=["documents"])`.
- `$LRAG/api/routers/query_routes.py:13` : `APIRouter(tags=["query"])`.
- `$LRAG/api/routers/graph_routes.py:13` : `APIRouter(tags=["graph"])`.
- `$LRAG/api/routers/ollama_api.py:226` : `APIRouter(tags=["ollama"])`, inclus avec `prefix="/api"` a `$LRAG/api/lightrag_server.py:1103`.

Endpoints principaux crees dans les routers :

- Documents : `/documents/scan` `$LRAG/api/routers/document_routes.py:2045`, `/documents/upload` `:2070`, `/documents/text` `:2145`, `/documents/texts` `:2223`, `DELETE /documents` `:2306`, `/documents/pipeline_status` `:2500`, `GET /documents` `:2601`, `DELETE /documents/delete_document` `:2711`, `/documents/clear_cache` `:2790`, `DELETE /documents/delete_entity` `:2824`, `DELETE /documents/delete_relation` `:2859`, `/documents/track_status/{track_id}` `:2897`, `/documents/paginated` `:2971`, `/documents/status_counts` `:3058`, `/documents/reprocess_failed` `:3085`, `/documents/cancel_pipeline` `:3131`.
- Query : `/query` `$LRAG/api/routers/query_routes.py:196`, `/query/stream` `:456`, `/query/data` `:742`.
- Graph : `/graph/label/list` `$LRAG/api/routers/graph_routes.py:92`, `/graph/label/popular` `:109`, `/graph/label/search` `:133`, `/graphs` `:159`, `/graph/entity/exists` `:197`, `/graph/entity/edit` `:220`, `/graph/relation/edit` `:410`, `/graph/entity/create` `:445`, `/graph/relation/create` `:518`, `/graph/entities/merge` `:607`.
- Ollama sous `/api` : `/api/version` `$LRAG/api/routers/ollama_api.py:233`, `/api/tags` `:238`, `/api/ps` `:261`, `/api/generate` `:285`, `/api/chat` `:462`.

## 4. Middlewares appliques

Un seul middleware FastAPI explicite : CORS, a `$LRAG/api/lightrag_server.py:447`.

```python
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-New-Token"
        ],  # Expose token renewal header for cross-origin requests
    )
```

`get_cors_origins()` lit `global_args.cors_origins` a `$LRAG/api/lightrag_server.py:438`.

Auth : pas un middleware ASGI global. C'est une dependance par route via `Depends(combined_auth)`. La dependance est creee a `$LRAG/api/lightrag_server.py:459` et definie dans `$LRAG/api/utils_api.py:80`.

```python
def get_combined_auth_dependency(api_key: Optional[str] = None):
```

Elle combine :

- whitelist de chemins calculee au module import depuis `global_args.whitelist_paths` : `$LRAG/api/utils_api.py:61`.
- OAuth2 bearer token : `$LRAG/api/utils_api.py:98`.
- header API key `X-API-Key` si configure : `$LRAG/api/utils_api.py:103`.
- auto-renouvellement de token via header `X-New-Token` : `$LRAG/api/utils_api.py:130`.

Routes non protegees par `Depends(combined_auth)` dans `lightrag_server.py` :

- `/docs`, `/docs/oauth2-redirect`, `/`, `/auth-status`, `/login`.
- `/webui` static mount. Le WebUI lui-meme est public ; ses appels API sont proteges par les dependances des endpoints.

Logging :

- Pas de middleware HTTP de logging applicatif.
- Configuration logging dictConfig dans `$LRAG/api/lightrag_server.py:1373`, avec handlers console+file et loggers `uvicorn`, `uvicorn.access`, `uvicorn.error`, `lightrag` aux lignes `$LRAG/api/lightrag_server.py:1420` a `:1443`.
- Filtre `lightrag.utils.LightragPathFilter` a `$LRAG/api/lightrag_server.py:1445`.

## 5. Lifespan / startup events

Il n'y a pas de `@app.on_event("startup")` dans les fichiers audites. Le startup/shutdown passe par un lifespan local a `create_app`, declare a `$LRAG/api/lightrag_server.py:349`, puis injecte dans `FastAPI` a `$LRAG/api/lightrag_server.py:397`.

```python
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()

        try:
            # Initialize database connections
            # Note: initialize_storages() now auto-initializes pipeline_status for rag.workspace
            await rag.initialize_storages()

            # Data migration regardless of storage implementation
            await rag.check_and_migrate_data()

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Clean up database connections
            await rag.finalize_storages()
```

Shutdown shared storage : `$LRAG/api/lightrag_server.py:371`.

```python
            if "LIGHTRAG_GUNICORN_MODE" not in os.environ:
                # Only perform cleanup in Uvicorn single-process mode
                logger.debug("Unvicorn Mode: finalizing shared storage...")
                finalize_share_data()
            else:
                # In Gunicorn mode with preload_app=True, cleanup is handled by on_exit hooks
                logger.debug(
                    "Gunicorn Mode: postpone shared storage finalization to master process"
                )
```

Peut-on hook avant ?

- Avant app creation : oui, en patchant `lightrag.api.lightrag_server.create_app` avant que `main()` ou `get_application()` soit appele.
- Apres app creation mais avant startup : oui, on peut modifier `app.router.routes` et `app.router.lifespan_context` dans le wrapper `create_app`.
- Avant l'initialisation `rag.initialize_storages()` : seulement en wrappant/remplacant `app.router.lifespan_context` ou en patchant `create_app` lui-meme. Un simple `app.mount()` apres creation ne permet pas d'executer du code avant le lifespan parent.
- Attention critique : une sous-app FastAPI montee avec `app.mount("/x", sub_app)` ne doit pas etre supposee initialiser son lifespan comme app principale. Notre `src/twindb_lightrag_memgraph/server/app.py:create_app()` initialise `_rag` dans son lifespan (`src/twindb_lightrag_memgraph/server/app.py:97` a `:137`). Si on la monte telle quelle, il faut chainer explicitement son lifespan dans le lifespan parent, ou exposer une variante sans lifespan qui partage le `rag` parent.

## 6. Points de monkey-patch viables pour `_patch_lightrag_server_ui()`

### Option A - wrapper `create_app`, remplacement du mount existant

Robustesse : meilleure option si notre launcher importe le patch avant `lightrag_server.main()` ou `run_with_gunicorn`. Couvre uvicorn programmatique et gunicorn, garde l'ordre de route existant, reversible en restaurant `orig`.

Attribut a patcher : `lightrag.api.lightrag_server.create_app`.

```python
def _patch_lightrag_server_ui(dist, twin_app):
    import lightrag.api.lightrag_server as srv
    from fastapi.staticfiles import StaticFiles
    if getattr(srv, "_twin_ui_patched", False): return
    orig = srv.create_app
    def wrapped(args):
        app = orig(args)
        next(r for r in app.router.routes if getattr(r, "name", None) == "webui").app = StaticFiles(directory=dist, html=True, check_dir=True)
        app.mount("/twin", twin_app, name="twin"); return app
    srv.create_app = wrapped; srv._twin_ui_patched = True
```

Limite : si `twin_app` est `server/app.py:create_app()`, son lifespan doit etre chaine manuellement (voir Option B), sinon les routes qui appellent `_get_rag()` risquent `RuntimeError("LightRAG not initialized")`.

### Option B - wrapper `create_app` avec chainage lifespan sous-app

Robustesse : meilleure option si la sous-app doit vraiment etre `src/twindb_lightrag_memgraph/server/app.py:create_app()`. Blast radius plus large que A, car on remplace `app.router.lifespan_context`, mais c'est necessaire pour initialiser/fermer la sous-app.

Attribut a patcher : `app.router.lifespan_context` sur l'app retournee par `lightrag_server.create_app`.

```python
from contextlib import AsyncExitStack, asynccontextmanager
def _chain_lifespan(parent, child):
    old = parent.router.lifespan_context
    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(old(app))
            await stack.enter_async_context(child.router.lifespan_context(child))
            yield
    parent.router.lifespan_context = lifespan
```

Combinaison : creer `twin_app = twindb_lightrag_memgraph.server.app.create_app()`, appeler `_chain_lifespan(app, twin_app)`, puis `app.mount("/twin", twin_app, name="twin")`. Si on veut que la sous-app demarre avant le parent, inverser les deux `enter_async_context`, mais ce n'est pas neutre car le parent initialise le `LightRAG` natif dans son propre lifespan.

### Option C - patch `get_application` seulement

Robustesse : utile pour Gunicorn ou pour un ASGI launcher qui cible `lightrag.api.lightrag_server:get_application`. Moins complet que A : le chemin `lightrag_server.main()` appelle directement `create_app(global_args)` a `$LRAG/api/lightrag_server.py:1505`, donc ce patch ne couvre pas le CLI uvicorn natif.

Attribut a patcher : `lightrag.api.lightrag_server.get_application`.

```python
def _patch_get_application(post_process):
    import lightrag.api.lightrag_server as srv
    if getattr(srv, "_twin_getapp_patched", False): return
    orig = srv.get_application
    def wrapped(args=None):
        return post_process(orig(args))
    srv.get_application = wrapped; srv._twin_getapp_patched = True
```

### Option D - overlay filesystem de `$LRAG/api/webui`

Robustesse UI seule : simple et stable si on controle l'image/container, mais pas suffisant pour monter la sous-app. Blast radius packaging plus fort : on modifie le contenu installe ou on bind-mount un repertoire par-dessus.

Attribut a patcher : aucun. Remplacer le repertoire servi par `static_dir = Path(__file__).parent / "webui"` a `$LRAG/api/lightrag_server.py:1343`.

```python
def overlay_webui(dist):
    import lightrag.api.lightrag_server as srv
    webui_dir = Path(srv.__file__).parent / "webui"
    assert (Path(dist) / "index.html").exists()
    # En container: bind-mount dist -> webui_dir.
    # En venv mutable: remplacer webui_dir par un symlink vers dist.
```

## 7. Risques upgrades 1.4.10 / 1.4.11 / 1.4.12

Comparaison locale avec `1.4.10` installe : les points HTTP/UI critiques sont stables. `create_app`, `get_application`, `FastAPI(**app_kwargs)`, les `include_router`, `app.mount("/webui", ...)` et le lifespan restent aux memes endroits conceptuels ; seuls des offsets de lignes changent legerement. Le diff local observe touche l'embedding Azure et un workaround Windows dans `main()`, pas la surface WebUI.

Le patch tient probablement si LightRAG garde :

- `lightrag.api.lightrag_server.create_app(args)`.
- `get_application(args=None)` qui appelle `create_app`.
- `app.mount("/webui", ..., name="webui")`.
- `check_frontend_build()` retournant `(assets_exist, is_outdated)`.
- Le WebUI empaquete dans `Path(__file__).parent / "webui"`.

Signaux a surveiller a chaque upgrade :

- Renommage ou changement de signature de `create_app(args)` ou `get_application(args=None)`.
- Passage d'une factory runtime vers une variable globale `app = FastAPI(...)` au module import.
- Refactor du WebUI : route `/webui` renommee, mount `/` direct, assets deplaces, suppression de `name="webui"`, remplacement de `StaticFiles`.
- Changement d'ordre : mount `/webui` avant les routes API, ou catch-all SPA ajoute sur `/`.
- Remplacement du lifespan local par `@app.on_event` ou par une factory externe.
- Disparition des routers globaux ou changement des prefixes `/documents`, `/query`, `/graph`, `/api`.
- Changements auth : passage de `Depends(combined_auth)` a middleware global, ce qui impacterait une sous-app montee.

Test de garde recommande : apres patch, inspecter `app.router.routes` et verifier :

- une route `name=="webui"` existe et pointe vers notre `dist`.
- `/` repond encore par redirect `/webui`.
- `/webui/index.html` vient de `lightrag_webui_twin/dist`.
- `/twin/health` ou equivalent repond et son lifespan a initialise ses dependances.

## Patch recommande

Strategie recommandee : Option A + chainage lifespan minimal de l'Option B. Le point d'injection principal doit etre un wrapper idempotent de `lightrag.api.lightrag_server.create_app`, charge par notre launcher avant LightRAG. Dans le wrapper, remplacer l'application du `Mount` nomme `webui` au lieu d'ajouter un deuxieme mount `/webui`; cela conserve la precedence exacte de LightRAG et evite les effets de bord sur `/`, `/docs`, `/health`, les routers et Swagger.

Pour la sous-app, ne pas se contenter de `app.mount("/twin", create_app())` si on utilise `src/twindb_lightrag_memgraph/server/app.py:create_app()`, car son initialisation est dans son lifespan. Il faut soit chainer explicitement `twin_app.router.lifespan_context`, soit extraire les routers Twin dans une integration sans lifespan qui partage le `rag` parent. Pour lundi, le chemin le moins risque est : remplacer le mount WebUI existant par notre `dist`, monter Twin sous un prefix non conflictuel (`/twin`), et ajouter un test de demarrage qui prouve que le lifespan Twin a ete execute.
