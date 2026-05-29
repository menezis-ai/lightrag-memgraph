# Prisme B - Boot lifecycle & monkey-patch hookpoints LightRAG 1.4.9.11

Audit realise sur le wheel exact `lightrag-hku==1.4.9.11` extrait dans `/private/tmp/lrag14911`. La venv locale contenait `1.4.10`; les references ci-dessous utilisent donc le chemin logique `$LRAG/...` pour la version cible.

## 1. Ordre d'import critique

Point important : en `1.4.9.11`, `python -m lightrag.api` n'est pas un entrypoint valide. Le package `$LRAG/api` n'a pas de `__main__.py`, donc Python sort avec :

```text
No module named lightrag.api.__main__; 'lightrag.api' is a package and cannot be directly executed
```

Les entrypoints valides du wheel sont :

- `lightrag-server = lightrag.api.lightrag_server:main`
- `lightrag-gunicorn = lightrag.api.run_with_gunicorn:main`
- equivalent direct : `python -m lightrag.api.lightrag_server`

Quand un utilisateur fait `from lightrag import LightRAG`, l'ordre est :

```text
user code
  import lightrag
    $LRAG/__init__.py
      from .lightrag import LightRAG, QueryParam          <- loads core class now
        $LRAG/lightrag.py
          imports lightrag.kg.STORAGES, verify_storage_implementation
          imports lightrag.kg.shared_storage helpers
          imports lightrag.operate.merge_nodes_and_edges by value
          defines @dataclass LightRAG
      __version__ = "v1.4.9.11"                          <- version source

  from twindb_lightrag_memgraph import register
  register()
    imports lightrag.kg                                <- already loaded, mutable
    mutates STORAGE_IMPLEMENTATIONS / STORAGE_ENV_REQUIREMENTS / STORAGES
    imports and patches lightrag.kg.memgraph_impl.MemgraphStorage
    patches lightrag.operate hot paths
    patches both lightrag.operate.merge_nodes_and_edges
      and lightrag.lightrag.merge_nodes_and_edges       <- patch hook here
    patches lightrag.lightrag.LightRAG._insert_done     <- patch hook here
    mutates lightrag.__version__                        <- patch hook here
```

Quand le serveur est ensuite lance via l'entrypoint valide :

```text
lightrag-server / python -m lightrag.api.lightrag_server
  $LRAG/api/lightrag_server.py module import
    imports FastAPI, uvicorn, StaticFiles, config, routers, auth
    from lightrag import LightRAG, __version__ as core_version
      core_version is bound once at module import         <- _patch_version_string must be before this
    webui_title = os.getenv(...)
    webui_description = os.getenv(...)
    auth_configured = bool(auth_handler.accounts)
    defines create_app(), get_application(), main()
  if __name__ == "__main__": main()
    initialize_config()
    check_env_file()
    check_and_install_dependencies()
    configure_logging()
    update_uvicorn_mode_config()
    display_splash_screen(global_args)
    app = create_app(global_args)                         <- _patch_lightrag_server_ui wrapper must be active before this
      check_frontend_build()
      config_cache = LLMConfigCache(args)
      doc_manager = DocumentManager(...)
      app = FastAPI(... lifespan=lifespan)
      app.add_middleware(CORSMiddleware, ...)
      rag = LightRAG(...)                                 <- storage registry patches must be active before this
        LightRAG.__post_init__()
          initialize_share_data()
          verify_storage_implementation(...)
          check_storage_env_vars(...)
          _get_storage_class(...)
            STORAGES[storage_name]
            lazy_external_import(...)(...)                <- our absolute module paths consumed here
      app.include_router(...)
      app.mount("/static/swagger-ui", ...)
      app.mount("/webui", ..., name="webui")              <- native UI mount; replace in wrapped create_app return path
    uvicorn.run(app=app, ...)
      lifespan startup
        await rag.initialize_storages()
        await rag.check_and_migrate_data()
```

Gunicorn path is similar, but `run_with_gunicorn` ends up using `get_application(global_args)`, which calls `create_app(args)`. Therefore patching `create_app` covers both CLI single-process and Gunicorn. Patching only `get_application` does not cover `lightrag_server.main()`, because `main()` calls `create_app(global_args)` directly at `$LRAG/api/lightrag_server.py:1505`.

## 2. Points de no-return

`lightrag.__version__`:

- No-return for `_patch_version_string()` is import evaluation of `$LRAG/api/lightrag_server.py:37`.
- At that line, `core_version` becomes a module local string copy via `from lightrag import LightRAG, __version__ as core_version`.
- Patching `lightrag.__version__` after `lightrag_server` is imported does not update `lightrag_server.core_version` unless we also mutate `lightrag.api.lightrag_server.core_version` explicitly.

`lightrag.api.lightrag_server.create_app`:

- The function object can be replaced after module import and before `main()` calls `create_app(global_args)`.
- No-return for a `create_app` wrapper is `$LRAG/api/lightrag_server.py:1505` in CLI mode, or the equivalent `get_application(...)->create_app(...)` call in Gunicorn.
- After `create_app()` returns, patching `create_app` is too late for the already-created `app`; at that point patch `app.router.routes`, existing `Mount` objects, or `app.router.lifespan_context` directly.

Native WebUI mount:

- The native mount is created inside `create_app()` at `$LRAG/api/lightrag_server.py:1345`.
- Adding another mount on `/webui` after this is too late because Starlette route order is insertion-order. The native mount will match first.
- Viable post-create mutation: find the route with `name == "webui"` and replace its `.app`, preserving route precedence. If assets are absent, LightRAG creates GET routes `/webui` and `/webui/` instead; replacing UI in that path requires removing/replacing those route entries or forcing `webui_assets_exist` path through a wrapper.

LightRAG storage registries:

- No-return for registry mutation is `LightRAG.__post_init__`, specifically storage verification at `$LRAG/lightrag.py:483` and class resolution at `$LRAG/lightrag.py:553`.
- Patching `STORAGE_IMPLEMENTATIONS`, `STORAGE_ENV_REQUIREMENTS`, and `STORAGES` after a `LightRAG(...)` instance is created does not affect that instance's already-created storage classes/objects.
- It can affect later `LightRAG(...)` instances in the same process.

Class/function monkey patches:

- `MemgraphStorage.initialize` and batch methods must be patched before the storage object calls those methods. Best timing is before `LightRAG(...)`, because `LightRAG.__post_init__` instantiates storage objects.
- `merge_nodes_and_edges` must patch both `lightrag.operate.merge_nodes_and_edges` and `lightrag.lightrag.merge_nodes_and_edges`; `$LRAG/lightrag.py` imports it by value at module import.
- `LightRAG._insert_done` can be patched after class import but before any insertion reaches `_insert_done`. Practically, keep it in `register()` before exposing the server.

## 3. Imports differes exploitables

There are a few lazy points, but none remove the need to call `register()` before the first `LightRAG(...)` construction.

- Storage fallback import is lazy: `$LRAG/lightrag.py:1110` reads `STORAGES[storage_name]`, and `$LRAG/utils.py:1867` returns a closure that calls `importlib.import_module(...)` only when the storage class is instantiated. This is why our absolute `STORAGES` paths work.
- Some provider-specific server imports are lazy inside `LLMConfigCache` and binding factories, for example `lightrag.llm.binding_options` and Bedrock helpers. These are not useful for UI/server hook timing.
- `lightrag_server` itself is not lazy: all top-level imports and top-level globals (`core_version`, `webui_title`, `auth_configured`) are evaluated during module import.
- `SmartStaticFiles` is local to `create_app()`, so it cannot be patched by module attribute before app creation. Patch the returned route/mount instead.

## 4. State global mutable

LightRAG has no single global `LightRAG` singleton in core. It has several mutable module globals that matter:

- `lightrag.kg.STORAGE_IMPLEMENTATIONS`, `STORAGE_ENV_REQUIREMENTS`, `STORAGES`: plain dict/list registry, safe to mutate before instance construction.
- `lightrag.kg.shared_storage`: process/global shared-state singleton set, including `_initialized`, `_shared_dicts`, `_init_flags`, `_update_flags`, `_default_workspace`, locks, and optional multiprocessing `Manager`. `initialize_share_data()` is idempotent once `_initialized` is true; `finalize_share_data()` resets these globals.
- `lightrag.api.lightrag_server`: module globals `core_version`, `webui_title`, `webui_description`, `auth_configured`, and imported function/class names. These are mutable, but values copied at import time will not follow later mutations of their source modules.
- Router modules under `$LRAG/api/routers`: routers are module-level objects. As noted in Prisme A, repeated `create_app()` calls in one process can accumulate routes on those global routers.
- Our package has `_registered` as a process-global guard and `_post_index_hooks` in `_hooks.py` as a mutable callback registry.

## 5. Idempotence and race conditions

Normal repeated call:

- `register()` returns immediately when `_registered` is true.
- Registry list insertion is individually guarded with `if class_name not in impls`.
- `STORAGE_ENV_REQUIREMENTS.update(...)` and `STORAGES.update(...)` are overwrite-idempotent.
- `_patch_version_string()` has its own marker guard.

Pitfalls if `_registered` is bypassed or two threads race before `_registered = True`:

- There is no lock around `_registered`. Python's import lock protects module import, not arbitrary concurrent calls to `register()`. Two concurrent calls can both pass the guard.
- `_patch_merge_write_path()` captures the current `operate.merge_nodes_and_edges`. A second patch can capture the first wrapper as its "original", stacking wrappers and causing duplicate proxy/flush behavior for Memgraph paths.
- `_patch_insert_done()` captures the current `LightRAG._insert_done`. A second patch stacks wrappers, so post-index hooks can run more than once per insert completion.
- `_patch_operate_hot_paths()` and `MemgraphStorage` method assignment are less dangerous because they mainly replace methods, but repeated patching still changes closure state and makes debugging harder.
- Test helpers that set `twindb_lightrag_memgraph._registered = False` do not undo class/function monkey patches. They only reopen the guard.

Recommendation for future patches:

- Give every wrapper patch its own sentinel on the target module/class/function, e.g. `srv._twindb_create_app_patched`, `LightRAG._twindb_insert_done_patched`, or `wrapped.__twindb_original__`.
- Set `_registered = True` only after all patches succeed, but protect the body with a module-level `threading.Lock` if server boot can call `register()` from multiple threads.
- For patches that wrap callables, never use only the top-level `_registered` flag as the sole idempotence boundary.

## 6. Recommendation for `_patch_lightrag_server_ui()` and `_mount_twin_server()`

Place both server/UI patches inside `register()` after `_patch_version_string()` and before `_registered = True`.

Rationale:

- `_patch_version_string()` must run before importing `lightrag.api.lightrag_server`, otherwise `core_version` is already bound to the old value.
- `_patch_lightrag_server_ui()` will necessarily import or patch `lightrag.api.lightrag_server`. If it runs before `_patch_version_string()`, it can accidentally trigger the no-return point for `core_version`.
- Storage and core behavior patches should remain first because they must be in place before any server-created `LightRAG(...)`.
- `_patch_lightrag_server_ui()` should patch `lightrag.api.lightrag_server.create_app` idempotently. The wrapper should call the original, then replace the existing `name=="webui"` mount app or handle the no-assets fallback.
- `_mount_twin_server()` should be called from inside the `create_app` wrapper, after the original app is returned and before the server starts. It should mount on a non-conflicting prefix such as `/twin` or `/twin-api`, and should not add a second `/webui` mount behind the native one.
- If `_mount_twin_server()` mounts `src/twindb_lightrag_memgraph.server.app:create_app()`, it must explicitly chain that sub-app lifespan or use a router-only integration. A mounted sub-app's lifespan should not be assumed to initialize the module-level `_rag` in `server/app.py`.

Suggested order:

```python
def register() -> None:
    ...
    _patch_builtin_memgraph_storage()
    _patch_merge_write_path()
    _patch_insert_done()
    _patch_version_string()          # must precede lightrag_server import
    _patch_lightrag_server_ui()      # wraps lightrag_server.create_app
    _mount_twin_server()             # or install hook used by the create_app wrapper
    _registered = True
```

If `_mount_twin_server()` is not an immediate patch but a helper invoked by `_patch_lightrag_server_ui()`'s wrapper, keep the call graph like this:

```text
register()
  _patch_version_string()
  _patch_lightrag_server_ui()
    import lightrag.api.lightrag_server as srv
    orig = srv.create_app
    srv.create_app = wrapped_create_app

wrapped_create_app(args)
  app = orig(args)
  replace native webui mount if configured
  _mount_twin_server(app)            <- patch X hook here
  return app
```

## ASCII lifecycle diagram

```text
Process start
  |
  | user imports twindb_lightrag_memgraph.register
  v
register()
  |
  +--> import lightrag.kg
  |      mutate STORAGE_IMPLEMENTATIONS / STORAGE_ENV_REQUIREMENTS / STORAGES
  |      <- storage registry hook here
  |
  +--> patch lightrag.kg.memgraph_impl.MemgraphStorage
  |      <- graph TLS/session/batch hook here
  |
  +--> patch lightrag.operate + lightrag.lightrag merge_nodes_and_edges
  |      <- double-patch hook here because lightrag.py imported by value
  |
  +--> patch LightRAG._insert_done
  |      <- post-index hook here
  |
  +--> patch lightrag.__version__
  |      <- must happen before lightrag_server imports core_version
  |
  +--> patch lightrag.api.lightrag_server.create_app
  |      <- _patch_lightrag_server_ui hook here
  |
  +--> set _registered = True
  |
  v
lightrag-server / python -m lightrag.api.lightrag_server
  |
  +--> import lightrag.api.lightrag_server
  |      from lightrag import LightRAG, __version__ as core_version
  |      <- no-return for _patch_version_string
  |
  +--> main()
         |
         +--> app = create_app(global_args)
         |      <- no-return for wrapping create_app
         |
         +--> wrapped_create_app()
                |
                +--> original create_app()
                |      app = FastAPI(...)
                |      rag = LightRAG(...)
                |        __post_init__()
                |          verify_storage_implementation()
                |          _get_storage_class()
                |          <- no-return for storage registry patches
                |      include native routers
                |      mount native /webui
                |
                +--> replace name=="webui" mount if needed
                |      <- _patch_lightrag_server_ui effective mutation
                |
                +--> mount /twin or /twin-api
                |      <- _mount_twin_server hook here
                |
                +--> return app
         |
         +--> uvicorn.run(app=app)
                |
                +--> lifespan startup
                       rag.initialize_storages()
                       rag.check_and_migrate_data()
                       <- too late for class/registry/server factory patches
```

## Bottom line

For LightRAG `1.4.9.11`, the reliable extension point is still "patch before instance construction", but the server/UI extension point is specifically "wrap `lightrag.api.lightrag_server.create_app` before `main()` or `get_application()` calls it". The only strict ordering inversion to avoid is importing `lightrag_server` before `_patch_version_string()`, because `core_version` is copied at import time.
