# WebUI Wiring Plan — Couches 2 & 3

> Companion to `lightrag_webui_twin/` (Vite + React 19 + TypeScript + Bun)
> and `src/twindb_lightrag_memgraph/` (Python storage backends + LightRAG
> registration). Documents what's already done (Couche 2 — Classification)
> and what remains (Couche 3 — LightRAG real-backend wiring), so anyone
> picking up this work has the full contract without reading the entire
> session log.

## TL;DR — state of play, 2026-05-31

| Couche | Scope | Status | Reference |
|---|---|---|---|
| **0** | Decisions + branch hygiene + visual snapshot | ✅ Done | session log |
| **1** | Visual port from `~/Downloads/prototype/` to React/TS | ✅ Done | PR [#158](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/pulls/158), [#159](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/pulls/159) |
| **2** | BNP classification (TS types + ClassPill + DocDetailPanel gating + Python extractor + pre-insert hook) | ✅ Done | PR [#157](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/pulls/157) (Python) + PR [#158](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/pulls/158) (TS UI) |
| **3** | LightRAG wiring real (server FastAPI sub-app, JWT, real fetch, X-Twin-Workspace header, drop MSW in prod) | ⏸ **TODO** | this document |

The standalone OVH demo at https://maquette.sigilum.fr/ uses the React
port with **MSW client-side** (everything mocked in the browser, no
backend). Couche 3 replaces MSW with the real LightRAG + Twin overlay.

---

## Couche 2 — Classification BNP (as-built)

The compliance layer that reads the Microsoft Information Protection
(MIP) sensitivity label on Office documents at ingestion time, persists
it on the DocStatus, gates retrieval-time UI, and refuses anything above
the configured ceiling.

### Files delivered

| File | Role |
|---|---|
| `src/twindb_lightrag_memgraph/classification.py` | Extractor module — OOXML (stdlib), legacy OLE (`olefile`), PDF (`pikepdf`). Returns `ClassificationResult` matching the TS shape. |
| `src/twindb_lightrag_memgraph/_classification_hook.py` | Pre-insert hook + `install_classification_hook(label_map_path, ceiling, audit_emit)` factory + `ClassificationRejection` exception (fail-closed on UNKNOWN classes). |
| `scripts/extract_msip.py` | CLI wrapper — `python scripts/extract_msip.py FILE [--label-map labels.json] [--json] [--exit-code-on-above C2]`. |
| `tests/test_classification.py` | 34 offline tests (synthetic OOXML built in-memory, no real fixture files). |
| `tests/test_classification_hook.py` | 8 offline tests covering gating + audit emission. |
| `lightrag_webui_twin/src/types/classification.ts` | TS mirror of `ClassificationResult.as_dict()`. `ClassId`, `ClassificationValue`, helpers (`isStructured`, `getClassId`, `getClassName`, `isAbove`, `isAboveInternal`). |
| `lightrag_webui_twin/src/components/ClassPill.tsx` | UI badge — 5 variants (`class-c1`/`c2`/`c3`/`c4`/`unknown`), silent on legacy string shape. |
| `lightrag_webui_twin/src/types/classification.test.ts` + `ClassPill.test.tsx` | 29 frontend tests. |
| `README.md` (top-level) | New "Classification (Microsoft Information Protection)" section + env vars `TWIN_MIP_LABEL_MAP` / `TWIN_MIP_MAX_CLASSIFICATION`. |

### Wiring contract

The Twin overlay endpoint `GET /twin/api/documents/{id}/metadata` MUST
return the structured classification when present:

```json
{
  "tags": ["rman", "oracle"],
  "workspace": "cib",
  "review": { ... },
  "classification": {
    "class_id": "C2",
    "class_name": "C2 Confidentiel",
    "label_guid": "22222222-2222-2222-2222-222222222222",
    "raw_name": "C2 Confidentiel",
    "set_date": "2026-03-12T14:22:01Z",
    "method": "Standard",
    "source_format": "ooxml",
    "reason": null,
    "meta": { "Enabled": "true", "SiteId": "{...}" }
  }
}
```

The WebUI reads `doc.metadata.classification` directly. The
`ClassPill` component renders only when the value is structured (the
legacy string shape `"internal"` / `"restricted"` is invisible — it
predates the hook). The `DocDetailPanel` chunks tab + "View raw"
notice gate on `isAboveInternal(cls)` = "above C2 on the BNP ladder".

### Activation matrix

| State | Behavior |
|---|---|
| No `TWIN_MIP_LABEL_MAP` env var | Empty map → every detected label → `class_id: "UNKNOWN"` → fail-closed reject by hook |
| `TWIN_MIP_LABEL_MAP=/etc/twin/labels.json`, GUID in map | `class_id` resolves to `"C1".."C4"`, allow / reject per `is_above(class_id, ceiling)` |
| `TWIN_MIP_MAX_CLASSIFICATION=C3` | Hook allows C1/C2/C3, rejects C4 + UNKNOWN |
| Hook not installed (default after `register()`) | No classification on `metadata.classification` — UI shows no pill |

### Activation steps (for a real BNP deploy)

1. **Acquire the tenant label map** from Louis HORVAT (Compliance Center
   → Sensitivity Labels → Export, JSON or CSV). Convert to:
   ```json
   {
     "<bnp-c1-guid>": {"id": "C1", "name": "C1 Public"},
     "<bnp-c2-guid>": {"id": "C2", "name": "C2 Confidentiel"},
     "<bnp-c3-guid>": {"id": "C3", "name": "C3 Strictement Confidentiel"},
     "<bnp-c4-guid>": {"id": "C4", "name": "C4 Secret"}
   }
   ```
   Save as `/etc/twin/labels.json` (or wherever your secret store mounts
   it). The file is **not** secret — but the GUIDs identify the tenant,
   so keep it out of public git.

2. **Set env vars** on the LightRAG host:
   ```bash
   TWIN_MIP_LABEL_MAP=/etc/twin/labels.json
   TWIN_MIP_MAX_CLASSIFICATION=C2   # adjust per workspace
   ```

3. **Install the hook** (Couche 3 wiring — see below) in the FastAPI
   sub-app, calling `install_classification_hook()` once at startup
   with the audit-emit callback bound to the Twin activity store.

4. **Smoke test** with `python scripts/extract_msip.py path/to/real.docx
   --label-map /etc/twin/labels.json`. Should resolve `class_id` to
   one of C1/C2/C3/C4. If `UNKNOWN`, the GUID mapping is missing.

### What's deliberately NOT in Couche 2

- **No `register()` integration** — the hook is opt-in via explicit
  `install_classification_hook()` + manual call before `LightRAG.insert()`.
  Couche 3 wires it as part of the FastAPI sub-app boot.
- **No BNP tenant label map shipped** — needs to come from Louis HORVAT.
- **No real-document fixtures** in the test suite — tests synthesize
  minimal OOXML packages in-memory (zip + xml).

---

## Couche 3 — LightRAG wiring real (TODO)

Replaces MSW with the real LightRAG + Twin overlay. After Couche 3, the
React port runs as a sub-app mounted by `register()` on the LightRAG
FastAPI server, talks to the real `/documents` + `/twin/api/*` endpoints,
uses real JWT auth from Keycloak / BNP IdP, and ingests with the
classification hook active.

### Architecture target

```
┌─────────────────────────────────────────────────────────────┐
│  LightRAG FastAPI (the wheel: twindb-lightrag-memgraph)     │
│  ├─ Native: /documents, /documents/{id}/chunks, /health,    │
│  │          /query, /openapi, /pipeline_status              │
│  ├─ Mounted via register(replace_ui=True):                  │
│  │   /webui/  → serves React port dist/index.html           │
│  │             with __TWIN_CONFIG_JSON__ substitution       │
│  └─ Mounted via register(mount_server=True):                │
│      /twin/api/* → Twin overlay sub-app                     │
│        ├─ /workspaces, /notifications, /tags, /activity     │
│        ├─ /documents/{id}/metadata, /approve, /reject       │
│        ├─ /graph/entities, /graph/relations                 │
│        └─ /auth/logout                                      │
├─────────────────────────────────────────────────────────────┤
│  Pre-insert hook (Couche 2)                                 │
│  install_classification_hook(label_map, ceiling, audit_emit)│
│  → DocStatus.metadata.classification = ClassificationResult │
├─────────────────────────────────────────────────────────────┤
│  Memgraph (KV + Vector + DocStatus + Graph)                 │
└─────────────────────────────────────────────────────────────┘
```

### Files to create / extend

| File | Action | Why |
|---|---|---|
| `src/twindb_lightrag_memgraph/server/__init__.py` | **NEW** | FastAPI sub-app factory `build_twin_overlay(audit_store, classification_hook, ...)` |
| `src/twindb_lightrag_memgraph/server/routes_documents.py` | **NEW** | `/twin/api/documents/{id}/metadata`, `/approve`, `/reject`, `/bulk-delete` |
| `src/twindb_lightrag_memgraph/server/routes_tags.py` | **NEW** | `/twin/api/tags` + mutations (proxy to a Memgraph-backed tag store) |
| `src/twindb_lightrag_memgraph/server/routes_activity.py` | **NEW** | `/twin/api/activity` audit feed (read from `activity` kind in Memgraph) |
| `src/twindb_lightrag_memgraph/server/routes_workspaces.py` | **NEW** | `/twin/api/workspaces`, `/twin/api/notifications` |
| `src/twindb_lightrag_memgraph/server/routes_graph.py` | **NEW** | `/twin/api/graph/entities`, `/twin/api/graph/relations` |
| `src/twindb_lightrag_memgraph/server/routes_auth.py` | **NEW** | `/twin/api/auth/logout` (revoke server-side session + Set-Cookie clear) |
| `src/twindb_lightrag_memgraph/server/middleware_workspace.py` | **NEW** | Read `X-Twin-Workspace` header on every request, scope downstream queries |
| `src/twindb_lightrag_memgraph/server/middleware_jwt.py` | **NEW** | Decode Keycloak/IdP JWT cookie → set `request.state.user: AuthenticatedUser` |
| `src/twindb_lightrag_memgraph/server/serve_webui.py` | **NEW** | `GET /webui/` reads `dist/index.html`, substitutes `__TWIN_CONFIG_JSON__` with JSON built from env + JWT claims |
| `src/twindb_lightrag_memgraph/__init__.py` | **EDIT** | Add `replace_ui: bool`, `mount_server: bool`, `webui_dist_path: str` kwargs to `register()`; wire the sub-app + WebUI mount when set |
| `tests/test_server/` | **NEW DIRECTORY** | Integration tests against a real Memgraph (`@pytest.mark.integration`) covering each route |
| `lightrag_webui_twin/index.html` | **VERIFY** | Confirm the `__TWIN_CONFIG_JSON__` placeholder is still in place (already there per `useAuth.ts`) |
| `lightrag_webui_twin/src/api/client.ts` | **EDIT** | Read `apiBaseUrl` from `window.__twinConfig`; switch off MSW when not `VITE_FORCE_MSW=true` |
| `lightrag_webui_twin/src/api/resources.ts` | **VERIFY** | The 1:1 endpoint mapping should already match — confirm no path drift |
| `pyproject.toml` | **EDIT** | Move `olefile` and `pikepdf` from optional to a new extra `pip install twindb-lightrag-memgraph[classification]` |
| `README.md` | **EDIT** | New "Couche 3 — Real backend wiring" section linking to this doc |

### Concrete tasks (ordered)

#### 3.1 — FastAPI sub-app skeleton (4-6h)

- [ ] Create `src/twindb_lightrag_memgraph/server/` package with
      `__init__.py` exposing `build_twin_overlay(...)` factory returning a
      FastAPI `APIRouter` (NOT a full FastAPI app — we mount it under
      the LightRAG server).
- [ ] Implement `routes_documents.py`:
  - `GET /documents/{id}/metadata` → reads DocStatus from
    `MemgraphDocStatusStorage`, returns `{tags, workspace, review,
    classification}` (extract `classification` from
    `DocStatus.metadata.classification`).
  - `POST /documents/{id}/approve` body `{actor, edits?}` → updates
    `review.state = "approved"`, optionally applies `edits` (merge into
    DocStatus), emits activity event `kind="doc-approved"`.
  - `POST /documents/{id}/reject` body `{reason, actor}` → updates
    `review.state = "rejected"` + `review.justification = reason`,
    emits activity event.
  - `POST /documents/bulk-delete` body `{doc_ids, actor}` → deletes
    each + emits one activity per doc.
- [ ] Implement `routes_tags.py`, `routes_activity.py`,
      `routes_workspaces.py`, `routes_graph.py`,
      `routes_auth.py` following the same shape as the MSW handlers in
      `lightrag_webui_twin/src/mocks/handlers.ts` (path-by-path 1:1).

#### 3.2 — Tag + activity + notifications persistence (4h)

The MSW handlers fake these; the real backend needs a place to put
them. Options:
- **Simplest**: new Memgraph labels `Twin_Tag`, `Twin_Activity`,
  `Twin_Notification` (one node per row, JSON `data` property).
  Reuses the existing `_pool.get_driver()` pattern from KV/Vector.
- **Alternative**: a tiny SQLite alongside Memgraph for these.
  Lower latency for the audit feed but adds a moving part.

Go with Memgraph labels — keeps the "one DB, three pools" story
intact and the audit feed query is a simple `MATCH (n:Twin_Activity)
WHERE n.workspace = $ws RETURN n ORDER BY n.ts DESC LIMIT $limit`.

#### 3.3 — JWT middleware + workspace scoping (3h)

- [ ] `middleware_jwt.py`: decode the BNP IdP / Keycloak JWT from
      cookie (HttpOnly, SameSite=Lax). Validate signature against the
      IdP's JWKS. On valid token: set `request.state.user =
      AuthenticatedUser(...)` matching the TS type.
      Failure modes: missing cookie → 401, expired → 401 with
      `WWW-Authenticate: Bearer error="expired"`, signature mismatch
      → 401.
- [ ] `middleware_workspace.py`: read `X-Twin-Workspace` header (set
      by the React WorkspaceSwitcher), validate it's in
      `request.state.user.workspaces`, set `request.state.workspace`.
      Downstream routes read this to scope every query (Memgraph
      labels are `KV_{workspace}_{namespace}` etc — no leakage by
      construction, but the middleware is the assertion point).
- [ ] Wire both in `register()` so every `/twin/api/*` request hits
      them before the route handlers.

#### 3.4 — index.html substitution + WebUI mount (2h)

- [ ] `serve_webui.py`: read `lightrag_webui_twin/dist/index.html` once
      at startup, cache it as a template, substitute
      `__TWIN_CONFIG_JSON__` per-request with:
      ```python
      json.dumps({
        "apiBaseUrl": "/twin/api",
        "lightragBaseUrl": "/api",
        "idpLogoutUrl": os.environ["TWIN_IDP_LOGOUT_URL"],
        "debugUser": None,  # PROD: no debug user, real JWT decoded server-side
      })
      ```
      Return `HTMLResponse(substituted_html)`.
- [ ] Mount under `/webui/` via `app.mount("/webui", ...)` — also serve
      the `dist/assets/*` and `dist/mockServiceWorker.js` (latter only
      relevant for OVH standalone; PROD won't load it because
      `VITE_FORCE_MSW` is unset).

#### 3.5 — Classification hook integration (1h)

- [ ] In the FastAPI startup (`@app.on_event("startup")` or lifespan),
      call `install_classification_hook(label_map_path,
      ceiling=os.environ.get("TWIN_MIP_MAX_CLASSIFICATION", "C2"),
      audit_emit=emit_to_memgraph_activity)`.
- [ ] Patch LightRAG's `insert()` call site to run the hook BEFORE
      passing the file to LightRAG. On `ClassificationRejection`, mark
      the DocStatus `status="FAILED"` + `error_msg=str(exc)` instead of
      ingesting.

#### 3.6 — Frontend cutover (2h)

- [ ] `client.ts`: read `apiBaseUrl` from `window.__twinConfig`. The
      MSW gate in `main.tsx` already turns off MSW unless
      `VITE_FORCE_MSW=true`, so a PROD build automatically hits the
      real backend.
- [ ] Add `X-Twin-Workspace` header to every fetch: extend `apiFetch`
      to read the current workspace from a React context (the
      `WorkspaceSwitcher` already updates it).
- [ ] `useAuth.signout()` is already wired correctly (POST
      `/twin/api/auth/logout` → `queryClient.clear()` → redirect IdP).
      Confirm the IdP URL is read from `window.__twinConfig.idpLogoutUrl`.

#### 3.7 — Integration tests (3h)

- [ ] Create `tests/test_server/` package.
- [ ] `test_metadata_endpoint.py`: insert a doc with structured
      `metadata.classification`, GET `/twin/api/documents/{id}/metadata`,
      assert the classification is in the response.
- [ ] `test_classification_rejection.py`: ingest a synthetic .docx
      tagged C3 with `TWIN_MIP_MAX_CLASSIFICATION=C2`, assert the
      DocStatus is `FAILED` with the expected `error_msg`, and an
      activity event of kind `classification-rejected` exists.
- [ ] `test_workspace_scoping.py`: insert docs in workspace `cib` and
      `wm`, GET `/documents` with `X-Twin-Workspace: cib`, assert only
      `cib` docs returned.
- [ ] `test_jwt_middleware.py`: GET `/twin/api/workspaces` without
      cookie → 401; with a valid JWT for a user with `workspaces:
      ["cib", "wm"]` → 200 with both listed.
- [ ] Add the new tests to the `integration-tests` job in
      `.forgejo/workflows/ci.yml` (already includes a Memgraph service
      container).

#### 3.8 — Deployment (2h)

- [ ] Build wheel: `python -m build` produces `dist/twindb_lightrag_memgraph-1.2.0-py3-none-any.whl`.
- [ ] Build webui dist: `cd lightrag_webui_twin && bun run build`
      (NOT with `VITE_FORCE_MSW=true` — that's only for the standalone
      OVH demo).
- [ ] Ship the wheel + webui dist to the LightRAG host (BNP infra,
      OVH staging, etc.).
- [ ] On host: `pip install twindb_lightrag_memgraph-1.2.0-py3-none-any.whl[classification]`
      (extra includes `olefile` + `pikepdf` for non-OOXML formats).
- [ ] Set env vars: `MEMGRAPH_URI`, `TWIN_MIP_LABEL_MAP`,
      `TWIN_MIP_MAX_CLASSIFICATION`, `TWIN_IDP_LOGOUT_URL`,
      `TWIN_JWKS_URL`, etc.
- [ ] Start the LightRAG server with `register(replace_ui=True,
      mount_server=True, webui_dist_path="/opt/twin/webui")`.
- [ ] Smoke check `https://<host>/webui/` loads the React port +
      hits real `/twin/api/*` endpoints.

### Risks + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Keycloak JWKS endpoint unreachable from the LightRAG host | Medium | Cache the JWKS at startup with a TTL; fail closed (401) if cache stale + refresh fails. |
| Memgraph labels for tags/activity grow unbounded | Low (90d retention per Settings) | Implement the retention sweep job — see `docs/operations/install-runbook.md` §5 |
| `register()` flag explosion (replace_ui, mount_server, classify, ...) | High if not designed carefully | Group under a single `extensions=ExtensionConfig(...)` dataclass; keep `register()` signature small |
| Frontend MSW removal breaks the dev story | Low (MSW stays on in DEV by default) | The activation matrix in `main.tsx` is already documented — don't regress |
| BNP tenant label map mismatched with Compliance Center | Medium | Add a `/twin/api/classification/_self_check` debug endpoint that returns the loaded map + lets ops validate visually |
| Real fetch + TanStack Query cache thrash on workspace switch | Medium | On `setActiveWorkspace`, call `queryClient.removeQueries()` for all `['documents', '...']` keys — already in App.tsx skeleton, just needs confirmation |

### Sequencing recommendation

```
3.1 sub-app skeleton (4-6h) ─┬─→ 3.5 classification hook (1h) ─┐
                             │                                  │
3.2 persistence (4h) ────────┤                                  │
                             │                                  ├─→ 3.8 deploy (2h)
3.3 JWT + workspace (3h) ────┤                                  │
                             │                                  │
3.4 index.html mount (2h) ───┤                                  │
                             │                                  │
                             └─→ 3.6 frontend cutover (2h) ─────┤
                                                                │
                              3.7 integration tests (3h) ───────┘
```

Total estimate: **18-22h of focused work** for a single engineer who
knows the codebase. Realistic shipping window: a focused 3-day sprint
once the tenant label map is in hand from Louis HORVAT.

---

## References

- **Brief sprint coder (Couches 1+2)** : `docs/handoff/SPRINT-2026-05-30-coder-brief.md`
- **Pitch deck Fabrice 2026-06-01** : `docs/presentations/pitch-fabrice-2026-06-01.md`
- **Install runbook (production)** : `docs/operations/install-runbook.md`
- **Rapport recette Alberto** : `docs/audits/TwinRAG - Rapport de recette v2.md`
- **Design-agent prototype** : `~/Downloads/prototype/` (Tier-1 visual reference)
- **PR #157** : MSIP classification (Python module + hook + 42 tests)
- **PR #158** : React port from prototype + Couche 2 UI (320/320 tests)
- **PR #159** : 11 visual fixes from dev smoke
- **PR #160** : VITE_FORCE_MSW + Dockerfile.react + standalone OVH deploy
- **Memory** : `~/.claude/projects/-Users-julien-twindb-lightrag-memgraph/memory/`
  - `project_louis_eric_meeting_2026-05-28.md` (compliance doctrine)
  - `project_twin_myaccess_rights_model.md` (palier ↔ classification mapping)
  - `project_webui_fork.md` (history of the React port)
