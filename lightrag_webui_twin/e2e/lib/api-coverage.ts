/**
 * Shared adversarial API-coverage protocol, run against the REAL backend.
 *
 * Two specs consume this:
 *   - api-coverage-real.spec.ts          → authenticates with the static
 *     LIGHTRAG_API_KEY (the infra key).
 *   - api-coverage-generated-key.spec.ts → mints a key through
 *     POST /twin/api/settings/api-keys (the Settings → API keys flow) and
 *     probes the whole surface with THAT non-admin operator key.
 *
 * Surface = every /twin/api route the live OpenAPI declares (discovered at
 * runtime, with no committed catalog) PLUS the native shim routes (an
 * explicit, fixed contract: they share root paths with LightRAG natives, so
 * OpenAPI discovery would collide).
 *
 * Per route: never 5xx on hostile input + no driver-error leak, wrong
 * method 4xx, missing-bearer 401/403 on non-public routes (when auth is
 * enforced), malformed JSON 422, unknown id 4xx. LLM-backed query routes
 * are probed only to the pre-LLM boundary (empty body → 422) because the
 * e2e CI has no model credentials.
 */

import { expect, type APIRequestContext, type TestType } from '@playwright/test';

export const HTTP_METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'] as const;
export type Method = (typeof HTTP_METHODS)[number];

const INJECTION = "x`) MATCH (n) DETACH DELETE n; //";
const NONEXISTENT = '__twin_e2e_nonexistent_0000__';
// A genuine driver/query error leaking to the client — error CLASS names,
// Neo/Memgraph error codes, or a Python traceback. Deliberately NOT the bare
// words "neo4j"/"memgraph"/"cypher": those appear in legitimate payloads such
// as the composite version string `v1.4.9.11+memgraph-1.0.0` that /health and
// /auth-status return (register() patches __version__), which must not trip it.
const DRIVER_LEAK =
  /Neo\.[A-Za-z.]+Error|Memgraph\.[A-Za-z.]+Error|neo4j\.exceptions|CypherSyntaxError|ServiceUnavailable|Failed to establish connection|Traceback \(most recent/i;

// Reachable without an auth backend by design.
const PUBLIC_TWIN_PATHS = new Set([
  '/twin/api/quota',
  '/twin/api/health',
  '/twin/api/openapi',
]);

export interface CoverageRoute {
  method: Method;
  path: string;
  /** All methods declared for this exact path, used to select a truly wrong verb. */
  supportedMethods?: readonly Method[];
  hasBody: boolean;
  isPublic: boolean;
  isAdminOnly: boolean;
  /** A *valid* body reaches retrieval/generation (LLM) — probe pre-LLM only. */
  preLlmOnly: boolean;
}

// Generated ``twk_`` keys are operator credentials, not RBAC-bearing admin
// identities. Keep this capability map explicit: a newly admin-gated route
// will make the generated-key job fail until its expected security boundary is
// reviewed and recorded here.
const ADMIN_ONLY_TWIN_OPERATIONS = new Set([
  'POST /twin/api/admin/portability/exports',
  'GET /twin/api/admin/portability/exports/{job_id}',
  'POST /twin/api/admin/portability/imports',
  'GET /twin/api/admin/portability/imports/{job_id}',
  'POST /twin/api/admin/portability/imports/{job_id}/approve',
  'POST /twin/api/admin/portability/imports/{job_id}/apply',
  'POST /twin/api/admin/portability/imports/{job_id}/validate',
  'POST /twin/api/admin/portability/imports/{job_id}/cancel',
  'GET /twin/api/settings/api-keys',
  'POST /twin/api/settings/api-keys',
  'DELETE /twin/api/settings/api-keys/{key_id}',
  // Vision curation knobs: GET is any-authenticated, PUT is admin-gated
  // (server/vision_settings_routes.py, feat/vision-settings-runtime).
  'PUT /twin/api/settings/vision',
  'GET /twin/api/documents/{doc_id}/folders',
  'POST /twin/api/documents/{doc_id}/folders',
  'DELETE /twin/api/documents/{doc_id}/folders/{folder_id}',
  'POST /twin/api/folders',
  'PATCH /twin/api/folders/{folder_id}',
  'DELETE /twin/api/folders/{folder_id}',
  'PATCH /twin/api/graph/entities/{entity_id}',
  'POST /twin/api/graph/entities',
  'DELETE /twin/api/graph/entities/{entity_id}',
  'POST /twin/api/graph/relations',
  'DELETE /twin/api/graph/relations/{rel_id}',
  'PATCH /twin/api/graph/relations/{rel_id}',
  'POST /twin/api/tags/categories/_import',
  'POST /twin/api/documents/_bulk-retag',
  // Audit 2026-08-06 remediation (R-03a/R-03b): destructive document
  // mutations and the client-side audit-write route are admin-gated.
  'POST /twin/api/documents/bulk-delete',
  'POST /twin/api/documents/{doc_id}/approve',
  'POST /twin/api/documents/{doc_id}/reject',
  // Document provenance is readable by operators, but every mutation is an
  // audited administrative action (server/webui/routes_source_links.py).
  'POST /twin/api/documents/{doc_id}/source-links',
  'PATCH /twin/api/documents/{doc_id}/source-links/{link_id}',
  'DELETE /twin/api/documents/{doc_id}/source-links/{link_id}',
  'POST /twin/api/documents/uploads/activity',
  'POST /twin/api/tags',
  'POST /twin/api/tags/{name}/suggest-edit',
  'POST /twin/api/tags/{name}/approve',
  'POST /twin/api/tags/{name}/reject',
  'PATCH /twin/api/tags/{name}',
  'POST /twin/api/tags/{name}/deprecate',
  'POST /twin/api/tags/{name}/reactivate',
  'POST /twin/api/tags/{name}/synonyms',
  'DELETE /twin/api/tags/{name}',
  // Procedure approval workflow (server/procedure_routes.py, PR 2):
  // the LIST is any-authenticated (folder-bound summaries); detail,
  // decisions and store recovery are admin-gated.
  'GET /twin/api/procedures/{bundle_id}',
  'POST /twin/api/procedures/{bundle_id}/approve',
  'POST /twin/api/procedures/{bundle_id}/reject',
  'POST /twin/api/procedures/{bundle_id}/retry',
  'POST /twin/api/procedures/{bundle_id}/reroute-standard',
  'GET /twin/api/procedures/store/health',
  'POST /twin/api/procedures/store/recover',
]);

// Native shim routes: a fixed contract (server/native_shims.py). Public set
// per the 2026-06-10 C1 audit: /auth-status, /login, /logout, /health.
// /openapi shim is protected; the FastAPI /openapi.json used for discovery
// is the public default and is covered via the /twin/api surface separately.
export const SHIM_ROUTES: CoverageRoute[] = [
  { method: 'GET', path: '/auth-status', hasBody: false, isPublic: true, isAdminOnly: false, preLlmOnly: false },
  { method: 'POST', path: '/login', hasBody: true, isPublic: true, isAdminOnly: false, preLlmOnly: false },
  { method: 'POST', path: '/logout', hasBody: false, isPublic: true, isAdminOnly: false, preLlmOnly: false },
  { method: 'GET', path: '/health', hasBody: false, isPublic: true, isAdminOnly: false, preLlmOnly: false },
  { method: 'GET', path: '/documents', hasBody: false, isPublic: false, isAdminOnly: false, preLlmOnly: false },
  { method: 'GET', path: '/documents/{doc_id}/chunks', hasBody: false, isPublic: false, isAdminOnly: false, preLlmOnly: false },
  { method: 'POST', path: '/documents/{doc_id}/scan', hasBody: false, isPublic: false, isAdminOnly: false, preLlmOnly: false },
  { method: 'DELETE', path: '/documents/{doc_id}', hasBody: false, isPublic: false, isAdminOnly: true, preLlmOnly: false },
  { method: 'GET', path: '/pipeline_status', hasBody: false, isPublic: false, isAdminOnly: false, preLlmOnly: false },
  { method: 'GET', path: '/openapi', hasBody: false, isPublic: false, isAdminOnly: false, preLlmOnly: false },
];

export interface CoverageConfig {
  /** Bearer used for *authenticated* requests (static key or generated key). */
  authToken: string;
  /** Whether the backend enforces auth (assert 401/403 on missing bearer). */
  expectAuth: boolean;
  defaultFolder: string;
  credentialTier: 'infrastructure-root' | 'operator';
}

function baseHeaders(cfg: CoverageConfig, withAuth: boolean): Record<string, string> {
  return {
    Accept: 'application/json',
    'X-Twin-Folder': cfg.defaultFolder,
    ...(withAuth && cfg.authToken ? { Authorization: `Bearer ${cfg.authToken}` } : {}),
  };
}

function concretePath(path: string, value = NONEXISTENT): string {
  return path.replace(/{[^}]+}/g, encodeURIComponent(value));
}

function injectionBody(): Record<string, unknown> {
  return {
    name: INJECTION, id: INJECTION, label: INJECTION, tag: INJECTION,
    rel_type: INJECTION, type: INJECTION, query: INJECTION, folder: INJECTION,
  };
}

/** Refuse to run a mutating adversarial battery against an unmarked target. */
export function requireEphemeralBackendTarget(
  backendUrl: string,
  ephemeralMarker: string | undefined,
): void {
  if (backendUrl && ephemeralMarker !== 'true') {
    throw new Error(
      'REAL_BACKEND_URL is set, but REAL_E2E_EPHEMERAL=true is required ' +
      'for the mutating adversarial route-reachability battery.',
    );
  }
}

/** The battery promises a client rejection, not merely an absence of 5xx. */
export function isClientErrorStatus(status: number): boolean {
  return status >= 400 && status < 500;
}

/** Pick a verb that is not declared for the exact path. */
export function selectWrongMethod(route: CoverageRoute): Method {
  const supported = new Set(route.supportedMethods ?? [route.method]);
  const wrong = HTTP_METHODS.find((method) => !supported.has(method));
  if (!wrong) {
    throw new Error(`No unsupported HTTP method available for ${route.path}`);
  }
  return wrong;
}

function expectClientError(status: number, message: string): void {
  expect(isClientErrorStatus(status), `${message}: received ${status}`).toBe(true);
}

/** Fetch the live OpenAPI schema, trying the conventional locations. */
export async function fetchOpenApi(
  ctx: APIRequestContext,
  cfg: CoverageConfig,
): Promise<Record<string, unknown>> {
  for (const p of ['/openapi.json', '/twin/api/openapi', '/openapi']) {
    const res = await ctx.get(p, { headers: baseHeaders(cfg, true) });
    if (res.ok()) {
      const json = (await res.json()) as Record<string, unknown>;
      if (json && typeof json === 'object' && 'paths' in json) return json;
    }
  }
  throw new Error('Could not fetch OpenAPI schema from the backend');
}

/** Every /twin/api route the live schema declares (drift-proof discovery). */
export function discoverTwinRoutes(schema: Record<string, unknown>): CoverageRoute[] {
  const paths = (schema.paths ?? {}) as Record<string, Record<string, unknown>>;
  const out: CoverageRoute[] = [];
  for (const [path, methods] of Object.entries(paths)) {
    if (!path.startsWith('/twin/api')) continue;
    const supportedMethods = Object.keys(methods)
      .map((method) => method.toUpperCase() as Method)
      .filter((method): method is Method => HTTP_METHODS.includes(method));
    for (const [m, op] of Object.entries(methods)) {
      const M = m.toUpperCase() as Method;
      if (!HTTP_METHODS.includes(M)) continue;
      out.push({
        method: M,
        path,
        supportedMethods,
        hasBody: !!(op && typeof op === 'object' && 'requestBody' in op),
        isPublic: PUBLIC_TWIN_PATHS.has(path),
        isAdminOnly: ADMIN_ONLY_TWIN_OPERATIONS.has(`${M} ${path}`),
        preLlmOnly: path.startsWith('/twin/api/query'),
      });
    }
  }
  return out;
}

/** Run the adversarial reachability battery for one route. Call inside a test.step. */
export async function coverRoute(
  ctx: APIRequestContext,
  route: CoverageRoute,
  cfg: CoverageConfig,
): Promise<void> {
  const { method, path, hasBody, isPublic, isAdminOnly, preLlmOnly } = route;
  const hasParam = /{[^}]+}/.test(path);
  const expectAdminForbidden =
    isAdminOnly && cfg.credentialTier === 'operator';

  // 0) Prove the authorization boundary with a syntactically clean request.
  //    Injection payloads may be rejected by the HTTP router before FastAPI
  //    reaches dependencies (for example an encoded slash can yield 404), so
  //    they cannot reliably prove whether the admin gate ran.
  if (expectAdminForbidden) {
    const res = await ctx.fetch(concretePath(path), {
      method,
      headers: baseHeaders(cfg, true),
      ...(hasBody ? { data: {} } : {}),
    });
    expect(res.status(), `${method} ${path} accepted a non-admin key`).toBe(403);
  }

  // 1) Hostile input must never 5xx or leak the driver. For body operations a
  //    JSON array is deliberately used against the object/multipart contracts:
  //    it cannot accidentally satisfy a create/update model and mutate state.
  //    Parametrised routes use a hostile identifier. Those guaranteed-invalid
  //    requests must be rejected with 4xx; static bodyless routes are only
  //    reachability probes and may legitimately succeed.
  {
    const data = hasBody ? [preLlmOnly ? {} : injectionBody()] : undefined;
    const res = await ctx.fetch(concretePath(path, INJECTION), {
      method,
      headers: {
        ...baseHeaders(cfg, !isPublic),
        ...(hasBody ? { 'Content-Type': 'application/json' } : {}),
      },
      ...(data === undefined ? {} : { data }),
    });
    const text = await res.text();
    expect(res.status(), `${method} ${path} 5xx on hostile input: ${text}`).toBeLessThan(500);
    expect(text, `${method} ${path} leaks driver error`).not.toMatch(DRIVER_LEAK);
    if (hasBody || hasParam) {
      expectClientError(
        res.status(),
        `${method} ${path} accepted guaranteed-invalid hostile input`,
      );
    }
  }

  // 2) A method absent from this path's OpenAPI operations must return 4xx.
  {
    const wrong = selectWrongMethod(route);
    const res = await ctx.fetch(concretePath(path), {
      method: wrong,
      headers: baseHeaders(cfg, !isPublic),
    });
    expectClientError(res.status(), `${wrong} ${path} wrong method was not rejected`);
  }

  // 3) Non-public route must reject a missing bearer (when auth is enforced).
  if (!isPublic && cfg.expectAuth) {
    const res = await ctx.fetch(concretePath(path), {
      method,
      headers: baseHeaders(cfg, false),
      ...(hasBody ? { data: {} } : {}),
    });
    expect([401, 403], `${method} ${path} did not reject anonymous`).toContain(res.status());
  }

  // 4) Malformed JSON body → 422/400, never 5xx. An admin dependency may
  //    reject an operator before body parsing; step 0 already proved that gate.
  if (hasBody) {
    const res = await ctx.fetch(concretePath(path), {
      method,
      headers: { ...baseHeaders(cfg, !isPublic), 'Content-Type': 'application/json' },
      data: '{ not valid json',
    });
    expect(res.status(), `${method} ${path} 5xx on bad JSON`).toBeLessThan(500);
    const expectedStatuses = expectAdminForbidden ? [400, 403, 422] : [400, 422];
    expect(expectedStatuses, `${method} ${path} bad JSON not rejected`).toContain(res.status());
  }

  // 5) Unknown id on a parametrised route must return 4xx. There are currently
  //    no idempotent-success exceptions in the backend contract; add one only
  //    against an explicitly documented operation if that contract changes.
  if (hasParam) {
    const res = await ctx.fetch(concretePath(path, NONEXISTENT), {
      method,
      headers: baseHeaders(cfg, !isPublic),
      ...(hasBody ? { data: {} } : {}),
    });
    expectClientError(res.status(), `${method} ${path} unknown id was not rejected`);
  }
}

/**
 * Discover and adversarially probe the route surface (/twin/api + shims).
 * Returns the list of probed "METHOD path" labels.
 */
export async function coverApiSurface(
  test: TestType<object, object>,
  ctx: APIRequestContext,
  cfg: CoverageConfig,
): Promise<string[]> {
  const schema = await fetchOpenApi(ctx, cfg);
  const routes = [...discoverTwinRoutes(schema), ...SHIM_ROUTES];

  const discovered = new Set(routes.map(({ method, path }) => `${method} ${path}`));
  for (const operation of ADMIN_ONLY_TWIN_OPERATIONS) {
    expect(
      discovered.has(operation),
      `admin operation missing from live surface: ${operation}`,
    ).toBe(true);
  }

  // Sanity floor: a broken/empty surface must fail loudly, not report a
  // vacuous reachability result. (45 /twin/api + 10 shims on the BNP target.)
  expect(routes.length, 'too few routes — surface looks broken').toBeGreaterThan(50);

  const probed: string[] = [];
  for (const route of routes) {
    await test.step(`${route.method} ${route.path}`, async () => {
      await coverRoute(ctx, route, cfg);
      probed.push(`${route.method} ${route.path}`);
    });
  }
  expect(probed.length, 'not every discovered route operation was probed').toBe(routes.length);
  return probed;
}
