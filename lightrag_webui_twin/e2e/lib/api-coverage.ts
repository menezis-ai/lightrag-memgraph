/**
 * Shared adversarial API-coverage protocol, run against the REAL backend.
 *
 * Two specs consume this:
 *   - api-coverage-real.spec.ts          → authenticates with the static
 *     LIGHTRAG_API_KEY (the infra key).
 *   - api-coverage-generated-key.spec.ts → mints a key through
 *     POST /twin/api/settings/api-keys (the Settings → API keys flow) and
 *     authenticates the whole surface with THAT key.
 *
 * Surface = every /twin/api route the live OpenAPI declares (discovered at
 * runtime — 100% by construction, no committed catalog) PLUS the native
 * shim routes (an explicit, fixed contract: they share root paths with
 * LightRAG natives, so OpenAPI discovery would collide).
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
  hasBody: boolean;
  isPublic: boolean;
  /** A *valid* body reaches retrieval/generation (LLM) — probe pre-LLM only. */
  preLlmOnly: boolean;
}

// Native shim routes: a fixed contract (server/native_shims.py). Public set
// per the 2026-06-10 C1 audit: /auth-status, /login, /logout, /health.
// /openapi shim is protected; the FastAPI /openapi.json used for discovery
// is the public default and is covered via the /twin/api surface separately.
export const SHIM_ROUTES: CoverageRoute[] = [
  { method: 'GET', path: '/auth-status', hasBody: false, isPublic: true, preLlmOnly: false },
  { method: 'POST', path: '/login', hasBody: true, isPublic: true, preLlmOnly: false },
  { method: 'POST', path: '/logout', hasBody: false, isPublic: true, preLlmOnly: false },
  { method: 'GET', path: '/health', hasBody: false, isPublic: true, preLlmOnly: false },
  { method: 'GET', path: '/documents', hasBody: false, isPublic: false, preLlmOnly: false },
  { method: 'GET', path: '/documents/{doc_id}/chunks', hasBody: false, isPublic: false, preLlmOnly: false },
  { method: 'POST', path: '/documents/{doc_id}/scan', hasBody: false, isPublic: false, preLlmOnly: false },
  { method: 'DELETE', path: '/documents/{doc_id}', hasBody: false, isPublic: false, preLlmOnly: false },
  { method: 'GET', path: '/pipeline_status', hasBody: false, isPublic: false, preLlmOnly: false },
  { method: 'GET', path: '/openapi', hasBody: false, isPublic: false, preLlmOnly: false },
];

export interface CoverageConfig {
  /** Bearer used for *authenticated* requests (static key or generated key). */
  authToken: string;
  /** Whether the backend enforces auth (assert 401/403 on missing bearer). */
  expectAuth: boolean;
  defaultFolder: string;
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
    for (const [m, op] of Object.entries(methods)) {
      const M = m.toUpperCase() as Method;
      if (!HTTP_METHODS.includes(M)) continue;
      out.push({
        method: M,
        path,
        hasBody: !!(op && typeof op === 'object' && 'requestBody' in op),
        isPublic: PUBLIC_TWIN_PATHS.has(path),
        preLlmOnly: path.startsWith('/twin/api/query'),
      });
    }
  }
  return out;
}

/** Run the adversarial battery for one route. Call inside a test.step. */
export async function coverRoute(
  ctx: APIRequestContext,
  route: CoverageRoute,
  cfg: CoverageConfig,
): Promise<void> {
  const { method, path, hasBody, isPublic, preLlmOnly } = route;
  const hasParam = /{[^}]+}/.test(path);

  // 1) Hostile input must never 5xx, never leak the driver. LLM routes get an
  //    empty body (→ 422) so a valid query never reaches an absent model.
  {
    const data = hasBody ? (preLlmOnly ? {} : injectionBody()) : undefined;
    const res = await ctx.fetch(concretePath(path, INJECTION), {
      method,
      headers: baseHeaders(cfg, !isPublic),
      ...(data === undefined ? {} : { data }),
    });
    const text = await res.text();
    expect(res.status(), `${method} ${path} 5xx on hostile input: ${text}`).toBeLessThan(500);
    expect(text, `${method} ${path} leaks driver error`).not.toMatch(DRIVER_LEAK);
  }

  // 2) Wrong method → 4xx, never 5xx.
  {
    const wrong = HTTP_METHODS.find((m) => m !== method)!;
    const res = await ctx.fetch(concretePath(path), {
      method: wrong,
      headers: baseHeaders(cfg, !isPublic),
    });
    expect(res.status(), `${wrong} ${path} not 4xx`).toBeLessThan(500);
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

  // 4) Malformed JSON body → 422/400, never 5xx.
  if (hasBody) {
    const res = await ctx.fetch(concretePath(path), {
      method,
      headers: { ...baseHeaders(cfg, !isPublic), 'Content-Type': 'application/json' },
      data: '{ not valid json',
    });
    expect(res.status(), `${method} ${path} 5xx on bad JSON`).toBeLessThan(500);
    expect([400, 422], `${method} ${path} bad JSON not rejected`).toContain(res.status());
  }

  // 5) Unknown id on a parametrised route → 4xx, never 5xx.
  if (hasParam) {
    const res = await ctx.fetch(concretePath(path, NONEXISTENT), {
      method,
      headers: baseHeaders(cfg, !isPublic),
      ...(hasBody ? { data: {} } : {}),
    });
    expect(res.status(), `${method} ${path} 5xx on unknown id`).toBeLessThan(500);
  }
}

/**
 * Discover + adversarially cover the entire surface (/twin/api + shims).
 * Returns the list of covered "METHOD path" labels.
 */
export async function coverApiSurface(
  test: TestType<object, object>,
  ctx: APIRequestContext,
  cfg: CoverageConfig,
): Promise<string[]> {
  const schema = await fetchOpenApi(ctx, cfg);
  const routes = [...discoverTwinRoutes(schema), ...SHIM_ROUTES];

  // Sanity floor: a broken/empty surface must fail loudly, not report a
  // vacuous "100%". (45 /twin/api + 10 shims on the BNP target.)
  expect(routes.length, 'too few routes — surface looks broken').toBeGreaterThan(50);

  const covered: string[] = [];
  for (const route of routes) {
    await test.step(`${route.method} ${route.path}`, async () => {
      await coverRoute(ctx, route, cfg);
      covered.push(`${route.method} ${route.path}`);
    });
  }
  expect(covered.length, 'not every route was exercised').toBe(routes.length);
  return covered;
}
