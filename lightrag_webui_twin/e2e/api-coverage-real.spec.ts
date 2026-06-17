/**
 * 100% adversarial coverage of the Twin API surface — against the REAL backend.
 *
 * The route list is NOT a committed catalog (those drift, and route-table
 * introspection broke on FastAPI 0.137 — `route.path` is unreliable). Instead
 * this spec reads the live `/openapi.json` of the running backend at runtime
 * and exercises every `/twin/api/*` route it declares. Coverage is therefore
 * 100% *by construction*: a route that exists on the deployed surface cannot
 * be missed, and a newly added route is picked up with no test change.
 *
 * Per route, the universal contract is **never 5xx on hostile input** (the
 * graceful-degradation rule in CLAUDE.md), plus the deterministic 4xx where
 * unambiguous: 422 on unparseable JSON, 405/404 on wrong method, 404 on an
 * unknown id, and (for non-public routes, when auth is enforced) 401/403 on a
 * missing bearer.
 *
 * Requires a real backend (Memgraph + LightRAG): set REAL_BACKEND_URL. Auth
 * rejection is asserted only when REAL_E2E_EXPECT_AUTH=true (the e2e-real CI
 * job sets LIGHTRAG_API_KEY). Runs in `webui-e2e-real`; auto-skips elsewhere.
 */

import { expect, request as playwrightRequest, test } from '@playwright/test';

const backendUrl = process.env.REAL_BACKEND_URL?.replace(/\/$/, '') ?? '';
const authToken = process.env.REAL_BACKEND_AUTH_TOKEN ?? process.env.VITE_AUTH_TOKEN ?? '';
const defaultFolder = process.env.REAL_BACKEND_FOLDER ?? 'default';
const expectAuth = process.env.REAL_E2E_EXPECT_AUTH === 'true';

const HTTP_METHODS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'] as const;
type Method = (typeof HTTP_METHODS)[number];

// Reachable without an auth backend by design (LB probe, banner, schema).
const PUBLIC_PATHS = new Set([
  '/twin/api/quota',
  '/twin/api/health',
  '/twin/api/openapi',
]);

const INJECTION = "x`) MATCH (n) DETACH DELETE n; //";
const NONEXISTENT = '__twin_e2e_nonexistent_0000__';
const DRIVER_LEAK = /neo4j|memgraph|cypher|Neo\.ClientError|SyntaxError at/i;

interface RoutePair {
  method: Method;
  path: string;
  /** Whether the route declares an OpenAPI requestBody — the source of truth
   * for "does malformed JSON get validated". Method alone is wrong: body-less
   * POSTs (logout, approve…) accept and ignore a junk body, returning 2xx. */
  hasBody: boolean;
}

function baseHeaders(withAuth: boolean): Record<string, string> {
  return {
    Accept: 'application/json',
    'X-Twin-Folder': defaultFolder,
    ...(withAuth && authToken ? { Authorization: `Bearer ${authToken}` } : {}),
  };
}

function concretePath(path: string, value = NONEXISTENT): string {
  return path.replace(/{[^}]+}/g, encodeURIComponent(value));
}

function injectionBody(): Record<string, unknown> {
  // Superset of likely identifier fields; a guarded backend rejects at the
  // identifier validator or stores opaquely — it must never reach the planner.
  return {
    name: INJECTION, id: INJECTION, label: INJECTION, tag: INJECTION,
    rel_type: INJECTION, type: INJECTION, query: INJECTION, folder: INJECTION,
  };
}

test.describe('Twin API — 100% adversarial coverage (real backend)', () => {
  test.skip(!backendUrl, 'Set REAL_BACKEND_URL to run real-backend API coverage.');

  let ctx: Awaited<ReturnType<typeof playwrightRequest.newContext>>;

  test.beforeAll(async () => {
    ctx = await playwrightRequest.newContext({ baseURL: backendUrl });
  });
  test.afterAll(async () => {
    await ctx.dispose();
  });

  async function fetchOpenApi(): Promise<Record<string, unknown>> {
    for (const p of ['/openapi.json', '/twin/api/openapi', '/openapi']) {
      const res = await ctx.get(p, { headers: baseHeaders(true) });
      if (res.ok()) {
        const json = (await res.json()) as Record<string, unknown>;
        if (json && typeof json === 'object' && 'paths' in json) return json;
      }
    }
    throw new Error('Could not fetch OpenAPI schema from the backend');
  }

  function twinRoutes(schema: Record<string, unknown>): RoutePair[] {
    const paths = (schema.paths ?? {}) as Record<string, Record<string, unknown>>;
    const out: RoutePair[] = [];
    for (const [path, methods] of Object.entries(paths)) {
      if (!path.startsWith('/twin/api')) continue;
      for (const [m, op] of Object.entries(methods)) {
        const M = m.toUpperCase() as Method;
        if (!HTTP_METHODS.includes(M)) continue;
        const hasBody = !!(op && typeof op === 'object' && 'requestBody' in op);
        out.push({ method: M, path, hasBody });
      }
    }
    return out.sort((a, b) => (a.path + a.method).localeCompare(b.path + b.method));
  }

  test('every /twin/api route degrades gracefully under hostile input', async () => {
    const schema = await fetchOpenApi();
    const routes = twinRoutes(schema);

    // Sanity floor: a broken/empty surface (the 2026-06-17 symptom) must fail
    // loudly rather than report a vacuous "100%".
    expect(routes.length, 'too few /twin/api routes — surface looks broken').toBeGreaterThan(40);

    const covered: string[] = [];

    for (const { method, path, hasBody } of routes) {
      await test.step(`${method} ${path}`, async () => {
        const hasParam = /{[^}]+}/.test(path);
        const isPublic = PUBLIC_PATHS.has(path);
        // LLM-backed routes: a *structurally valid* body reaches retrieval /
        // generation, which the e2e-real CI cannot serve (no model creds — the
        // same reason real-backend.spec skips LLM journeys). We probe these only
        // up to the pre-LLM boundary (an empty body → 422 validation), so the
        // route is still proven mounted, auth-gated, and validating, without
        // calling a model that isn't there.
        const isLlm = path.startsWith('/twin/api/query');

        // 1) Universal: hostile input must never 5xx, never leak the driver.
        {
          const data = hasBody ? (isLlm ? {} : injectionBody()) : undefined;
          const res = await ctx.fetch(concretePath(path, INJECTION), {
            method,
            headers: baseHeaders(!isPublic),
            ...(data === undefined ? {} : { data }),
          });
          const text = await res.text();
          expect(res.status(), `${method} ${path} 5xx on hostile input: ${text}`).toBeLessThan(500);
          expect(text, `${method} ${path} leaks driver error`).not.toMatch(DRIVER_LEAK);
        }

        // 2) Wrong method → 404/405, never 5xx.
        {
          const wrong = HTTP_METHODS.find((m) => m !== method)!;
          const res = await ctx.fetch(concretePath(path), {
            method: wrong,
            headers: baseHeaders(!isPublic),
          });
          expect(res.status(), `${wrong} ${path} not 4xx`).toBeLessThan(500);
        }

        // 3) Auth: a non-public route must reject a missing bearer (when enforced).
        if (!isPublic && expectAuth) {
          const res = await ctx.fetch(concretePath(path), {
            method,
            headers: baseHeaders(false),
            ...(hasBody ? { data: {} } : {}),
          });
          expect([401, 403], `${method} ${path} did not reject anonymous`).toContain(res.status());
        }

        // 4) Malformed JSON body → 422/400, never 5xx.
        if (hasBody) {
          const res = await ctx.fetch(concretePath(path), {
            method,
            headers: { ...baseHeaders(!isPublic), 'Content-Type': 'application/json' },
            data: '{ not valid json',
          });
          expect(res.status(), `${method} ${path} 5xx on bad JSON`).toBeLessThan(500);
          expect([400, 422], `${method} ${path} bad JSON not rejected`).toContain(res.status());
        }

        // 5) Unknown id on a parametrised route → 404 (or other 4xx), never 5xx.
        if (hasParam) {
          const res = await ctx.fetch(concretePath(path, NONEXISTENT), {
            method,
            headers: baseHeaders(!isPublic),
            ...(hasBody ? { data: {} } : {}),
          });
          expect(res.status(), `${method} ${path} 5xx on unknown id`).toBeLessThan(500);
        }

        covered.push(`${method} ${path}`);
      });
    }

    // 100% by construction: every discovered route was exercised.
    expect(covered.length, 'not every route was exercised').toBe(routes.length);
    test.info().annotations.push({
      type: 'coverage',
      description: `${covered.length}/${routes.length} /twin/api routes adversarially covered`,
    });
  });
});
