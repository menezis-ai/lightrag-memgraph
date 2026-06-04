import { expect, test, type Page } from '@playwright/test';

const backendUrl = process.env.REAL_BACKEND_URL?.replace(/\/$/, '') ?? '';
const authToken = process.env.REAL_BACKEND_AUTH_TOKEN ?? process.env.VITE_AUTH_TOKEN;
const defaultSpace = process.env.REAL_BACKEND_SPACE ?? 'default';

function authHeaders(): Record<string, string> {
  return {
    Accept: 'application/json',
    'X-Twin-Space': defaultSpace,
    'X-Twin-Workspace': defaultSpace,
    ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
  };
}

async function fetchFromBrowser<T>(
  page: Page,
  path: string,
  init: { method?: string; body?: unknown } = {},
): Promise<{ status: number; ok: boolean; body: T; contentType: string | null }> {
  return page.evaluate(
    async ({ base, path: requestPath, headers, method, body }) => {
      const res = await fetch(`${base}${requestPath}`, {
        method,
        headers: {
          ...headers,
          ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        credentials: 'include',
      });
      const text = await res.text();
      let parsed: unknown = text;
      try {
        parsed = text ? JSON.parse(text) : null;
      } catch {
        // Keep raw text for clearer assertion output.
      }
      return {
        status: res.status,
        ok: res.ok,
        body: parsed,
        contentType: res.headers.get('content-type'),
      };
    },
    {
      base: backendUrl,
      path,
      headers: authHeaders(),
      method: init.method ?? 'GET',
      body: init.body,
    },
  ) as Promise<{ status: number; ok: boolean; body: T; contentType: string | null }>;
}

test.describe('real backend smoke', () => {
  test.skip(!backendUrl, 'Set REAL_BACKEND_URL to run real-backend e2e smoke.');

  test.beforeEach(async ({ page }) => {
    await page.addInitScript(
      ({ apiBaseUrl, lightragBaseUrl, space }) => {
        window.localStorage.setItem(
          'twin.onboarding.v1',
          JSON.stringify({ step: 'completion', dismissed: true, tasks: [] }),
        );
        window.localStorage.removeItem('twin-rag.threads.v2');
        window.__twinE2eRuntimeConfig = {
          apiBaseUrl,
          lightragBaseUrl,
          idpLogoutUrl: 'https://idp.example.invalid/logout',
          defaultSpaceId: space,
          maxSpaces: 5,
          spaces: [
            {
              id: space,
              label: space,
              kind: 'primary',
              description: 'Real backend e2e smoke space',
            },
          ],
          debugUser: {
            sso_subject: 'real-e2e@local',
            email: 'real-e2e@local',
            name: 'Real E2E',
            palier: {
              level: 3,
              label: 'Steward',
              scopes: ['twin:read', 'twin:write', 'twin:approve'],
            },
            workspaces: [space],
            idp: 'e2e',
            idp_realm: 'real-backend',
            sub: 'real-e2e',
            session_expires: '2099-12-31T23:59:00Z',
            gateway_scopes: ['read:documents', 'read:query', 'read:activity'],
          },
        };
      },
      {
        apiBaseUrl: `${backendUrl}/twin/api`,
        lightragBaseUrl: backendUrl,
        space: defaultSpace,
      },
    );
  });

  test('browser app boots with MSW disabled and reaches real read endpoints', async ({
    page,
  }) => {
    await page.goto('/');
    await expect(
      page.getByRole('button', { name: 'Documents', exact: true }),
    ).toBeVisible();

    const workerRegistrations = await page.evaluate(async () => {
      if (!('serviceWorker' in navigator)) {
        return [];
      }
      return (await navigator.serviceWorker.getRegistrations()).map(
        (registration) => registration.active?.scriptURL ?? '',
      );
    });
    expect(workerRegistrations.filter((url) => url.includes('mockServiceWorker'))).toEqual(
      [],
    );

    const health = await fetchFromBrowser<{ status: string }>(page, '/health');
    expect(health.ok, JSON.stringify(health.body)).toBe(true);
    expect(health.body).toHaveProperty('status');

    const documents = await fetchFromBrowser<{ items: unknown[]; total: number }>(
      page,
      '/documents',
    );
    expect(documents.ok, JSON.stringify(documents.body)).toBe(true);
    expect(Array.isArray(documents.body.items)).toBe(true);
    expect(typeof documents.body.total).toBe('number');

    const spaces = await fetchFromBrowser<unknown[]>(page, '/twin/api/spaces');
    expect(spaces.ok, JSON.stringify(spaces.body)).toBe(true);
    expect(Array.isArray(spaces.body)).toBe(true);

    const graphEntities = await fetchFromBrowser<unknown[]>(
      page,
      '/twin/api/graph/entities',
    );
    expect(graphEntities.ok, JSON.stringify(graphEntities.body)).toBe(true);
    expect(Array.isArray(graphEntities.body)).toBe(true);
  });

  test('optional retrieval query reaches the real backend', async ({ page }) => {
    test.skip(
      process.env.REAL_E2E_QUERY !== 'true',
      'Set REAL_E2E_QUERY=true to run the potentially LLM-backed /query smoke.',
    );

    await page.goto('/');
    const result = await fetchFromBrowser<{ response: string }>(page, '/query', {
      method: 'POST',
      body: { query: 'health check', mode: 'hybrid', top_k: 1 },
    });
    expect(result.ok, JSON.stringify(result.body)).toBe(true);
    expect(typeof result.body.response).toBe('string');
  });
});
