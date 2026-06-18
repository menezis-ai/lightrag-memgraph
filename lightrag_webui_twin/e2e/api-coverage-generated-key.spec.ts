/**
 * 100% adversarial coverage of the API surface, authenticated with a key
 * MINTED THROUGH THE INTERFACE — POST /twin/api/settings/api-keys, the exact
 * endpoint the WebUI "Create API key" button hits.
 *
 * This proves the full lifecycle: a key an operator generates in
 * Settings → API keys is a real, accepted credential across the entire API
 * (require_auth validates it via api_key_store.validate_bearer), not a
 * display-only artefact. If a generated key did NOT authenticate, the
 * sanity probe below would fail loudly.
 *
 * Designed to run on its OWN CI runner (webui-e2e-keygen), separate from the
 * static-key job, against the same real Memgraph + LightRAG backend. The
 * static infra key (REAL_BACKEND_AUTH_TOKEN) is used ONLY to mint the key;
 * every coverage request then uses the generated key.
 */

import { expect, request as playwrightRequest, test } from '@playwright/test';

import { coverApiSurface, type CoverageConfig } from './lib/api-coverage';

const backendUrl = process.env.REAL_BACKEND_URL?.replace(/\/$/, '') ?? '';
const adminToken = process.env.REAL_BACKEND_AUTH_TOKEN ?? process.env.VITE_AUTH_TOKEN ?? '';
const expectAuth = process.env.REAL_E2E_EXPECT_AUTH === 'true';
const defaultFolder = process.env.REAL_BACKEND_FOLDER ?? 'default';

test.describe('Twin API — 100% coverage with a Settings-generated key', () => {
  test.skip(!backendUrl, 'Set REAL_BACKEND_URL to run generated-key coverage.');

  let ctx: Awaited<ReturnType<typeof playwrightRequest.newContext>>;
  let generatedKey = '';

  test.beforeAll(async () => {
    ctx = await playwrightRequest.newContext({ baseURL: backendUrl });
    // Mint a key through the same endpoint the WebUI uses. Authenticated with
    // the static admin key (palier-1 require_admin_user accepts any
    // authenticated identity when the IdP is dormant).
    const res = await ctx.post('/twin/api/settings/api-keys', {
      headers: {
        Accept: 'application/json',
        'X-Twin-Folder': defaultFolder,
        Authorization: `Bearer ${adminToken}`,
        'Content-Type': 'application/json',
      },
      data: { name: `e2e-coverage-key-${process.pid}` },
    });
    expect(res.status(), `mint failed: ${await res.text()}`).toBe(201);
    generatedKey = ((await res.json()) as { full_value?: string }).full_value ?? '';
    expect(generatedKey, 'POST returned no full_value').toBeTruthy();
  });

  test.afterAll(async () => {
    await ctx?.dispose();
  });

  test('a Settings-generated key authenticates the entire surface', async () => {
    // Sanity: prove the generated key (not the static key) is what authenticates.
    const probe = await ctx.get('/twin/api/folders', {
      headers: {
        Accept: 'application/json',
        'X-Twin-Folder': defaultFolder,
        Authorization: `Bearer ${generatedKey}`,
      },
    });
    expect(probe.status(), 'generated key did not authenticate').toBe(200);

    const cfg: CoverageConfig = { authToken: generatedKey, expectAuth, defaultFolder };
    const covered = await coverApiSurface(test, ctx, cfg);
    test.info().annotations.push({
      type: 'coverage',
      description: `${covered.length} routes adversarially covered with a Settings-generated key`,
    });
  });
});
