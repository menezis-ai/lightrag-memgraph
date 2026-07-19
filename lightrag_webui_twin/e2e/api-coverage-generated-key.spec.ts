/**
 * 100% adversarial coverage of the API surface, authenticated with a key
 * MINTED THROUGH THE INTERFACE — POST /twin/api/settings/api-keys, the exact
 * endpoint the WebUI "Create API key" button hits.
 *
 * This proves the full lifecycle: a key an operator generates in
 * Settings → API keys is a real credential accepted by ``require_auth``, not
 * a display-only artefact. It intentionally remains non-admin: routes guarded
 * by ``require_admin_user`` return 403 because only the separately managed
 * infrastructure root key may administer a deployment without an IdP. If a
 * generated key did NOT authenticate, the sanity probe below would fail loudly.
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
    // Mint a key through the same endpoint the WebUI uses. Only the static
    // infrastructure root key may call this admin endpoint without an IdP.
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

  test('a Settings-generated key authenticates operator routes and is denied admin routes', async () => {
    // Sanity: prove the generated key (not the static key) is what authenticates.
    const probe = await ctx.get('/twin/api/folders', {
      headers: {
        Accept: 'application/json',
        'X-Twin-Folder': defaultFolder,
        Authorization: `Bearer ${generatedKey}`,
      },
    });
    expect(probe.status(), 'generated key did not authenticate').toBe(200);

    // The same key must not cross the dormant-IdP administration boundary.
    const adminProbe = await ctx.post(
      '/twin/api/documents/__twin_e2e_nonexistent_0000__/folders',
      {
        headers: {
          Accept: 'application/json',
          'X-Twin-Folder': defaultFolder,
          Authorization: `Bearer ${generatedKey}`,
          'Content-Type': 'application/json',
        },
        data: { folder_id: defaultFolder },
      },
    );
    expect(adminProbe.status(), 'generated key unexpectedly has admin power').toBe(403);

    const cfg: CoverageConfig = {
      authToken: generatedKey,
      expectAuth,
      defaultFolder,
      credentialTier: 'operator',
    };
    const covered = await coverApiSurface(test, ctx, cfg);
    test.info().annotations.push({
      type: 'coverage',
      description: `${covered.length} routes adversarially covered with a Settings-generated key`,
    });
  });
});
