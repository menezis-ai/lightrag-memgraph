/**
 * 100% adversarial coverage of the API surface (/twin/api + native shims),
 * authenticated with the static infra key (LIGHTRAG_API_KEY).
 *
 * The surface is discovered from the live /openapi.json at runtime, so a
 * route that exists cannot be missed. See ./lib/api-coverage for the shared
 * protocol. A sibling spec re-runs the same protocol with a key minted
 * through Settings → API keys (api-coverage-generated-key.spec.ts).
 *
 * Requires a real backend: set REAL_BACKEND_URL. Auth rejection is asserted
 * only when REAL_E2E_EXPECT_AUTH=true. Runs in webui-e2e-real; auto-skips
 * elsewhere.
 */

import { request as playwrightRequest, test } from '@playwright/test';

import { coverApiSurface, type CoverageConfig } from './lib/api-coverage';

const backendUrl = process.env.REAL_BACKEND_URL?.replace(/\/$/, '') ?? '';
const cfg: CoverageConfig = {
  authToken: process.env.REAL_BACKEND_AUTH_TOKEN ?? process.env.VITE_AUTH_TOKEN ?? '',
  expectAuth: process.env.REAL_E2E_EXPECT_AUTH === 'true',
  defaultFolder: process.env.REAL_BACKEND_FOLDER ?? 'default',
  credentialTier: 'infrastructure-root',
};

test.describe('Twin API — 100% adversarial coverage (static API key)', () => {
  test.skip(!backendUrl, 'Set REAL_BACKEND_URL to run real-backend API coverage.');

  let ctx: Awaited<ReturnType<typeof playwrightRequest.newContext>>;
  test.beforeAll(async () => {
    ctx = await playwrightRequest.newContext({ baseURL: backendUrl });
  });
  test.afterAll(async () => {
    await ctx?.dispose();
  });

  test('every API route degrades gracefully under hostile input', async () => {
    const covered = await coverApiSurface(test, ctx, cfg);
    test.info().annotations.push({
      type: 'coverage',
      description: `${covered.length} routes adversarially covered with the static key`,
    });
  });
});
