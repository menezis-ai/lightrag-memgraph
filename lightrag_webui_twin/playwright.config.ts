import { defineConfig, devices } from '@playwright/test';

// Frontend dev-server port. Configurable so CI jobs that run Playwright with
// `--network host` (webui-e2e-real, webui-e2e-keygen) can bind distinct host
// ports and coexist on the same runner without colliding on 4173. Defaults to
// 4173 for local dev and the container-isolated MSW e2e job.
const PORT = Number(process.env.PLAYWRIGHT_PORT ?? 4173);
const E2E_SUITE = process.env.TWIN_E2E_SUITE;

// The default MSW gate must have zero structural skips. Real-backend/API
// batteries and the documentation screenshot harness have dedicated commands;
// loading them only to skip them made missing CI configuration indistinguishable
// from intentional suite selection.
const MSW_TEST_IGNORE = [
  '**/capture-qa-screenshots.spec.ts',
  '**/real-backend.spec.ts',
  '**/api-coverage-real.spec.ts',
  '**/api-coverage-generated-key.spec.ts',
];

export default defineConfig({
  testDir: './e2e',
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  // MSW keeps mutable mock state in the browser worker. Until e2e runs against
  // the real Couche 3 backend, run specs serially so /__e2e/reset from one file
  // cannot wipe another file mid-journey.
  workers: 1,
  reporter: process.env.CI ? [['html'], ['github']] : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: 'chromium',
      testIgnore: E2E_SUITE === 'msw' ? MSW_TEST_IGNORE : [],
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
