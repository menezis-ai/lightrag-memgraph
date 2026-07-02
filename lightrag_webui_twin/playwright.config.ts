import { defineConfig, devices } from '@playwright/test';

// Frontend dev-server port. Configurable so CI jobs that run Playwright with
// `--network host` (webui-e2e-real, webui-e2e-keygen) can bind distinct host
// ports and coexist on the same runner without colliding on 4173. Defaults to
// 4173 for local dev and the container-isolated MSW e2e job.
const PORT = Number(process.env.PLAYWRIGHT_PORT ?? 4173);

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
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
