/**
 * MSW Service Worker bootstrap for browser dev.
 *
 * `main.tsx` calls `startMsw()` before mounting React when `VITE_USE_MSW` is
 * truthy. The worker is registered against the public `/mockServiceWorker.js`
 * artifact (added by `bun msw init`), so dev fetches are intercepted without
 * any network egress. Playwright's MSW scripts set `VITE_E2E_MSW_STRICT=true`
 * so a missing API handler fails closed; interactive development keeps the
 * permissive bypass needed for local experimentation.
 */

import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';

export const worker = setupWorker(...handlers);

export async function startMsw(): Promise<void> {
  if (globalThis.window === undefined) return;
  await worker.start({
    onUnhandledRequest:
      import.meta.env.VITE_E2E_MSW_STRICT === 'true' ? 'error' : 'bypass',
    serviceWorker: { url: '/mockServiceWorker.js' },
  });
}
