/**
 * MSW Service Worker bootstrap for browser dev.
 *
 * `main.tsx` calls `startMsw()` before mounting React when `VITE_USE_MSW` is
 * truthy. The worker is registered against the public `/mockServiceWorker.js`
 * artifact (added by `bun msw init`), so dev fetches are intercepted without
 * any network egress.
 */

import { setupWorker } from 'msw/browser';
import { handlers } from './handlers';

export const worker = setupWorker(...handlers);

export async function startMsw(): Promise<void> {
  if (typeof globalThis.window === 'undefined') return;
  await worker.start({
    onUnhandledRequest: 'bypass',
    serviceWorker: { url: '/mockServiceWorker.js' },
  });
}
