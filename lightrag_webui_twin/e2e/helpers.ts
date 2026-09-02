import { expect, type Page } from '@playwright/test';

const QUERY_IDLE_STABILITY_MS = 100;
const DOCUMENTS_BOOTSTRAP_QUERY_KEYS = [
  'quota',
  'procedures',
  'vision-settings',
] as const;

type E2eControlPath =
  | '/__e2e/reset'
  | '/__e2e/scenario'
  | '/__e2e/quota'
  | '/__e2e/stats'
  | '/__e2e/documents'
  | '/__e2e/activity'
  | '/__e2e/procedures';

export type E2eStats = {
  approveCalls: Record<string, number>;
  tagApproveCalls: Record<string, number>;
  portabilityImportStarts: number;
  portabilityApproveCalls: number;
  portabilityApproveTransitions: number;
  portabilityApplyCalls: number;
  portabilityApplyTransitions: number;
  portabilityCancelCalls: number;
  portabilityCancelTransitions: number;
  linkedSourcePreviewCalls: number;
  linkedSourceCreateCalls: number;
  linkedSourceCreateTransitions: number;
  linkedSourceDisableCalls: number;
  linkedSourceDisableTransitions: number;
  folderRequests: Array<{
    path: string;
    folder: string | null;
  }>;
  queryRequests: Array<{
    path: string;
    body: Record<string, unknown>;
  }>;
  uploadRequests: Array<{
    name: string;
    docType: string | null;
    classification: string | null;
  }>;
};

async function controlFetch<Payload>(
  page: Page,
  path: E2eControlPath,
  body?: unknown,
): Promise<Payload> {
  const method = path === '/__e2e/stats' ? 'GET' : 'POST';
  return page.evaluate(
    async ({ controlPath, method, body }) => {
      let response: globalThis.Response;
      try {
        response = await fetch(controlPath, {
          method,
          headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
          body: body === undefined ? undefined : JSON.stringify(body),
        });
      } catch (error) {
        const cause = error instanceof Error ? error.message : String(error);
        throw new Error(
          `E2E control ${method} ${controlPath} failed: status=<unavailable>; body=<unavailable>; cause=${cause}`,
          { cause: error },
        );
      }

      let responseBody: string;
      try {
        responseBody = await response.text();
      } catch (error) {
        const cause = error instanceof Error ? error.message : String(error);
        throw new Error(
          `E2E control ${method} ${controlPath} failed: status=${response.status} ${response.statusText}; body=<unreadable>; cause=${cause}`,
          { cause: error },
        );
      }
      if (!response.ok) {
        throw new Error(
          `E2E control ${method} ${controlPath} failed: status=${response.status} ${response.statusText}; body=${responseBody || '<empty>'}`,
        );
      }

      try {
        return JSON.parse(responseBody) as Payload;
      } catch (error) {
        const cause = error instanceof Error ? error.message : String(error);
        throw new Error(
          `E2E control ${method} ${controlPath} returned invalid JSON: status=${response.status} ${response.statusText}; body=${responseBody || '<empty>'}; cause=${cause}`,
          { cause: error },
        );
      }
    },
    { controlPath: path, method, body },
  );
}

export async function boot(page: Page) {
  await page.addInitScript(() => {
    // Clear every retrieval-thread key, including the per-folder
    // (twin-rag.threads.v<n>:<folder>) variants introduced by the
    // cross-folder-leak fix — not just the bare base keys.
    for (let i = window.localStorage.length - 1; i >= 0; i -= 1) {
      const k = window.localStorage.key(i);
      if (k && k.startsWith('twin-rag.threads.v')) {
        window.localStorage.removeItem(k);
      }
    }
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await waitForQueryIdle(page, DOCUMENTS_BOOTSTRAP_QUERY_KEYS);
  await controlFetch<{ ok: true }>(page, '/__e2e/reset');
  await page.reload();
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await waitForQueryIdle(page, DOCUMENTS_BOOTSTRAP_QUERY_KEYS);
}

export async function waitForQueryIdle(
  page: Page,
  requiredQueryKeys: readonly string[] = [],
): Promise<void> {
  let idleSince: number | null = null;
  await expect
    .poll(
      async () => {
        const state = await page.evaluate((requiredKeys) => {
          const client = (
            window as Window & {
              __TWIN_E2E_QUERY_CLIENT?: {
                isFetching: () => number;
                getQueryCache: () => {
                  getAll: () => Array<{ queryKey: readonly unknown[] }>;
                };
              };
            }
          ).__TWIN_E2E_QUERY_CLIENT;
          if (!client) return { fetching: -1, hasRequiredKeys: false };
          const presentKeys = new Set(
            client
              .getQueryCache()
              .getAll()
              .map((query) => query.queryKey[0])
              .filter((key): key is string => typeof key === 'string'),
          );
          return {
            fetching: client.isFetching(),
            hasRequiredKeys: requiredKeys.every((key) => presentKeys.has(key)),
          };
        }, requiredQueryKeys);
        if (state.fetching !== 0 || !state.hasRequiredKeys) {
          idleSince = null;
          return false;
        }

        // A first zero can be observed between React mounting the shell and
        // running the effects that register its queries. Require a short,
        // continuous idle window so a slow CI runner cannot navigate away
        // while the Documents bootstrap fetches are only about to start.
        idleSince ??= Date.now();
        return Date.now() - idleSince >= QUERY_IDLE_STABILITY_MS;
      },
      {
        intervals: [25],
        message: 'MSW query client did not settle before the E2E transition',
      },
    )
    .toBe(true);
}

export async function openTab(page: Page, name: string) {
  // Changing tabs unmounts the current pane. Let its React Query work finish
  // first so a loaded runner does not turn a normal component teardown into
  // transport-level ERR_ABORTED noise in the browser guard. On Documents,
  // wait until all three dependent bootstrap queries have actually registered
  // as well: isFetching() can briefly be zero between chained React renders.
  const documentsActive = await page
    .getByRole('navigation')
    .getByRole('button', { name: 'Documents', exact: true })
    .evaluate((button) => button.classList.contains('active'));
  await waitForQueryIdle(
    page,
    documentsActive ? DOCUMENTS_BOOTSTRAP_QUERY_KEYS : [],
  );
  // Scope to the topbar nav — tab labels (e.g. "Settings", "Retrieval")
  // also exist as filter pills / buttons inside some tab panes.
  await page
    .getByRole('navigation')
    .getByRole('button', { name, exact: true })
    .click();
}

export async function setMswScenario(page: Page, scenario: Record<string, unknown>) {
  await controlFetch<{ ok: true }>(page, '/__e2e/scenario', scenario);
}

export async function setMswQuota(page: Page, quota: Record<string, unknown>) {
  await controlFetch<{ ok: true }>(page, '/__e2e/quota', quota);
  await page.evaluate(async () => {
    await (
      window as Window & {
        __TWIN_E2E_QUERY_CLIENT?: {
          invalidateQueries: (args: { queryKey: readonly string[] }) => Promise<unknown>;
        };
      }
    ).__TWIN_E2E_QUERY_CLIENT?.invalidateQueries({ queryKey: ['quota'] });
  });
}

export async function getMswStats(page: Page): Promise<E2eStats> {
  return controlFetch<E2eStats>(page, '/__e2e/stats');
}

export async function seedDocuments(
  page: Page,
  documents: readonly Record<string, unknown>[],
) {
  await controlFetch<{ ok: true }>(page, '/__e2e/documents', { documents });
  await page.evaluate(async () => {
    await (
      window as Window & {
        __TWIN_E2E_QUERY_CLIENT?: {
          invalidateQueries: (args: { queryKey: readonly string[] }) => Promise<unknown>;
        };
      }
    ).__TWIN_E2E_QUERY_CLIENT?.invalidateQueries({ queryKey: ['documents'] });
  });
}

export async function seedActivity(
  page: Page,
  events: readonly Record<string, unknown>[],
) {
  await controlFetch<{ ok: true }>(page, '/__e2e/activity', { events });
}

export async function addSourceFile(
  page: Page,
  file: { name: string; mimeType: string; buffer: Buffer },
  tag?: string,
) {
  await page.getByRole('button', { name: 'Add source' }).click();
  await page.getByTestId('addsource-file-input').setInputFiles(file);
  if (tag) {
    await page.getByLabel('Tag input').fill(tag);
    await page.getByTestId(`tag-sugg-${tag}`).click();
  }
  await expect(page.getByRole('button', { name: 'Add 1 source' })).toBeEnabled({
    timeout: 12_000,
  });
  await page.getByRole('button', { name: 'Add 1 source' }).click();
}

export async function seedProcedures(
  page: Page,
  bundles: readonly Record<string, unknown>[],
) {
  await controlFetch<{ ok: true }>(page, '/__e2e/procedures', { bundles });
  await page.evaluate(async () => {
    await (
      window as Window & {
        __TWIN_E2E_QUERY_CLIENT?: {
          invalidateQueries: (args: { queryKey: readonly string[] }) => Promise<unknown>;
        };
      }
    ).__TWIN_E2E_QUERY_CLIENT?.invalidateQueries({ queryKey: ['procedures'] });
  });
}
