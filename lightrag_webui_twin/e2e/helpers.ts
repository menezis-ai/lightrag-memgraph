import { expect, type Page } from '@playwright/test';

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
  await page.evaluate(async () => {
    await fetch('/__e2e/reset', { method: 'POST' });
  });
  await page.reload();
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
}

export async function openTab(page: Page, name: string) {
  // Scope to the topbar nav — tab labels (e.g. "Settings", "Retrieval")
  // also exist as filter pills / buttons inside some tab panes.
  await page
    .getByRole('navigation')
    .getByRole('button', { name, exact: true })
    .click();
}

export async function setMswScenario(page: Page, scenario: Record<string, unknown>) {
  await page.evaluate(async (body) => {
    await fetch('/__e2e/scenario', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }, scenario);
}

export async function setMswQuota(page: Page, quota: Record<string, unknown>) {
  await page.evaluate(async (body) => {
    await fetch('/__e2e/quota', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    await (
      window as Window & {
        __TWIN_E2E_QUERY_CLIENT?: {
          invalidateQueries: (args: { queryKey: readonly string[] }) => Promise<unknown>;
        };
      }
    ).__TWIN_E2E_QUERY_CLIENT?.invalidateQueries({ queryKey: ['quota'] });
  }, quota);
}

export async function getMswStats(
  page: Page,
): Promise<{
  approveCalls: Record<string, number>;
  tagApproveCalls: Record<string, number>;
  folderRequests: Array<{
    path: string;
    folder: string | null;
  }>;
  queryRequests: Array<{
    path: string;
    body: Record<string, unknown>;
  }>;
}> {
  return page.evaluate(async () => {
    const res = await fetch('/__e2e/stats');
    return res.json();
  });
}

export async function seedDocuments(
  page: Page,
  documents: readonly Record<string, unknown>[],
) {
  await page.evaluate(async (docs) => {
    await fetch('/__e2e/documents', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ documents: docs }),
    });
    await (
      window as Window & {
        __TWIN_E2E_QUERY_CLIENT?: {
          invalidateQueries: (args: { queryKey: readonly string[] }) => Promise<unknown>;
        };
      }
    ).__TWIN_E2E_QUERY_CLIENT?.invalidateQueries({ queryKey: ['documents'] });
  }, documents);
}

export async function seedActivity(
  page: Page,
  events: readonly Record<string, unknown>[],
) {
  await page.evaluate(async (activityEvents) => {
    await fetch('/__e2e/activity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events: activityEvents }),
    });
  }, events);
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
