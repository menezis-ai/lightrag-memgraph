import { expect, type Page } from '@playwright/test';

export async function boot(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem(
      'twin.onboarding.v1',
      JSON.stringify({ step: 'completion', dismissed: true, tasks: [] }),
    );
    window.localStorage.removeItem('twin-rag.threads.v2');
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
  await page.getByRole('button', { name, exact: true }).click();
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

export async function getMswStats(
  page: Page,
): Promise<{ approveCalls: Record<string, number> }> {
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
