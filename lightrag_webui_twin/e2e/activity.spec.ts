import { expect, test } from '@playwright/test';
import { boot, openTab, seedActivity } from './helpers';

test.describe('Activity RC-1 refresh and immutable ledger', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openTab(page, 'Activity');
  });

  test('@activity @rc1 refresh fetches newly available events and survives reload', async ({
    page,
  }) => {
    await page.getByLabel('Search events').fill('e2e-activity-refresh');
    await expect(page.getByText('No events match the current filter')).toBeVisible();

    await seedActivity(page, [
      {
        id: 'evt_e2e_activity_refresh',
        ts: '2026-05-11T09:59:00Z',
        rel: 'now',
        day: 'Today',
        kind: 'settings',
        sev: 'info',
        actor: { user: 'system', role: 'pipeline' },
        target: { type: 'folder', label: 'e2e-activity-refresh' },
        summary: 'e2e-activity-refresh · newly fetched by explicit Refresh',
        meta: { source: 'playwright' },
      },
    ]);

    await page.getByRole('button', { name: 'Refresh' }).click();
    await expect(page.getByRole('heading', { name: 'e2e-activity-refresh' })).toBeVisible();
    await expect(page.getByRole('complementary')).toContainText(
      'newly fetched by explicit Refresh',
    );

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Activity');
    await expect(page.getByLabel('Search events')).toHaveValue('e2e-activity-refresh');
    await expect(page.getByRole('heading', { name: 'e2e-activity-refresh' })).toBeVisible();
  });

  test('@activity @rc1 no manual clear affordance exists on immutable audit ledger', async ({
    page,
  }) => {
    await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible();
    await expect(page.getByRole('button', { name: /Clear activity events/i })).toHaveCount(0);
    await expect(page.getByRole('button', { name: /Purge expired events/i })).toHaveCount(0);
    await expect(page.getByRole('dialog', { name: /Clear activity events/i })).toHaveCount(0);
    await expect(page.locator('.toast-viewport')).not.toContainText('events removed');
  });
});

test.describe('Activity filters and event detail', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openTab(page, 'Activity');
    await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible();
  });

  const statsCount = async (page: import('@playwright/test').Page, label: string) => {
    const stat = page.locator('.activity-stats .stat', { hasText: label }).locator('b');
    return Number(await stat.innerText());
  };

  test('@activity @filters severity filter narrows the timeline to matching rows', async ({
    page,
  }) => {
    const errorCount = await statsCount(page, 'errors');
    expect(errorCount).toBeGreaterThan(0);

    await page.getByLabel('Severity filter').selectOption('error');
    await expect
      .poll(() => statsCount(page, 'events'))
      .toBe(errorCount);
    await expect(page.locator('.activity-row.sev-error').first()).toBeVisible();
    await expect(page.locator('.activity-row.sev-info')).toHaveCount(0);

    await page.getByLabel('Severity filter').selectOption('any');
    await expect
      .poll(() => statsCount(page, 'events'))
      .toBeGreaterThan(errorCount);
  });

  test('@activity @filters kind pill isolates one event kind', async ({ page }) => {
    const retrievals = await statsCount(page, 'retrievals');
    expect(retrievals).toBeGreaterThan(0);

    const retrievalPill = page
      .locator('.activity-kinds')
      .getByRole('button', { name: 'Retrieval', exact: true });
    await retrievalPill.click();
    await expect(retrievalPill).toHaveAttribute('aria-pressed', 'true');
    await expect.poll(() => statsCount(page, 'events')).toBe(retrievals);

    await retrievalPill.click();
    await expect.poll(() => statsCount(page, 'events')).toBeGreaterThan(retrievals);
  });

  test('@activity @filters search, clear-search and time range stay coherent', async ({
    page,
  }) => {
    await page.getByLabel('Search events').fill('zzz-no-such-event');
    await expect(page.getByText('No events match the current filter')).toBeVisible();
    await page.getByLabel('Clear search').click();
    await expect(page.getByLabel('Search events')).toHaveValue('');
    await expect(page.locator('.activity-row').first()).toBeVisible();

    const allCount = await statsCount(page, 'events');
    await page.getByRole('tab', { name: '24h' }).click();
    await expect(page.getByRole('tab', { name: '24h' })).toHaveAttribute(
      'aria-selected',
      'true',
    );
    const dayCount = await statsCount(page, 'events');
    expect(dayCount).toBeLessThanOrEqual(allCount);

    await page.getByRole('tab', { name: 'All' }).click();
    await expect.poll(() => statsCount(page, 'events')).toBe(allCount);
  });

  test('@activity @detail selecting a failed source exposes Replay ingestion', async ({
    page,
  }) => {
    await page.getByLabel('Severity filter').selectOption('error');
    await page.locator('.activity-row.sev-error').first().click();

    const detail = page.getByRole('complementary');
    await expect(detail).toContainText('Event ID');
    await expect(detail).toContainText('Severity');

    await detail.getByRole('button', { name: 'Replay ingestion' }).click();
    // Audit C7: ActivityTab's "Replay ingestion" routes through the
    // failed-batch endpoint; the toast wording reflects the real
    // action (POST /documents/reprocess_failed) instead of the
    // misleading "Replay queued".
    await expect(page.locator('.toast-viewport')).toContainText('Re-processing failed sources');
  });

  test('@activity @detail "Open source" drills down to the Documents tab', async ({
    page,
  }) => {
    await page.getByLabel('Severity filter').selectOption('error');
    await page.locator('.activity-row.sev-error').first().click();
    const detail = page.getByRole('complementary');
    const target = (await detail.locator('h3').innerText()).trim();

    await detail.getByRole('button', { name: 'Open source' }).click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page.getByLabel('Search source')).toHaveValue(target);
  });
});
