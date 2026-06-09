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
