import { expect, test } from '@playwright/test';
import { boot, openTab } from './helpers';

test.describe('Toast lifecycle', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@toasts manual dismiss removes a toast immediately', async ({ page }) => {
    await page.getByTestId('docs-row-delete-d1').click();
    await page.getByTestId('doc-detail-reprocess').click();
    const toast = page.locator('.toast-viewport .toast');
    await expect(toast).toHaveCount(1);
    await toast.getByLabel('Dismiss').click();
    await expect(toast).toHaveCount(0);
  });

  test('@toasts retag success toast carries an Undo affordance', async ({ page }) => {
    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page
      .getByLabel('Bulk actions')
      .getByRole('button', { name: /Retag 1 sources/ })
      .click();
    await page.getByRole('textbox', { name: 'Tag input' }).fill('memgraph');
    await page.getByTestId('sugg-memgraph').click();
    await page.getByRole('button', { name: 'Apply tag' }).click();

    const toast = page.locator('.toast-viewport .toast', { hasText: 'memgraph' });
    await expect(toast).toBeVisible();
    await toast.getByRole('button', { name: 'Undo' }).click();
    await expect(toast).toHaveCount(0);
  });

  test('@toasts overflow collapses older toasts under a "+N more" dismiss button', async ({
    page,
  }) => {
    await page.getByTestId('docs-row-delete-d1').click();
    // TOAST_MAX_VISIBLE = 3 — the 4th pushes the oldest into the overflow pill.
    for (let i = 0; i < 4; i += 1) {
      await page.getByTestId('doc-detail-reprocess').click();
    }
    const more = page.locator('.toast-stack-more');
    await expect(more).toContainText('+1 more');
    await expect(page.locator('.toast-viewport .toast')).toHaveCount(3);

    await more.click();
    await expect(more).toHaveCount(0);
    await expect(page.locator('.toast-viewport .toast')).toHaveCount(3);
  });
});

test.describe('Topbar shell', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@topbar brand button always returns to Documents', async ({ page }) => {
    await openTab(page, 'Graph');
    await expect(page.getByTestId('kg-canvas')).toBeVisible();
    await page.getByRole('button', { name: 'Open Documents' }).click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
  });

  test('@topbar @notifications "View full activity log" deep-links to Activity', async ({
    page,
  }) => {
    await page.getByRole('button', { name: /Notifications/ }).click();
    await page.getByRole('button', { name: 'View full activity log →' }).click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toBeHidden();
    await expect(page.getByLabel('Search events')).toBeVisible();
  });

  test('@topbar @folders "Manage folders" opens the Settings folder section', async ({
    page,
  }) => {
    await page.getByTitle('Switch folder').click();
    await page.getByRole('button', { name: 'Manage folders →' }).click();
    await expect(page.getByTestId('settings-tab')).toBeVisible();
    await expect(page.getByTestId('settings-folder')).toBeVisible();
  });

  test('@topbar theme toggle flips dark and back to light', async ({ page }) => {
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
    await page.getByLabel('Theme').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    await page.getByLabel('Theme').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  });
});
