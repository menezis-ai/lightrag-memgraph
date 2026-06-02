import { expect, test } from '@playwright/test';
import { boot, openTab } from './helpers';

test.describe('Settings guardrails', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openTab(page, 'Settings');
  });

  test('TWIN-SET-03 bearer token revoke requires confirmation', async ({ page }) => {
    await page.getByTestId('settings-rail-api').click();
    await page.getByRole('button', { name: 'Authorize' }).click();
    await page.getByLabel('Value').fill('e2e-token');
    await page
      .getByRole('dialog', { name: 'Authorize' })
      .getByRole('button', { name: 'Authorize' })
      .click();
    await expect(page.getByRole('button', { name: 'Authorized' })).toBeVisible();

    await page.getByRole('button', { name: 'Authorized' }).click();
    const dialog = page.getByRole('dialog', { name: 'Authorize' });
    await dialog.getByRole('button', { name: 'Revoke token' }).click();
    await expect(dialog.getByRole('button', { name: 'Confirm revoke token' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Authorized' })).toBeVisible();

    await dialog.getByRole('button', { name: 'Confirm revoke token' }).click();
    await expect(page.getByRole('button', { name: 'Authorize' })).toBeVisible();
  });

  test('TWIN-SET-04/05/06 editable settings remain out of scope', async ({ page }) => {
    await expect(page.getByTestId('settings-rail-profile')).toBeVisible();
    await expect(page.getByTestId('settings-rail-api')).toBeVisible();
    await expect(page.getByTestId('settings-rail-workspace')).toBeVisible();

    await expect(page.getByTestId('settings-tab')).not.toContainText(
      /Default ingestion tags|Invite member|Delete member|Revoke token/,
    );
    await expect(page.getByTestId('settings-rail-members')).toHaveCount(0);
    await expect(page.getByTestId('settings-rail-tokens')).toHaveCount(0);
    await expect(page.getByTestId('settings-rail-providers')).toHaveCount(0);
  });
});
