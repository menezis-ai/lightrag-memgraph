import { expect, test } from '@playwright/test';
import { boot, openTab } from './helpers';

test.describe('Settings guardrails', () => {
  // The API-key reveal panel copies the secret via navigator.clipboard, which
  // throws in headless Chromium without an explicit grant (the button would
  // then read "Copy failed" instead of "Copied").
  test.use({ permissions: ['clipboard-read', 'clipboard-write'] });

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

  test('TWIN-SET-07 @doctrine API key create reveals the full secret exactly once', async ({
    page,
  }) => {
    await page.getByTestId('settings-rail-api-keys').click();
    const section = page.getByTestId('settings-api-keys');
    await expect(section).toBeVisible();
    // Fresh MSW state (reset in boot) → empty list.
    await expect(page.getByTestId('settings-api-keys-empty')).toBeVisible();

    await page.getByTestId('settings-api-keys-create-btn').click();

    // Modal autofocus (~30-60ms) can steal early keystrokes — wait for the
    // input then force focus before typing into it.
    const nameInput = page.getByTestId('settings-api-keys-create-name');
    await expect(nameInput).toBeVisible();
    await page.waitForTimeout(80);
    await nameInput.focus();
    await nameInput.fill('ingestion-agent');
    await expect(nameInput).toHaveValue('ingestion-agent');

    await page.getByTestId('settings-api-keys-create-submit').click();

    // One-time reveal panel: full value shown, scoped to its own testid so the
    // ARIA live "will not be shown again" warning doesn't collide.
    const revealValue = page.getByTestId('settings-api-keys-reveal-value');
    await expect(revealValue).toBeVisible();
    await expect(page.getByTestId('settings-api-keys-reveal-name')).toHaveText(
      'ingestion-agent',
    );
    const fullSecret = (await revealValue.textContent())?.trim() ?? '';
    expect(fullSecret).toMatch(/^twk_/);

    // Copy affordance flips to "Copied" (MSW-backed, navigator.clipboard).
    await page.getByTestId('settings-api-keys-reveal-copy').click();
    await expect(page.getByTestId('settings-api-keys-reveal-copy')).toContainText(
      'Copied',
    );

    // Dismiss → secret is gone and never retrievable again.
    await page.getByTestId('settings-api-keys-reveal-dismiss').click();
    await expect(revealValue).toHaveCount(0);
    await expect(page.getByText(fullSecret, { exact: false })).toHaveCount(0);

    // The new key now lists by prefix only (GET returns no full_value).
    const table = page.getByTestId('settings-api-keys-table');
    await expect(table).toBeVisible();
    await expect(table.getByText('ingestion-agent')).toBeVisible();
    await expect(table.locator('.settings-api-keys-prefix-value')).toHaveCount(1);
    const prefix =
      (await table.locator('.settings-api-keys-prefix-value').textContent())?.trim() ??
      '';
    expect(prefix).toMatch(/^twk_/);
    // Prefix is a truncated preview, not the full secret.
    expect(prefix).not.toBe(fullSecret);
  });

  test('TWIN-SET-08 @doctrine API key revoke is double-confirm and survives reload', async ({
    page,
  }) => {
    await page.getByTestId('settings-rail-api-keys').click();
    await expect(page.getByTestId('settings-api-keys')).toBeVisible();

    // Mint a key to revoke.
    await page.getByTestId('settings-api-keys-create-btn').click();
    const nameInput = page.getByTestId('settings-api-keys-create-name');
    await expect(nameInput).toBeVisible();
    await page.waitForTimeout(80);
    await nameInput.focus();
    await nameInput.fill('revoke-me');
    await page.getByTestId('settings-api-keys-create-submit').click();
    await expect(page.getByTestId('settings-api-keys-reveal-value')).toBeVisible();
    await page.getByTestId('settings-api-keys-reveal-dismiss').click();

    const row = page.getByTestId(/^settings-api-keys-row-/);
    await expect(row).toHaveCount(1);
    await expect(row.locator('.status-pill.active')).toBeVisible();

    const id = (await row.getAttribute('data-testid'))?.replace(
      'settings-api-keys-row-',
      '',
    );
    expect(id, 'row id missing').toBeTruthy();

    // First click arms the confirm; the active state is unchanged until confirm.
    await page.getByTestId(`settings-api-keys-revoke-${id}`).click();
    const confirmBtn = page.getByTestId(`settings-api-keys-revoke-confirm-${id}`);
    await expect(confirmBtn).toBeVisible();
    await expect(row.locator('.status-pill.active')).toBeVisible();
    await expect(row.locator('.status-pill.revoked')).toHaveCount(0);

    // Second click confirms → DELETE → revoked state.
    await confirmBtn.click();
    await expect(row.locator('.status-pill.revoked')).toBeVisible();
    await expect(page.getByTestId(`settings-api-keys-revoke-${id}`)).toHaveCount(0);

    // Survives reload: the revoked row persists from MSW state (audit trail) —
    // the mock now mirrors the durable Memgraph-backed api_key_store.
    await page.reload();
    await openTab(page, 'Settings');
    await page.getByTestId('settings-rail-api-keys').click();
    const reloadedRow = page.getByTestId(`settings-api-keys-row-${id}`);
    await expect(reloadedRow).toBeVisible();
    await expect(reloadedRow.locator('.status-pill.revoked')).toBeVisible();
    await expect(reloadedRow.getByText('revoke-me')).toBeVisible();
  });

  test('TWIN-SET-04/05/06 editable settings remain out of scope', async ({ page }) => {
    await expect(page.getByTestId('settings-rail-profile')).toBeVisible();
    await expect(page.getByTestId('settings-rail-api')).toBeVisible();
    await expect(page.getByTestId('settings-rail-folder')).toBeVisible();

    await expect(page.getByTestId('settings-tab')).not.toContainText(
      /Default ingestion tags|Invite member|Delete member|Revoke token/,
    );
    await expect(page.getByTestId('settings-rail-members')).toHaveCount(0);
    await expect(page.getByTestId('settings-rail-tokens')).toHaveCount(0);
    await expect(page.getByTestId('settings-rail-providers')).toHaveCount(0);
  });
});
