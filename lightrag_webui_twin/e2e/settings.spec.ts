import { allowRequestAbort, expect, test } from './fixtures';
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

  test('TWIN-SET-03 bearer token revoke requires confirmation', async ({
    allowBrowserIssues,
    page,
  }) => {
    const revokeReloadReason =
      'Revoking the bearer token reloads the shell and cancels its active queries.';
    allowBrowserIssues(
      allowRequestAbort(/\/twin\/api\/quota$/, revokeReloadReason),
      allowRequestAbort(/\/twin\/api\/procedures$/, revokeReloadReason),
      allowRequestAbort(/\/twin\/api\/settings\/vision$/, revokeReloadReason),
      allowRequestAbort(/\/openapi\.json$/, revokeReloadReason),
    );
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
    allowBrowserIssues,
    page,
  }) => {
    allowBrowserIssues(
      allowRequestAbort(
        /\/twin\/api\/settings\/api-keys$/,
        'Creating a key invalidates and replaces the active API-key list query.',
      ),
    );
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
    allowBrowserIssues,
    page,
  }) => {
    allowBrowserIssues(
      allowRequestAbort(
        /\/twin\/api\/settings\/api-keys$/,
        'Create and revoke each invalidate the preceding API-key list query.',
        2,
      ),
    );
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

  test('TWIN-SET-09 vision procedure toggle persists through save and reload', async ({
    page,
  }) => {
    await page.getByTestId('settings-rail-vision').click();
    const section = page.getByTestId('settings-vision');
    await expect(section).toBeVisible();

    // Fresh state: env-default provenance, procedure ingestion off, no
    // unavailable warning (MSW reports the deployment as procedure-ready).
    await expect(page.getByTestId('settings-vision-provenance-env')).toBeVisible();
    const toggle = page.getByTestId('settings-vision-procedure-toggle');
    await expect(toggle).toHaveAttribute('aria-checked', 'false');
    await expect(
      page.getByTestId('settings-vision-procedure-unavailable'),
    ).toHaveCount(0);

    await toggle.click();
    await expect(toggle).toHaveAttribute('aria-checked', 'true');
    await page.getByTestId('settings-vision-save').click();
    await expect(page.getByRole('status')).toContainText('Vision settings saved');
    await expect(page.getByTestId('settings-vision-provenance-runtime')).toBeVisible();

    // The PUT reached the stateful mock, not just local component state: the
    // audit entry carries the enabled wording…
    await openTab(page, 'Activity');
    await expect(page.getByText('procedure ingestion enabled').first()).toBeVisible();

    // …and the runtime value survives a full reload (server-side state).
    await page.reload();
    await openTab(page, 'Settings');
    await page.getByTestId('settings-rail-vision').click();
    await expect(
      page.getByTestId('settings-vision-procedure-toggle'),
    ).toHaveAttribute('aria-checked', 'true');
    await expect(page.getByTestId('settings-vision-provenance-runtime')).toBeVisible();
  });

  test('TWIN-SET-10 portability import follows the approved stateful workflow', async ({
    allowBrowserIssues,
    page,
  }) => {
    const portabilityReloadReason =
      'The deliberate mid-workflow reload replaces active Settings shell queries.';
    allowBrowserIssues(
      allowRequestAbort(
        /\/twin\/api\/admin\/portability\/imports\/imp_[A-Za-z0-9_-]+$/,
        'The deliberate mid-workflow reload cancels the current portability poll before resuming it.',
      ),
      allowRequestAbort(/\/twin\/api\/quota$/, portabilityReloadReason),
      allowRequestAbort(/\/twin\/api\/procedures$/, portabilityReloadReason),
      allowRequestAbort(/\/twin\/api\/settings\/vision$/, portabilityReloadReason),
    );
    await page.getByTestId('settings-rail-portability').click();
    const section = page.getByTestId('settings-portability');
    await expect(section).toBeVisible();

    await page.getByTestId('portability-import-file').setInputFiles({
      name: 'staging.tar.gz',
      mimeType: 'application/gzip',
      buffer: Buffer.from('canonical twin-kb-bundle'),
    });
    await page
      .getByTestId('portability-folder-map')
      .fill('{"staging":"production"}');
    await page.getByTestId('portability-import-start').click();

    const report = page.getByTestId('portability-report');
    await expect(report).toContainText('ready for approval');
    await expect(report).toContainText('all three probe cosines');
    await expect(report).toContainText('C2');
    await expect
      .poll(() =>
        page.evaluate(() =>
          sessionStorage.getItem('twin.portability.import-job.v1'),
        ),
      )
      .toMatch(/^imp_[A-Za-z0-9_-]+$/);

    // The opaque job id is session-persisted so an operator can resume the
    // server-side workflow after an accidental reload.
    await page.reload();
    await expect
      .poll(() =>
        page.evaluate(() =>
          sessionStorage.getItem('twin.portability.import-job.v1'),
        ),
      )
      .toMatch(/^imp_[A-Za-z0-9_-]+$/);
    await openTab(page, 'Settings');
    const resumedRequest = page.waitForResponse((response) =>
      /\/twin\/api\/admin\/portability\/imports\/imp_/.test(response.url()),
    );
    await page.getByTestId('settings-rail-portability').click();
    expect((await resumedRequest).status()).toBe(200);
    await expect(page.getByTestId('portability-report')).toContainText(
      'ready for approval',
    );

    await page.getByTestId('portability-approve').click();
    await page.getByTestId('portability-apply').click();
    await page.getByTestId('portability-validate').click();
    await expect(page.getByTestId('portability-validation')).toContainText(
      'Validation passed',
    );

    // The mock records the same immutable Activity event as the real backend,
    // proving that the controls are wired to server-side state transitions.
    await openTab(page, 'Activity');
    await expect(
      page.getByText('KB bundle imported into workspace base').first(),
    ).toBeVisible();
  });

  test('TWIN-SET-04/05/06 editable settings remain out of scope', async ({ page }) => {
    await expect(page.getByTestId('settings-rail-profile')).toBeVisible();
    await expect(page.getByTestId('settings-rail-api')).toBeVisible();
    await expect(page.getByTestId('settings-rail-folder')).toBeVisible();
    await expect(page.getByTestId('settings-rail-portability')).toBeVisible();

    await expect(page.getByTestId('settings-tab')).not.toContainText(
      /Default ingestion tags|Invite member|Delete member|Revoke token/,
    );
    await expect(page.getByTestId('settings-rail-members')).toHaveCount(0);
    await expect(page.getByTestId('settings-rail-tokens')).toHaveCount(0);
    await expect(page.getByTestId('settings-rail-providers')).toHaveCount(0);
  });
});
