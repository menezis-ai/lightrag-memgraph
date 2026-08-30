import { allowRequestAbort, expect, test, type Page } from './fixtures';
import { boot, getMswStats, openTab, setMswScenario } from './helpers';

async function openPortability(page: Page): Promise<void> {
  await openTab(page, 'Settings');
  await page.getByTestId('settings-rail-portability').click();
  await expect(page.getByTestId('settings-portability')).toBeVisible();
}

async function startImport(page: Page, name = 'staging.tar.gz'): Promise<void> {
  await page.getByTestId('portability-import-file').setInputFiles({
    name,
    mimeType: 'application/gzip',
    buffer: Buffer.from('canonical twin-kb-bundle'),
  });
  await page.getByTestId('portability-import-start').click();
  await expect(page.getByTestId('portability-report')).toBeVisible();
}

test.describe('Portability fail-closed journeys', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openPortability(page);
  });

  test('a blocking dry-run stays blocked across reload without an approval transition', async ({
    allowBrowserIssues,
    page,
  }) => {
    allowBrowserIssues(
      allowRequestAbort(
        /\/twin\/api\/admin\/portability\/imports\/imp_000000000000000000000001$/,
        'Reloading the blocking report cancels its one in-flight import-status query.',
      ),
    );
    await setMswScenario(page, { portabilityBlockingDryRun: true });
    await startImport(page, 'classified.tar.gz');

    const report = page.getByTestId('portability-report');
    await expect(report).toContainText('1 blocking');
    await expect(report).toContainText('CLASSIFICATION_CEILING');
    await expect(page.getByTestId('portability-approve')).toBeDisabled();
    await expect(page.getByTestId('portability-apply')).toHaveCount(0);
    await expect.poll(() => getMswStats(page)).toMatchObject({
      portabilityApproveCalls: 0,
      portabilityApproveTransitions: 0,
      portabilityApplyCalls: 0,
      portabilityApplyTransitions: 0,
    });

    await page.reload();
    await openPortability(page);
    await expect(page.getByTestId('portability-report')).toContainText(
      'CLASSIFICATION_CEILING',
    );
    await expect(page.getByTestId('portability-approve')).toBeDisabled();
    await expect(page.getByTestId('portability-apply')).toHaveCount(0);
  });

  test('rapid double activation produces one approve and one apply request and transition', async ({
    allowBrowserIssues,
    page,
  }) => {
    const appliedReloadReason =
      'Reloading the applied report cancels one active shell query for this endpoint.';
    allowBrowserIssues(
      allowRequestAbort(
        /\/twin\/api\/admin\/portability\/imports\/imp_000000000000000000000001$/,
        'Reloading the applied report cancels its current import-status poll.',
      ),
      allowRequestAbort(/\/twin\/api\/quota$/, appliedReloadReason),
      allowRequestAbort(/\/twin\/api\/procedures$/, appliedReloadReason),
      allowRequestAbort(/\/twin\/api\/settings\/vision$/, appliedReloadReason),
    );
    await setMswScenario(page, {
      portabilityApproveDelayMs: 120,
      portabilityApplyDelayMs: 120,
    });
    await startImport(page);

    await page.getByTestId('portability-approve').dblclick({ delay: 5 });
    await expect(page.getByTestId('portability-apply')).toBeVisible();
    await page.getByTestId('portability-apply').dblclick({ delay: 5 });
    await expect(page.getByTestId('portability-validate')).toBeVisible();

    await expect.poll(() => getMswStats(page)).toMatchObject({
      portabilityApproveCalls: 1,
      portabilityApproveTransitions: 1,
      portabilityApplyCalls: 1,
      portabilityApplyTransitions: 1,
    });

    await page.reload();
    await openPortability(page);
    await expect(page.getByTestId('portability-import-status')).toContainText(
      'applied',
    );
    await expect(page.getByTestId('portability-validate')).toBeVisible();
  });

  test('cancel survives reload and releases the workspace for a fresh dry-run', async ({
    allowBrowserIssues,
    page,
  }) => {
    allowBrowserIssues(
      allowRequestAbort(
        /\/twin\/api\/admin\/portability\/imports\/imp_000000000000000000000001$/,
        'Reloading the cancelled report cancels its current import-status poll.',
      ),
    );
    await startImport(page, 'cancelled.tar.gz');
    await page
      .getByTestId('portability-report')
      .getByRole('button', { name: 'Cancel', exact: true })
      .click();
    await expect(page.getByTestId('portability-import-status')).toContainText(
      'cancelled',
    );
    await expect.poll(() => getMswStats(page)).toMatchObject({
      portabilityImportStarts: 1,
      portabilityCancelCalls: 1,
      portabilityCancelTransitions: 1,
    });

    await page.reload();
    await openPortability(page);
    await expect(page.getByTestId('portability-import-status')).toContainText(
      'cancelled',
    );
    await page.getByTestId('portability-import-file').setInputFiles({
      name: 'retry.tar.gz',
      mimeType: 'application/gzip',
      buffer: Buffer.from('fresh canonical twin-kb-bundle'),
    });
    await expect(page.getByTestId('portability-import-start')).toBeEnabled();
    await page.getByTestId('portability-import-start').click();
    await expect(page.getByTestId('portability-report')).toContainText(
      'ready for approval',
    );

    // The browser-side request counters restart with the reloaded MSW module;
    // the persisted cancelled job does not. One accepted start after reload
    // proves that the terminal job released the workspace conflict guard.
    await expect.poll(() => getMswStats(page)).toMatchObject({
      portabilityImportStarts: 1,
      portabilityCancelCalls: 0,
      portabilityCancelTransitions: 0,
    });
  });
});
