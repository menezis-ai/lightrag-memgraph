import { expect, test } from '@playwright/test';
import { boot, openTab, seedDocuments, setMswQuota } from './helpers';

test.describe('Operator user flows', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@documents @quota blocked quota disables ingestion actions and recovers', async ({
    page,
  }) => {
    await seedDocuments(page, [
      {
        doc_id: 'quota_failed_doc',
        file_path: '/cib/e2e/quota-failed.md',
        content_summary: 'Failed source used to verify retry gating',
        status: 'FAILED',
        chunks_count: 0,
        error_msg: 'Synthetic ingestion failure',
        tags: ['oracle'],
        folder: 'default',
      },
    ]);

    await setMswQuota(page, {
      configured: true,
      status: 'blocked',
      used_bytes: 2_147_483_648,
      limit_bytes: 2_147_483_648,
      used_pct: 1,
      warn_threshold: 0.85,
    });

    await expect(page.getByTestId('quota-banner-blocked')).toContainText(
      'ingestion disabled',
    );
    await expect(page.getByTestId('docs-add-source')).toBeDisabled();
    await expect(
      page.getByRole('button', { name: /Re-process failed \(current folder\)/ }),
    ).toBeDisabled();

    await setMswQuota(page, {
      configured: true,
      status: 'ok',
      used_bytes: 10,
      limit_bytes: 2_147_483_648,
      used_pct: 0.00000001,
      warn_threshold: 0.85,
    });

    await expect(page.getByTestId('quota-banner-blocked')).toHaveCount(0);
    await expect(page.getByTestId('docs-add-source')).toBeEnabled();
    await expect(
      page.getByRole('button', { name: /Re-process failed \(current folder\)/ }),
    ).toBeEnabled();
  });

  test('@documents @folders document copy is visible after switching folders', async ({
    page,
  }) => {
    await seedDocuments(page, [
      {
        doc_id: 'folder_flow_doc',
        file_path: '/cib/e2e/folder-flow.md',
        content_summary: 'Document copied between Twin folders',
        status: 'PROCESSED',
        chunks_count: 1,
        tags: ['oracle'],
        folder: 'default',
      },
    ]);

    await page.getByLabel('Search source').fill('folder-flow');
    await expect(page.getByTestId('docs-row-folder_flow_doc')).toBeVisible();
    await page.getByTestId('docs-row-folders-folder_flow_doc').click();

    const dialog = page.getByRole('dialog', {
      name: 'Manage folders for /cib/e2e/folder-flow.md',
    });
    await expect(dialog).toBeVisible();
    await dialog.getByLabel('Target folder').selectOption('sandbox');
    await dialog.getByTestId('document-folder-copy').click();
    await expect(page.locator('.toast-viewport')).toContainText(
      'Document copied to folder',
    );
    await dialog.getByRole('button', { name: 'Close dialog' }).click();

    await page.getByTitle('Switch folder').click();
    await page.getByRole('menuitemradio', { name: /sandbox/i }).click();
    await expect(page.getByTitle('Switch folder')).toContainText('sandbox');
    await expect(page.getByTestId('docs-row-folder_flow_doc')).toBeVisible();

    await page.getByTestId('docs-row-folders-folder_flow_doc').click();
    const sandboxDialog = page.getByRole('dialog', {
      name: 'Manage folders for /cib/e2e/folder-flow.md',
    });
    await expect(sandboxDialog).toContainText('Default folder (default)');
    await expect(sandboxDialog).toContainText('Sandbox (sandbox)');
  });

  test('@settings @folders runtime folder creation reaches switcher and activity', async ({
    page,
  }) => {
    await openTab(page, 'Settings');
    await page.getByTestId('settings-rail-folder').click();
    await page.getByTestId('settings-add-folder-btn').click();
    await page.getByTestId('settings-add-folder-id').fill('qa-flow');
    await page.getByTestId('settings-add-folder-label').fill('QA Flow');
    await page.getByTestId('settings-add-folder-kind').selectOption('project');
    await page.getByTestId('settings-add-folder-submit').click();

    await expect(page.getByTestId('settings-folder-row-qa-flow')).toContainText(
      'QA Flow',
    );

    await page.getByTitle('Switch folder').click();
    await page.getByTestId('topbar-folder-pick-qa-flow').click();
    await expect(page.getByTitle('Switch folder')).toContainText('qa-flow');

    await openTab(page, 'Activity');
    await page.getByLabel('Search events').fill('qa-flow');
    await expect(page.getByRole('heading', { name: 'QA Flow' })).toBeVisible();
    await expect(page.getByRole('complementary')).toContainText(
      "Folder 'qa-flow' created",
    );
  });
});
