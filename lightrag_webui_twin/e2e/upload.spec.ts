import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import { boot } from './helpers';

test.describe('Add source validation', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@documents @upload @rc3 browse opens the native file chooser', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'Add source' }).click();
    const chooserPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Drop files or click to browse' }).click();
    const chooser = await chooserPromise;
    await chooser.setFiles({
      name: 'browse-opens.md',
      mimeType: 'text/markdown',
      buffer: Buffer.from('# Browse opens'),
    });
    await expect(page.getByText('browse-opens.md')).toBeVisible();
  });

  test('@documents @upload @rc3 unsupported and oversized files are visibly rejected', async ({
    page,
  }, testInfo) => {
    const unsupportedPath = testInfo.outputPath('unsupported.zip');
    const oversizedPath = testInfo.outputPath('oversized.pdf');
    await writeFile(unsupportedPath, Buffer.from('zip payload'));
    await writeFile(oversizedPath, Buffer.alloc(51 * 1024 * 1024));

    await page.getByRole('button', { name: 'Add source' }).click();
    await page.getByTestId('addsource-file-input').setInputFiles([
      unsupportedPath,
      oversizedPath,
    ]);

    await expect(page.getByText('unsupported.zip')).toBeVisible();
    await expect(page.getByText('unsupported type')).toBeVisible();
    await expect(page.getByText('oversized.pdf')).toBeVisible();
    await expect(page.getByText('Exceeds 50 MB')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Add 0 sources' })).toBeDisabled();
  });

  test('@documents @upload @rc3 mixed upload counts and submits valid files only', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'Add source' }).click();
    await page.getByTestId('addsource-file-input').setInputFiles([
      {
        name: 'valid-only.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# Valid only'),
      },
      {
        name: 'invalid-archive.zip',
        mimeType: 'application/zip',
        buffer: Buffer.from('zip payload'),
      },
    ]);

    await expect(page.getByText('invalid-archive.zip')).toBeVisible();
    await expect(page.getByText('unsupported type')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Add 1 source' })).toBeEnabled({
      timeout: 12_000,
    });
    await expect(page.getByRole('button', { name: 'Add 2 sources' })).toHaveCount(0);
    await page.getByRole('button', { name: 'Add 1 source' }).click();
    await expect(page.getByRole('status')).toContainText('Sources queued for ingestion');

    await page.getByLabel('Search source').fill('valid-only');
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText('valid-only.md');
    await page.getByLabel('Search source').fill('invalid-archive');
    await expect(page.getByTestId('docs-empty')).toBeVisible();
  });
});
