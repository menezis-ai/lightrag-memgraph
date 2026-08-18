import { expect, test } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
import { boot } from './helpers';

test.describe('Add source validation', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@documents @upload @classification operator MIP class travels on the upload request', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'Add source' }).click();
    await page.getByTestId('addsource-file-input').setInputFiles([
      {
        name: 'cft-faq.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# CFT FAQ'),
      },
    ]);
    await expect(page.getByText('cft-faq.md')).toBeVisible();
    // A .md carries no embedded MIP label — the operator sets the class here.
    await page
      .getByTestId('addsource-classification-cft-faq.md')
      .selectOption('C2');

    const submit = page.getByRole('button', { name: 'Add 1 source' });
    await expect(submit).toBeEnabled({ timeout: 15000 });

    const [request] = await Promise.all([
      page.waitForRequest(
        (r) =>
          new URL(r.url()).pathname.endsWith('/documents/upload') &&
          r.method() === 'POST',
      ),
      submit.click(),
    ]);
    // The operator choice is carried as the X-Twin-Classification header the
    // backend ingestion middleware reads (server-side floor policy applies).
    expect(request.headers()['x-twin-classification']).toBe('C2');
  });

  test('@documents @upload @classification bulk sensitivity is C1/C2 only and applies to every valid file', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'Add source' }).click();
    await page.getByTestId('addsource-file-input').setInputFiles([
      {
        name: 'bulk-one.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# Bulk one'),
      },
      {
        name: 'bulk-two.txt',
        mimeType: 'text/plain',
        buffer: Buffer.from('Bulk two'),
      },
    ]);

    const bulkSelect = page.getByLabel('Sensitivity for all files');
    await expect(bulkSelect.locator('option')).toHaveText([
      'no MIP',
      'C1 · Public',
      'C2 · Internal',
    ]);

    const perFileSelect = page.getByTestId('addsource-classification-bulk-one.md');
    await expect(perFileSelect.locator('option')).toHaveText([
      'no MIP',
      'C1 · Public',
      'C2 · Internal',
    ]);
    await expect(page.getByRole('option', { name: 'C3 · Confidential' })).toHaveCount(
      0,
    );
    await expect(page.getByRole('option', { name: 'C4 · Secret' })).toHaveCount(0);

    await bulkSelect.selectOption('C2');
    await expect(page.getByTestId('addsource-classification-bulk-one.md')).toHaveValue(
      'C2',
    );
    await expect(page.getByTestId('addsource-classification-bulk-two.txt')).toHaveValue(
      'C2',
    );

    const uploadHeaders: string[] = [];
    page.on('request', (request) => {
      if (
        new URL(request.url()).pathname.endsWith('/documents/upload') &&
        request.method() === 'POST'
      ) {
        uploadHeaders.push(request.headers()['x-twin-classification'] ?? '');
      }
    });

    await page.getByRole('button', { name: 'Add 2 sources' }).click();
    await expect.poll(() => uploadHeaders.length).toBe(2);
    expect(uploadHeaders).toEqual(['C2', 'C2']);
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
    await expect(page.getByText('ZIP format is not supported')).toBeVisible();
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
    await expect(page.getByText('ZIP format is not supported')).toBeVisible();
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
