import { expect, test } from '@playwright/test';
import { boot, openTab } from './helpers';

test.describe('Retrieval citations', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openTab(page, 'Retrieval');
  });

  test('@retrieval @rc4 clicking a citation opens the referenced source in Documents', async ({
    page,
  }) => {
    await expect(page.getByTestId('source-1')).toContainText(
      'oracle-restart-procedure.pdf',
    );

    await page.getByTestId('citation-1').click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page.getByLabel('Search source')).toHaveValue('oracle-restart-procedure.pdf');
    await expect(page.getByTestId('docs-row-d1')).toContainText(
      'oracle-restart-procedure.pdf',
    );
  });
});

test.describe('Retrieval threads and parameters', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openTab(page, 'Retrieval');
  });

  test('@retrieval @threads new conversation sends a query and lands in the sidebar', async ({
    page,
  }) => {
    await expect(page.locator('.history-item').first()).toBeVisible();
    const threadCount = await page.locator('.history-item').count();

    await page.getByRole('button', { name: /New/ }).click();
    await expect(
      page.getByText('Ask a question to retrieve from the knowledge base'),
    ).toBeVisible();

    await page.getByLabel('Query input').fill('What is the Oracle restart runbook?');
    await page.getByRole('button', { name: 'Send' }).click();

    await expect(page.locator('.retrieval-conv')).toContainText(
      'Mock retrieval response for: What is the Oracle restart runbook?',
      { timeout: 20_000 },
    );
    await expect(page.locator('.history-item')).toHaveCount(threadCount + 1);
    await expect(
      page.getByRole('button', { name: /Open conversation What is the Oracle/ }),
    ).toBeVisible();
  });

  test('@retrieval @threads switching threads swaps the conversation pane', async ({
    page,
  }) => {
    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Query input').fill('Thread switch probe');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.retrieval-conv')).toContainText(
      'Thread switch probe',
      { timeout: 20_000 },
    );

    const other = page
      .locator('.history-item')
      .filter({ hasNotText: 'Thread switch probe' })
      .first();
    await other.click();
    await expect(page.locator('.retrieval-conv')).not.toContainText(
      'Thread switch probe',
    );

    await page
      .getByRole('button', { name: /Open conversation Thread switch probe/ })
      .click();
    await expect(page.locator('.retrieval-conv')).toContainText('Thread switch probe');
  });

  test('@retrieval @threads deleting a thread removes it from the sidebar', async ({
    page,
  }) => {
    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Query input').fill('Thread to delete');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.retrieval-conv')).toContainText('Thread to delete', {
      timeout: 20_000,
    });

    await page.getByLabel(/Delete Thread to delete/).click();
    await expect(
      page.getByRole('button', { name: /Open conversation Thread to delete/ }),
    ).toHaveCount(0);
  });

  test('@retrieval @params top-k drives the number of returned sources', async ({
    page,
  }) => {
    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Top K', { exact: true }).fill('1');
    await page.getByLabel('Query input').fill('Single source probe');
    await page.getByRole('button', { name: 'Send' }).click();

    await expect(page.getByTestId('source-1')).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId('source-2')).toHaveCount(0);
  });

  test('@retrieval @params empty-state suggestion chips fire a query', async ({
    page,
  }) => {
    await page.getByRole('button', { name: /New/ }).click();
    await page.locator('.suggestion').first().click();
    await expect(page.locator('.retrieval-conv')).toContainText(
      'Mock retrieval response for:',
      { timeout: 20_000 },
    );
  });

  test('@retrieval @params tag filter chips add and remove from the thesaurus', async ({
    page,
  }) => {
    await page.getByLabel('Retrieval tag input').fill('oracle');
    await page.getByTestId('rtag-sugg-oracle').click();
    await expect(page.getByLabel('Remove oracle')).toBeVisible();

    await page.getByLabel('Remove oracle').click();
    await expect(page.getByLabel('Remove oracle')).toHaveCount(0);
  });
});
