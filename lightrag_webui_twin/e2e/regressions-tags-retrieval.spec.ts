import { expect, test, type Page } from '@playwright/test';
import { boot, getMswStats, openTab } from './helpers';

async function approveArgocd(page: Page) {
  await openTab(page, 'Tags');
  await page
    .getByTestId('pending-argocd')
    .getByRole('button', { name: 'Approve', exact: true })
    .click();
  await expect(page.getByTestId('pending-argocd')).toBeHidden();
  await page.getByLabel('Search tags').fill('argocd');
  await expect(page.getByTestId('tag-card-argocd')).toContainText('Active');
}

async function addDocumentsTagFilter(
  page: Page,
  tag: string,
) {
  await page.getByRole('button', { name: '+ Add tag' }).click();
  await page.getByLabel('Add tag filter').fill(tag);
  await page.getByTestId(`docs-tag-sugg-${tag}`).click();
}

test.describe('Regression guards: canonical tags and retrieval contracts', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@regression @tags @filters approved tag is usable across retag, document filters, graph filters, and retrieval filters', async ({
    page,
  }) => {
    await openTab(page, 'Retrieval');
    await page.getByLabel('Retrieval tag input').fill('argo');
    await expect(page.getByTestId('rtag-sugg-argocd')).toHaveCount(0);

    await approveArgocd(page);

    await openTab(page, 'Documents');
    await page.getByLabel('Search source').fill('');
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page
      .getByLabel('Bulk actions')
      .getByRole('button', { name: /Retag 1 sources/ })
      .click();
    await expect(page.getByRole('dialog', { name: 'Retag document' })).toBeVisible();
    await page.getByRole('textbox', { name: 'Tag input' }).fill('argo');
    await expect(page.getByTestId('sugg-argocd')).toContainText('argocd');
    await page.getByTestId('sugg-argocd').click();
    await page.getByRole('button', { name: 'Apply tag' }).click();
    await expect(page.getByRole('status')).toContainText('Tag argocd applied');
    await page.getByRole('button', { name: 'Clear selection' }).click();

    await page.getByLabel('Search source').fill('oracle-restart-procedure');
    await addDocumentsTagFilter(page, 'argocd');
    await expect(page).toHaveURL(/tag=argocd/);
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await expect(page.getByTestId('docs-row-d2')).toBeHidden();

    await openTab(page, 'Graph');
    await page.getByLabel('Filter by tag').fill('argo');
    await expect(page.getByTestId('kg-pick-argocd')).toHaveText('argocd');
    await page.getByTestId('kg-pick-argocd').click();
    await expect(page.getByTestId('kg-picked-argocd')).toBeVisible();
    await expect(page.getByTestId('kg-node-e_oracle')).toBeVisible();
    await expect(page.getByTestId('kg-node-e_memgraph')).toBeHidden();

    await openTab(page, 'Retrieval');
    await page.getByLabel('Retrieval tag input').fill('argocd');
    await expect(page.getByTestId('rtag-sugg-argocd')).toContainText('argocd');
  });

  test('@regression @retrieval sends tag filter and real thread history to the backend', async ({
    page,
  }) => {
    await openTab(page, 'Retrieval');
    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Retrieval tag input').fill('oracle');
    await page.getByTestId('rtag-sugg-oracle').click();
    await page.getByLabel('Top K', { exact: true }).fill('1');
    await page.getByLabel('History turns').fill('3');

    await page.getByLabel('Query input').fill('First retrieval history probe');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.retrieval-conv')).toContainText(
      'Mock retrieval response for: First retrieval history probe',
      { timeout: 20_000 },
    );

    await page.getByLabel('Query input').fill('Second retrieval history probe');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.retrieval-conv')).toContainText(
      'Mock retrieval response for: Second retrieval history probe',
      { timeout: 20_000 },
    );

    const stats = await getMswStats(page);
    const queryRequests = stats.queryRequests.filter((request) =>
      request.path.endsWith('/twin/api/query/stream'),
    );
    expect(queryRequests).toHaveLength(2);

    const firstBody = queryRequests[0].body;
    expect(firstBody).toMatchObject({
      query: 'First retrieval history probe',
      top_k: 1,
      history_turns: 3,
      tag_filter: { all: ['oracle'] },
      conversation_history: [],
    });

    const secondBody = queryRequests[1].body;
    expect(secondBody).toMatchObject({
      query: 'Second retrieval history probe',
      top_k: 1,
      history_turns: 3,
      tag_filter: { all: ['oracle'] },
    });
    expect(secondBody.conversation_history).toEqual([
      { role: 'user', content: 'First retrieval history probe' },
      {
        role: 'assistant',
        content: 'Mock retrieval response for: First retrieval history probe',
      },
    ]);
  });
});
