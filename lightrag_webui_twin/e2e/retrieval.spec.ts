import { expect, test } from '@playwright/test';
import { boot, openTab } from './helpers';

test.describe('Retrieval citations', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openTab(page, 'Retrieval');
  });

  // Enabled 2026-06-26: the MSW /query/stream handler now maps source #1 to a
  // real seeded document (DOCUMENT_FIXTURES[0] = d1, oracle-restart-procedure.pdf),
  // which is the deliberate, single-source handler tune the prior TODO called
  // for. The drilldown therefore lands on a document that actually exists in the
  // Documents seed, so this asserts a true citation -> document linkage.
  test('@retrieval @rc4 clicking a source opens the referenced document in Documents', async ({
    page,
  }) => {
    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Query input').fill('What is the Oracle restart procedure?');
    await page.getByRole('button', { name: 'Send' }).click();

    // The first source is the real seeded document.
    const source = page.getByTestId('source-1');
    await expect(source).toBeVisible({ timeout: 20_000 });
    await expect(source).toContainText('oracle-restart-procedure.pdf');

    await source.click();

    // Drilldown navigates to Documents, scoped to that source (doc=d1).
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page).toHaveURL(/doc=d1/);
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
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
    // boot helper clears the threads localStorage key + reloads, and
    // App.tsx now passes `initialThreads={[]}`. The sidebar starts
    // empty — sending a query is what creates the first thread.
    await expect(page.locator('.history-item')).toHaveCount(0);

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
    await expect(page.locator('.history-item')).toHaveCount(1);
    await expect(
      page.getByRole('button', { name: /Open conversation What is the Oracle/ }),
    ).toBeVisible();
  });

  test('@retrieval @threads switching threads swaps the conversation pane', async ({
    page,
  }) => {
    // The sidebar starts empty (no seed threads) — create two threads
    // explicitly so the test owns its own setup rather than depending
    // on whatever the fixtures happened to provide.
    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Query input').fill('Thread switch probe one');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.retrieval-conv')).toContainText(
      'Thread switch probe one',
      { timeout: 20_000 },
    );

    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Query input').fill('Thread switch probe two');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.locator('.retrieval-conv')).toContainText(
      'Thread switch probe two',
      { timeout: 20_000 },
    );

    // Sanity: both threads landed in the sidebar.
    await expect(page.locator('.history-item')).toHaveCount(2);

    // Click thread "one" — pane must swap to its content (no longer
    // showing the "two" text we just emitted).
    await page
      .getByRole('button', { name: /Open conversation Thread switch probe one/ })
      .click();
    await expect(page.locator('.retrieval-conv')).toContainText(
      'Thread switch probe one',
    );
    await expect(page.locator('.retrieval-conv')).not.toContainText(
      'Thread switch probe two',
    );

    // Click "two" back — pane swaps again.
    await page
      .getByRole('button', { name: /Open conversation Thread switch probe two/ })
      .click();
    await expect(page.locator('.retrieval-conv')).toContainText(
      'Thread switch probe two',
    );
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

  // Removed: App.tsx now passes `suggestions={[]}` to RetrievalTab
  // (commit 1d1b0a0 "Remove runtime UI fixture fallbacks"), so the
  // empty-state chip rail is always empty in prod and demo alike.
  // The "click a chip → fire a query" affordance has nothing to
  // exercise. If product wants the suggestion rail back, it has to
  // own its own runtime data source first, then this test returns.
  // (The placeholder `test.skip` stub was dropped — this comment is the
  // tombstone; an empty skipped test is itself a lint smell.)

  // TR-RET-02 step 3 / audit C1: the "Tag filter chips" scenario was
  // dropped here because the affordance itself was removed — LightRAG
  // 1.4.x silently ignored the param and we now reject it 422 on
  // /query and /query/stream. The unit-test "does not render a Tag
  // filter affordance" in RetrievalTab.test.tsx is the replacement
  // guard; restoring this e2e would mean restoring the lie.
});
