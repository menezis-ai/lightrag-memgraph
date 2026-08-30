import { allowRequestAbort, expect, test, type Page } from './fixtures';
import { boot, openTab } from './helpers';

async function expectTagsViewportIsContained(page: Page) {
  const metrics = await page.evaluate(() => {
    const doc = document.documentElement;
    const body = document.body;
    const selectors = [
      '.tags-screen',
      '.tags-header',
      '.tags-header-actions',
      '.pending-section',
      '.tags-filters',
      '.tags-body',
      '.tags-rail',
      '.tags-grid-wrap',
      '.tag-detail',
    ];
    const viewportWidth = window.innerWidth;
    const offenders = selectors.flatMap((selector) =>
      Array.from(document.querySelectorAll<HTMLElement>(selector))
        .filter((el) => {
          const style = window.getComputedStyle(el);
          const rect = el.getBoundingClientRect();
          return style.display !== 'none' && rect.width > 0 && rect.height > 0;
        })
        .map((el) => {
          const rect = el.getBoundingClientRect();
          return {
            selector,
            left: Math.floor(rect.left),
            right: Math.ceil(rect.right),
            width: Math.ceil(rect.width),
          };
        })
        .filter((rect) => rect.left < -1 || rect.right > viewportWidth + 1),
    );
    return {
      viewportWidth,
      documentScrollWidth: Math.ceil(doc.scrollWidth),
      documentClientWidth: Math.floor(doc.clientWidth),
      bodyScrollWidth: Math.ceil(body.scrollWidth),
      bodyClientWidth: Math.floor(body.clientWidth),
      offenders,
    };
  });

  expect(metrics.documentScrollWidth, JSON.stringify(metrics)).toBeLessThanOrEqual(
    metrics.documentClientWidth + 1,
  );
  expect(metrics.bodyScrollWidth, JSON.stringify(metrics)).toBeLessThanOrEqual(
    metrics.bodyClientWidth + 1,
  );
  expect(metrics.offenders, JSON.stringify(metrics)).toEqual([]);
}

test.describe('Twin WebUI tag governance persistence', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openTab(page, 'Tags');
  });

  test('@doctrine @tags @rc1 tag edit remains visible after reload', async ({ page }) => {
    await page.getByLabel('Search tags').fill('oracle');
    await page.getByTestId('tag-card-oracle').click();
    await expect(page.locator('.tag-detail-name')).toHaveText('oracle');

    await page.getByRole('button', { name: 'Edit', exact: true }).click();
    await expect(page.getByRole('dialog', { name: 'Edit tag' })).toBeVisible();
    await page
      .getByLabel('Short definition')
      .fill('Oracle database operations and recovery runbooks edited by e2e.');
    await page.getByRole('button', { name: 'Save' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('oracle');
    await expect(page.locator('.toast-viewport')).toContainText('updated');
    await expect(page.getByTestId('tag-card-oracle')).toContainText('edited by e2e');

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('oracle');
    await expect(page.getByTestId('tag-card-oracle')).toContainText('edited by e2e');
  });

  test('@doctrine @tags @rc1 requested tag remains in pending queue after reload', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'Request new tag' }).click();
    const requestDialog = page.getByRole('dialog', { name: /Request new tag/ });
    await page.getByLabel(/Proposed name/).fill('golden-signal');
    await page
      .getByLabel(/Definition/)
      .fill('Operational signal used to triage reliability regressions.');
    await requestDialog.getByLabel('Domain', { exact: true }).selectOption('infra');
    await page
      .getByLabel('Justification')
      .fill('Required for SLO triage runbooks.');
    await page.getByRole('button', { name: 'Submit request' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('golden-signal');
    await expect(page.getByTestId('pending-golden-signal')).toBeVisible();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Tags');
    await expect(page.getByTestId('pending-golden-signal')).toContainText('golden-signal');
  });

  test('@doctrine @tags @rc1 rejected request is purged and can be requested again after reload', async ({
    page,
  }) => {
    await page
      .getByTestId('pending-pacs008')
      .getByRole('button', { name: 'Reject', exact: true })
      .click();
    await expect(page.getByRole('dialog', { name: /Reject request/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Reject request' })).toBeDisabled();
    await page.getByLabel('Reason').fill('Covered by iso20022 taxonomy for now.');
    await page.getByRole('button', { name: 'Reject request' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('pacs008');
    await expect(page.getByTestId('pending-pacs008')).toBeHidden();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Tags');
    await expect(page.getByTestId('pending-pacs008')).toBeHidden();
    await page.getByLabel('Search tags').fill('pacs008');
    await expect(page.getByTestId('tag-card-pacs008')).toHaveCount(0);

    await page.getByRole('button', { name: 'Request new tag' }).click();
    const requestDialog = page.getByRole('dialog', { name: /Request new tag/ });
    await page.getByLabel(/Proposed name/).fill('pacs008');
    await page
      .getByLabel(/Definition/)
      .fill('ISO 20022 payment message tag requested again after rejection.');
    await requestDialog.getByLabel('Domain', { exact: true }).selectOption('infra');
    await page
      .getByLabel('Justification')
      .fill('Required for payment operations retrieval.');
    await page.getByRole('button', { name: 'Submit request' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('pacs008');
    await expect(page.getByTestId('pending-pacs008')).toContainText('pacs008');
  });

  test('@doctrine @tags @rc1 edit-approve commits steward edits after reload', async ({
    allowBrowserIssues,
    page,
  }) => {
    const approvalReloadReason =
      'The explicit persistence reload replaces the just-invalidated tag and notification queries.';
    allowBrowserIssues(
      allowRequestAbort(/\/twin\/api\/tags$/, approvalReloadReason),
      allowRequestAbort(/\/twin\/api\/notifications$/, approvalReloadReason),
    );
    await page
      .getByTestId('pending-argocd')
      .getByRole('button', { name: 'Edit & approve' })
      .click();
    await expect(page.getByRole('dialog', { name: 'Edit & approve request' })).toBeVisible();
    await page
      .getByLabel('Short definition')
      .fill('Argo CD runtime deployment controller approved with steward edits.');
    await page.getByRole('button', { name: 'Approve with edits' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('approved (edited)');
    await expect(page.getByTestId('pending-argocd')).toBeHidden();

    await page.getByLabel('Search tags').fill('argocd');
    await expect(page.getByTestId('tag-card-argocd')).toContainText(
      'steward edits',
    );

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('argocd');
    await expect(page.getByTestId('tag-card-argocd')).toContainText(
      'steward edits',
    );
    await expect(page.getByTestId('pending-argocd')).toBeHidden();
  });

  test('@doctrine @tags @rc1 synonym update remains visible after reload', async ({
    page,
  }) => {
    await page.getByLabel('Search tags').fill('memgraph');
    await page.getByTestId('tag-card-memgraph').click();
    await expect(page.locator('.tag-detail-name')).toHaveText('memgraph');

    await page.getByRole('button', { name: 'Manage synonyms' }).click();
    await expect(page.getByRole('dialog', { name: /Manage synonyms/ })).toBeVisible();
    await page.getByLabel('Add synonym').fill('graphdb');
    await page.getByRole('button', { name: 'Save synonyms' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('synonyms updated');
    await expect(page.getByTestId('tag-card-memgraph')).toContainText('graphdb');

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('memgraph');
    await expect(page.getByTestId('tag-card-memgraph')).toContainText('graphdb');
  });

  test('@doctrine @tags @rc1 delete migration removes tag after reload', async ({
    page,
  }) => {
    await page.getByLabel('Search tags').fill('ansible');
    await page.getByTestId('tag-card-ansible').click();
    await expect(page.locator('.tag-detail-name')).toHaveText('ansible');

    await page.getByRole('button', { name: 'Deprecate' }).click();
    await expect(page.getByRole('dialog', { name: /Deprecate tag/ })).toBeVisible();
    await page
      .getByRole('dialog', { name: /Deprecate tag/ })
      .getByRole('button', { name: 'Deprecate' })
      .click();
    await expect(page.locator('.toast-viewport')).toContainText('deprecated');

    await page.getByRole('button', { name: 'Delete' }).click();
    await expect(page.getByRole('dialog', { name: /Delete tag/ })).toBeVisible();
    await page.getByLabel('Replacement tag').selectOption('memgraph');
    await expect(page.getByRole('button', { name: 'Migrate and delete' })).toBeEnabled();
    await page.getByRole('button', { name: 'Migrate and delete' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('migrated to memgraph');
    await expect(page.getByTestId('tag-card-ansible')).toBeHidden();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('ansible');
    await expect(page.getByTestId('tag-card-ansible')).toBeHidden();
  });

  test('@tags @dark deprecate modal keeps tag identity readable', async ({
    page,
  }) => {
    await page.getByLabel('Theme').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');

    await page.getByLabel('Search tags').fill('ansible');
    await page.getByTestId('tag-card-ansible').click();
    await page.getByRole('button', { name: 'Deprecate' }).click();

    const dialog = page.getByRole('dialog', { name: /Deprecate tag/ });
    await expect(dialog).toBeVisible();
    const tagCode = dialog.locator('.modal-h-sub code');
    const statusBadge = dialog.locator('.status-badge');
    await expect(tagCode).toHaveText('ansible');
    await expect(statusBadge).toBeVisible();

    const styleSnapshot = await tagCode.evaluate((el) => {
      const code = window.getComputedStyle(el);
      const modal = window.getComputedStyle(el.closest('.modal') as Element);
      return {
        color: code.color,
        backgroundColor: code.backgroundColor,
        modalBackgroundColor: modal.backgroundColor,
      };
    });
    expect(styleSnapshot.color).not.toBe('rgba(0, 0, 0, 0)');
    expect(styleSnapshot.color).not.toBe(styleSnapshot.backgroundColor);
    expect(styleSnapshot.color).not.toBe(styleSnapshot.modalBackgroundColor);
  });
});

test.describe('Twin WebUI tag governance responsive layout', () => {
  for (const viewport of [
    { name: 'mobile', width: 390, height: 760 },
    { name: 'tablet', width: 820, height: 900 },
  ] as const) {
    test(`@tags @responsive ${viewport.name} stays contained and actions remain reachable`, async ({
      page,
    }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await boot(page);
      await openTab(page, 'Tags');

      await expect(page.getByRole('heading', { name: 'Tags' })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Request new tag' })).toBeVisible();
      await expect(page.getByLabel('Search tags')).toBeVisible();
      await expect(page.getByTestId('rail-all')).toBeVisible();
      await expect(page.getByTestId('tag-card-oracle')).toBeVisible();

      await expectTagsViewportIsContained(page);

      await page.getByTestId('tag-card-oracle').click();
      await page.getByRole('button', { name: 'Edit', exact: true }).scrollIntoViewIfNeeded();
      await expect(page.getByRole('button', { name: 'Edit', exact: true })).toBeVisible();
      await expect(page.getByRole('button', { name: 'Manage synonyms' })).toBeVisible();

      await expectTagsViewportIsContained(page);
    });
  }
});
