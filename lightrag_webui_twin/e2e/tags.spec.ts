import { expect, test } from '@playwright/test';
import { boot, openTab } from './helpers';

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
    await page.getByLabel(/Proposed name/).fill('golden-signal');
    await page
      .getByLabel(/Definition/)
      .fill('Operational signal used to triage reliability regressions.');
    await page.getByLabel('Domain').selectOption('infra');
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

  test('@doctrine @tags @rc1 rejected request leaves pending queue after reload', async ({
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
    await expect(page.getByTestId('tag-card-pacs008')).toContainText('pacs008');
    await expect(page.getByTestId('tag-card-pacs008')).toContainText('Rejected');
  });

  test('@doctrine @tags @rc1 edit-approve commits steward edits after reload', async ({
    page,
  }) => {
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
});
