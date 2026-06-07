import { expect, test, type Page } from '@playwright/test';
import { boot } from './helpers';

async function addDocumentTagFilter(page: Page, tag: string) {
  await page.getByRole('button', { name: '+ Add tag' }).click();
  await page.getByLabel('Add tag filter').fill(tag);
  await page.getByTestId(`docs-tag-sugg-${tag}`).click();
}

test.describe('Documents RC-1 persistence', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@documents @rc1 bulk retag is committed and survives reload', async ({ page }) => {
    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page.getByLabel('Bulk actions').getByRole('button', { name: /Retag 1 sources/ }).click();
    await expect(page.getByRole('dialog', { name: 'Retag document' })).toBeVisible();
    await page.getByRole('textbox', { name: 'Tag input' }).fill('memgraph');
    await page.getByTestId('sugg-memgraph').click();
    await page.getByRole('button', { name: 'Apply tag' }).click();
    await expect(page.getByRole('status')).toContainText('Tag memgraph applied');

    await page.getByRole('button', { name: 'Clear selection' }).click();
    await page.getByLabel('Search source').fill('oracle-restart-procedure');
    await addDocumentTagFilter(page, 'memgraph');
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await expect(page.getByLabel('Search source')).toHaveValue('oracle-restart-procedure');
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
  });

  test('@documents @rc1 bulk delete is committed and survives reload', async ({ page }) => {
    await page.getByLabel('Select memgraph-mage-3.8-release-notes.md').check();
    await page.getByTestId('docs-bulk-delete').click();
    await expect(page.getByRole('status')).toContainText('Confirm bulk delete');
    await page.getByTestId('docs-bulk-delete').click();
    await expect(page.getByRole('status')).toContainText('1 source deleted');

    await page.getByLabel('Search source').fill('memgraph-mage');
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await expect(page.getByLabel('Search source')).toHaveValue('memgraph-mage');
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();
  });

  test('@documents @rc1 edit-approve decision is committed and survives reload', async ({
    page,
  }) => {
    await expect(page.getByTestId('pending-doc-d6')).toContainText(
      'cft-vendor-api-spec-draft.pdf',
    );

    await page.getByTestId('pending-doc-edit-approve-d6').click();
    await expect(page.getByRole('dialog', { name: 'Edit & approve document' })).toBeVisible();
    await page
      .getByTestId('pending-doc-edit-summary')
      .fill('Vendor integration contract reviewed and approved by e2e.');
    await page.getByTestId('pending-doc-edit-tags').fill('cft, network, e2e');
    await page.getByTestId('pending-doc-edit-submit').click();
    await expect(page.getByRole('status')).toContainText('Document approved');
    await expect(page.getByTestId('pending-doc-d6')).toBeHidden();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await expect(page.getByTestId('pending-doc-d6')).toBeHidden();
    await page.getByLabel('Search source').fill('cft-vendor-api-spec');
    await expect(page.getByTestId('docs-row-d6')).toContainText('reviewed and approved');
  });

  test('@documents @rc1 reject decision is committed and survives reload', async ({ page }) => {
    await expect(page.getByTestId('pending-doc-d7')).toContainText(
      'incident-2026-Q2-postmortem-draft',
    );

    await page.getByTestId('pending-doc-reject-d7').click();
    await expect(page.getByRole('dialog', { name: 'Reject document' })).toBeVisible();
    await page.getByLabel('Rejection reason').fill('Needs legal review before indexing.');
    await page.getByTestId('pending-doc-reject-submit').click();
    await expect(page.getByRole('status')).toContainText('Document rejected');
    await expect(page.getByTestId('pending-doc-d7')).toBeHidden();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await expect(page.getByTestId('pending-doc-d7')).toBeHidden();
  });
});

test.describe('Documents RC-2 filters and counters', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@documents @rc2 counters follow the active search filter', async ({ page }) => {
    await page.getByLabel('Search source').fill('oracle');

    await expect(page.getByRole('button', { name: /^All \(1\)/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /^Completed \(1\)/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /^Failed \(0\)/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /^Processing \(0\)/ })).toBeVisible();
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();
  });

  test('@documents @rc2 status and tag filters are URL-backed and update rows', async ({
    page,
  }) => {
    await page.getByRole('button', { name: /^Failed \(1\)/ }).click();
    await expect(page).toHaveURL(/status=failed/);
    await expect(page.getByTestId('docs-row-d3')).toBeVisible();
    await expect(page.getByTestId('docs-row-d1')).toBeHidden();

    // Reset the status filter to All (the dedicated header Clear button
    // was removed — pills carry their own dismiss UX).
    await page.getByRole('button', { name: /^All/ }).click();
    await addDocumentTagFilter(page, 'rman');
    await expect(page).toHaveURL(/tag=rman/);
    await expect(page.getByRole('button', { name: /^All \(1\)/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /^Completed \(1\)/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /^Failed \(0\)/ })).toBeVisible();
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await expect(page.getByTestId('docs-row-d2')).toBeHidden();
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await expect(page).toHaveURL(/tag=rman/);
    await expect(page.getByRole('button', { name: /^All \(1\)/ })).toBeVisible();
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await expect(page.getByTestId('docs-row-d2')).toBeHidden();
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();
  });
});
