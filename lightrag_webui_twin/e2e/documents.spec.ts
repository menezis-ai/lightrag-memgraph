import { expect, test, type Page } from '@playwright/test';
import { boot, openTab, setMswScenario } from './helpers';

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
    await page.getByRole('combobox', { name: 'Tag input' }).fill('memgraph');
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

  test('@documents @rc1 deleting state stays visible during the round-trip and the Graph tab refetches', async ({
    page,
  }) => {
    // Force a 1.5 s delay on the bulk-delete handler so the optimistic
    // "deleting…" badge is observably present before the row vanishes.
    await setMswScenario(page, { bulkDeleteDelayMs: 1500 });

    await page.getByLabel('Select memgraph-mage-3.8-release-notes.md').check();
    await page.getByTestId('docs-bulk-delete').click();
    await expect(page.getByRole('status')).toContainText('Confirm bulk delete');
    await page.getByTestId('docs-bulk-delete').click();

    // While the round-trip is in flight, the doc row stays + status is
    // the UI-only "deleting…" badge (not PROCESSING / not gone).
    const row = page.getByTestId('docs-row-d4');
    await expect(row).toBeVisible();
    await expect(
      row.locator('[data-testid="status-deleting"]'),
    ).toHaveText(/deleting/i);

    // After the cascade resolves the row disappears and the toast lands.
    await expect(row).toBeHidden({ timeout: 5000 });
    await expect(page.getByRole('status')).toContainText('1 source deleted');

    // Navigate to the Graph tab — the entities sourced ONLY by d4 must
    // be gone from the refetched graph (cascade invalidation of the
    // ['graph-entities'] query key, not stale fixture data).
    await openTab(page, 'Graph');
    await expect(page.getByTestId('kg-node-e_memgraph')).toBeHidden();
    await expect(page.getByTestId('kg-node-e_mage')).toBeHidden();
    await expect(page.getByTestId('kg-node-e_lightrag')).toBeHidden();
    await expect(page.getByTestId('kg-node-e_cypher')).toBeHidden();
  });

  test('@documents @rc1 deleting the selected graph node leaves the inspector empty (no auto-fallback)', async ({
    page,
  }) => {
    // First, in the Graph tab, select e_memgraph explicitly.
    await openTab(page, 'Graph');
    await page.getByTestId('kg-node-e_memgraph').click();
    await expect(page.getByTestId('kg-detail-entity')).toContainText('Memgraph');

    // Now go delete d4 (the only doc sourcing e_memgraph). After cascade
    // the inspector should NOT silently re-select another node — it
    // should surface the empty state.
    await openTab(page, 'Documents');
    await page.getByLabel('Select memgraph-mage-3.8-release-notes.md').check();
    await page.getByTestId('docs-bulk-delete').click();
    await page.getByTestId('docs-bulk-delete').click(); // confirm
    await expect(page.getByTestId('docs-row-d4')).toBeHidden({ timeout: 5000 });

    await openTab(page, 'Graph');
    await expect(page.getByText('Select a node to inspect')).toBeVisible();
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
    // Real contract (audit 2026-07-02, DUP-2d): the approval persists as
    // review.state='approved' (pending card stays gone after reload) and the
    // operator edits are recorded in metadata.review.edits — the backend
    // does NOT merge them into the document fields, so the row keeps its
    // ORIGINAL summary. Asserting the edited text here would validate an
    // imaginary backend.
    await expect(page.getByTestId('docs-row-d6')).toContainText(
      'cft-vendor-api-spec-draft.pdf',
    );
    await expect(page.getByTestId('docs-row-d6')).not.toContainText(
      'reviewed and approved',
    );
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

  test('@documents @rc2 counters keep total while match count follows search filter', async ({ page }) => {
    await page.getByLabel('Search source').fill('oracle');

    await expect(page.getByRole('button', { name: /^All \(4\)/ })).toBeVisible();
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
    await expect(page.getByRole('button', { name: /^All \(4\)/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /^Completed \(1\)/ })).toBeVisible();
    await expect(page.getByRole('button', { name: /^Failed \(0\)/ })).toBeVisible();
    await expect(page.getByText('1 document match')).toBeVisible();
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await expect(page.getByTestId('docs-row-d2')).toBeHidden();
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await expect(page).toHaveURL(/tag=rman/);
    await expect(page.getByRole('button', { name: /^All \(4\)/ })).toBeVisible();
    await expect(page.getByText('1 document match')).toBeVisible();
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await expect(page.getByTestId('docs-row-d2')).toBeHidden();
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();
  });
});

test.describe('Documents classification badge (MIP)', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@documents @doctrine @classification C2 shield renders for classified docs and is silent otherwise', async ({
    page,
  }) => {
    // d1 carries a STRUCTURED MIP classification (class_id "C2",
    // class_name "C2 Confidentiel") → the read-only ClassPill shield renders.
    // C2 maps to the "internal" tone, so getMipDisplayName() shows "Internal"
    // and the aria-label combines it with the raw class name.
    const classifiedRow = page.getByTestId('docs-row-d1');
    await expect(classifiedRow).toBeVisible();

    // Scope the pill query to the row — class-pill aria-labels would collide
    // across rows if queried page-wide.
    const pill = classifiedRow.getByTestId('class-pill-d1');
    await expect(pill).toBeVisible();
    await expect(pill).toHaveAttribute(
      'aria-label',
      'Classification: Internal · C2 Confidentiel',
    );
    await expect(pill).toHaveAttribute('data-class-id', 'C2');
    await expect(pill).toHaveAttribute('data-class-tone', 'internal');
    // The visible label text is the display name only ("Internal").
    await expect(pill.locator('.class-pill-label')).toHaveText('Internal');

    // d5 carries a LEGACY string classification ("restricted"), which is NOT
    // a structured payload → the pill is silent (renders nothing).
    const legacyRow = page.getByTestId('docs-row-d5');
    await expect(legacyRow).toBeVisible();
    await expect(legacyRow.getByTestId('class-pill-d5')).toHaveCount(0);

    // d3 has no classification at all → also silent (pure-absence case).
    const unclassifiedRow = page.getByTestId('docs-row-d3');
    await expect(unclassifiedRow).toBeVisible();
    await expect(unclassifiedRow.getByTestId('class-pill-d3')).toHaveCount(0);
  });
});
