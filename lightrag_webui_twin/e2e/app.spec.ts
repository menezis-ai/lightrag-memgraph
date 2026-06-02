import { expect, test, type Page } from '@playwright/test';

async function boot(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.setItem(
      'twin.onboarding.v1',
      JSON.stringify({ step: 'completion', dismissed: true, tasks: [] }),
    );
    window.localStorage.removeItem('twin-rag.threads.v2');
  });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await fetch('/__e2e/reset', { method: 'POST' });
  });
  await page.reload();
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
}

async function openTab(page: Page, name: string) {
  await page.getByRole('button', { name, exact: true }).click();
}

async function setMswScenario(page: Page, scenario: Record<string, unknown>) {
  await page.evaluate(async (body) => {
    await fetch('/__e2e/scenario', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  }, scenario);
}

async function getMswStats(page: Page): Promise<{ approveCalls: Record<string, number> }> {
  return page.evaluate(async () => {
    const res = await fetch('/__e2e/stats');
    return res.json();
  });
}

test.describe('Twin WebUI operator journeys', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('documents: filter, add source, retag and delete flows stay wired', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page.getByTestId('pending-docs-section')).toContainText(
      'documents awaiting your sign-off',
    );

    await page.getByRole('button', { name: /Failed/ }).click();
    await expect(page.getByTestId('docs-row-d3')).toContainText('huge-archive.zip');
    await page.getByLabel('Search source').fill('oracle');
    await expect(page.getByTestId('docs-empty')).toBeVisible();
    await page.getByRole('button', { name: /Clear/ }).click();

    await page.getByRole('button', { name: 'Add source' }).click();
    await expect(page.getByRole('dialog', { name: /Add source/ })).toBeVisible();
    await page.getByTestId('addsource-file-input').setInputFiles({
      name: 'runbook-smoke.md',
      mimeType: 'text/markdown',
      buffer: Buffer.from('# Smoke\nE2E upload fixture'),
    });
    await expect(page.getByText('runbook-smoke.md')).toBeVisible();
    await page.getByLabel('Tag input').fill('oracle');
    await page.getByTestId('tag-sugg-oracle').click();
    await expect(page.getByRole('button', { name: 'Add 0 sources' })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Add 1 source' })).toBeEnabled({
      timeout: 12_000,
    });
    await page.getByRole('button', { name: 'Add 1 source' }).click();
    await expect(page.getByRole('status')).toContainText('Sources queued for ingestion');

    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page.getByLabel('Bulk actions').getByRole('button', { name: /Retag 1 sources/ }).click();
    await expect(page.getByRole('dialog', { name: 'Retag document' })).toBeVisible();
    await page.getByRole('textbox', { name: 'Tag input' }).fill('memgraph');
    await page.getByTestId('sugg-memgraph').click();
    await expect(page.getByTestId('preview-impact')).toContainText('Adding');
    await page.getByRole('button', { name: 'Apply tag' }).click();
    await expect(page.getByRole('status')).toContainText('Tag memgraph applied');

    await page.getByRole('button', { name: 'Clear selection' }).click();
    await page.getByLabel('Select memgraph-mage-3.8-release-notes.md').check();
    await page.getByTestId('docs-bulk-delete').click();
    await expect(page.getByRole('status')).toContainText('Confirm bulk delete');
    await page.getByTestId('docs-bulk-delete').click();
    await expect(page.getByRole('status')).toContainText('1 source deleted');
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();
  });

  test('topbar, tags, retrieval, graph and activity cross-navigation work', async ({ page }) => {
    await page.getByRole('button', { name: /Notifications/ }).click();
    await expect(page.getByText('Mark all read')).toBeVisible();
    await page.getByText('Mark all read').click();
    await expect(page.getByRole('button', { name: 'Notifications' })).toBeVisible();

    await page.getByLabel('Theme').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');

    await page.getByTitle('Switch workspace').click();
    await page.getByRole('menuitemradio', { name: /infra/ }).click();
    await expect(page.getByTitle('Switch workspace')).toContainText('infra');

    await openTab(page, 'Tags');
    await expect(page.getByRole('heading', { name: 'Tags' })).toBeVisible();
    await page.getByLabel('Search tags').fill('oracle');
    await expect(page.getByTestId('tag-card-oracle')).toBeVisible();
    await page.getByTestId('rail-network').click();
    await expect(page.getByTestId('tags-empty-filtered')).toBeVisible();
    await page.getByRole('button', { name: /Clear filters/ }).click();
    await page
      .getByTestId('pending-argocd')
      .getByRole('button', { name: 'Approve', exact: true })
      .click();
    await expect(page.getByRole('status')).toContainText('Tag argocd approved');

    await openTab(page, 'Retrieval');
    await page.getByLabel('Query input').fill('How do I restart Oracle?');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.getByText('How do I restart Oracle?')).toBeVisible();
    await expect(page.getByTestId('source-1')).toBeVisible({ timeout: 12_000 });
    await page.getByLabel('Retrieval tag input').fill('oracle');
    await page.getByTestId('rtag-sugg-oracle').click();
    await page.getByLabel('Only need context').click();
    await expect(page.getByLabel('Only need context')).toHaveAttribute('aria-checked', 'true');

    await openTab(page, 'Graph');
    await expect(page.getByRole('heading', { name: 'Knowledge Graph' })).toBeVisible();
    await page.getByLabel('Search entities').fill('Oracle');
    await page.getByLabel(/Select entity Oracle/).click();
    await page.getByLabel('Zoom in').click();
    await expect(page.getByTestId('kg-zoom-value')).not.toHaveText('100%');
    await page.getByRole('button', { name: /View .* sources mentioning this entity/ }).click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page.getByLabel('Search source')).toHaveValue(/Oracle/i);

    await openTab(page, 'Activity');
    await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible();
    await page.getByLabel('Search events').fill('huge-archive');
    await expect(page.getByRole('heading', { name: 'huge-archive.zip' })).toBeVisible();
    await page.getByRole('button', { name: /Replay ingestion/ }).click();
    await expect(page.getByRole('status')).toContainText('Replay queued');
  });

  test('reviewer queue: read extracted text, edit-approve and reject pending sources', async ({
    page,
  }) => {
    await expect(page.getByTestId('pending-doc-d6')).toContainText(
      'cft-vendor-api-spec-draft.pdf',
    );

    await page.getByTestId('pending-doc-read-d6').click();
    await expect(page.getByRole('dialog', { name: 'Extracted text' })).toBeVisible();
    await expect(page.getByText('CFT Vendor API specification')).toBeVisible();
    await expect(page.getByTestId('rs-pill-pending')).toContainText(
      'awaiting reviewer sign-off',
    );
    await page.getByRole('dialog', { name: 'Extracted text' }).getByLabel('Close').click();

    await page.getByTestId('pending-doc-edit-approve-d6').click();
    await expect(
      page.getByRole('dialog', { name: 'Edit & approve document' }),
    ).toBeVisible();
    await page.getByTestId('pending-doc-edit-summary').fill(
      'Vendor integration contract reviewed by e2e before retrieval exposure.',
    );
    await page.getByTestId('pending-doc-edit-tags').fill('cft, network, e2e');
    await page.getByTestId('pending-doc-edit-submit').click();
    await expect(page.getByRole('status')).toContainText('Document approved');

    await page.getByTestId('pending-doc-reject-d7').click();
    await expect(page.getByRole('dialog', { name: 'Reject document' })).toBeVisible();
    await expect(page.getByTestId('pending-doc-reject-submit')).toBeDisabled();
    await page.getByLabel('Rejection reason').fill('Needs legal review before indexing.');
    await expect(page.getByTestId('pending-doc-reject-submit')).toBeEnabled();
    await page.getByTestId('pending-doc-reject-submit').click();
    await expect(page.getByRole('status')).toContainText('Document rejected');
  });

  test('document detail panel exposes chunks, lineage, audit and gated raw notice', async ({
    page,
  }) => {
    await page.getByTestId('docs-row-delete-d3').click();
    await expect(page.getByTestId('doc-detail-panel')).toBeVisible();
    await expect(page.getByRole('dialog', { name: /Detail: huge-archive.zip/ })).toBeVisible();
    await expect(page.getByTestId('doc-detail-chunks-empty')).toBeVisible();

    await page.getByTestId('doc-detail-tab-lineage').click();
    await expect(page.getByTestId('doc-detail-lineage')).toContainText('tk_2026-05-29_002');
    await expect(page.getByTestId('doc-detail-lineage')).toContainText('FAILED');

    await page.getByTestId('doc-detail-tab-audit').click();
    await expect(page.getByTestId('doc-detail-audit-empty')).toBeVisible();

    await page.getByTestId('doc-detail-view-raw').click();
    await expect(page.getByRole('dialog', { name: 'View raw notice' })).toBeVisible();
    await expect(page.getByRole('dialog', { name: 'View raw notice' })).toContainText(
      'Source classification',
    );
    await page
      .getByRole('dialog', { name: 'View raw notice' })
      .getByRole('button', { name: 'Close', exact: true })
      .last()
      .click();

    await page.getByTestId('doc-detail-reprocess').click();
    await expect(page.getByRole('status')).toContainText('Re-process queued');
  });

  test('settings profile, workspace and API explorer remain usable', async ({ page }) => {
    await openTab(page, 'Settings');
    await expect(page.getByTestId('settings-profile')).toBeVisible();
    await expect(page.getByTestId('settings-profile-name')).toBeVisible();

    await page.getByTestId('settings-restart-tutorial').click();
    await expect(page.getByRole('status')).toContainText('Tutorial restarted');

    await page.getByTestId('settings-rail-workspace').click();
    await expect(page.getByTestId('settings-workspace')).toBeVisible();
    await expect(page.getByTestId('settings-active-ws')).toContainText('cib');
    await expect(page.getByText('Retention policy')).toBeVisible();

    await page.getByTestId('settings-rail-api').click();
    await expect(page.getByTestId('settings-api')).toBeVisible();
    await page.getByLabel('Filter endpoints').fill('label');
    await expect(page.getByTestId('endpoint-GET-/graph/label/list')).toBeVisible();
    await page.getByTestId('endpoint-GET-/graph/label/list').getByRole('button').first().click();
    await page.getByRole('button', { name: 'Try it out' }).click();
    await page.getByRole('button', { name: /Execute/ }).click();
    await expect(page.getByText('Unauthorized')).toBeVisible();

    await page.getByRole('button', { name: 'Authorize' }).click();
    await page.getByLabel('Value').fill('e2e-token');
    await page.getByRole('dialog', { name: 'Authorize' }).getByRole('button', { name: 'Authorize' }).click();
    await expect(page.getByRole('button', { name: 'Authorized' })).toBeVisible();
  });

  test('tags governance: request and reject review workflows are enforced', async ({ page }) => {
    await openTab(page, 'Tags');
    await page.getByRole('button', { name: 'Request new tag' }).click();
    await expect(page.getByRole('dialog', { name: /Request new tag/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Submit request' })).toBeDisabled();
    await page.getByLabel(/Proposed name/).fill('golden-signal');
    await page.getByLabel(/Definition/).fill('Operational telemetry signal used for SLO triage.');
    await page.getByLabel('Domain').selectOption('infra');
    await page.getByLabel(/Synonyms/).fill('sli, service-level-indicator');
    await page.getByLabel('Justification').fill(
      'Needed to classify runbooks that describe SLO-driven incident triage.',
    );
    await expect(page.getByRole('button', { name: 'Submit request' })).toBeEnabled();
    await page.getByRole('button', { name: 'Submit request' }).click();
    await expect(page.getByRole('status')).toContainText(
      'Tag golden-signal requested for review',
    );

    await page
      .getByTestId('pending-pacs008')
      .getByRole('button', { name: 'Reject', exact: true })
      .click();
    await expect(page.getByRole('dialog', { name: /Reject request/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Reject request' })).toBeDisabled();
    await page.getByLabel('Reason').fill('Covered by iso20022 taxonomy for now.');
    await expect(page.getByRole('button', { name: 'Reject request' })).toBeEnabled();
    await page.getByRole('button', { name: 'Reject request' }).click();
    await expect(page.getByRole('status')).toContainText('Tag pacs008 rejected');
  });

  test('tags governance: synonyms, deprecate and delete migration actions persist', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('memgraph');
    await page.getByTestId('tag-card-memgraph').click();
    await expect(page.locator('.tag-detail-name')).toHaveText('memgraph');

    await page.getByRole('button', { name: 'Manage synonyms' }).click();
    await expect(page.getByRole('dialog', { name: /Manage synonyms/ })).toBeVisible();
    await page.getByLabel('Add synonym').fill('graphdb');
    await page.getByRole('button', { name: 'Save synonyms' }).click();
    await expect(page.getByRole('status')).toContainText('Tag memgraph synonyms updated');

    await page.getByLabel('Search tags').fill('ansible');
    await page.getByTestId('tag-card-ansible').click();
    await expect(page.locator('.tag-detail-name')).toHaveText('ansible');
    await page.getByRole('button', { name: 'Deprecate' }).click();
    await expect(page.getByRole('dialog', { name: /Deprecate tag/ })).toBeVisible();
    await page
      .getByRole('dialog', { name: /Deprecate tag/ })
      .getByRole('button', { name: 'Deprecate' })
      .click();
    await expect(page.getByRole('status')).toContainText('Tag ansible deprecated');

    await page.getByRole('button', { name: 'Delete' }).click();
    await expect(page.getByRole('dialog', { name: /Delete tag/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Migrate and delete' })).toBeDisabled();
    await page.getByLabel('Replacement tag').selectOption('memgraph');
    await expect(page.getByRole('button', { name: 'Migrate and delete' })).toBeEnabled();
    await page.getByRole('button', { name: 'Migrate and delete' }).click();
    await expect(page.getByRole('status')).toContainText('Tag ansible migrated to memgraph');
  });

  test('tags discovery: filters, related tags and document drill-down keep URL state wired', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    await page.getByTestId('rail-oracle').click();
    await expect(page.getByTestId('tag-card-oracle')).toBeVisible();
    await expect(page).toHaveURL(/cat=oracle/);

    await page.getByTestId('tag-card-rman').click();
    await page.getByTestId('related-oracle').click();
    await expect(page.locator('.tag-detail-name')).toHaveText('oracle');

    await page.getByRole('button', { name: /View all .* docs in Documents/ }).click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page).toHaveURL(/tag=oracle/);
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
  });

  test('doctrine: bulk retag backend failure rolls back optimistic tags', async ({ page }) => {
    await setMswScenario(page, { bulkRetagStatus: 500 });
    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page.getByLabel('Bulk actions').getByRole('button', { name: /Retag 1 sources/ }).click();
    await page.getByRole('textbox', { name: 'Tag input' }).fill('memgraph');
    await page.getByTestId('sugg-memgraph').click();
    await page.getByRole('button', { name: 'Apply tag' }).click();
    await expect(page.getByRole('alert')).toContainText('Tag mutation failed');
    await expect(page.getByTestId('docs-row-d1')).not.toContainText('memgraph');
  });

  test('doctrine: double-click approve sends one mutation for a pending doc', async ({ page }) => {
    await setMswScenario(page, { approveDelayMs: 600 });
    const approve = page.getByTestId('pending-doc-approve-d6');
    await approve.click();
    await expect(approve).toBeDisabled();
    await approve.click({ timeout: 200 }).catch(() => undefined);
    await expect(page.getByRole('status')).toContainText('Document approved');
    const stats = await getMswStats(page);
    expect(stats.approveCalls.d6).toBe(1);
  });

  test('doctrine: upload with initial tags auto-applies them after processed track status', async ({
    page,
  }) => {
    await setMswScenario(page, { trackStatusMode: 'processed' });
    await page.getByRole('button', { name: 'Add source' }).click();
    await page.getByTestId('addsource-file-input').setInputFiles({
      name: 'auto-tagged-runbook.md',
      mimeType: 'text/markdown',
      buffer: Buffer.from('# Auto tagged\nProcessed track status fixture'),
    });
    await page.getByLabel('Tag input').fill('oracle');
    await page.getByTestId('tag-sugg-oracle').click();
    await expect(page.getByRole('button', { name: 'Add 1 source' })).toBeEnabled({
      timeout: 12_000,
    });
    await page.getByRole('button', { name: 'Add 1 source' }).click();
    await expect(page.getByRole('status')).toContainText('Sources queued for ingestion');
    await expect(page.getByRole('status')).toContainText('Initial tags applied', {
      timeout: 6_000,
    });
    await page.getByLabel('Search source').fill('auto-tagged-runbook');
    await expect(page.getByText('auto-tagged-runbook.md')).toBeVisible();
    await expect(page.getByText('oracle')).toBeVisible();
  });

  test('doctrine: add source click accepts multiple files and they appear in documents', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'Add source' }).click();
    const chooserPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Drop files or click to browse' }).click();
    const chooser = await chooserPromise;
    await chooser.setFiles([
      {
        name: 'multi-a.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# A'),
      },
      {
        name: 'multi-b.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# B'),
      },
    ]);
    await expect(page.getByText('multi-a.md')).toBeVisible();
    await expect(page.getByText('multi-b.md')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Add 2 sources' })).toBeEnabled({
      timeout: 12_000,
    });
    await page.getByRole('button', { name: 'Add 2 sources' }).click();
    await expect(page.getByRole('status')).toContainText('Sources queued for ingestion');
    await page.getByLabel('Search source').fill('multi-');
    await expect(page.getByText('multi-a.md')).toBeVisible();
    await expect(page.getByText('multi-b.md')).toBeVisible();
  });

  test('doctrine: single delete requires confirm and removes the document', async ({ page }) => {
    await page.getByTestId('docs-row-delete-d4').click();
    await expect(page.getByTestId('doc-detail-panel')).toBeVisible();
    await page.getByTestId('doc-detail-delete').click();
    await expect(page.getByTestId('doc-detail-delete')).toHaveText(/Confirm delete/);
    await page.getByTestId('doc-detail-delete').click();
    await expect(page.getByRole('status')).toContainText('Document deleted');
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();
  });

  test('doctrine: taxonomy template download and valid/invalid category imports', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    const downloadPromise = page.waitForEvent('download');
    await page.getByTestId('taxonomy-download-template').click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toBe('twin-categories.template.json');

    await page.getByTestId('taxonomy-import-file').setInputFiles({
      name: 'bad-categories.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify([{ id: 'bad' }])),
    });
    await expect(page.getByTestId('taxonomy-import-status')).toContainText(
      'Category[0] missing required fields',
    );

    await page.getByTestId('taxonomy-import-file').setInputFiles({
      name: 'good-categories.json',
      mimeType: 'application/json',
      buffer: Buffer.from(
        JSON.stringify([
          { id: 'reliability', label: 'Reliability', color: '#3366cc' },
          { id: 'oracle', label: 'Oracle', color: '#B85A1E' },
        ]),
      ),
    });
    await expect(page.getByTestId('taxonomy-import-status')).toContainText(
      'Categories imported',
    );
    await expect(page.getByTestId('rail-reliability')).toContainText('Reliability');
  });

  test('doctrine: sign out purges retrieval thread localStorage', async ({ page }) => {
    await openTab(page, 'Retrieval');
    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Query input').fill('Persisted before signout');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect
      .poll(() => page.evaluate(() => localStorage.getItem('twin-rag.threads.v2')))
      .not.toBeNull();

    await openTab(page, 'Settings');
    await page.getByTestId('settings-signout').click();
    await expect
      .poll(() => page.evaluate(() => localStorage.getItem('twin-rag.threads.v2')))
      .toBeNull();
  });
});
