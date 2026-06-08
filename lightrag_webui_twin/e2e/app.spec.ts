import { expect, test } from '@playwright/test';
import {
  addSourceFile,
  boot,
  getMswStats,
  openTab,
  seedDocuments,
  setMswScenario,
} from './helpers';

test.describe('Twin WebUI operator journeys', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@documents @smoke filter, add source, retag and delete flows stay wired', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page.getByTestId('pending-docs-section')).toContainText(
      'documents awaiting your sign-off',
    );

    await page.getByRole('button', { name: /Failed/ }).click();
    await expect(page.getByTestId('docs-row-d3')).toContainText('huge-archive.zip');
    await page.getByLabel('Search source').fill('oracle');
    await expect(page.getByTestId('docs-empty')).toBeVisible();
    // Reset filters by clicking All + clearing the search input — the
    // dedicated "Clear" button was removed (filters carry their own
    // dismiss UX and the bulk bar handles selection).
    await page.getByRole('button', { name: /^All/ }).click();
    await page.getByLabel('Search source').fill('');

    await addSourceFile(
      page,
      {
        name: 'runbook-smoke.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# Smoke\nE2E upload fixture'),
      },
      'oracle',
    );
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

  test('@smoke @navigation topbar, tags, retrieval, graph and activity cross-navigation work', async ({ page }) => {
    await page.getByRole('button', { name: /Notifications/ }).click();
    await expect(page.getByText('Mark all read')).toBeVisible();
    await page.getByText('Mark all read').click();
    await expect(page.getByRole('button', { name: 'Notifications' })).toBeVisible();

    await page.getByLabel('Theme').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');

    await page.getByTitle('Switch folder').click();
    await page.getByRole('menuitemradio', { name: /sandbox/ }).click();
    await expect(page.getByTitle('Switch folder')).toContainText('sandbox');

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
    // `exact: true` avoids a strict-mode collision with the MSW
    // streaming response, which echoes the query inside the
    // assistant message ("Mock retrieval response for: How do I
    // restart Oracle?").
    await expect(
      page.getByText('How do I restart Oracle?', { exact: true }),
    ).toBeVisible();
    // The seed thread already paints `source-1` from its sample
    // turn, and the new assistant reply now also carries its own
    // `source-1` once the NDJSON stream completes — scope to the
    // latest assistant turn so the assertion is unambiguous.
    const latestAssistant = page.locator('.msg-assistant').last();
    await expect(
      latestAssistant.getByTestId('source-1'),
    ).toBeVisible({ timeout: 12_000 });
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
    // Mock-kill F3 — the CTA label is now "View documents mentioning
    // this entity" and navigation falls back to `?q=<entity name>`
    // (the per-entity `?source=` filter was dropped because the fixture
    // map keyed on prototype ids missed real Memgraph entities).
    await page
      .getByRole('button', { name: /View documents mentioning this entity/ })
      .click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page).toHaveURL(/[?&]q=/);

    await openTab(page, 'Activity');
    await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible();
    await page.getByLabel('Search events').fill('huge-archive');
    await expect(page.getByRole('heading', { name: 'huge-archive.zip' })).toBeVisible();
    await page.getByRole('button', { name: /Replay ingestion/ }).click();
    await expect(page.getByRole('status')).toContainText('Replay queued');
  });

  test('@doctrine @a11y keyboard activates graph nodes, retrieval history and switches', async ({
    page,
  }) => {
    await openTab(page, 'Graph');
    await page.getByTestId('kg-node-e_rman').focus();
    await page.keyboard.press('Space');
    await expect(page.getByRole('heading', { name: 'RMAN' })).toBeVisible();

    await openTab(page, 'Retrieval');
    await page.getByTestId('thread-th_seed_2').focus();
    await page.keyboard.press('Enter');
    await expect(page.getByTestId('thread-th_seed_2')).toHaveAttribute('aria-current', 'true');
    const onlyContext = page.getByRole('switch', { name: 'Only need context' });
    await onlyContext.focus();
    await page.keyboard.press('Space');
    await expect(onlyContext).toHaveAttribute('aria-checked', 'true');
    const onlyPrompt = page.getByRole('switch', { name: 'Only need prompt' });
    await onlyPrompt.focus();
    await page.keyboard.press('Enter');
    await expect(onlyPrompt).toHaveAttribute('aria-checked', 'true');
  });

  test('@doctrine @workspace space switch refreshes documents and clears local filters', async ({
    page,
  }) => {
    await seedDocuments(page, [
      {
        doc_id: 'sandbox_doc_1',
        file_path: '/sandbox/runbooks/kernel-panic-runbook.md',
        content_summary: 'Sandbox space document seeded by e2e',
        tags: ['rhel9'],
        workspace: 'sandbox',
      },
    ]);
    await page.getByLabel('Search source').fill('oracle');
    await expect(page).toHaveURL(/q=oracle/);
    await page.getByTitle('Switch folder').click();
    await page.getByRole('menuitemradio', { name: /sandbox/ }).click();
    await expect(page.getByTitle('Switch folder')).toContainText('sandbox');
    await expect(page).not.toHaveURL(/q=oracle/);
    await expect(page.getByLabel('Search source')).toHaveValue('');
    await expect(page.getByTestId('docs-row-sandbox_doc_1')).toContainText(
      'kernel-panic-runbook.md',
    );
    await expect(page.getByTestId('docs-row-d1')).toBeHidden();
  });

  test('@documents @reviewer read extracted text, edit-approve and reject pending sources', async ({
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

  test('@documents document detail panel exposes chunks, lineage, audit and gated raw notice', async ({
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

  test('@settings @auth settings profile, workspace and API explorer remain usable', async ({ page }) => {
    await openTab(page, 'Settings');
    await expect(page.getByTestId('settings-profile')).toBeVisible();
    await expect(page.getByTestId('settings-profile-name')).toBeVisible();

    await page.getByTestId('settings-restart-tutorial').click();
    await expect(page.getByRole('status')).toContainText('Tutorial restarted');

    await page.getByTestId('settings-rail-workspace').click();
    await expect(page.getByTestId('settings-workspace')).toBeVisible();
    await expect(page.getByTestId('settings-active-ws')).toContainText('default');
    // Mock-kill F1 — Visibility / Region / Retention cards were
    // dropped because their values were fixture-only inventions
    // (eu-west-3 dc-paris, hardcoded TTLs). Identity card is now the
    // single source of truth for the active-space view.
    await expect(page.getByTestId('settings-space-display-name')).toBeVisible();

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

  test('@tags @governance request and reject review workflows are enforced', async ({ page }) => {
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

  test('@doctrine @tags double-click tag approve sends one mutation', async ({
    page,
  }) => {
    await setMswScenario(page, { tagApproveDelayMs: 600 });
    await openTab(page, 'Tags');
    const approve = page
      .getByTestId('pending-argocd')
      .getByRole('button', { name: 'Approve', exact: true });
    await approve.click();
    await expect(page.getByTestId('pending-argocd')).toContainText('Approving');
    await approve.click({ timeout: 200 }).catch(() => undefined);
    await expect(page.locator('.toast-viewport')).toContainText('argocd');
    await expect(page.locator('.toast-viewport')).toContainText('approved');
    const stats = await getMswStats(page);
    expect(stats.tagApproveCalls.argocd).toBe(1);
  });

  test('@doctrine @tags @notifications tag approval appears in bell and activity', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    await page
      .getByTestId('pending-argocd')
      .getByRole('button', { name: 'Approve', exact: true })
      .click();
    await expect(page.locator('.toast-viewport')).toContainText('argocd');
    await expect(page.locator('.toast-viewport')).toContainText('approved');

    await page.getByRole('button', { name: /Notifications, \d+ unread/ }).click();
    const popover = page.getByRole('dialog', { name: 'Notifications' });
    await expect(popover).toContainText('argocd');
    await expect(popover).toContainText('approved');

    await openTab(page, 'Activity');
    await page.getByLabel('Search events').fill('argocd');
    await expect(page.getByRole('heading', { name: 'argocd' })).toBeVisible();
    await expect(page.getByRole('button', { name: /Tag argocd approved/ })).toBeVisible();

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('argocd');
    await expect(page.getByTestId('tag-card-argocd')).toBeVisible();
    await expect(page.getByTestId('pending-argocd')).toBeHidden();

    await page.getByRole('button', { name: /Notifications, \d+ unread/ }).click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toContainText('argocd');

    await openTab(page, 'Activity');
    await page.getByLabel('Search events').fill('argocd');
    await expect(page.getByRole('button', { name: /Tag argocd approved/ })).toBeVisible();
  });

  test('@tags @governance synonyms, deprecate and delete migration actions persist', async ({
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

  test('@tags @navigation filters, related tags and document drill-down keep URL state wired', async ({
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

  test('@doctrine @tags bulk retag backend failure rolls back optimistic tags', async ({ page }) => {
    await setMswScenario(page, { bulkRetagStatus: 500 });
    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page.getByLabel('Bulk actions').getByRole('button', { name: /Retag 1 sources/ }).click();
    await page.getByRole('textbox', { name: 'Tag input' }).fill('memgraph');
    await page.getByTestId('sugg-memgraph').click();
    await page.getByRole('button', { name: 'Apply tag' }).click();
    await expect(page.getByRole('alert')).toContainText('Tag mutation failed');
    await expect(page.getByTestId('docs-row-d1')).not.toContainText('memgraph');
  });

  test('@doctrine @reviewer double-click approve sends one mutation for a pending doc', async ({ page }) => {
    await setMswScenario(page, { approveDelayMs: 600 });
    const approve = page.getByTestId('pending-doc-approve-d6');
    await approve.click();
    await expect(page.getByTestId('pending-doc-d6')).toBeHidden();
    await approve.click({ timeout: 200 }).catch(() => undefined);
    await expect(page.getByRole('status')).toContainText('Document approved');
    const stats = await getMswStats(page);
    expect(stats.approveCalls.d6).toBe(1);
  });

  test('@doctrine @a11y focus is restored after approving a pending doc', async ({
    page,
  }) => {
    await page.getByTestId('pending-doc-approve-d6').click();
    await expect(page.getByRole('status')).toContainText('Document approved');
    await expect(page.getByTestId('pending-doc-d6')).toBeHidden();
    await expect
      .poll(() =>
        page.evaluate(() => {
          const active = document.activeElement as HTMLElement | null;
          return {
            tag: active?.tagName ?? '',
            testId: active?.dataset?.testid ?? '',
            fallback: active?.dataset?.focusFallback ?? '',
          };
        }),
      )
      .toMatchObject({ tag: /^(BUTTON|MAIN)$/ });
  });

  test('@doctrine @upload @tags upload with initial tags auto-applies them after processed track status', async ({
    page,
  }) => {
    await setMswScenario(page, { trackStatusMode: 'processed' });
    await addSourceFile(
      page,
      {
        name: 'auto-tagged-runbook.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# Auto tagged\nProcessed track status fixture'),
      },
      'oracle',
    );
    await expect(page.getByRole('status')).toContainText('Sources queued for ingestion');
    await expect(page.locator('.toast-viewport')).toContainText('Initial tags applied', {
      timeout: 10_000,
    });
    await page.getByLabel('Search source').fill('auto-tagged-runbook');
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText('auto-tagged-runbook.md');
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText('oracle');
    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await expect(page.getByLabel('Search source')).toHaveValue('auto-tagged-runbook');
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText('auto-tagged-runbook.md');
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText('oracle');
  });

  test('@doctrine @upload @resilience initial tag polling timeout surfaces manual retag guidance', async ({
    page,
  }) => {
    await setMswScenario(page, { trackStatusMode: 'timeout' });
    await page.evaluate(() => {
      (
        window as Window & {
          __TWIN_E2E_INITIAL_TAG_POLL?: { intervalMs: number; maxPolls: number };
        }
      ).__TWIN_E2E_INITIAL_TAG_POLL = { intervalMs: 50, maxPolls: 2 };
    });
    await addSourceFile(
      page,
      {
        name: 'slow-ingestion-runbook.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# Slow\nTimeout fixture'),
      },
      'oracle',
    );
    await expect(page.locator('.toast-viewport')).toContainText('Initial tags not applied', {
      timeout: 5_000,
    });
    await expect(page.locator('.toast-viewport')).toContainText('Retag manually');
    await page.getByLabel('Search source').fill('slow-ingestion-runbook');
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText(
      'slow-ingestion-runbook.md',
    );
    await expect(page.getByTestId('docs-row-uploaded_1')).not.toContainText('oracle');
  });

  test('@doctrine @upload add source click accepts multiple files and they appear in documents', async ({
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
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText('multi-a.md');
    await expect(page.getByTestId('docs-row-uploaded_2')).toContainText('multi-b.md');
  });

  test('@doctrine @upload add source drag-drop accepts multiple files', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'Add source' }).click();
    // The Add Source modal body is lazy-loaded via Suspense (perf
    // optimization) — wait for the dropzone to actually be in the DOM
    // before we reach into it with `page.evaluate`.
    await expect(
      page.getByLabel('Drop files or click to browse'),
    ).toBeVisible();
    await page.evaluate(() => {
      const dropzone = document.querySelector('[aria-label="Drop files or click to browse"]');
      if (!dropzone) throw new Error('dropzone not found');
      const transfer = new DataTransfer();
      transfer.items.add(new File(['# Drag A'], 'drag-a.md', { type: 'text/markdown' }));
      transfer.items.add(new File(['# Drag B'], 'drag-b.md', { type: 'text/markdown' }));
      dropzone.dispatchEvent(
        new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: transfer }),
      );
    });
    await expect(page.getByText('drag-a.md')).toBeVisible();
    await expect(page.getByText('drag-b.md')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Add 2 sources' })).toBeEnabled({
      timeout: 12_000,
    });
    await page.getByRole('button', { name: 'Add 2 sources' }).click();
    await expect(page.getByRole('status')).toContainText('Sources queued for ingestion');
    await page.getByLabel('Search source').fill('drag-');
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText('drag-a.md');
    await expect(page.getByTestId('docs-row-uploaded_2')).toContainText('drag-b.md');
  });

  test('@doctrine @upload partial multi-file failure reports accurate counts', async ({
    page,
  }) => {
    await setMswScenario(page, { uploadFailureNames: ['partial-fail.md'] });
    await page.getByRole('button', { name: 'Add source' }).click();
    const chooserPromise = page.waitForEvent('filechooser');
    await page.getByRole('button', { name: 'Drop files or click to browse' }).click();
    const chooser = await chooserPromise;
    await chooser.setFiles([
      {
        name: 'partial-ok.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# OK'),
      },
      {
        name: 'partial-fail.md',
        mimeType: 'text/markdown',
        buffer: Buffer.from('# Fail'),
      },
    ]);
    await expect(page.getByRole('button', { name: 'Add 2 sources' })).toBeEnabled({
      timeout: 12_000,
    });
    await page.getByRole('button', { name: 'Add 2 sources' }).click();
    await expect(page.getByRole('alert')).toContainText('1 upload failed');
    await expect(page.getByRole('alert')).toContainText('1 ok · 1 ko');
    await page.getByLabel('Search source').fill('partial-');
    await expect(page.getByTestId('docs-row-uploaded_1')).toContainText('partial-ok.md');
    await expect(page.getByText('partial-fail.md')).toBeHidden();
  });

  test('@doctrine @documents single delete requires confirm and removes the document', async ({ page }) => {
    await page.getByTestId('docs-row-delete-d4').click();
    await expect(page.getByTestId('doc-detail-panel')).toBeVisible();
    await page.getByTestId('doc-detail-delete').click();
    await expect(page.getByTestId('doc-detail-delete')).toHaveText(/Confirm delete/);
    await page.getByTestId('doc-detail-delete').click();
    await expect(page.getByRole('status')).toContainText('Document deleted');
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();
  });

  test('@doctrine @tags taxonomy template download and valid/invalid category imports', async ({
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

  test('@doctrine @tags invalid category JSON syntax shows a precise banner', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    await page.getByTestId('taxonomy-import-file').setInputFiles({
      name: 'syntax-error-categories.json',
      mimeType: 'application/json',
      buffer: Buffer.from('[{"id":"x",]'),
    });
    await expect(page.getByTestId('taxonomy-import-status')).toContainText(
      'Invalid JSON file',
    );
  });

  test('@doctrine @tags invalid category duplicates and colors show server messages', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    await page.getByTestId('taxonomy-import-file').setInputFiles({
      name: 'duplicate-categories.json',
      mimeType: 'application/json',
      buffer: Buffer.from(
        JSON.stringify([
          { id: 'reliability', label: 'Reliability', color: '#3366cc' },
          { id: 'reliability', label: 'Reliability again', color: '#224466' },
        ]),
      ),
    });
    await expect(page.getByTestId('taxonomy-import-status')).toContainText(
      'duplicate id: reliability',
    );

    await page.getByTestId('taxonomy-import-file').setInputFiles({
      name: 'bad-color-categories.json',
      mimeType: 'application/json',
      buffer: Buffer.from(
        JSON.stringify([{ id: 'reliability', label: 'Reliability', color: 'blue' }]),
      ),
    });
    await expect(page.getByTestId('taxonomy-import-status')).toContainText(
      'color must be a #RRGGBB hex value',
    );
  });

  test('@doctrine @tags imported tag domain drives request, approve and filtering workflow', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    await page.getByTestId('taxonomy-import-file').setInputFiles({
      name: 'saad-workflow-categories.json',
      mimeType: 'application/json',
      buffer: Buffer.from(
        JSON.stringify([
          { id: 'oracle', label: 'Oracle', color: '#B85A1E' },
          { id: 'infra', label: 'Infrastructure', color: '#5A7FB4' },
          { id: 'network', label: 'Network', color: '#1F8A7A' },
          { id: 'payment', label: 'Payment', color: '#7B5BB8' },
          { id: 'lifecycle', label: 'Lifecycle', color: '#8A5C0E' },
          { id: 'governance', label: 'Governance', color: '#2C3E50' },
          { id: 'reliability', label: 'Reliability', color: '#3366cc' },
        ]),
      ),
    });
    await expect(page.getByTestId('rail-reliability')).toContainText('Reliability');

    await page.getByRole('button', { name: 'Request new tag' }).click();
    await page.getByLabel(/Proposed name/).fill('golden-signal');
    await page
      .getByLabel(/Definition/)
      .fill('Operational signal used to triage reliability regressions.');
    await page.getByLabel('Domain').selectOption('reliability');
    await page
      .getByLabel('Justification')
      .fill('Required for Saad demo workflow and SLO runbooks.');
    await page.getByRole('button', { name: 'Submit request' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('golden-signal');
    await expect(page.locator('.toast-viewport')).toContainText('requested for review');

    await page
      .getByTestId('pending-golden-signal')
      .getByRole('button', { name: 'Approve', exact: true })
      .click();
    await expect(page.locator('.toast-viewport')).toContainText('golden-signal');
    await expect(page.locator('.toast-viewport')).toContainText('approved');

    await page.getByTestId('rail-reliability').click();
    await expect(page.getByTestId('tag-card-golden-signal')).toBeVisible();
    await page.getByTestId('tag-card-golden-signal').click();
    await expect(page.locator('.tag-detail-name')).toHaveText('golden-signal');

    await page.getByRole('button', { name: 'Edit', exact: true }).click();
    await page
      .getByLabel('Short definition')
      .fill('Updated reliability signal definition from e2e.');
    await page.getByRole('button', { name: 'Save' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('golden-signal');
    await expect(page.locator('.toast-viewport')).toContainText('updated');
    await expect(page.getByTestId('tag-card-golden-signal')).toContainText(
      'Updated reliability signal definition from e2e.',
    );

    await page.getByTestId('tag-card-golden-signal').click();
    await page.getByRole('button', { name: 'Manage synonyms' }).click();
    await page.getByLabel('Add synonym').fill('slo-signal');
    await page.getByRole('button', { name: 'Save synonyms' }).click();
    await expect(page.locator('.toast-viewport')).toContainText('synonyms updated');
    await expect(page.getByTestId('tag-card-golden-signal')).toContainText('slo-signal');

    await page.getByTestId('tag-card-golden-signal').click();
    await page.getByRole('button', { name: 'Deprecate' }).click();
    await page
      .getByRole('dialog', { name: /Deprecate tag/ })
      .getByRole('button', { name: 'Deprecate' })
      .click();
    await expect(page.locator('.toast-viewport')).toContainText('deprecated');
    await page.getByLabel('Status').selectOption('deprecated');
    await expect(page.getByTestId('tag-card-golden-signal')).toBeVisible();
  });

  test('@doctrine @tags bulk retag 413 is surfaced as a red error and rolls back', async ({
    page,
  }) => {
    await setMswScenario(page, { bulkRetagStatus: 413 });
    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page.getByLabel('Bulk actions').getByRole('button', { name: /Retag 1 sources/ }).click();
    await page.getByRole('textbox', { name: 'Tag input' }).fill('memgraph');
    await page.getByTestId('sugg-memgraph').click();
    await page.getByRole('button', { name: 'Apply tag' }).click();
    await expect(page.getByRole('alert')).toContainText('Tag mutation failed');
    await expect(page.getByRole('alert')).toContainText('413');
    await expect(page.getByTestId('docs-row-d1')).not.toContainText('memgraph');
  });

  test('@doctrine @tags bulk retag 600 selected documents surfaces backend 413', async ({
    page,
  }) => {
    const docs = Array.from({ length: 600 }, (_, index) => {
      const n = String(index + 1).padStart(3, '0');
      return {
        doc_id: `bulk_${n}`,
        file_path: `/cib/e2e/bulk-massive-${n}.md`,
        content_summary: 'Bulk 413 fixture',
        tags: ['oracle'],
        workspace: 'default',
      };
    });
    await seedDocuments(page, docs);
    await setMswScenario(page, { bulkRetagStatus: 413 });
    await page.getByLabel('Search source').fill('bulk-massive-');
    await expect(page.getByTestId('docs-row-bulk_001')).toBeVisible();
    await page.getByLabel('Select all visible').check();
    await expect(page.getByLabel('Bulk actions')).toContainText('600 selected');
    await page.getByLabel('Bulk actions').getByRole('button', { name: /Retag 600 sources/ }).click();
    await page.getByRole('textbox', { name: 'Tag input' }).fill('memgraph');
    await page.getByTestId('sugg-memgraph').click();
    await page.getByRole('button', { name: 'Apply tag' }).click();
    await expect(page.getByRole('alert')).toContainText('Tag mutation failed');
    await expect(page.getByRole('alert')).toContainText('413');
  });

  test('@doctrine @auth twin api without auth returns 401 challenge', async ({ page }) => {
    await setMswScenario(page, { authGate: true });
    const response = await page.evaluate(async () => {
      const res = await fetch('/twin/api/tags');
      return {
        status: res.status,
        body: await res.json(),
        challenge: res.headers.get('WWW-Authenticate'),
      };
    });
    expect(response.status).toBe(401);
    expect(response.body.detail).toContain('Basic Auth required');
    expect(response.challenge).toContain('Basic');
  });

  test('@doctrine @documents reprocess on a non-failed document tells the truth', async ({ page }) => {
    await page.getByTestId('docs-row-delete-d1').click();
    await expect(page.getByRole('dialog', { name: /Detail: oracle-restart-procedure.pdf/ })).toBeVisible();
    await page.getByTestId('doc-detail-reprocess').click();
    await expect(page.locator('.toast-viewport')).toContainText('Re-process not applicable');
    await expect(page.locator('.toast-viewport')).not.toContainText('Re-process queued');
  });

  test('@doctrine @resilience long document paths ellipsize with a native tooltip', async ({
    page,
  }) => {
    const longPath =
      '/cib/runbooks/' +
      'oracle-rman-disaster-recovery-cross-region-data-guard-failover-validation-'.repeat(3) +
      'final-checklist.md';
    await seedDocuments(page, [
      {
        doc_id: 'long_path',
        file_path: longPath,
        content_summary: 'Long path overflow fixture',
        tags: ['oracle', 'rman', 'runbook'],
      },
    ]);
    await page.getByLabel('Search source').fill('final-checklist');
    const sourceName = page.getByTestId('docs-row-long_path').locator('.source-name');
    await expect(sourceName).toHaveAttribute('title', longPath);
    await expect(sourceName).toHaveCSS('text-overflow', 'ellipsis');
    const box = await sourceName.boundingBox();
    expect(box?.width ?? 0).toBeLessThan(420);
  });

  test('@doctrine @notifications notifications can be marked read and cleared without refresh', async ({
    page,
  }) => {
    await page.getByRole('button', { name: /Notifications, \d+ unread/ }).click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toContainText('unread');
    await page.getByRole('button', { name: 'Mark all read' }).click();
    await expect(page.getByRole('button', { name: 'Notifications' })).toBeVisible();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).not.toContainText('unread');
    await page.getByRole('button', { name: 'Clear all' }).click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toContainText(
      "You're all caught up.",
    );
  });

  test('@doctrine @reviewer pending docs section disappears after the queue is drained', async ({
    page,
  }) => {
    await page.getByTestId('pending-doc-approve-d6').click();
    await expect(page.getByRole('status')).toContainText('Document approved');
    await page.getByTestId('pending-doc-reject-d7').click();
    await page.getByLabel('Rejection reason').fill('Out of scope for retrieval.');
    await page.getByTestId('pending-doc-reject-submit').click();
    await expect(page.getByRole('status')).toContainText('Document rejected');
    await page.getByRole('button', { name: 'Approve update' }).click();
    await expect(page.getByRole('status')).toContainText('Update approved');
    await expect(page.getByTestId('pending-docs-section')).toBeHidden();
  });

  test('@doctrine @auth sign out purges retrieval thread localStorage', async ({ page }) => {
    await openTab(page, 'Retrieval');
    await page.evaluate(() => {
      localStorage.setItem('twin-rag.extra-cache.v1', 'stale');
    });
    await page.getByRole('button', { name: /New/ }).click();
    await page.getByLabel('Query input').fill('Persisted before signout');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect
      .poll(() => page.evaluate(() => localStorage.getItem('twin-rag.threads.v2')))
      .not.toBeNull();

    await openTab(page, 'Settings');
    await page.getByTestId('settings-signout').click();
    await expect
      .poll(async () =>
        page
          .evaluate(() => ({
            threads: localStorage.getItem('twin-rag.threads.v2'),
            extra: localStorage.getItem('twin-rag.extra-cache.v1'),
          }))
          .catch(() => 'navigation-in-progress'),
      )
      .toMatchObject({ threads: null, extra: null });
  });
});
