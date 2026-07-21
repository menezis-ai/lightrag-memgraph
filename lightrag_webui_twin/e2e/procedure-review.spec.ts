/**
 * Operator journey — procedure PDF review (MSW-backed).
 *
 * Backend contract: `server/procedure_routes.py`. The MSW handlers are
 * STATEFUL (approve flips the bundle AND enqueues a document row), so a
 * falsely-green pass without mutation is not possible here.
 */

import { expect, test } from '@playwright/test';
import { boot, getMswStats, openTab } from './helpers';

test.describe('Procedure review journey', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@documents @procedure pending bundle card renders in the review section', async ({
    page,
  }) => {
    // The seeded proc-1 bundle (pending) surfaces as a third card variant in
    // the "To be validated" section of the Documents tab.
    const card = page.getByTestId('pending-proc-proc-1');
    await expect(card).toBeVisible();
    await expect(card).toContainText('oracle-failover-procedure.pdf');
    await expect(page.getByTestId('pending-proc-state-proc-1')).toContainText(
      'Procedure review',
    );
    await expect(card).toContainText('2/2 schematics described');

    // The failed bundle renders with the red state pill.
    await expect(page.getByTestId('pending-proc-state-proc-2')).toContainText(
      'Procedure failed',
    );
  });

  test('@documents @procedure review modal shows PNG, informed description and divergence report', async ({
    page,
  }) => {
    await page.getByTestId('pending-proc-review-proc-1').click();
    const modal = page.getByTestId('procedure-review-modal');
    await expect(modal).toBeVisible();

    // Left pane: the base64 schematic rendering with its page number.
    const png = page.getByTestId('procedure-review-png');
    await expect(png).toBeVisible();
    await expect(png).toHaveAttribute('src', /^data:image\/png;base64,/);

    // Right pane: the informed pass (title + description + tasks).
    const informed = page.getByTestId('procedure-review-informed');
    await expect(informed).toContainText('Failover decision tree');
    await expect(informed).toContainText('Check Data Guard lag');

    // Page 1 is coherent → green summary line.
    await expect(
      page.getByTestId('procedure-review-divergence'),
    ).toHaveAttribute('data-coherent', 'true');

    // Page 2 diverges → highlighted panel + itemised divergences.
    await page.getByTestId('procedure-review-next').click();
    const divergence = page.getByTestId('procedure-review-divergence');
    await expect(divergence).toHaveAttribute('data-coherent', 'false');
    await expect(divergence).toContainText(
      'Blind and informed readings diverge',
    );
    await expect(divergence).toContainText('ISAB gate missing from the diagram');

    // Blind reading stays behind its collapsible.
    await expect(page.getByTestId('procedure-review-blind')).toBeHidden();
    await page.getByTestId('procedure-review-blind-toggle').click();
    await expect(page.getByTestId('procedure-review-blind')).toContainText(
      'Switchback sequence (blind)',
    );
  });

  test('@documents @procedure @rc1 approve releases the bundle and the document lands', async ({
    page,
  }) => {
    await page.getByTestId('pending-proc-review-proc-1').click();
    await expect(page.getByTestId('procedure-review-modal')).toBeVisible();

    await page.getByTestId('procedure-review-approve').click();
    await expect(page.getByRole('status')).toContainText('Procedure approved');

    // MSW state mutated: the bundle leaves the pending list…
    await expect(page.getByTestId('pending-proc-proc-1')).toBeHidden();
    // …the failed sibling is untouched…
    await expect(page.getByTestId('pending-proc-proc-2')).toBeVisible();
    // …and the approved bundle became a real document row (stateful
    // documents mock — the enqueued doc polls PENDING → PROCESSED).
    await page.getByLabel('Search source').fill('oracle-failover-procedure');
    await expect(page.getByTestId('docs-row-proc_doc_proc-1_1')).toBeVisible();
  });

  test('@documents @procedure quick reject keeps the bundle visible and recoverable', async ({
    page,
  }) => {
    await expect(page.getByTestId('pending-proc-proc-2')).toBeVisible();
    await page.getByTestId('pending-proc-reject-proc-2').click();
    await expect(page.getByRole('status')).toContainText('Procedure rejected');
    // Rejected is NOT terminal-hidden: the review modal is the only surface
    // offering retry/reroute recovery, so the card stays visible with the
    // rejected pill and the quick-reject action gone (review-pass finding #3).
    await expect(page.getByTestId('pending-proc-proc-2')).toBeVisible();
    await expect(page.getByTestId('pending-proc-state-proc-2')).toContainText(
      'Procedure rejected',
    );
    await expect(page.getByTestId('pending-proc-reject-proc-2')).toHaveCount(0);
    // The pending sibling stays reviewable.
    await expect(page.getByTestId('pending-proc-proc-1')).toBeVisible();
  });

  test('@documents @procedure decisions land in the Activity feed', async ({
    page,
  }) => {
    await page.getByTestId('pending-proc-reject-proc-2').click();
    await expect(page.getByRole('status')).toContainText('Procedure rejected');

    await openTab(page, 'Activity');
    await expect(
      page
        .getByText(
          "Procedure 'network-segmentation-procedure.pdf' rejected",
        )
        .first(),
    ).toBeVisible();
  });

  test('@upload @procedure document type selector rides the upload as X-Twin-Doc-Type', async ({
    page,
  }) => {
    // The MSW upload handler records the received X-Twin-Doc-Type header in
    // the e2e stats (page.route cannot see service-worker-handled fetches),
    // so the assertion covers the real request the client sent.
    await page.getByRole('button', { name: 'Add source' }).click();
    await page.getByTestId('addsource-file-input').setInputFiles({
      name: 'forced-procedure.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('%PDF-1.4 procedure'),
    });
    await page
      .getByTestId('addsource-doc-type')
      .selectOption('procedure');
    await page.getByRole('button', { name: 'Add 1 source' }).click();
    await expect(page.getByRole('status')).toContainText(
      'Sources queued for ingestion',
    );

    const stats = await getMswStats(page);
    const upload = stats.uploadRequests.find(
      (u) => u.name === 'forced-procedure.pdf',
    );
    expect(upload?.docType).toBe('procedure');
  });

  test('@upload @procedure a parked upload resolves into a review card, not a dangling document row', async ({
    page,
  }) => {
    // The backend deliberately creates NO document for a parked procedure:
    // the optimistic upload row must reconcile against the approval queue
    // (track_id) instead of dangling forever, and the review card must
    // appear without a manual refresh.
    await page.getByRole('button', { name: 'Add source' }).click();
    await page.getByTestId('addsource-file-input').setInputFiles({
      name: 'parked-procedure.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('%PDF-1.4 parked procedure'),
    });
    await page.getByTestId('addsource-doc-type').selectOption('procedure');
    await page.getByRole('button', { name: 'Add 1 source' }).click();

    // Reconciliation polls every 2s: the parked toast fires once resolved.
    await expect(page.getByRole('status')).toContainText('Parked for review', {
      timeout: 15_000,
    });
    // The review card is live in the pending section…
    const card = page.getByTestId(/pending-proc-proc_up_/);
    await expect(card).toBeVisible();
    await expect(card).toContainText('parked-procedure.pdf');
    // …and NO document row dangles in the table (the optimistic row was
    // removed; the backend never created a document).
    await expect(
      page.getByTestId(/docs-row-filename-/).filter({
        hasText: 'parked-procedure.pdf',
      }),
    ).toHaveCount(0);
  });
});
