import { expect, test, type Page } from '@playwright/test';
import { boot, openTab } from './helpers';

/**
 * UI sweep ("moulinette") — systematic pass over every screen and modal.
 *
 * Instead of asserting one journey, these tests walk the whole surface and
 * enforce global invariants:
 *   - no console errors / uncaught page errors anywhere,
 *   - no request hits MSW without a matching handler (contract drift),
 *   - every tab renders real content (no blank pane behind the tab switch),
 *   - every visible control exposes an accessible name.
 *
 * A regression anywhere in the shell (lazy chunk that fails to load, a
 * handler that starts 500ing, an unnamed icon button) fails here even if no
 * journey test covers that exact spot.
 */

const TABS = ['Documents', 'Tags', 'Retrieval', 'Graph', 'Activity', 'Settings'] as const;

interface SweepLog {
  errors: string[];
  unhandled: string[];
}

function track(page: Page): SweepLog {
  const log: SweepLog = { errors: [], unhandled: [] };
  page.on('console', (msg) => {
    const text = msg.text();
    if (msg.type() === 'error') log.errors.push(text);
    // MSW prints "captured a request without a matching request handler"
    // as a warning — surface it as contract drift.
    if (msg.type() === 'warning' && text.includes('[MSW]')) log.unhandled.push(text);
  });
  page.on('pageerror', (err) => {
    log.errors.push(`pageerror: ${err.message}`);
  });
  return log;
}

function assertClean(log: SweepLog) {
  expect(log.errors, 'console/page errors during sweep').toEqual([]);
  expect(log.unhandled, 'requests unhandled by MSW during sweep').toEqual([]);
}

async function expectRenderedPane(page: Page) {
  // Secondary tabs are lazy-loaded — poll past the Suspense fallback.
  await expect
    .poll(
      async () => (await page.locator('main .tab-pane').innerText()).trim().length,
      { message: 'tab pane should render visible content' },
    )
    .toBeGreaterThan(20);
}

test.describe('UI sweep', () => {
  test('@sweep every tab renders content with a clean console, in light and dark theme', async ({
    page,
  }) => {
    const log = track(page);
    await boot(page);

    for (const tab of TABS) {
      await openTab(page, tab);
      await expect(page.locator('main .tab-pane')).toBeVisible();
      await expectRenderedPane(page);
    }

    await page.getByLabel('Theme').click();
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    for (const tab of TABS) {
      await openTab(page, tab);
      await expectRenderedPane(page);
    }

    assertClean(log);
  });

  test('@sweep every modal opens and closes without errors or stray mutations', async ({
    page,
  }) => {
    const log = track(page);
    await boot(page);

    // Documents — Add source
    await page.getByRole('button', { name: 'Add source' }).click();
    await expect(page.getByRole('dialog', { name: 'Add source' })).toBeVisible();
    await page.getByLabel('Close dialog').click();
    await expect(page.getByRole('dialog', { name: 'Add source' })).toBeHidden();

    // Documents — Retag (bulk path)
    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page
      .getByLabel('Bulk actions')
      .getByRole('button', { name: /Retag 1 sources/ })
      .click();
    await expect(page.getByRole('dialog', { name: 'Retag document' })).toBeVisible();
    await page.getByLabel('Close dialog').click();
    await expect(page.getByRole('dialog', { name: 'Retag document' })).toBeHidden();
    await page.getByRole('button', { name: 'Clear selection' }).click();

    // Documents — pending review trio
    // Audit C6: dialog aria-label is "Indexed chunks" since the
    // modal renders the real /documents/{id}/chunks projection.
    await page.getByTestId('pending-doc-read-d6').click();
    await expect(page.getByRole('dialog', { name: 'Indexed chunks' })).toBeVisible();
    await page.getByRole('dialog', { name: 'Indexed chunks' }).getByLabel('Close').click();
    await page.getByTestId('pending-doc-edit-approve-d6').click();
    await expect(
      page.getByRole('dialog', { name: 'Edit & approve document' }),
    ).toBeVisible();
    await page.getByRole('button', { name: 'Cancel' }).click();
    await page.getByTestId('pending-doc-reject-d7').click();
    await expect(page.getByRole('dialog', { name: 'Reject document' })).toBeVisible();
    await page.getByRole('button', { name: 'Cancel' }).click();

    // Documents — detail panel + gated raw notice
    await page.getByTestId('docs-row-delete-d3').click();
    await expect(page.getByTestId('doc-detail-panel')).toBeVisible();
    await page.getByTestId('doc-detail-view-raw').click();
    await expect(page.getByRole('dialog', { name: 'View raw notice' })).toBeVisible();
    await page
      .getByRole('dialog', { name: 'View raw notice' })
      .getByRole('button', { name: 'Close', exact: true })
      .last()
      .click();
    await page
      .getByRole('dialog', { name: /Detail:/ })
      .getByLabel('Close')
      .first()
      .click();
    await expect(page.getByTestId('doc-detail-panel')).toBeHidden();

    // Tags — governance dialogs (edit + request)
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('oracle');
    await page.getByTestId('tag-card-oracle').click();
    await page.getByRole('button', { name: 'Edit', exact: true }).click();
    await expect(page.getByRole('dialog', { name: 'Edit tag' })).toBeVisible();
    await page.getByLabel('Close dialog').click();
    await expect(page.getByRole('dialog', { name: 'Edit tag' })).toBeHidden();
    await page.getByRole('button', { name: 'Request new tag' }).click();
    await expect(page.getByRole('dialog', { name: /Request/ })).toBeVisible();
    await page.getByLabel('Close dialog').click();
    await expect(page.getByRole('dialog', { name: /Request/ })).toBeHidden();

    // Settings — API authorize
    await openTab(page, 'Settings');
    await page.getByTestId('settings-rail-api').click();
    await page.getByRole('button', { name: 'Authorize' }).click();
    const authorizeDialog = page.getByRole('dialog', { name: 'Authorize' });
    await expect(authorizeDialog).toBeVisible();
    await expect
      .poll(() =>
        page.evaluate(
          () => document.activeElement?.closest('[role="dialog"]') !== null,
        ),
      )
      .toBe(true);
    await page.keyboard.press('Escape');
    await expect(authorizeDialog).toBeHidden();

    // Topbar — popovers
    await page.getByRole('button', { name: /Notifications/ }).click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toBeVisible();
    await page.keyboard.press('Escape');
    await page.getByTitle('Switch folder').click();
    await expect(page.getByRole('menu', { name: 'Switch folder' })).toBeVisible();
    await page.keyboard.press('Escape');

    // No dialog stayed behind, nothing committed, console stayed clean.
    await expect(page.getByRole('dialog')).toHaveCount(0);
    await expect(page.locator('.toast-viewport .toast')).toHaveCount(0);
    assertClean(log);
  });

  test('@sweep @a11y every visible control exposes an accessible name on every tab', async ({
    page,
  }) => {
    await boot(page);

    const offenders: string[] = [];
    for (const tab of TABS) {
      await openTab(page, tab);
      await expect(page.locator('main .tab-pane')).toBeVisible();
      const anonymous = await page.evaluate(() => {
        const named = (el: Element): boolean => {
          const aria = el.getAttribute('aria-label');
          if (aria && aria.trim()) return true;
          const labelledBy = el.getAttribute('aria-labelledby');
          if (labelledBy && document.getElementById(labelledBy)?.textContent?.trim())
            return true;
          if (el.getAttribute('title')?.trim()) return true;
          if ((el.textContent ?? '').trim()) return true;
          if (el instanceof HTMLInputElement || el instanceof HTMLSelectElement || el instanceof HTMLTextAreaElement) {
            if (el.labels && Array.from(el.labels).some((l) => l.textContent?.trim()))
              return true;
            if (el instanceof HTMLInputElement && el.placeholder.trim()) return true;
          }
          return false;
        };
        const visible = (el: Element): boolean => {
          const rect = el.getBoundingClientRect();
          if (rect.width === 0 || rect.height === 0) return false;
          return window.getComputedStyle(el).visibility !== 'hidden';
        };
        return Array.from(
          document.querySelectorAll(
            'button, [role="button"], input:not([type=hidden]), select, textarea, a[href]',
          ),
        )
          .filter((el) => visible(el) && !named(el))
          .map(
            (el) =>
              `${el.tagName.toLowerCase()}${el.id ? `#${el.id}` : ''}.${Array.from(
                el.classList,
              ).join('.')}`,
          );
      });
      offenders.push(...anonymous.map((sel) => `[${tab}] ${sel}`));
    }

    expect(offenders, 'controls without an accessible name').toEqual([]);
  });

  test('@sweep folder switch keeps every tab consistent and the console clean', async ({
    page,
  }) => {
    const log = track(page);
    await boot(page);

    await page.getByTitle('Switch folder').click();
    const target = page.getByRole('menuitemradio').filter({ hasNotText: 'default' }).first();
    if ((await target.count()) > 0) {
      await target.click();
      for (const tab of TABS) {
        await openTab(page, tab);
        await expect(page.locator('main .tab-pane')).toBeVisible();
      }
    } else {
      await page.keyboard.press('Escape');
    }

    assertClean(log);
  });
});
