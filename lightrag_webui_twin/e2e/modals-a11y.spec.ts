import { expect, test, type Page } from '@playwright/test';
import { boot, openTab } from './helpers';

/**
 * Systematic dialog contract: every modal in the WebUI must open, keep
 * focus inside (focus trap), close on Escape and on its close affordance,
 * and hand focus back to the trigger. One test per modal so a regression
 * pinpoints the broken surface immediately.
 */

async function expectFocusInsideDialog(page: Page) {
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          document.activeElement?.closest('dialog,[role="dialog"]') !== null,
      ),
    )
    .toBe(true);
}

test.describe('Modal dialogs a11y contract', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
  });

  test('@a11y @modals Add Source traps focus, closes on Escape and restores the trigger', async ({
    page,
  }) => {
    const trigger = page.getByRole('button', { name: 'Add source' });
    await trigger.click();
    const dialog = page.getByRole('dialog', { name: 'Add source' });
    await expect(dialog).toBeVisible();
    await expectFocusInsideDialog(page);

    for (let i = 0; i < 15; i += 1) {
      await page.keyboard.press('Tab');
    }
    await expectFocusInsideDialog(page);

    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    await expect(trigger).toBeFocused();
  });

  test('@a11y @modals @bug04 Add Source tag input keeps focus while typing char-by-char', async ({
    page,
  }) => {
    // Regression for BUG-04 (Alberto recette): the modal a11y autofocus
    // re-fired on every keystroke (unstable onClose in deps), yanking focus
    // out of the non-first "Apply tags to all" input mid-typing. Use
    // pressSequentially (real per-key events) — fill() set the value
    // atomically and masked the bug.
    await page.getByRole('button', { name: 'Add source' }).click();
    const input = page.getByRole('combobox', { name: 'Tag input' });
    await input.click();
    await input.pressSequentially('rman', { delay: 40 });
    await expect(input).toHaveValue('rman');
    await expect(input).toBeFocused();
  });

  test('@a11y @modals Add Source closes from the backdrop and the close button', async ({
    page,
  }) => {
    await page.getByRole('button', { name: 'Add source' }).click();
    const dialog = page.getByRole('dialog', { name: 'Add source' });
    await expect(dialog).toBeVisible();
    await page.getByLabel('Close dialog').click();
    await expect(dialog).toBeHidden();

    await page.getByRole('button', { name: 'Add source' }).click();
    await expect(dialog).toBeVisible();
    await page.getByTestId('addsource-backdrop').click({ position: { x: 5, y: 5 } });
    await expect(dialog).toBeHidden();
  });

  test('@a11y @modals Retag traps focus and closes on Escape without mutating', async ({
    page,
  }) => {
    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page
      .getByLabel('Bulk actions')
      .getByRole('button', { name: /Retag 1 sources/ })
      .click();
    const dialog = page.getByRole('dialog', { name: 'Retag document' });
    await expect(dialog).toBeVisible();
    await expectFocusInsideDialog(page);

    // Escape inside the tag input only clears the draft text (autocomplete
    // UX); a second Escape from outside the input closes the dialog.
    const tagInput = page.getByRole('combobox', { name: 'Tag input' });
    await tagInput.fill('memgraph');
    await tagInput.press('Escape');
    await expect(tagInput).toHaveValue('');
    await expect(dialog).toBeVisible();

    await page.keyboard.press('Tab');
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    // Abandoning the draft must not mutate anything — no toast, no tag.
    await expect(page.locator('.toast-viewport .toast')).toHaveCount(0);
  });

  test('@a11y @modals Tag edit dialog closes on Escape and restores the Edit trigger', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('oracle');
    await page.getByTestId('tag-card-oracle').click();
    const trigger = page.getByRole('button', { name: 'Edit', exact: true });
    await trigger.click();
    const dialog = page.getByRole('dialog', { name: 'Edit tag' });
    await expect(dialog).toBeVisible();
    await expectFocusInsideDialog(page);

    for (let i = 0; i < 12; i += 1) {
      await page.keyboard.press('Tab');
    }
    await expectFocusInsideDialog(page);

    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
    await expect(trigger).toBeFocused();
  });

  test('@a11y @modals tag action backdrop click cancels without committing', async ({
    page,
  }) => {
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('oracle');
    await page.getByTestId('tag-card-oracle').click();
    await page.getByRole('button', { name: 'Edit', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Edit tag' });
    await expect(dialog).toBeVisible();
    await page.getByTestId('tagaction-backdrop').click({ position: { x: 5, y: 5 } });
    await expect(dialog).toBeHidden();
    await expect(page.locator('.toast-viewport .toast')).toHaveCount(0);
  });

  test('@a11y @modals API authorize dialog closes on Escape', async ({ page }) => {
    await openTab(page, 'Settings');
    await page.getByTestId('settings-rail-api').click();
    await page.getByRole('button', { name: 'Authorize' }).click();
    const dialog = page.getByRole('dialog', { name: 'Authorize' });
    await expect(dialog).toBeVisible();
    await expectFocusInsideDialog(page);
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
  });

  test('@a11y @modals Indexed chunks closes on Escape', async ({ page }) => {
    // Audit C6: the dialog aria-label is now "Indexed chunks" to
    // match the real ``/documents/{id}/chunks`` projection rendered
    // by ReadSourceModal.
    await page.getByTestId('pending-doc-read-d6').click();
    const dialog = page.getByRole('dialog', { name: 'Indexed chunks' });
    await expect(dialog).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(dialog).toBeHidden();
  });

  test('@a11y @modals document detail panel closes on Escape and on its close button', async ({
    page,
  }) => {
    await page.getByTestId('docs-row-delete-d1').click();
    await expect(page.getByTestId('doc-detail-panel')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByTestId('doc-detail-panel')).toBeHidden();

    await page.getByTestId('docs-row-delete-d1').click();
    await expect(page.getByTestId('doc-detail-panel')).toBeVisible();
    await page
      .getByTestId('doc-detail-panel')
      .getByLabel('Close')
      .first()
      .click();
    await expect(page.getByTestId('doc-detail-panel')).toBeHidden();
  });

  test('@a11y @modals topbar popovers close on Escape and on outside click', async ({
    page,
  }) => {
    await page.getByTitle('Switch folder').click();
    await expect(page.getByRole('menu', { name: 'Switch folder' })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('menu', { name: 'Switch folder' })).toBeHidden();

    await page.getByRole('button', { name: /Notifications/ }).click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toBeHidden();

    await page.getByRole('button', { name: /Notifications/ }).click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toBeVisible();
    await page.getByRole('heading', { name: 'Document management' }).click();
    await expect(page.getByRole('dialog', { name: 'Notifications' })).toBeHidden();
  });
});
