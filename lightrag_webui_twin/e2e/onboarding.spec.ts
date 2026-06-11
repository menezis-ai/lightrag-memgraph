import { expect, test, type Page } from '@playwright/test';
import { boot, openTab } from './helpers';

/**
 * Onboarding wizard journeys. Unlike `boot()` (which pre-dismisses the wizard
 * so other suites never see it), these tests boot with a virgin
 * `twin.onboarding.v1` so the 6-step tour shows up exactly as it would for a
 * first-time operator.
 */
async function bootFirstVisit(page: Page) {
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await fetch('/__e2e/reset', { method: 'POST' });
    window.localStorage.removeItem('twin.onboarding.v1');
    window.localStorage.removeItem('twin-rag.threads.v2');
  });
  await page.reload();
  await expect(page.getByRole('dialog', { name: 'Onboarding' })).toBeVisible();
}

const wizard = (page: Page) => page.getByRole('dialog', { name: 'Onboarding' });

test.describe('Onboarding wizard', () => {
  test('@onboarding full 6-step walkthrough completes and stays dismissed after reload', async ({
    page,
  }) => {
    await bootFirstVisit(page);
    await expect(page.getByTestId('onboarding-step-welcome')).toBeVisible();
    await expect(page.getByTestId('onboarding-prev')).toHaveCount(0);

    await page.getByTestId('onboarding-next').click();
    await expect(page.getByTestId('onboarding-step-kb-empty')).toBeVisible();

    await page.getByTestId('onboarding-next').click();
    await expect(page.getByTestId('onboarding-step-checklist')).toBeVisible();
    await page.getByTestId('onboarding-task-add-source').check();
    await expect(page.getByTestId('onboarding-task-add-source')).toBeChecked();

    await page.getByTestId('onboarding-next').click();
    await expect(page.getByTestId('onboarding-step-first-source')).toBeVisible();

    await page.getByTestId('onboarding-next').click();
    await expect(page.getByTestId('onboarding-step-first-query')).toBeVisible();

    await page.getByTestId('onboarding-prev').click();
    await expect(page.getByTestId('onboarding-step-first-source')).toBeVisible();
    await page.getByTestId('onboarding-next').click();

    await page.getByTestId('onboarding-next').click();
    await expect(page.getByTestId('onboarding-step-completion')).toBeVisible();
    await expect(page.getByTestId('onboarding-next')).toHaveCount(0);

    await page.getByTestId('onboarding-done').click();
    await expect(wizard(page)).toBeHidden();

    await page.reload();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(wizard(page)).toBeHidden();
  });

  test('@onboarding skip dismisses for good across reloads', async ({ page }) => {
    await bootFirstVisit(page);
    await page.getByTestId('onboarding-skip').click();
    await expect(wizard(page)).toBeHidden();

    await page.reload();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(wizard(page)).toBeHidden();
  });

  test('@onboarding backdrop click closes the wizard', async ({ page }) => {
    await bootFirstVisit(page);
    await page
      .getByTestId('onboarding-backdrop')
      .click({ position: { x: 5, y: 5 } });
    await expect(wizard(page)).toBeHidden();
  });

  test('@onboarding checklist progress and current step survive a reload', async ({
    page,
  }) => {
    await bootFirstVisit(page);
    await page.getByTestId('onboarding-next').click();
    await page.getByTestId('onboarding-next').click();
    await expect(page.getByTestId('onboarding-step-checklist')).toBeVisible();
    await page.getByTestId('onboarding-task-first-query').check();

    await page.reload();
    await expect(page.getByTestId('onboarding-step-checklist')).toBeVisible();
    await expect(page.getByTestId('onboarding-task-first-query')).toBeChecked();
    await expect(page.getByTestId('onboarding-task-add-source')).not.toBeChecked();
  });

  test('@onboarding first-source step hands off to the Add Source modal', async ({
    page,
  }) => {
    await bootFirstVisit(page);
    for (let i = 0; i < 3; i += 1) {
      await page.getByTestId('onboarding-next').click();
    }
    await expect(page.getByTestId('onboarding-step-first-source')).toBeVisible();
    await page.getByTestId('onboarding-add-source').click();
    await expect(wizard(page)).toBeHidden();
    await expect(page.getByRole('dialog', { name: 'Add source' })).toBeVisible();

    await page.getByLabel('Close dialog').click();
    await page.reload();
    await expect(wizard(page)).toBeHidden();
  });

  test('@onboarding first-query step hands off to the Retrieval tab', async ({
    page,
  }) => {
    await bootFirstVisit(page);
    for (let i = 0; i < 4; i += 1) {
      await page.getByTestId('onboarding-next').click();
    }
    await expect(page.getByTestId('onboarding-step-first-query')).toBeVisible();
    await page.getByTestId('onboarding-go-retrieval').click();
    await expect(wizard(page)).toBeHidden();
    await expect(page.getByLabel('Query input')).toBeVisible();
  });

  test('@onboarding @settings restart tutorial reopens the wizard at the welcome step', async ({
    page,
  }) => {
    await boot(page);
    await openTab(page, 'Settings');
    await page.getByTestId('settings-restart-tutorial').click();
    await expect(page.getByRole('status')).toContainText('Tutorial restarted');
    await expect(wizard(page)).toBeVisible();
    await expect(page.getByTestId('onboarding-step-welcome')).toBeVisible();

    await page.getByTestId('onboarding-skip').click();
    await expect(wizard(page)).toBeHidden();
  });
});
