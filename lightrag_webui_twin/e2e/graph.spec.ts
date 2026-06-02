import { expect, test } from '@playwright/test';
import { boot, openTab } from './helpers';

test.describe('Knowledge Graph filters and drill-downs', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await page.evaluate(() => window.localStorage.removeItem('twin.kg.pinned.v1'));
    await openTab(page, 'Graph');
  });

  test('@graph @rc2 entity type counters follow active graph filters', async ({
    page,
  }) => {
    await page.getByLabel('Search entities').fill('swift');

    await expect(page.getByTestId('kg-node-e_swift')).toBeVisible();
    await expect(page.getByTestId('kg-node-e_iso20022')).toBeVisible();
    await expect(page.getByTestId('kg-node-e_oracle')).toBeHidden();
    await expect(page.getByTestId('kg-type-ORG')).toContainText('1');
    await expect(page.getByTestId('kg-type-CONCEPT')).toContainText('1');
    await expect(page.getByTestId('kg-type-PRODUCT')).toContainText('0');
  });

  test('@graph @rc2 entity drill-down opens Documents with exact source filters', async ({
    page,
  }) => {
    await page.getByLabel('Search entities').fill('Oracle');
    await page.getByLabel('Select entity Oracle Database').click();
    await page
      .getByRole('button', { name: /View .* sources mentioning this entity/ })
      .click();

    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page).toHaveURL(/source=/);
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();

    const sourceParam = await page.evaluate(
      () => new URLSearchParams(window.location.search).get('source'),
    );
    expect(sourceParam?.split(',')).toEqual([
      'oracle-restart-procedure.pdf',
      '/cib/runbooks/oracle-pga-tuning',
    ]);
  });

  test('@graph @rc1 pinned entity state survives reload', async ({ page }) => {
    await page.getByTestId('kg-node-e_memgraph').click();
    await expect(page.getByTestId('kg-detail-entity')).toContainText('Memgraph');
    await page.getByTestId('kg-entity-pin').click();
    await expect(page.getByTestId('kg-entity-pin')).toContainText('Pinned');
    await expect(page.getByTestId('kg-entity-pin')).toHaveAttribute(
      'aria-pressed',
      'true',
    );

    await page.reload();
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
    await openTab(page, 'Graph');
    await page.getByTestId('kg-node-e_memgraph').click();
    await expect(page.getByTestId('kg-entity-pin')).toContainText('Pinned');
    await expect(page.getByTestId('kg-entity-pin')).toHaveAttribute(
      'aria-pressed',
      'true',
    );
  });
});
