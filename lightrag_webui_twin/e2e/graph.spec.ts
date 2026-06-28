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
      .getByRole('button', { name: /View documents mentioning this entity/ })
      .click();

    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page).toHaveURL(/[?&]source=/);
    await expect(page).not.toHaveURL(/[?&]q=/);

    const sourceParam = await page.evaluate(
      () => new URLSearchParams(window.location.search).get('source'),
    );
    expect(sourceParam?.split(',')).toEqual([
      'oracle-restart-procedure.pdf',
      '/cib/runbooks/oracle-pga-tuning',
    ]);
    await expect(
      page.getByTestId('source-filter-oracle-restart-procedure.pdf'),
    ).toBeVisible();
    await expect(
      page.getByTestId('source-filter-/cib/runbooks/oracle-pga-tuning'),
    ).toBeVisible();
    await expect(page.getByTestId('docs-row-d1')).toBeVisible();
    await expect(page.getByTestId('docs-row-d4')).toBeHidden();
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

test.describe('Knowledge Graph entity and relation lifecycle', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await page.evaluate(() => window.localStorage.removeItem('twin.kg.pinned.v1'));
    await openTab(page, 'Graph');
    await expect(page.getByTestId('kg-canvas')).toBeVisible();
  });

  test('@graph @crud add entity creates a node and rejects duplicates', async ({
    page,
  }) => {
    await page.getByTestId('kg-add-entity-btn').click();
    await expect(page.getByTestId('kg-add-entity-submit')).toBeDisabled();
    await page.getByTestId('kg-add-entity-name').fill('E2E Probe Entity');
    await page.getByTestId('kg-add-entity-summary').fill('Created by the e2e crud suite.');
    await expect(page.getByTestId('kg-add-entity-submit')).toBeEnabled();
    await page.getByTestId('kg-add-entity-submit').click();

    await expect(page.getByTestId('kg-node-kg_E2E Probe Entity')).toBeVisible();

    // Same name again → inline duplicate guard, no second node.
    await page.getByTestId('kg-add-entity-btn').click();
    await page.getByTestId('kg-add-entity-name').fill('E2E Probe Entity');
    await expect(page.getByTestId('kg-add-entity-duplicate')).toBeVisible();
    await expect(page.getByTestId('kg-add-entity-submit')).toBeDisabled();
  });

  test('@graph @crud edit entity metadata persists in the detail panel', async ({
    page,
  }) => {
    await page.getByTestId('kg-node-e_cypher').click();
    await expect(page.getByTestId('kg-detail-entity')).toContainText('Cypher');
    await page.getByTestId('kg-entity-edit').click();
    await page
      .getByLabel('Entity summary')
      .fill('Graph query language — summary rewritten by e2e.');
    await page.getByTestId('kg-entity-save').click();
    await expect(page.getByTestId('kg-detail-entity')).toContainText(
      'summary rewritten by e2e',
    );
  });

  test('@graph @crud @rc1 rename, retype and tag an entity — persists after reload', async ({
    page,
  }) => {
    // Contract: a graph mutation survives a reload. The real backend writes it
    // to Memgraph (`SET n += $props`); the MSW mock mirrors that via
    // sessionStorage (persistGraphState). Re-selection after reload uses the
    // STABLE id testid (kg-node-e_cypher), never the renamed label.
    await page.getByTestId('kg-node-e_cypher').click();
    const detail = page.getByTestId('kg-detail-entity');
    await expect(detail).toContainText('Cypher');

    await page.getByTestId('kg-entity-edit').click();
    const nameInput = page.getByLabel('Entity name');
    await expect(nameInput).toBeVisible();
    await page.waitForTimeout(80);
    await nameInput.click();
    await nameInput.fill('');
    await nameInput.fill('Cypher Query Language');
    await page.getByLabel('Entity type').selectOption('PRODUCT');

    const tagInput = page.getByLabel('Add node tag');
    await tagInput.click();
    await tagInput.fill('memgraph');
    await detail.getByTestId('kg-tag-sugg-memgraph').click();
    await expect(detail.getByLabel('Remove memgraph')).toBeVisible();

    await page.getByTestId('kg-entity-save').click();
    await expect(
      detail.getByRole('heading', { name: 'Cypher Query Language' }),
    ).toBeVisible();
    await expect(detail).toContainText('Product');
    await expect(detail.getByText('memgraph', { exact: true })).toBeVisible();

    await page.reload();
    await expect(
      page.getByRole('button', { name: 'Documents', exact: true }),
    ).toBeVisible();
    await openTab(page, 'Graph');
    await expect(page.getByTestId('kg-canvas')).toBeVisible();
    await page.getByTestId('kg-node-e_cypher').click();

    const reloaded = page.getByTestId('kg-detail-entity');
    await expect(
      reloaded.getByRole('heading', { name: 'Cypher Query Language' }),
    ).toBeVisible();
    await expect(reloaded).toContainText('Product');
    await expect(reloaded.getByText('memgraph', { exact: true })).toBeVisible();
  });

  test('@graph @crud delete entity is double-armed and cancellable', async ({
    page,
  }) => {
    await page.getByTestId('kg-node-e_mage').click();
    await expect(page.getByTestId('kg-detail-entity')).toContainText('MAGE 3.8');

    await page.getByTestId('kg-entity-delete').click();
    await expect(page.getByTestId('kg-entity-delete')).toContainText(
      'Click again to confirm',
    );
    await page.getByTestId('kg-entity-delete-cancel').click();
    await expect(page.getByTestId('kg-entity-delete')).toContainText('Delete entity');
    await expect(page.getByTestId('kg-node-e_mage')).toBeVisible();

    await page.getByTestId('kg-entity-delete').click();
    await page.getByTestId('kg-entity-delete').click();
    await expect(page.getByTestId('kg-node-e_mage')).toBeHidden();
  });

  test('@graph @crud add relation links two existing entities', async ({ page }) => {
    await page.getByTestId('kg-node-e_cft').click();
    await expect(page.getByTestId('kg-detail-entity')).toContainText('CFT');

    await page.getByTestId('kg-add-rel-btn').click();
    // e_cft → e_swift already exists in the fixtures (duplicate guard keeps
    // the submit disabled) — link towards DC Paris instead.
    await page.getByTestId('kg-add-rel-target').selectOption('e_paris');
    await page.getByTestId('kg-add-rel-label').fill('DEPLOYED_AT');
    await page.getByTestId('kg-add-rel-submit').click();

    await expect(
      page.getByTestId('kg-detail-entity').getByText('DEPLOYED_AT'),
    ).toBeVisible();
  });
});
