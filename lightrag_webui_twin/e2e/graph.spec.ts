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

  test('@graph @rc2 entity drill-down opens Documents with the entity name as text query', async ({
    page,
  }) => {
    // Mock-kill F3 — the per-entity `?source=...` filter was dropped
    // because the fixture map `GRAPH_ENTITY_DOCS` was keyed on
    // prototype entity ids and always missed real Memgraph entities.
    // The drill-down now navigates with `?q=<entity name>` so the
    // Documents tab filters by text content instead of explicit
    // source-id list.
    await page.getByLabel('Search entities').fill('Oracle');
    await page.getByLabel('Select entity Oracle Database').click();
    await page
      .getByRole('button', { name: /View documents mentioning this entity/ })
      .click();

    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page).toHaveURL(/[?&]q=/);

    const qParam = await page.evaluate(
      () => new URLSearchParams(window.location.search).get('q'),
    );
    expect(qParam).toBe('Oracle Database');
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
