import {
  allowRequestAbort,
  expect,
  test,
  type Locator,
  type Page,
} from './fixtures';
import { boot, getMswStats, openTab, setMswScenario } from './helpers';

const CATALOG_ADMIN_CONFIG = {
  apiBaseUrl: '/twin/api',
  lightragBaseUrl: '',
  idpLogoutUrl: 'https://idp.example.com/logout',
  defaultFolderId: 'default',
  maxFolders: 5,
  catalogEnabled: true,
  folders: [
    { id: 'default', label: 'Default folder', kind: 'primary', sources: 12 },
    { id: 'sandbox', label: 'Sandbox', kind: 'sandbox', sources: 2 },
  ],
  debugUser: {
    sso_subject: 'catalog.admin@example.com',
    email: 'catalog.admin@example.com',
    name: 'Catalogue Admin',
    palier: {
      level: 3,
      label: 'Steward',
      scopes: ['twin:read', 'twin:write', 'twin:approve'],
    },
    folders: ['default', 'sandbox'],
    idp: 'test',
    idp_realm: 'test-realm',
    sub: 'catalog-admin',
    session_expires: '2099-12-31T23:59:00Z',
    gateway_scopes: [
      'read:documents',
      'write:documents',
      'read:activity',
      'admin:folders',
    ],
  },
};

function sourceCard(page: Page, text: string): Locator {
  return page.locator('article.rag-source-card').filter({ hasText: text });
}

async function openSourcesFromAddSource(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Add source', exact: true }).click();
  await page.getByRole('button', { name: 'Manage Sources RAG' }).click();
  await expect(page.getByRole('heading', { name: 'Sources RAG' })).toBeVisible();
}

test.describe('Sources RAG catalogue journeys', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript((config) => {
      window.__twinE2eRuntimeConfig = config;
    }, CATALOG_ADMIN_CONFIG);
    await boot(page);
  });

  test('Add Source navigation requires visibility, creates once, reloads, and remains folder-scoped', async ({
    allowBrowserIssues,
    page,
  }) => {
    const sourceNavigationReason =
      'The asserted reload and folder switch cancel superseded catalogue-shell queries.';
    allowBrowserIssues(
      allowRequestAbort(/\/twin\/api\/linked-sources$/, sourceNavigationReason, 3),
      allowRequestAbort(/\/twin\/api\/notifications$/, sourceNavigationReason, 2),
      allowRequestAbort(/\/twin\/api\/tags$/, sourceNavigationReason, 2),
      allowRequestAbort(/\/twin\/api\/tags\/categories$/, sourceNavigationReason, 2),
      allowRequestAbort(
        /\/twin\/api\/activity\?range=7d&limit=200$/,
        sourceNavigationReason,
        2,
      ),
    );
    await setMswScenario(page, { linkedSourceCreateDelayMs: 120 });
    await openSourcesFromAddSource(page);
    await page.getByRole('button', { name: 'Add linked source' }).click();

    const url =
      'https://tenant.sharepoint.com/sites/pf/Shared/Documents/finger-slip.pdf';
    await page.getByRole('textbox', { name: 'URL' }).fill(url);
    await page.getByRole('button', { name: 'Preview change' }).click();
    await expect(page.locator('.rag-state.error[role="alert"]')).toContainText(
      'Choose Public or Restricted before previewing',
    );
    await expect.poll(() => getMswStats(page)).toMatchObject({
      linkedSourcePreviewCalls: 0,
      linkedSourceCreateCalls: 0,
      linkedSourceCreateTransitions: 0,
    });

    await page.getByRole('radio', { name: 'Public', exact: true }).click();
    await page.getByRole('button', { name: 'Preview change' }).click();
    await expect(page.getByLabel('Mutation preview')).toContainText(
      'Catalogue preview · create',
    );
    await page.getByRole('button', { name: 'Confirm change' }).dblclick({
      delay: 5,
    });

    const createdCard = sourceCard(page, 'finger-slip.pdf');
    await expect(createdCard).toBeVisible();
    await expect(createdCard).toContainText('Public');
    await expect.poll(() => getMswStats(page)).toMatchObject({
      linkedSourcePreviewCalls: 1,
      linkedSourceCreateCalls: 1,
      linkedSourceCreateTransitions: 1,
    });

    await page.reload();
    await openTab(page, 'Sources RAG');
    await expect(sourceCard(page, 'finger-slip.pdf')).toContainText('Public');

    await page.getByTitle('Switch folder').click();
    await page.getByRole('menuitemradio', { name: /sandbox/ }).click();
    await expect(page.getByRole('heading', { name: 'Sources RAG' })).toBeVisible();
    await expect(page.getByText('Folder sandbox')).toBeVisible();
    await expect(sourceCard(page, 'finger-slip.pdf')).toHaveCount(0);

    await page.getByTitle('Switch folder').click();
    await page.getByRole('menuitemradio', { name: /default/ }).click();
    await expect(sourceCard(page, 'finger-slip.pdf')).toBeVisible();
    await expect.poll(async () => {
      const stats = await getMswStats(page);
      return stats.folderRequests
        .filter((request) => request.path === '/twin/api/linked-sources')
        .map((request) => request.folder);
    }).toEqual(expect.arrayContaining(['default', 'sandbox']));
  });

  test('a stale disable is explained, changes nothing, then recovers once after reload', async ({
    allowBrowserIssues,
    page,
  }) => {
    allowBrowserIssues(
      allowRequestAbort(
        /\/twin\/api\/linked-sources$/,
        'Reloading after the asserted stale conflict replaces one catalogue list query.',
      ),
    );
    await setMswScenario(page, { linkedSourceDisableConflictOnce: true });
    await openTab(page, 'Sources RAG');

    let card = sourceCard(page, 'PF Move2Cloud');
    await card.getByRole('button', { name: 'Disable' }).click();
    await page.getByRole('button', { name: 'Confirm change' }).click();
    await expect(page.locator('.rag-state.error[role="alert"]')).toContainText(
      'row_version mismatch — reload and retry',
    );
    await expect(card).toContainText('active');
    await expect(card.getByRole('button', { name: 'Disable' })).toBeVisible();
    await expect.poll(() => getMswStats(page)).toMatchObject({
      linkedSourceDisableCalls: 1,
      linkedSourceDisableTransitions: 0,
    });

    await page.reload();
    await openTab(page, 'Sources RAG');
    card = sourceCard(page, 'PF Move2Cloud');
    await expect(card).toContainText('active');
    await card.getByRole('button', { name: 'Disable' }).click();
    await page.getByRole('button', { name: 'Confirm change' }).click();
    await expect(card).toContainText('disabled');
    await expect(card.getByRole('button', { name: 'Disable' })).toHaveCount(0);
    await expect.poll(() => getMswStats(page)).toMatchObject({
      linkedSourceDisableCalls: 1,
      linkedSourceDisableTransitions: 1,
    });

    await page.reload();
    await openTab(page, 'Sources RAG');
    await expect(sourceCard(page, 'PF Move2Cloud')).toContainText('disabled');
  });
});
