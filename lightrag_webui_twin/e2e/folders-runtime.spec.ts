import { expect, test, type Page } from './fixtures';
import { getMswStats } from './helpers';

const runtimeUser = {
  sso_subject: 'demo.steward@example.com',
  email: 'demo.steward@example.com',
  name: 'Demo Steward',
  palier: {
    level: 3,
    label: 'Steward',
    scopes: ['twin:read', 'twin:write', 'twin:approve'],
  },
  folders: ['default', 'sandbox', 'ops'],
  idp: 'keycloak',
  idp_realm: 'demo-realm',
  sub: 'clb-7f4e',
  session_expires: '2026-05-19T23:59:00Z',
  gateway_scopes: ['read:documents', 'write:documents', 'read:activity', 'admin:tags'],
};

async function bootWithRuntimeConfig(page: Page, config: Record<string, unknown>) {
  await page.addInitScript((runtimeConfig) => {
    for (let i = window.localStorage.length - 1; i >= 0; i -= 1) {
      const k = window.localStorage.key(i);
      if (k && k.startsWith('twin-rag.threads.v')) window.localStorage.removeItem(k);
    }
    window.__twinE2eRuntimeConfig = runtimeConfig;
  }, config);
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await fetch('/__e2e/reset', { method: 'POST' });
  });
  await page.reload();
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
}

test.describe('Twin folders runtime config', () => {
  test('@folders runtime defaultFolderId and configured folders drive the topbar', async ({
    page,
  }) => {
    await bootWithRuntimeConfig(page, {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultFolderId: 'sandbox',
      maxFolders: 5,
      debugUser: runtimeUser,
      folders: [
        { id: 'default', label: 'Default runtime', kind: 'primary', sources: 12 },
        { id: 'sandbox', label: 'Sandbox runtime', kind: 'sandbox', sources: 2 },
        { id: 'ops', label: 'Ops archive', kind: 'custom', sources: 4 },
      ],
    });

    await expect(page.getByTitle('Switch folder')).toContainText('sandbox');
    await page.getByTitle('Switch folder').click();
    const menu = page.getByRole('menu', { name: 'Switch folder' });
    await expect(menu).toContainText('Folders');
    await expect(menu).toContainText('default');
    await expect(menu).toContainText('sandbox');
    await expect(menu).toContainText('ops');
    await expect(menu).toContainText('Ops archive');
  });

  test('@folders switching folder sends subsequent Twin requests with the new header', async ({
    page,
  }) => {
    await bootWithRuntimeConfig(page, {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultFolderId: 'default',
      maxFolders: 5,
      debugUser: runtimeUser,
      folders: [
        { id: 'default', label: 'Default runtime', kind: 'primary', sources: 12 },
        { id: 'sandbox', label: 'Sandbox runtime', kind: 'sandbox', sources: 2 },
      ],
    });

    await page.getByTitle('Switch folder').click();
    await page.getByRole('menuitemradio', { name: /sandbox/ }).click();
    await expect(page.getByTitle('Switch folder')).toContainText('sandbox');

    await expect
      .poll(async () => {
        const stats = await getMswStats(page);
        return stats.folderRequests
          .filter((request) => request.folder === 'sandbox')
          .map((request) => request.path)
          .sort();
      })
      .toContain('/twin/api/tags');

    const stats = await getMswStats(page);
    expect(
      stats.folderRequests.some(
        (request) =>
          request.path === '/twin/api/activity' &&
          request.folder === 'sandbox',
      ),
    ).toBe(true);
  });

  test('@folders empty configured folder list shows administrator guidance', async ({
    page,
  }) => {
    await bootWithRuntimeConfig(page, {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultFolderId: '',
      maxFolders: 5,
      debugUser: { ...runtimeUser, folders: [] },
      folders: [],
    });

    await page.getByTitle('Switch folder').click();
    await expect(page.getByTestId('topbar-folder-empty')).toHaveText(
      'No folder is provisioned for this knowledge base. Ask your platform administrator to provision one.',
    );
  });
});
