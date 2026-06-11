import { expect, test, type Page } from '@playwright/test';
import { getMswStats } from './helpers';

const runtimeUser = {
  sso_subject: 'claire.benoit@demo.local',
  email: 'claire.benoit@demo.local',
  name: 'Claire Benoit',
  palier: {
    level: 3,
    label: 'Steward',
    scopes: ['twin:read', 'twin:write', 'twin:approve'],
  },
  folders: ['default', 'sandbox', 'ops'],
  idp: 'keycloak',
  idp_realm: 'twin-cib',
  sub: 'clb-7f4e',
  session_expires: '2026-05-19T23:59:00Z',
  gateway_scopes: ['read:documents', 'write:documents', 'read:activity', 'admin:tags'],
};

async function bootWithRuntimeConfig(page: Page, config: Record<string, unknown>) {
  await page.addInitScript((runtimeConfig) => {
    window.localStorage.setItem(
      'twin.onboarding.v1',
      JSON.stringify({ step: 'completion', dismissed: true, tasks: [] }),
    );
    window.localStorage.removeItem('twin-rag.threads.v3');
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

  test('@folders empty configured folder list shows the Twincore guidance', async ({
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
      'No folder available for this KB. Please contact Twincore Team',
    );
  });
});
