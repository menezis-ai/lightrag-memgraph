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
  workspaces: ['default', 'sandbox', 'ops'],
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
    window.localStorage.removeItem('twin-rag.threads.v2');
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

test.describe('Twin spaces runtime config', () => {
  test('@spaces runtime defaultSpaceId and configured spaces drive the topbar', async ({
    page,
  }) => {
    await bootWithRuntimeConfig(page, {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultSpaceId: 'sandbox',
      maxSpaces: 5,
      debugUser: runtimeUser,
      spaces: [
        { id: 'default', label: 'Default runtime', kind: 'primary', sources: 12 },
        { id: 'sandbox', label: 'Sandbox runtime', kind: 'sandbox', sources: 2 },
        { id: 'ops', label: 'Ops archive', kind: 'custom', sources: 4 },
      ],
    });

    await expect(page.getByTitle('Switch space')).toContainText('sandbox');
    await page.getByTitle('Switch space').click();
    const menu = page.getByRole('menu', { name: 'Switch space' });
    await expect(menu).toContainText('Spaces');
    await expect(menu).toContainText('default');
    await expect(menu).toContainText('sandbox');
    await expect(menu).toContainText('ops');
    await expect(menu).toContainText('Ops archive');
  });

  test('@spaces switching space sends subsequent Twin requests with the new header', async ({
    page,
  }) => {
    await bootWithRuntimeConfig(page, {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultSpaceId: 'default',
      maxSpaces: 5,
      debugUser: runtimeUser,
      spaces: [
        { id: 'default', label: 'Default runtime', kind: 'primary', sources: 12 },
        { id: 'sandbox', label: 'Sandbox runtime', kind: 'sandbox', sources: 2 },
      ],
    });

    await page.getByTitle('Switch space').click();
    await page.getByRole('menuitemradio', { name: /sandbox/ }).click();
    await expect(page.getByTitle('Switch space')).toContainText('sandbox');

    await expect
      .poll(async () => {
        const stats = await getMswStats(page);
        return stats.spaceRequests
          .filter((request) => request.space === 'sandbox' && request.workspace === 'sandbox')
          .map((request) => request.path)
          .sort();
      })
      .toContain('/twin/api/tags');

    const stats = await getMswStats(page);
    expect(
      stats.spaceRequests.some(
        (request) =>
          request.path === '/twin/api/activity' &&
          request.space === 'sandbox' &&
          request.workspace === 'sandbox',
      ),
    ).toBe(true);
  });

  test('@spaces empty configured space list shows the Twincore guidance', async ({
    page,
  }) => {
    await bootWithRuntimeConfig(page, {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: 'https://idp.example.com/logout',
      defaultSpaceId: '',
      maxSpaces: 5,
      debugUser: { ...runtimeUser, workspaces: [] },
      spaces: [],
    });

    await page.getByTitle('Switch space').click();
    await expect(page.getByTestId('topbar-workspace-empty')).toHaveText(
      'No space available for this KB. Please contact Twincore Team',
    );
  });
});
