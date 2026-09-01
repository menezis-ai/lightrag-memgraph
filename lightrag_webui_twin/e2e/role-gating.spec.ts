import { allowRequestAbort, expect, test, type Page } from './fixtures';
import { openTab } from './helpers';

/**
 * Role-based access gating on the folder-admin surface.
 *
 * Folder administration is the one operator surface whose UI affordances are
 * gated by the *authenticated* user: SettingsTab reads useAuth().user and
 * passes it to FoldersAdminSection, which renders the Add/Edit/Delete controls
 * only when canManageFolders(user) — i.e. the user carries the `admin:folders`
 * gateway scope. We inject the identity through the same `__twinE2eRuntimeConfig`
 * debugUser hook the folders-runtime / login specs use.
 *
 * Tag governance now derives its palier from the same authenticated user and is
 * covered by the frontendIdentity + TagsTab unit suites. This browser spec stays
 * focused on folder administration; Graph and API-key controls remain backend
 * responsibilities.
 */

const BASE = {
  apiBaseUrl: '/twin/api',
  lightragBaseUrl: '',
  idpLogoutUrl: 'https://idp.example.com/logout',
  defaultFolderId: 'default',
  maxFolders: 5,
  folders: [
    { id: 'default', label: 'Default folder', kind: 'primary', sources: 12 },
    { id: 'sandbox', label: 'Sandbox', kind: 'sandbox', sources: 2 },
  ],
};

function user(level: 1 | 2 | 3, label: string, gatewayScopes: string[]) {
  return {
    sso_subject: `${label.toLowerCase()}@example.com`,
    email: `${label.toLowerCase()}@example.com`,
    name: `Test ${label}`,
    palier: { level, label, scopes: ['twin:read'] },
    folders: ['default', 'sandbox'],
    idp: 'test',
    idp_realm: 'test-realm',
    sub: `${label.toLowerCase()}-sub`,
    session_expires: '2099-12-31T23:59:00Z',
    gateway_scopes: gatewayScopes,
  };
}

const READER_CONFIG = {
  ...BASE,
  debugUser: user(1, 'Reader', ['read:documents', 'read:activity']),
};

const ADMIN_CONFIG = {
  ...BASE,
  debugUser: user(3, 'Steward', [
    'read:documents',
    'write:documents',
    'read:activity',
    'admin:folders',
  ]),
};

async function bootAs(page: Page, config: Record<string, unknown>) {
  await page.addInitScript((cfg) => {
    for (let i = window.localStorage.length - 1; i >= 0; i -= 1) {
      const k = window.localStorage.key(i);
      if (k && k.startsWith('twin-rag.threads.v')) window.localStorage.removeItem(k);
    }
    (window as unknown as { __twinE2eRuntimeConfig: unknown }).__twinE2eRuntimeConfig = cfg;
  }, config);
  await page.goto('/');
  // debugUser bypasses the auth gate; the app shell renders directly.
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await fetch('/__e2e/reset', { method: 'POST' });
  });
  await openTab(page, 'Settings');
  await page.getByTestId('settings-rail-folder').click();
}

test.describe('Role gating — folder administration', () => {
  test.beforeEach(async ({ allowBrowserIssues }) => {
    allowBrowserIssues(
      allowRequestAbort(
        /\/twin\/api\/quota$/,
        'Opening Settings may replace the still-active quota bootstrap query.',
      ),
    );
  });

  test('@rbac a Reader (no admin:folders) sees folders read-only', async ({ page }) => {
    await bootAs(page, READER_CONFIG);
    await expect(page.getByTestId('folders-admin-readonly-badge')).toBeVisible();
    await expect(page.getByTestId('settings-add-folder-btn')).toHaveCount(0);
  });

  test('@rbac an Admin (admin:folders) can administer folders', async ({ page }) => {
    await bootAs(page, ADMIN_CONFIG);
    await expect(page.getByTestId('folders-admin-readonly-badge')).toHaveCount(0);
    await expect(page.getByTestId('settings-add-folder-btn')).toBeVisible();
  });
});
