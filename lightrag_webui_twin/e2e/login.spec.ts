import { expect, test, type Page } from '@playwright/test';
import { setMswScenario } from './helpers';

/**
 * Full local-auth journey against the MSW auth gate.
 *
 * The dev fallback config injects a `debugUser`, which bypasses the gate by
 * design — so these tests boot with an explicit runtime config WITHOUT a
 * debug user (same `__twinE2eRuntimeConfig` hook as folders-runtime). The
 * gate flag and the logged-in user are persisted in sessionStorage by the
 * mock layer, so the login screen survives the reload that follows
 * `setMswScenario`.
 */
const GATED_RUNTIME_CONFIG = {
  apiBaseUrl: '/twin/api',
  lightragBaseUrl: '',
  idpLogoutUrl: 'https://idp.example.com/logout',
  defaultFolderId: 'default',
  maxFolders: 5,
  folders: [
    { id: 'default', label: 'Default folder', kind: 'primary', sources: 12 },
  ],
};

async function bootToLogin(page: Page) {
  await page.addInitScript((cfg) => {
    window.localStorage.removeItem('twin-rag.threads.v3');
    window.__twinE2eRuntimeConfig = cfg;
  }, GATED_RUNTIME_CONFIG);
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await fetch('/__e2e/reset', { method: 'POST' });
  });
  await setMswScenario(page, { authGate: true });
  await page.reload();
  await expect(page.getByTestId('login-screen')).toBeVisible();
}

test.describe('Login screen', () => {
  test('@auth @login auth gate lands on the login screen with a disabled submit', async ({
    page,
  }) => {
    await bootToLogin(page);
    await expect(page.getByRole('heading', { name: 'Twin KMS' })).toBeVisible();
    await expect(page.getByTestId('login-submit')).toBeDisabled();

    await page.getByTestId('login-username').fill('claire.benoit');
    await expect(page.getByTestId('login-submit')).toBeDisabled();
    await page.getByTestId('login-password').fill('s3cret');
    await expect(page.getByTestId('login-submit')).toBeEnabled();

    // The app shell must not leak behind the auth gate.
    await expect(page.getByRole('button', { name: 'Documents', exact: true })).toHaveCount(0);
  });

  test('@auth @login bad credentials surface an error, valid ones enter the app', async ({
    page,
  }) => {
    await bootToLogin(page);
    await page.getByTestId('login-username').fill('claire.benoit');
    await page.getByTestId('login-password').fill('invalid-password');
    await page.getByTestId('login-submit').click();
    await expect(page.getByTestId('login-error')).toBeVisible();
    await expect(page.getByTestId('login-screen')).toBeVisible();

    await page.getByTestId('login-password').fill('s3cret');
    await page.getByTestId('login-submit').click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page.getByTestId('login-screen')).toBeHidden();
  });

  test('@auth @login session survives a reload and sign-out returns to the gate', async ({
    page,
  }) => {
    await bootToLogin(page);
    await page.getByTestId('login-username').fill('claire.benoit');
    await page.getByTestId('login-password').fill('s3cret');
    await page.getByTestId('login-submit').click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();

    await page.reload();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();

    await page.getByRole('button', { name: 'Settings', exact: true }).click();
    await expect(page.getByTestId('settings-profile-name')).toContainText('claire.benoit');

    await page.getByTestId('settings-signout').click();
    await expect(page.getByTestId('login-screen')).toBeVisible();

    const residual = await page.evaluate(() =>
      Object.keys(window.localStorage).filter((key) => key.startsWith('twin-rag.')),
    );
    expect(residual).toEqual([]);
  });
});
