import {
  allowRequestAbort,
  expect,
  test,
  type BrowserIssueAllowance,
  type Page,
} from './fixtures';
import { expectNoBlockingAxeViolations } from './a11y';
import { boot, openTab, setMswScenario } from './helpers';

const OPERATOR_TABS = [
  'Documents',
  'Tags',
  'Retrieval',
  'Graph',
  'Activity',
  'Settings',
] as const;

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

function tabNavigationAllowances(tab: (typeof OPERATOR_TABS)[number]): BrowserIssueAllowance[] {
  if (tab === 'Documents') return [];
  return [
    allowRequestAbort(
      /\/twin\/api\/procedures$/,
      `The axe ${tab} scan intentionally unmounts the Documents procedure query.`,
    ),
  ];
}

async function bootToLogin(page: Page): Promise<void> {
  await page.addInitScript((config) => {
    window.__twinE2eRuntimeConfig = config;
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

test.describe('axe serious/critical accessibility gate', () => {
  test('@axe login has no blocking axe violation', async ({ page }) => {
    await bootToLogin(page);
    await expectNoBlockingAxeViolations(page);
  });

  for (const tab of OPERATOR_TABS) {
    test(`@axe ${tab} has no blocking axe violation`, async ({
      allowBrowserIssues,
      page,
    }) => {
      allowBrowserIssues(...tabNavigationAllowances(tab));
      await boot(page);
      if (tab !== 'Documents') await openTab(page, tab);
      await expect(page.locator('main .tab-pane')).toBeVisible();
      await expectNoBlockingAxeViolations(page);
    });
  }

  test('@axe document modals have no blocking axe violation', async ({ page }) => {
    await boot(page);

    await page.getByRole('button', { name: 'Add source' }).click();
    await expect(page.getByRole('dialog', { name: 'Add source' })).toBeVisible();
    await expectNoBlockingAxeViolations(page);
    await page.getByLabel('Close dialog').click();

    await page.getByLabel('Select oracle-restart-procedure.pdf').check();
    await page
      .getByLabel('Bulk actions')
      .getByRole('button', { name: /Retag 1 sources/ })
      .click();
    await expect(page.getByRole('dialog', { name: 'Retag document' })).toBeVisible();
    await expectNoBlockingAxeViolations(page);
    await page.getByLabel('Close dialog').click();

    await page.getByTestId('pending-doc-read-d6').click();
    await expect(page.getByRole('dialog', { name: 'Indexed chunks' })).toBeVisible();
    await expectNoBlockingAxeViolations(page);
  });

  test('@axe tag governance modals have no blocking axe violation', async ({
    allowBrowserIssues,
    page,
  }) => {
    allowBrowserIssues(...tabNavigationAllowances('Tags'));
    await boot(page);
    await openTab(page, 'Tags');
    await page.getByLabel('Search tags').fill('oracle');
    await page.getByTestId('tag-card-oracle').click();

    await page.getByRole('button', { name: 'Edit', exact: true }).click();
    await expect(page.getByRole('dialog', { name: 'Edit tag' })).toBeVisible();
    await expectNoBlockingAxeViolations(page);
    await page.getByLabel('Close dialog').click();

    await page.getByRole('button', { name: 'Request new tag' }).click();
    await expect(page.getByRole('dialog', { name: /Request/ })).toBeVisible();
    await expectNoBlockingAxeViolations(page);
  });

  test('@axe API authorize modal has no blocking axe violation', async ({
    allowBrowserIssues,
    page,
  }) => {
    allowBrowserIssues(
      ...tabNavigationAllowances('Settings'),
      allowRequestAbort(
        /\/twin\/api\/quota$/,
        'Opening the API settings rail replaces its bootstrap quota query.',
      ),
      allowRequestAbort(
        /\/twin\/api\/settings\/vision$/,
        'Opening the API settings rail replaces its bootstrap vision query.',
      ),
      allowRequestAbort(
        /\/openapi\.json$/,
        'Closing the axe API modal may cancel the explorer schema query at teardown.',
      ),
    );
    await boot(page);
    await openTab(page, 'Settings');
    await page.getByTestId('settings-rail-api').click();
    await page.getByRole('button', { name: 'Authorize' }).click();
    await expect(page.getByRole('dialog', { name: 'Authorize' })).toBeVisible();
    await expectNoBlockingAxeViolations(page);
  });
});
