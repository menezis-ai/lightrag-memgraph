/**
 * QA documentation screenshot harness — NOT a behavioural test.
 *
 * Drives the MSW-backed WebUI through every operator screen and the key modals,
 * writing a numbered PNG per surface into `qa-screenshots/`. The Python docx
 * builder (`scripts/qa/build_qa_docx.py`) consumes that folder.
 *
 * Run with:  npm run test:e2e:qa
 *
 * Each surface is its own test so a selector drift on one modal cannot abort the
 * whole capture run. Optional interactions are wrapped in try/catch: we always
 * emit a screenshot of whatever rendered, and log what failed for triage.
 */
import { expect, test, type Page } from './fixtures';
import { boot, openTab, setMswScenario } from './helpers';

const DIR = 'qa-screenshots';

test.use({ viewport: { width: 1440, height: 1100 } });

// This file regenerates the QA documentation screenshots; it is NOT a
// regression spec. Skip it in normal/CI runs so it does not burden the e2e
// job. Regenerate with:  npm run test:e2e:qa
test.beforeEach(() => {
  test.skip(!process.env.CAPTURE_QA, 'QA screenshot harness — run with CAPTURE_QA=1');
});

async function shot(page: Page, name: string) {
  await page.waitForTimeout(400); // let transitions/animations settle
  await page.screenshot({ path: `${DIR}/${name}.png` });
}

async function tryStep(label: string, fn: () => Promise<void>) {
  try {
    await fn();
  } catch (err) {
    console.log(`[capture] optional step skipped: ${label} -> ${(err as Error).message}`);
  }
}

const GATED_RUNTIME_CONFIG = {
  apiBaseUrl: '/twin/api',
  lightragBaseUrl: '',
  idpLogoutUrl: 'https://idp.example.com/logout',
  defaultFolderId: 'default',
  maxFolders: 5,
  folders: [{ id: 'default', label: 'Default folder', kind: 'primary', sources: 12 }],
};

test('01 login screen', async ({ page }) => {
  await page.addInitScript((cfg) => {
    for (let i = window.localStorage.length - 1; i >= 0; i -= 1) {
      const k = window.localStorage.key(i);
      if (k && k.startsWith('twin-rag.threads.v')) window.localStorage.removeItem(k);
    }
    (window as unknown as { __twinE2eRuntimeConfig: unknown }).__twinE2eRuntimeConfig = cfg;
  }, GATED_RUNTIME_CONFIG);
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Documents', exact: true })).toBeVisible();
  await page.evaluate(async () => {
    await fetch('/__e2e/reset', { method: 'POST' });
  });
  await setMswScenario(page, { authGate: true });
  await page.reload();
  await expect(page.getByTestId('login-screen')).toBeVisible();
  await page.getByTestId('login-username').fill('demo.steward');
  await page.getByTestId('login-password').fill('s3cret');
  await shot(page, '01-login');
});

test('02 documents list', async ({ page }) => {
  await boot(page);
  await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
  await shot(page, '02-documents-list');
});

test('03 add source modal', async ({ page }) => {
  await boot(page);
  await page.getByRole('button', { name: 'Add source' }).click();
  await page.waitForTimeout(600);
  await shot(page, '03-add-source-modal');
});

test('04 document detail panel', async ({ page }) => {
  await boot(page);
  await tryStep('open doc row', async () => {
    await page.getByTestId('docs-row-d1').click();
    await page.waitForTimeout(500);
  });
  await shot(page, '04-document-detail');
});

test('05 retag modal', async ({ page }) => {
  await boot(page);
  await tryStep('select a doc and open retag', async () => {
    await page.getByLabel(/^Select /).first().check();
    await page
      .getByLabel('Bulk actions')
      .getByRole('button', { name: /Retag/ })
      .click();
    await page.waitForTimeout(500);
  });
  await shot(page, '05-retag-modal');
});

test('06 tags governance', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Tags');
  await page.waitForTimeout(400);
  await shot(page, '06-tags-list');
});

test('07 tag detail selected', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Tags');
  await tryStep('select a tag card', async () => {
    await page.getByTestId('tag-card-oracle').click();
    await page.waitForTimeout(400);
  });
  await shot(page, '07-tag-detail');
});

test('08 request new tag modal', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Tags');
  await tryStep('open request tag modal', async () => {
    await page.getByRole('button', { name: 'Request new tag' }).click();
    await page.waitForTimeout(400);
  });
  await shot(page, '08-request-tag-modal');
});

test('09 retrieval tab', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Retrieval');
  await page.waitForTimeout(400);
  await shot(page, '09-retrieval-empty');
});

test('10 retrieval answer with sources', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Retrieval');
  await tryStep('ask a question', async () => {
    await page.getByRole('button', { name: /New/ }).first().click();
    await page.getByLabel('Query input').fill('What is the Oracle restart procedure?');
    await page.getByRole('button', { name: 'Send' }).click();
    await expect(page.getByTestId('source-1')).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(800);
  });
  await shot(page, '10-retrieval-answer');
});

test('11 graph tab', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Graph');
  await expect(page.getByTestId('kg-canvas')).toBeVisible();
  await page.waitForTimeout(500);
  await shot(page, '11-graph-canvas');
});

test('12 graph entity inspector', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Graph');
  await tryStep('select an entity node', async () => {
    await page.getByTestId('kg-node-e_memgraph').click();
    await expect(page.getByTestId('kg-detail-entity')).toBeVisible();
    await page.waitForTimeout(400);
  });
  await shot(page, '12-graph-inspector');
});

test('13 graph add entity form', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Graph');
  await tryStep('open add-entity form', async () => {
    await page.getByTestId('kg-add-entity-btn').click();
    await page.waitForTimeout(400);
  });
  await shot(page, '13-graph-add-entity');
});

test('14 activity feed', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Activity');
  await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible();
  await page.waitForTimeout(400);
  await shot(page, '14-activity-feed');
});

test('15 activity event detail', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Activity');
  await tryStep('open an event detail', async () => {
    await page.locator('.activity-row').first().click();
    await expect(page.getByRole('complementary')).toContainText('Event ID');
    await page.waitForTimeout(400);
  });
  await shot(page, '15-activity-detail');
});

test('16 settings profile', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Settings');
  await page.waitForTimeout(400);
  await shot(page, '16-settings-profile');
});

test('17 settings api explorer', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Settings');
  await tryStep('open API rail', async () => {
    await page.getByTestId('settings-rail-api').click();
    await page.waitForTimeout(800);
  });
  await shot(page, '17-settings-api');
});

test('18 settings api keys', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Settings');
  await tryStep('open API keys rail', async () => {
    await page.getByTestId('settings-rail-api-keys').click();
    await page.waitForTimeout(500);
  });
  await shot(page, '18-settings-api-keys');
});

test('19 settings folders admin', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Settings');
  await tryStep('open folder rail', async () => {
    await page.getByTestId('settings-rail-folder').click();
    await page.waitForTimeout(500);
  });
  await shot(page, '19-settings-folders');
});

test('20 notifications popover', async ({ page }) => {
  await boot(page);
  await tryStep('open notifications', async () => {
    await page.getByRole('button', { name: /Notifications/ }).click();
    await page.waitForTimeout(400);
  });
  await shot(page, '20-notifications');
});

test('21 folder switcher', async ({ page }) => {
  await boot(page);
  await tryStep('open folder switcher', async () => {
    await page.getByTitle('Switch folder').click();
    await expect(page.getByRole('menu', { name: 'Switch folder' })).toBeVisible();
    await page.waitForTimeout(400);
  });
  await shot(page, '21-folder-switcher');
});

test('22 settings portability', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Settings');
  await tryStep('open portability rail', async () => {
    await page.getByTestId('settings-rail-portability').click();
    await expect(page.getByTestId('settings-portability')).toBeVisible();
    await page.waitForTimeout(500);
  });
  await shot(page, '22-settings-portability');
});

test('23 portability dry-run report', async ({ page }) => {
  await boot(page);
  await openTab(page, 'Settings');
  await tryStep('run a dry-run', async () => {
    await page.getByTestId('settings-rail-portability').click();
    await page.getByTestId('portability-import-file').setInputFiles({
      name: 'staging-kb.tar.gz',
      mimeType: 'application/gzip',
      buffer: Buffer.from('canonical twin-kb-bundle'),
    });
    await page.getByTestId('portability-import-start').click();
    await expect(page.getByTestId('portability-report')).toBeVisible();
    await page.waitForTimeout(500);
  });
  await shot(page, '23-portability-dry-run');
});
