import { expect, test } from '@playwright/test';
import { boot, openTab } from './helpers';

test.describe('Retrieval citations', () => {
  test.beforeEach(async ({ page }) => {
    await boot(page);
    await openTab(page, 'Retrieval');
  });

  test('@retrieval @rc4 clicking a citation opens the referenced source in Documents', async ({
    page,
  }) => {
    await expect(page.getByTestId('source-1')).toContainText(
      'oracle-restart-procedure.pdf',
    );

    await page.getByTestId('citation-1').click();
    await expect(page.getByRole('heading', { name: 'Document management' })).toBeVisible();
    await expect(page.getByLabel('Search source')).toHaveValue('oracle-restart-procedure.pdf');
    await expect(page.getByTestId('docs-row-d1')).toContainText(
      'oracle-restart-procedure.pdf',
    );
  });
});
