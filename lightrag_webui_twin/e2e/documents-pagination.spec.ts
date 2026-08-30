import { expect, test } from './fixtures';
import { boot, seedDocuments } from './helpers';

/**
 * Document list pagination. The mock paginates at 50 documents/page with a
 * cursor; seeding 60 documents forces a second page so the Previous/Next
 * controls and the page label become exercisable. Closes the "pagination
 * boundary" gap flagged in the QA plan (§7.1).
 */
test.describe('Documents pagination', () => {
  test('@documents @pagination Next advances to a second page and Previous re-enables', async ({
    page,
  }) => {
    await boot(page);

    const docs = Array.from({ length: 60 }, (_, i) => {
      const n = String(i + 1).padStart(3, '0');
      return {
        doc_id: `page_doc_${n}`,
        file_path: `/demo/runbooks/page-doc-${n}.md`,
        content_summary: `Pagination probe document ${n}`,
        status: 'PROCESSED',
        folder: 'default',
      };
    });
    await seedDocuments(page, docs);

    const pagination = page.getByTestId('docs-pagination');
    await expect(pagination).toBeVisible();
    await expect(pagination.locator('.docs-pagination-label')).toContainText('Page 1');

    const prev = page.getByTestId('docs-page-prev');
    const next = page.getByTestId('docs-page-next');
    await expect(prev).toBeDisabled();
    await expect(next).toBeEnabled();

    await next.click();

    await expect(pagination.locator('.docs-pagination-label')).toContainText('Page 2');
    await expect(prev).toBeEnabled();
  });
});
