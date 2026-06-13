/**
 * Responsive topbar regression (TR-RESP-01 / recette Alberto
 * 2026-06-12). Below 1100px the brand + KB name used to overlap the
 * absolutely-centered tabs nav. The fix in ``styles/legacy.css``
 * switches ``.tabs`` to in-flow flex at the same breakpoint where
 * brand text already truncates; this test pins the no-overlap
 * contract at a representative sub-1100px viewport.
 */

import { expect, test } from '@playwright/test';
import { boot } from './helpers';

test.describe('Responsive topbar', () => {
  test('@regression @responsive brand and tabs do not overlap below 1100px', async ({
    page,
  }) => {
    // Pick a viewport well inside the Tier-B band (1024-1099px is
    // where the absolute-centered ``.tabs`` overlapped the brand;
    // 900px is the value Alberto's recette report described).
    await page.setViewportSize({ width: 900, height: 720 });
    await boot(page);

    const brand = await page.locator('.topbar .brand').boundingBox();
    const tabs = await page.locator('.topbar .tabs').boundingBox();

    expect(brand, 'brand button must exist').not.toBeNull();
    expect(tabs, 'tabs nav must exist').not.toBeNull();

    // The brand sits at the start of the row; the tabs container
    // sits to its right with at least the 28px flex gap between
    // them. Pre-fix, the brand could span past the tabs' left edge
    // because tabs were positioned absolutely with translateX(-50%).
    const brandRight = brand!.x + brand!.width;
    expect(brandRight).toBeLessThanOrEqual(tabs!.x);
  });

  test('@regression @responsive brand and tabs do not overlap at the small mobile viewport', async ({
    page,
  }) => {
    // Belt and braces: an even narrower viewport. The fix should
    // still keep the row sane (tabs may wrap into a tighter row but
    // must never sit underneath the brand).
    await page.setViewportSize({ width: 760, height: 720 });
    await boot(page);

    const brand = await page.locator('.topbar .brand').boundingBox();
    const tabs = await page.locator('.topbar .tabs').boundingBox();

    expect(brand).not.toBeNull();
    expect(tabs).not.toBeNull();

    const brandRight = brand!.x + brand!.width;
    expect(brandRight).toBeLessThanOrEqual(tabs!.x);
  });
});
