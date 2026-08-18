/**
 * Responsive topbar regression (TR-RESP-01 / QA report
 * 2026-06-12). The brand + KB name used to overlap the
 * absolutely-centered tabs nav. The 901–1100px Tier-B band keeps the in-flow
 * horizontal layout; at the V8 narrow breakpoint the topbar switches to two
 * grid rows, with the tabs rail below the brand/actions row. These tests pin
 * both no-overlap contracts.
 */

import { expect, test } from '@playwright/test';
import { boot } from './helpers';

test.describe('Responsive topbar', () => {
  test('@regression @responsive brand and tabs stay horizontally separated in Tier-B', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 1000, height: 720 });
    await boot(page);

    const brand = await page.locator('.topbar .brand').boundingBox();
    const tabs = await page.locator('.topbar .tabs').boundingBox();

    expect(brand, 'brand button must exist').not.toBeNull();
    expect(tabs, 'tabs nav must exist').not.toBeNull();

    const brandRight = brand!.x + brand!.width;
    expect(brandRight).toBeLessThanOrEqual(tabs!.x);
  });

  test('@regression @responsive brand and tabs do not overlap at 900px', async ({
    page,
  }) => {
    // 900px is the inclusive V8 breakpoint and the value described by QA.
    await page.setViewportSize({ width: 900, height: 720 });
    await boot(page);

    const brand = await page.locator('.topbar .brand').boundingBox();
    const tabs = await page.locator('.topbar .tabs').boundingBox();

    expect(brand, 'brand button must exist').not.toBeNull();
    expect(tabs, 'tabs nav must exist').not.toBeNull();

    // The tabs intentionally occupy the second grid row. Comparing only the
    // x-axis would report a false overlap because both rows start at x=12.
    const brandBottom = brand!.y + brand!.height;
    expect(brandBottom).toBeLessThanOrEqual(tabs!.y);
  });

  test('@regression @responsive brand and tabs do not overlap at the small mobile viewport', async ({
    page,
  }) => {
    // Belt and braces: the two-row contract remains intact on a narrower
    // viewport while the tabs rail scrolls horizontally within its own row.
    await page.setViewportSize({ width: 760, height: 720 });
    await boot(page);

    const brand = await page.locator('.topbar .brand').boundingBox();
    const tabs = await page.locator('.topbar .tabs').boundingBox();

    expect(brand).not.toBeNull();
    expect(tabs).not.toBeNull();

    const brandBottom = brand!.y + brand!.height;
    expect(brandBottom).toBeLessThanOrEqual(tabs!.y);
  });
});
