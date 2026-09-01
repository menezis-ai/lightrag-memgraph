import AxeBuilder from '@axe-core/playwright';
import type { Page } from '@playwright/test';
import { expect } from './fixtures';

const BLOCKING_IMPACTS = new Set(['serious', 'critical']);

interface AxeScanOptions {
  include?: string;
}

/**
 * Run the complete axe ruleset and fail only on the impacts that #111 makes
 * blocking. Keep this helper exception-free: any future exclusion must be
 * scoped to one selector/rule and carry an owner plus a review date beside the
 * call site.
 */
export async function expectNoBlockingAxeViolations(
  page: Page,
  options: Readonly<AxeScanOptions> = {},
): Promise<void> {
  // Scan the settled UI, not a translucent intermediate animation frame:
  // contrast computed while a modal/tab fades in is both nondeterministic and
  // different from the state an operator can actually read.
  await page.addStyleTag({
    content:
      '*, *::before, *::after { animation-duration: 0s !important; transition-duration: 0s !important; }',
  });
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())));
  const builder = new AxeBuilder({ page });
  if (options.include) builder.include(options.include);
  const results = await builder.analyze();
  const violations = results.violations
    .filter((violation) => BLOCKING_IMPACTS.has(violation.impact ?? ''))
    .map((violation) => ({
      id: violation.id,
      impact: violation.impact,
      help: violation.help,
      helpUrl: violation.helpUrl,
      nodes: violation.nodes.map((node) => ({
        target: node.target,
        html: node.html,
        failureSummary: node.failureSummary,
      })),
    }));

  expect(
    violations,
    `axe serious/critical violations:\n${JSON.stringify(violations, null, 2)}`,
  ).toEqual([]);
}
