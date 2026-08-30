import {
  expect,
  test as base,
  type ConsoleMessage,
  type Page,
} from '@playwright/test';

export interface BrowserIssue {
  kind: 'console.error' | 'pageerror' | 'requestfailed';
  detail: string;
  message: string;
  url: string;
}

type BrowserGuardFixtures = {
  allowBrowserIssues: (...allowances: BrowserIssueAllowance[]) => void;
  browserGuard: void;
  browserIssueRegistry: BrowserIssueAllowance[];
};

export interface BrowserIssueAllowance {
  kind: BrowserIssue['kind'];
  url: RegExp;
  message: RegExp;
  maxOccurrences: number;
  reason: string;
}

const HTTP_RESPONSE_CONSOLE_ERROR =
  /^Failed to load resource: the server responded with a status of 4\d{2}\b/;

export function allowRequestAbort(
  url: RegExp,
  reason: string,
  maxOccurrences = 1,
): BrowserIssueAllowance {
  return {
    kind: 'requestfailed',
    url,
    message: /^net::ERR_ABORTED$/,
    maxOccurrences,
    reason,
  };
}

export function allowHttpConsoleError(
  url: RegExp,
  status: number,
  reason: string,
): BrowserIssueAllowance {
  return {
    kind: 'console.error',
    url,
    message: new RegExp(
      `^Failed to load resource: the server responded with a status of ${status}\\b`,
    ),
    maxOccurrences: 1,
    reason,
  };
}

function consoleLocation(message: ConsoleMessage, page: Page): string {
  const location = message.location();
  const url = location.url || page.url();
  return `${url}:${location.lineNumber + 1}:${location.columnNumber + 1}`;
}

function matches(pattern: RegExp, value: string): boolean {
  pattern.lastIndex = 0;
  return pattern.test(value);
}

export function unexpectedIssues(
  issues: BrowserIssue[],
  allowlist: BrowserIssueAllowance[],
): BrowserIssue[] {
  const allowanceCounts = allowlist.map(() => 0);
  return issues.filter((issue) => {
    const allowanceIndex = allowlist.findIndex((allowance, index) => {
      return (
        allowanceCounts[index] < allowance.maxOccurrences &&
        allowance.kind === issue.kind &&
        matches(allowance.url, issue.url) &&
        matches(allowance.message, issue.message)
      );
    });
    if (allowanceIndex === -1) return true;
    allowanceCounts[allowanceIndex] += 1;
    return false;
  });
}

/**
 * Browser-journey fixture. Import `test`, `expect`, and Playwright types from
 * `./fixtures` so console errors, uncaught page errors (including unhandled
 * rejections), and transport-level request failures fail the owning test.
 *
 * HTTP 4xx/5xx results are responses, not `requestfailed` events, and remain
 * available to journeys that deliberately exercise an error response.
 */
export const test = base.extend<BrowserGuardFixtures>({
  browserIssueRegistry: async ({ context }, provide) => {
    void context;
    await provide([]);
  },
  allowBrowserIssues: async ({ browserIssueRegistry }, provide, testInfo) => {
    await provide((...allowances) => {
      allowances.forEach((allowance) => {
        if (
          !Number.isInteger(allowance.maxOccurrences) ||
          allowance.maxOccurrences < 1 ||
          allowance.reason.trim().length === 0
        ) {
          throw new Error(
            'Browser issue allowances require a positive maxOccurrences and a reason.',
          );
        }
        testInfo.annotations.push({
          type: 'browser-issue-allowance',
          description: `${allowance.kind} ${allowance.url} ${allowance.message} (max ${allowance.maxOccurrences}): ${allowance.reason}`,
        });
        browserIssueRegistry.push(allowance);
      });
    });
  },
  browserGuard: [
    async ({ context, page, browserIssueRegistry }, provide) => {
      const issues: BrowserIssue[] = [];
      const guardedPages = new Set<Page>();

      const guardPage = (page: Page) => {
        if (guardedPages.has(page)) return;
        guardedPages.add(page);
        let lastMainFrameNavigationAt = Number.NEGATIVE_INFINITY;

        page.on('request', (request) => {
          if (
            request.isNavigationRequest() &&
            request.frame() === page.mainFrame()
          ) {
            lastMainFrameNavigationAt = Date.now();
          }
        });

        page.on('console', (message) => {
          if (message.type() !== 'error') return;
          // Chromium reports an HTTP error response as a console error even
          // though the request completed. Error-status journeys must assert
          // those responses themselves; the guard owns transport failures.
          if (HTTP_RESPONSE_CONSOLE_ERROR.test(message.text())) return;
          const url = message.location().url || page.url();
          issues.push({
            kind: 'console.error',
            url,
            message: message.text(),
            detail: `${consoleLocation(message, page)}: ${message.text()}`,
          });
        });
        page.on('pageerror', (error) => {
          const url = page.url();
          const message = error.stack ?? error.message;
          issues.push({
            kind: 'pageerror',
            url,
            message,
            detail: `${url}: ${message}`,
          });
        });
        page.on('requestfailed', (request) => {
          const url = request.url();
          const message = request.failure()?.errorText ?? 'unknown network failure';
          // A full-page navigation can cancel subresources. API fetch aborts
          // are deliberately NOT swallowed here: each expected cancellation
          // must use allowRequestAbort(url, reason, maxOccurrences), so an
          // unexpected URL or an exhausted budget fails the owning journey.
          if (
            Date.now() - lastMainFrameNavigationAt <= 1_000 &&
            !request.isNavigationRequest() &&
            message === 'net::ERR_ABORTED'
          ) {
            return;
          }
          issues.push({
            kind: 'requestfailed',
            url,
            message,
            detail: `${request.method()} ${url} (${request.resourceType()}): ${message}`,
          });
        });
      };

      const guardNewPage = (page: Page) => guardPage(page);
      // Depending explicitly on Playwright's page fixture makes this
      // teardown run before the page is closed. Page-close cancellations are
      // harness cleanup, not application failures, and must not consume an
      // API abort allowance nondeterministically.
      guardPage(page);
      context.pages().forEach(guardPage);
      context.on('page', guardNewPage);

      await provide();

      context.off('page', guardNewPage);
      const unexpected = unexpectedIssues(issues, browserIssueRegistry);
      if (unexpected.length > 0) {
        const diagnostics = unexpected
          .map((issue, index) => `  ${index + 1}. [${issue.kind}] ${issue.detail}`)
          .join('\n');
        throw new Error(
          `Browser guard detected ${unexpected.length} unexpected page issue(s):\n${diagnostics}`,
        );
      }
    },
    { auto: true },
  ],
});

export { expect };
export type {
  APIRequestContext,
  BrowserContext,
  ConsoleMessage,
  Locator,
  Page,
  Request,
  Response,
  TestInfo,
} from '@playwright/test';
