import { expect, test } from '@playwright/test';

import {
  allowRequestAbort,
  isNavigationRelatedAbort,
  unexpectedIssues,
  type BrowserIssue,
} from '../fixtures';

test('navigation abort detection is event-order independent and bounded', () => {
  const failedAt = 10_000;

  expect(isNavigationRelatedAbort(failedAt, [9_001])).toBe(true);
  expect(isNavigationRelatedAbort(failedAt, [10_999])).toBe(true);
  expect(isNavigationRelatedAbort(failedAt, [8_999, 11_001])).toBe(false);
});

test('browser issue allowance reports aborts beyond its occurrence budget', () => {
  const abort = (detail: string): BrowserIssue => ({
    kind: 'requestfailed',
    detail,
    message: 'net::ERR_ABORTED',
    url: 'http://127.0.0.1:4173/twin/api/quota',
  });
  const first = abort('first expected cancellation');
  const exhausted = abort('second unexpected cancellation');

  expect(
    unexpectedIssues(
      [first, exhausted],
      [
        allowRequestAbort(
          /\/twin\/api\/quota$/,
          'The journey permits exactly one quota cancellation.',
          1,
        ),
      ],
    ),
  ).toEqual([exhausted]);
});
