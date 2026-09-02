import { expect, test } from '@playwright/test';

import {
  DOCUMENTS_LIFECYCLE_ABORT_ALLOWANCES,
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

test('Documents lifecycle abort defaults are exact and bounded', () => {
  const abort = (path: string, detail: string): BrowserIssue => ({
    kind: 'requestfailed',
    detail,
    message: 'net::ERR_ABORTED',
    url: `http://127.0.0.1:4173${path}`,
  });
  const quotaAborts = Array.from({ length: 4 }, (_, index) =>
    abort('/twin/api/quota', `quota cancellation ${index + 1}`),
  );
  const unrelated = abort('/documents', 'unrelated cancellation');

  expect(
    unexpectedIssues(
      [...quotaAborts, unrelated],
      DOCUMENTS_LIFECYCLE_ABORT_ALLOWANCES,
    ),
  ).toEqual([quotaAborts[3], unrelated]);
});
