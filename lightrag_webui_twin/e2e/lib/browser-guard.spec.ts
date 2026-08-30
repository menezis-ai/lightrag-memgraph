import { expect, test } from '@playwright/test';

import {
  allowRequestAbort,
  unexpectedIssues,
  type BrowserIssue,
} from '../fixtures';

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
