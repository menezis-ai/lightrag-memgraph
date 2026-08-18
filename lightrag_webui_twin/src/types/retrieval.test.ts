/**
 * Pure-function tests for parseAnswer + relTime. No DOM, no fixtures.
 */

import { describe, expect, it } from 'vitest';
import {
  parseAnswer,
  relTime,
  stripTrailingReferencesSection,
} from './retrieval';

describe('parseAnswer', () => {
  it('returns empty array for an empty token list', () => {
    expect(parseAnswer([])).toEqual([]);
  });

  it('emits a plain text part for a marker-free token', () => {
    expect(parseAnswer(['Hello world'])).toEqual([
      { type: 'text', value: 'Hello world' },
    ]);
  });

  it('extracts a citation', () => {
    expect(parseAnswer(['See ', '{cite:3}', ' for details'])).toEqual([
      { type: 'text', value: 'See ' },
      { type: 'cite', value: 3 },
      { type: 'text', value: ' for details' },
    ]);
  });

  it('extracts a code span', () => {
    expect(parseAnswer(['Run `shutdown immediate` now'])).toEqual([
      { type: 'text', value: 'Run ' },
      { type: 'code', value: 'shutdown immediate' },
      { type: 'text', value: ' now' },
    ]);
  });

  it('handles cite + code in the same token', () => {
    expect(
      parseAnswer(['Use `lsnrctl` then check {cite:7} for kernel notes']),
    ).toEqual([
      { type: 'text', value: 'Use ' },
      { type: 'code', value: 'lsnrctl' },
      { type: 'text', value: ' then check ' },
      { type: 'cite', value: 7 },
      { type: 'text', value: ' for kernel notes' },
    ]);
  });

  it('extracts a [N] citation (LightRAG prompt format)', () => {
    expect(parseAnswer(['See [2] for the runbook'])).toEqual([
      { type: 'text', value: 'See ' },
      { type: 'cite', value: 2 },
      { type: 'text', value: ' for the runbook' },
    ]);
  });

  it('treats {cite:N} and [N] interchangeably in one token', () => {
    expect(parseAnswer(['first {cite:1} then [3]'])).toEqual([
      { type: 'text', value: 'first ' },
      { type: 'cite', value: 1 },
      { type: 'text', value: ' then ' },
      { type: 'cite', value: 3 },
    ]);
  });

  it('extracts bold text and footnote-style citations', () => {
    expect(parseAnswer(['Use **RMAN** with [^4]'])).toEqual([
      { type: 'text', value: 'Use ' },
      { type: 'bold', value: 'RMAN' },
      { type: 'text', value: ' with ' },
      { type: 'cite', value: 4 },
    ]);
  });

  it('extracts emphasis and safe HTTP links', () => {
    expect(parseAnswer(['Read *carefully* in [the runbook](https://kb.example/runbook)'])).toEqual([
      { type: 'text', value: 'Read ' },
      { type: 'italic', value: 'carefully' },
      { type: 'text', value: ' in ' },
      { type: 'link', label: 'the runbook', href: 'https://kb.example/runbook' },
    ]);
  });

  it('does not treat snake_case identifiers as emphasis', () => {
    expect(parseAnswer(['Set TWIN_MIP_MAX_CLASSIFICATION to C2'])).toEqual([
      { type: 'text', value: 'Set TWIN_MIP_MAX_CLASSIFICATION to C2' },
    ]);
  });

  it('does not treat spaced multiplication as emphasis', () => {
    expect(parseAnswer(['compute 3 * 4 * 5'])).toEqual([
      { type: 'text', value: 'compute 3 * 4 * 5' },
    ]);
  });

  it('renders minimal Markdown blocks for headings and lists', () => {
    expect(parseAnswer(['### Runbook\n- check backup\n1. open incident'])).toEqual([
      {
        type: 'heading',
        level: 3,
        children: [{ type: 'text', value: 'Runbook' }],
      },
      {
        type: 'listItem',
        ordered: false,
        marker: '•',
        children: [{ type: 'text', value: 'check backup' }],
      },
      {
        type: 'listItem',
        ordered: true,
        marker: '1.',
        children: [{ type: 'text', value: 'open incident' }],
      },
    ]);
  });

  it('strips a trailing References section from generated answers', () => {
    expect(
      parseAnswer([
        'Conclusion\n\nAnswer text.\n\n### References\n- [1] runbook.pdf\n- [2] guide.pdf',
      ]),
    ).toEqual([
      { type: 'text', value: 'Conclusion' },
      { type: 'paragraphBreak' },
      { type: 'text', value: 'Answer text.' },
    ]);
  });

  it('parses GitHub-flavored Markdown tables into structured cells', () => {
    expect(
      parseAnswer([
        '| Step | Command |\n| --- | --- |\n| Stop | `shutdown immediate` |\n| Start | `startup` |',
      ]),
    ).toEqual([
      {
        type: 'table',
        headers: [
          [{ type: 'text', value: 'Step' }],
          [{ type: 'text', value: 'Command' }],
        ],
        rows: [
          [
            [{ type: 'text', value: 'Stop' }],
            [{ type: 'code', value: 'shutdown immediate' }],
          ],
          [
            [{ type: 'text', value: 'Start' }],
            [{ type: 'code', value: 'startup' }],
          ],
        ],
      },
    ]);
  });
});

describe('stripTrailingReferencesSection', () => {
  it('strips reference headings with a descriptive suffix', () => {
    expect(
      stripTrailingReferencesSection(
        'Visible answer\n\n### References — cited docs\n- hidden.pdf',
      ),
    ).toBe('Visible answer');
  });
});

describe('relTime', () => {
  it('returns empty for null/undefined/empty', () => {
    expect(relTime(null)).toBe('');
    expect(relTime(undefined)).toBe('');
    expect(relTime('')).toBe('');
  });

  it('returns "" for unparseable strings', () => {
    expect(relTime('garbage')).toBe('');
  });

  it('returns "now" for fresh timestamps', () => {
    expect(relTime(Date.now())).toBe('now');
    expect(relTime(Date.now() - 10_000)).toBe('now');
  });

  it('returns Xm minutes', () => {
    expect(relTime(Date.now() - 5 * 60_000)).toBe('5m');
  });

  it('returns Xh hours', () => {
    expect(relTime(Date.now() - 4 * 3_600_000)).toBe('4h');
  });

  it('returns Xd days', () => {
    expect(relTime(Date.now() - 3 * 86_400_000)).toBe('3d');
  });
});
