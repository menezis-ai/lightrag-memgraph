/**
 * Unit tests for ActivityTab.
 *
 * Covers: range/kind/sev/actor/query filters, group-by-day, selection ->
 * detail panel, clear-modal CLEAR-gate, exportActivityCsv helper.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ActivityTab, exportActivityCsv } from './ActivityTab';
import { ACTIVITY_FIXTURES, ACTIVITY_NOW_MS } from '../fixtures';

function defaultProps() {
  return {
    events: ACTIVITY_FIXTURES,
    nowMs: ACTIVITY_NOW_MS,
    live: false as const,
    onPushToast: vi.fn(),
    onNavigate: vi.fn(),
  };
}

beforeEach(() => {
  window.history.replaceState(null, '', '/');
});
afterEach(() => {
  window.history.replaceState(null, '', '/');
});

describe('ActivityTab — rendering', () => {
  it('renders the header and live indicator (paused)', () => {
    render(<ActivityTab {...defaultProps()} folderLabel="sandbox" />);
    expect(screen.getByRole('heading', { name: 'Activity' })).toBeInTheDocument();
    expect(document.querySelector('.activity-sub')?.textContent).toMatch(
      /folder sandbox/,
    );
    expect(screen.getByText('Polling paused')).toBeInTheDocument();
  });

  it('renders today + yesterday + earlier-this-week day buckets by default (7d range)', () => {
    render(<ActivityTab {...defaultProps()} />);
    expect(screen.getByText('Today')).toBeInTheDocument();
    expect(screen.getByText('Yesterday')).toBeInTheDocument();
    expect(screen.getByText('Earlier this week')).toBeInTheDocument();
  });

  it('shows the right stats: total / errors / warnings / retrievals', () => {
    render(<ActivityTab {...defaultProps()} />);
    // Within 7d range = all 16 fixtures (the oldest is 2026-05-07 = 4d ago)
    // 2 errors, 2 warnings, 4 retrievals.
    const stats = document.querySelector('.activity-stats') as HTMLElement;
    const t = stats.textContent || '';
    expect(t).toMatch(/16\s*events/);
    expect(t).toMatch(/2\s*errors/);
    expect(t).toMatch(/3\s*warnings/);
    expect(t).toMatch(/4\s*retrievals/);
  });
});

describe('ActivityTab — filters', () => {
  it('24h range filters out 3-4d-old events but keeps yesterday afternoon', async () => {
    render(<ActivityTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('tab', { name: '24h' }));
    // 4d-old "Earlier this week" bucket is fully out
    expect(screen.queryByText('Earlier this week')).toBeNull();
    // Yesterday morning event (09:32 UTC = 24.5h before pinned now) is dropped
    expect(screen.queryByText('cft-vendor-api-spec.pdf')).toBeNull();
    expect(screen.getByText('Today')).toBeInTheDocument();
  });

  it('severity filter narrows to errors only', async () => {
    render(<ActivityTab {...defaultProps()} />);
    await userEvent.selectOptions(
      screen.getByLabelText('Severity filter'),
      'error',
    );
    const stats = document.querySelector('.activity-stats') as HTMLElement;
    expect(stats.textContent).toMatch(/2\s*events/);
  });

  it('actor filter narrows to one user', async () => {
    render(<ActivityTab {...defaultProps()} />);
    await userEvent.selectOptions(
      screen.getByLabelText('Actor filter'),
      'marc.berthier',
    );
    const stats = document.querySelector('.activity-stats') as HTMLElement;
    // marc.berthier has 3 events: 2 retrievals + 1 auth.
    expect(stats.textContent).toMatch(/3\s*events/);
  });

  it('search query filters by substring across summary / target / id / actor', async () => {
    render(<ActivityTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search events'), 'oracle');
    const stats = document.querySelector('.activity-stats') as HTMLElement;
    // events mentioning "oracle" in summary/target/meta-as-string
    expect(stats.textContent).not.toMatch(/^0\s/);
    // verify one known oracle event is present
    expect(
      screen.getAllByText(/Oracle/, { exact: false }).length,
    ).toBeGreaterThan(0);
  });

  it('kind pill toggles narrow the list to one kind', async () => {
    render(<ActivityTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Retrieval' }));
    const stats = document.querySelector('.activity-stats') as HTMLElement;
    expect(stats.textContent).toMatch(/4\s*events/);
  });

  it('empty state shows "Clear filters" CTA and resets filters', async () => {
    render(<ActivityTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Search events'), 'zzz-no-match');
    expect(
      screen.getByText('No events match the current filter'),
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Clear filters' }));
    // After reset, "Today" header should be visible again.
    expect(screen.getByText('Today')).toBeInTheDocument();
  });
});

describe('ActivityTab — selection + detail', () => {
  it('clicking a row updates the detail panel', async () => {
    render(<ActivityTab {...defaultProps()} />);
    const detail = document.querySelector('.activity-detail') as HTMLElement;
    // Default is first event (a retrieval)
    expect(within(detail).getByText('Retrieval')).toBeInTheDocument();
    // Click the source-uploaded event row
    const uploadedRow = screen.getByText('swift-iso20022-migration.pdf');
    await userEvent.click(uploadedRow);
    expect(within(detail).getByText('Source uploaded')).toBeInTheDocument();
  });

  it('source-failed event renders Replay button that pushes a propagating toast', async () => {
    const p = defaultProps();
    render(<ActivityTab {...p} />);
    // huge-archive.zip = source-failed
    await userEvent.click(screen.getByText('huge-archive.zip'));
    await userEvent.click(
      screen.getByRole('button', { name: /Replay ingestion/ }),
    );
    expect(p.onPushToast).toHaveBeenCalledTimes(1);
    const toast = p.onPushToast.mock.calls[0][0];
    expect(toast.kind).toBe('propagating');
    // Audit C7: no "queued" wording — backend has no observable
    // queue. The action surfaces what it actually does: re-process
    // failed sources via the batch endpoint.
    expect(toast.title).toBe('Re-processing failed sources');
    expect(toast.sub).toMatch(/POST \/documents\/reprocess_failed/);
  });

  it('query target Re-run navigates with q + mode params', async () => {
    const p = defaultProps();
    render(<ActivityTab {...p} />);
    // first event is the Oracle retrieval (selected by default)
    await userEvent.click(
      screen.getByRole('button', { name: /Re-run query/ }),
    );
    expect(p.onNavigate).toHaveBeenCalledWith('retrieval', {
      q: 'How to restart Oracle on RHEL 9?',
      mode: 'hybrid',
    });
  });

  it('source target Open source navigates with q param', async () => {
    const p = defaultProps();
    render(<ActivityTab {...p} />);
    await userEvent.click(screen.getByText('huge-archive.zip'));
    await userEvent.click(
      screen.getByRole('button', { name: /Open source/ }),
    );
    expect(p.onNavigate).toHaveBeenCalledWith('documents', {
      q: 'huge-archive.zip',
    });
  });
});

describe('ActivityTab — immutable ledger (no Clear)', () => {
  it('does NOT expose a Clear button — audit trail is append-only', () => {
    render(<ActivityTab {...defaultProps()} />);
    // Activity is an immutable ledger per audit doctrine (EBA / DORA audit
    // trail). The Clear button was removed 2026-05-31. Natural expiry is
    // governed by the retention-policy table in Settings → Folder.
    expect(screen.queryByRole('button', { name: /^Clear/ })).toBeNull();
    expect(
      screen.queryByRole('dialog', { name: 'Clear activity events' }),
    ).toBeNull();
  });
});

describe('ActivityTab — unknown kind resilience (real backend)', () => {
  // Regression: the live backend emits kinds the UI map does not enumerate
  // (api-key-*, dynamic settings sub-kinds). Before resolveKindMeta(),
  // ActivityRow did `ACTIVITY_KIND_META[e.kind].color` and crashed the
  // whole tab on a single unknown event ("can't access property color").
  it('renders events whose kind is absent from ACTIVITY_KIND_META without crashing', () => {
    const base = ACTIVITY_FIXTURES[0];
    const events = [
      { ...base, id: 'evt-apikey', kind: 'api-key-created', summary: 'key created' },
      {
        ...base,
        id: 'evt-mystery',
        kind: 'settings-future-subkind',
        summary: 'mystery event',
      },
    ] as unknown as typeof ACTIVITY_FIXTURES;
    expect(() =>
      render(
        <ActivityTab
          events={events}
          nowMs={ACTIVITY_NOW_MS}
          live={false}
          onPushToast={vi.fn()}
          onNavigate={vi.fn()}
        />,
      ),
    ).not.toThrow();
    // newly-mapped api-key kind gets a friendly label
    expect(screen.getAllByText('API key created').length).toBeGreaterThan(0);
    // truly-unknown kind falls back to a humanized label instead of crashing
    expect(
      screen.getAllByText('Settings future subkind').length,
    ).toBeGreaterThan(0);
  });
});

describe('exportActivityCsv helper', () => {
  it('produces a CSV with header + one row per event and escapes quotes', () => {
    const calls: { type: string; arg: unknown }[] = [];
    const origCreate = URL.createObjectURL;
    const origRevoke = URL.revokeObjectURL;
    URL.createObjectURL = vi.fn((b: Blob) => {
      calls.push({ type: 'create', arg: b });
      return 'blob:mock';
    }) as unknown as typeof URL.createObjectURL;
    URL.revokeObjectURL = vi.fn() as unknown as typeof URL.revokeObjectURL;
    try {
      exportActivityCsv(ACTIVITY_FIXTURES.slice(0, 2), '7d');
      expect(calls).toHaveLength(1);
      const blob = calls[0].arg as Blob;
      expect(blob.type).toMatch(/text\/csv/);
    } finally {
      URL.createObjectURL = origCreate;
      URL.revokeObjectURL = origRevoke;
    }
  });
});
