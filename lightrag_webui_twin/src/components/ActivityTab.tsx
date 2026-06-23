/**
 * ActivityTab — split timeline (left) + event detail (right).
 *
 * Ported from Desktop/UI/activity.jsx. Mixed audit feed: source lifecycle,
 * tag mutations, retrievals, pipeline events, auth, settings.
 *
 * Behavior delta vs the proto:
 *   - Events are injected via the `events` prop (no window.MOCK_ACTIVITY).
 *   - `nowMs` is a pinned timestamp for deterministic range filtering during
 *     dev; pass `Date.now()` in prod.
 *   - `onPushToast` is the structured Toast emitter from the host (App.tsx).
 *   - `onNavigate(tab, params)` is invoked instead of pushing window history
 *     directly, so the host owns routing.
 *   - "Clear activity" modal a11y via useModalA11y.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon } from './Icon';
import { useModalA11y } from '../hooks/useModalA11y';
import { useUrlParam } from '../hooks/useUrlParam';
import {
  ACTIVITY_KIND_META,
  ACTIVITY_RANGE_MS,
  ACTIVITY_RANGES,
  resolveKindMeta,
  type ActivityEvent,
  type ActivityKind,
  type ActivityRange,
  type ActivitySeverity,
} from '../types/activity';
import type { Toast } from '../types/toast';

export type ActivityDensity = 'comfortable' | 'compact';

export interface ActivityTabProps {
  events: readonly ActivityEvent[];
  /** Pinned "now" for deterministic ranges. Defaults to `Date.now()`. */
  nowMs?: number;
  folderLabel?: string;
  density?: ActivityDensity;
  /** When true, simulate live polling (pending-event counter ticks). */
  live?: boolean;
  groupByDay?: boolean;
  onPushToast?: (toast: Omit<Toast, 'id'>) => void;
  /** Host-controlled tab navigation (replaces direct window.history mutation). */
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
  /** Host-controlled query refresh. Used by the explicit Refresh affordance. */
  onRefresh?: () => void | Promise<unknown>;
}

const RANGE_IDS = ACTIVITY_RANGES.map((r) => r.id);

function kindPillStateClass(explicit: boolean, active: boolean): string {
  if (explicit) return 'is-explicit';
  if (active) return 'is-dim';
  return 'is-off';
}

export function ActivityTab({
  events,
  nowMs,
  folderLabel = 'default',
  density = 'comfortable',
  live = true,
  groupByDay = true,
  onPushToast,
  onNavigate,
  onRefresh,
}: Readonly<ActivityTabProps>) {
  const [range, setRange] = useUrlParam<ActivityRange>('range', '7d', {
    validate: (v) => (RANGE_IDS as readonly string[]).includes(v as string),
  });
  const [kinds, setKinds] = useUrlParam<Set<string>>('kind', new Set<string>(), {
    parse: (s) => new Set(s.split(',').filter(Boolean)),
    serialize: (set) => (set?.size ? [...set].join(',') : ''),
    validate: (v) => v instanceof Set,
  });
  const [sev, setSev] = useUrlParam<'any' | ActivitySeverity>('sev', 'any', {
    validate: (v) =>
      (['any', 'info', 'warning', 'error', 'critical'] as const).includes(
        v as 'any' | ActivitySeverity,
      ),
  });
  const [q, setQ] = useUrlParam<string>('q', '');
  const [actor, setActor] = useUrlParam<string>('actor', 'any');
  const [selectedId, setSelectedId] = useState<string>(events[0]?.id ?? '');
  const [clearOpen, setClearOpen] = useState(false);
  const [clearConfirm, setClearConfirm] = useState('');
  const [initialNowMs] = useState(() => Date.now());
  const clearModalRef = useRef<HTMLDialogElement>(null);
  useModalA11y({ open: clearOpen, onClose: () => setClearOpen(false), ref: clearModalRef });

  // Real polling: refetch from the backend on an interval. The previous
  // implementation incremented a fake "N new events" counter every 9s
  // with no network call — pure demo theater that is not credible in
  // production (one operator alone saw "43 new events").
  useEffect(() => {
    if (!live || !onRefresh) return undefined;
    const t = setInterval(() => {
      void onRefresh();
    }, 30_000);
    return () => clearInterval(t);
  }, [live, onRefresh]);

  const actors = useMemo(() => {
    const s = new Set<string>(events.map((e) => e.actor.user));
    return ['any', ...s];
  }, [events]);

  const toggleKind = (k: ActivityKind) => {
    const next = new Set(kinds);
    if (next.has(k)) next.delete(k);
    else next.add(k);
    setKinds(next);
  };

  const filtered = useMemo(() => {
    const effectiveNow = nowMs ?? initialNowMs;
    return events.filter((e) => {
      if (range !== 'all') {
        const cutoff =
          effectiveNow - (ACTIVITY_RANGE_MS[range] ?? ACTIVITY_RANGE_MS['7d']);
        const ts = Date.parse(e.ts);
        if (!Number.isNaN(ts) && ts < cutoff) return false;
      }
      if (kinds.size && !kinds.has(e.kind)) return false;
      if (sev !== 'any' && e.sev !== sev) return false;
      if (actor !== 'any' && e.actor.user !== actor) return false;
      if (q.trim()) {
        const needle = q.trim().toLowerCase();
        const hay = (
          e.summary +
          ' ' +
          e.target.label +
          ' ' +
          e.actor.user +
          ' ' +
          e.id
        ).toLowerCase();
        if (!hay.includes(needle)) return false;
      }
      return true;
    });
  }, [events, range, kinds, sev, actor, q, nowMs, initialNowMs]);

  const selected = filtered.find((e) => e.id === selectedId) ?? filtered[0] ?? null;

  const grouped = useMemo(() => {
    if (!groupByDay) return [['', filtered] as const];
    const acc = new Map<string, ActivityEvent[]>();
    filtered.forEach((e) => {
      const bucket = acc.get(e.day) ?? [];
      bucket.push(e);
      acc.set(e.day, bucket);
    });
    return Array.from(acc.entries());
  }, [filtered, groupByDay]);

  const clearFilters = () => {
    setKinds(new Set());
    setSev('any');
    setActor('any');
    setQ('');
  };
  const refreshEvents = () => {
    void onRefresh?.();
  };

  return (
    <div className={'activity' + (density === 'compact' ? ' is-compact' : '')}>
      <div className="activity-main">
        <div className="activity-header">
          <h1>Activity</h1>
          <div className="activity-sub">
            <span>
              Audit trail · folder <code>{folderLabel}</code>
            </span>
            <span className="dot-sep">·</span>
            <span
              className={'activity-live ' + (live ? 'is-on' : 'is-paused')}
              title={
                live
                  ? 'Polling /activity every 9s'
                  : 'Polling disabled — new events will not surface until re-enabled in Tweaks'
              }
            >
              <span className="live-dot" /> {live ? 'Live polling' : 'Polling paused'}
            </span>
          </div>
        </div>

        <div className="activity-filters">
          <div className="seg-range" role="tablist" aria-label="Time range">
            {ACTIVITY_RANGES.map((r) => (
              <button
                key={r.id}
                role="tab"
                aria-selected={range === r.id}
                className={'seg ' + (range === r.id ? 'is-active' : '')}
                onClick={() => setRange(r.id)}
              >
                {r.label}
              </button>
            ))}
          </div>

          <div className="activity-kinds">
            {(Object.entries(ACTIVITY_KIND_META) as [ActivityKind, (typeof ACTIVITY_KIND_META)[ActivityKind]][]).map(
              ([k, m]) => {
                const active = kinds.size === 0 || kinds.has(k);
                const explicit = kinds.has(k);
                return (
                  <button
                    key={k}
                    className={'kind-pill ' + kindPillStateClass(explicit, active)}
                    onClick={() => toggleKind(k)}
                    title={m.label}
                    aria-pressed={explicit}
                  >
                    <Icon name={m.icon} size={11} color={m.color} />
                    {m.label}
                  </button>
                );
              },
            )}
          </div>

          <div className="activity-secondary">
            <select
              className="mini-select"
              value={sev}
              onChange={(e) => setSev(e.target.value as 'any' | ActivitySeverity)}
              aria-label="Severity filter"
            >
              <option value="any">All severities</option>
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
            </select>

            <select
              className="mini-select"
              value={actor}
              onChange={(e) => setActor(e.target.value)}
              aria-label="Actor filter"
            >
              {actors.map((a) => (
                <option key={a} value={a}>
                  {a === 'any' ? 'All actors' : a}
                </option>
              ))}
            </select>

            <div className="activity-search">
              <Icon name="search" size={13} color="var(--color-text-tertiary)" />
              <input
                type="text"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search summary, target, event ID…"
                aria-label="Search events"
              />
              {q && (
                <button
                  className="x"
                  onClick={() => setQ('')}
                  aria-label="Clear search"
                >
                  <Icon name="x" size={11} color="var(--color-text-tertiary)" />
                </button>
              )}
            </div>

            <div className="activity-actions">
              <button
                className="ghost-btn"
                onClick={refreshEvents}
                title="Refresh activity events"
              >
                <Icon name="refresh" size={12} />
                Refresh
              </button>
              <button
                className="ghost-btn"
                onClick={() => exportActivityCsv(filtered, range)}
                title={`Download ${filtered.length} event${filtered.length === 1 ? '' : 's'} as CSV`}
              >
                <Icon name="external-link" size={12} />
                Export
              </button>
              {/*
                Clear button removed — Activity is an IMMUTABLE LEDGER.
                Audit events are append-only by doctrine (EBA/DORA audit trail
                requirements). The retention-policy table
                in Settings → Folder governs natural expiry; no operator
                affordance to wipe rows. Modal + clearOpen state still wired in
                case it gets reintroduced behind a Steward-only ops escape.
              */}
            </div>
          </div>
        </div>

        <div className="activity-stats">
          <span className="stat">
            <b>{filtered.length}</b> events
          </span>
          <span className="dot-sep">·</span>
          <span className="stat">
            <b>{filtered.filter((e) => e.sev === 'error').length}</b> errors
          </span>
          <span className="dot-sep">·</span>
          <span className="stat">
            <b>{filtered.filter((e) => e.sev === 'warning').length}</b> warnings
          </span>
          <span className="dot-sep">·</span>
          <span className="stat">
            <b>{filtered.filter((e) => e.kind === 'retrieval').length}</b> retrievals
          </span>
        </div>

        <div className="activity-timeline">
          {grouped.map(([day, evts]) => (
            <div key={day || '_all'} className="day-group">
              {day && (
                <div className="day-h">
                  <span>{day}</span>
                  <span className="day-line" />
                  <span className="day-count">{evts.length}</span>
                </div>
              )}
              {evts.map((e) => (
                <ActivityRow
                  key={e.id}
                  e={e}
                  selected={!!selected && selected.id === e.id}
                  onClick={() => setSelectedId(e.id)}
                />
              ))}
            </div>
          ))}
          {!filtered.length && (
            <div className="empty-state" style={{ padding: 60 }}>
              <Icon name="activity" size={24} color="var(--color-text-tertiary)" />
              <div className="title">No events match the current filter</div>
              <button className="suggestion" onClick={clearFilters}>
                Clear filters
              </button>
            </div>
          )}
        </div>
      </div>

      <ActivityDetail
        e={selected}
        onPushToast={onPushToast}
        onNavigate={onNavigate}
      />

      {clearOpen && (
        <div
          className="modal-bg"
        >
          <button
            type="button"
            className="modal-backdrop-dismiss"
            onClick={() => setClearOpen(false)}
            aria-label="Close clear activity dialog"
            data-testid="clear-modal-bg"
          />
          <dialog
            open
            ref={clearModalRef}
            className="modal"
            style={{ width: 480 }}
            aria-modal="true"
            aria-labelledby="clear-title"
            tabIndex={-1}
          >
            <div className="modal-h">
              <h3 id="clear-title">Clear activity events</h3>
              <div className="modal-h-sub">Palier 3 · admin action</div>
              <button
                className="modal-x"
                onClick={() => setClearOpen(false)}
                aria-label="Close dialog"
              >
                <Icon name="x" size={14} />
              </button>
            </div>
            <div className="modal-body">
              <div className="impact-box danger">
                <Icon name="alert-triangle" size={13} color="var(--twin-red-vivid)" />
                <span>
                  Purges events <b>past their retention window</b> only. Events still within retention (e.g.{' '}
                  <code>system.policy_violation</code> kept for 7 years) are untouched. Action is recorded as{' '}
                  <code>admin.clear</code> in this log itself.
                </span>
              </div>
              <div className="retention-grid">
                <div>
                  <span>Source mgmt</span>
                  <code>90d</code>
                </div>
                <div>
                  <span>Tag mgmt</span>
                  <code>90d</code>
                </div>
                <div>
                  <span>Retrieval</span>
                  <code>30d</code>
                </div>
                <div>
                  <span>Admin</span>
                  <code>1y</code>
                </div>
                <div>
                  <span>Auth</span>
                  <code>1y</code>
                </div>
                <div>
                  <span>Policy / System</span>
                  <code>7y</code>
                </div>
              </div>
              <label className="field-label" htmlFor="clear-confirm-input">
                Type <code>CLEAR</code> to confirm
              </label>
              <input
                id="clear-confirm-input"
                className="text-input"
                value={clearConfirm}
                onChange={(e) => setClearConfirm(e.target.value)}
                placeholder="CLEAR"
                autoFocus
              />
            </div>
            <div className="modal-footer">
              <button className="ghost-btn" onClick={() => setClearOpen(false)}>
                Cancel
              </button>
              <button
                className="primary-btn danger"
                disabled={clearConfirm !== 'CLEAR'}
                onClick={() => {
                  setClearOpen(false);
                  setClearConfirm('');
                  onPushToast?.({
                    kind: 'done',
                    title: 'Activity',
                    titleSuffix: 'events past retention purged',
                    sub: 'admin.clear emitted · 1,247 events removed',
                  });
                }}
              >
                Purge expired events
              </button>
            </div>
          </dialog>
        </div>
      )}
    </div>
  );
}

interface ActivityRowProps {
  e: ActivityEvent;
  selected: boolean;
  onClick: () => void;
}

function ActivityRow({ e, selected, onClick }: Readonly<ActivityRowProps>) {
  const m = resolveKindMeta(e.kind);
  return (
    <button
      className={'activity-row ' + (selected ? 'is-selected' : '') + ' sev-' + e.sev}
      onClick={onClick}
      aria-current={selected ? 'true' : undefined}
    >
      <span className="row-time">{e.rel}</span>
      <span className="row-rail" style={{ background: m.color }} />
      <span className="row-icon" style={{ color: m.color }}>
        <Icon name={m.icon} size={14} />
      </span>
      <span className="row-body">
        <span className="row-line1">
          <span className="row-actor">{e.actor.user}</span>
          <span className="row-kind">{m.label}</span>
          <span className="row-target">{e.target.label}</span>
        </span>
        <span className="row-summary">{e.summary}</span>
      </span>
      {e.sev !== 'info' && <span className={'sev-badge sev-' + e.sev}>{e.sev}</span>}
    </button>
  );
}

interface ActivityDetailProps {
  e: ActivityEvent | null;
  onPushToast?: (toast: Omit<Toast, 'id'>) => void;
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

function ActivityDetail({ e, onPushToast, onNavigate }: Readonly<ActivityDetailProps>) {
  const [copied, setCopied] = useState(false);
  if (!e) {
    return (
      <aside className="activity-detail">
        <div className="empty-state">
          <div className="title">Select an event</div>
        </div>
      </aside>
    );
  }
  const m = resolveKindMeta(e.kind);
  const copyId = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      void navigator.clipboard.writeText(e.id);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <aside className="activity-detail">
      <div className="detail-head">
        <div className="detail-kind" style={{ color: m.color }}>
          <Icon name={m.icon} size={14} color={m.color} />
          {m.label}
          {e.sev !== 'info' && <span className={'sev-badge sev-' + e.sev}>{e.sev}</span>}
        </div>
        <h3>{e.target.label}</h3>
        <div className="detail-summary">{e.summary}</div>
      </div>

      <div className="detail-grid">
        <div className="kv">
          <span>Event ID</span>
          <button
            type="button"
            className="copyable"
            onClick={copyId}
            title="Copy"
          >
            {e.id} {copied ? '✓' : ''}
          </button>
        </div>
        <div className="kv">
          <span>Timestamp</span>
          <code>{e.ts}</code>
        </div>
        <div className="kv">
          <span>Relative</span>
          <span>{e.rel}</span>
        </div>
        <div className="kv">
          <span>Actor</span>
          <span>
            {e.actor.user} <em>({e.actor.role})</em>
          </span>
        </div>
        <div className="kv">
          <span>Target</span>
          <span>
            {e.target.type} · {e.target.label}
          </span>
        </div>
        <div className="kv">
          <span>Severity</span>
          <span className={'sev-text sev-' + e.sev}>{e.sev}</span>
        </div>
      </div>

      <div className="detail-section">
        <div className="detail-section-h">Metadata</div>
        <pre className="detail-meta">{JSON.stringify(e.meta, null, 2)}</pre>
      </div>

      <div className="detail-actions">
        {e.kind === 'source-failed' && (
          <button
            className="primary-btn"
            onClick={() =>
              onPushToast?.({
                kind: 'propagating',
                title: 'Re-processing failed sources',
                // Audit C7: targeted ``/documents/{id}/scan`` is rejected
                // because LightRAG has no safe per-document rescan. The
                // honest action that includes this row is the failed-batch
                // endpoint.
                sub: `${e.target.label} · POST /documents/reprocess_failed`,
              })
            }
          >
            <Icon name="refresh" size={12} /> Replay ingestion
          </button>
        )}
        {e.target.type === 'source' && (
          <button
            className="ghost-btn"
            onClick={() =>
              onNavigate?.('documents', e.target.label ? { q: e.target.label } : undefined)
            }
          >
            <Icon name="arrow-right" size={12} /> Open source
          </button>
        )}
        {e.target.type === 'query' && (
          <button
            className="ghost-btn"
            onClick={() => {
              const params: Record<string, string> = {};
              if (e.target.label) params.q = e.target.label;
              const meta = e.meta as { mode?: string };
              if (meta?.mode) params.mode = meta.mode;
              onNavigate?.('retrieval', params);
            }}
          >
            <Icon name="arrow-right" size={12} /> Re-run query
          </button>
        )}
        <button className="ghost-btn" onClick={copyId}>
          <Icon name="external-link" size={12} /> Copy payload
        </button>
      </div>
    </aside>
  );
}

/**
 * Flatten an ActivityEvent list to a CSV blob and trigger a download. Exported
 * so it can be unit-tested without rendering the whole tab.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function exportActivityCsv(
  rows: readonly ActivityEvent[],
  range: ActivityRange,
): void {
  const esc = (v: unknown): string => {
    if (v === null || v === undefined) return '';
    const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const cols = [
    'id',
    'ts',
    'kind',
    'sev',
    'actor',
    'role',
    'target_type',
    'target_label',
    'summary',
    'meta',
  ];
  const lines = [cols.join(',')];
  rows.forEach((e) => {
    lines.push(
      [
        e.id,
        e.ts,
        e.kind,
        e.sev,
        e.actor.user,
        e.actor.role,
        e.target.type,
        e.target.label,
        e.summary,
        e.meta,
      ]
        .map(esc)
        .join(','),
    );
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const stamp = new Date().toISOString().slice(0, 10);
  a.href = url;
  a.download = `twin-rag-activity-${range}-${stamp}.csv`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    a.remove();
  }, 0);
}
