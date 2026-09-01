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
 *   - The immutable ledger deliberately exposes no destructive clear action.
 */

/* eslint-disable react-refresh/only-export-components -- compatibility re-export keeps the established ActivityTab helper contract. */
import { useEffect, useMemo, useState } from 'react';
import { Icon } from './Icon';
import { ActivityDetail } from './Activity/ActivityDetail';
import { ActivityRow } from './Activity/ActivityRow';
import { exportActivityCsv } from './Activity/activityExport';
import { useUrlParam } from '../hooks/useUrlParam';
import { relativeTime } from '../utils/relativeTime';
import {
  ACTIVITY_KIND_META,
  ACTIVITY_RANGE_MS,
  ACTIVITY_RANGES,
  type ActivityEvent,
  type ActivityKind,
  type ActivityRange,
  type ActivitySeverity,
} from '../types/activity';
import type { Toast } from '../types/toast';
import type { ActivityQuery } from '../api/resources';

export { exportActivityCsv } from './Activity/activityExport';

export type ActivityDensity = 'comfortable' | 'compact';
export const DEFAULT_ACTIVITY_LIMIT = 200;

export interface ActivityTabProps {
  events: readonly ActivityEvent[];
  /** Backend-filtered total for the current activity query. */
  total?: number;
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
  /** Publishes UI filters so the host can issue authoritative backend queries. */
  onQueryChange?: (query: ActivityQuery) => void;
  /** Page size requested from the backend. */
  limit?: number;
  /** Optional resource scope, used by embedded audit views. */
  resourceId?: string;
}

const RANGE_IDS = ACTIVITY_RANGES.map((r) => r.id);
const DAY_MS = 86_400_000;

function utcDayStart(ms: number): number {
  const d = new Date(ms);
  return Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
}

function activityRelativeLabel(event: ActivityEvent, nowMs: number): string {
  const ts = Date.parse(event.ts);
  if (Number.isNaN(ts)) return event.rel;
  return relativeTime(event.ts, nowMs);
}

function activityDayLabel(event: ActivityEvent, nowMs: number): string {
  const ts = Date.parse(event.ts);
  if (Number.isNaN(ts)) return event.day;
  const dayDelta = Math.floor((utcDayStart(nowMs) - utcDayStart(ts)) / DAY_MS);
  if (dayDelta <= 0) return 'Today';
  if (dayDelta === 1) return 'Yesterday';
  if (dayDelta < 7) return 'Earlier this week';
  return new Date(ts).toISOString().slice(0, 10);
}

function kindPillStateClass(explicit: boolean, active: boolean): string {
  if (explicit) return 'is-explicit';
  if (active) return 'is-dim';
  return 'is-off';
}

export function ActivityTab({
  events,
  total,
  nowMs,
  folderLabel = 'default',
  density = 'comfortable',
  live = true,
  groupByDay = true,
  onPushToast,
  onNavigate,
  onRefresh,
  onQueryChange,
  limit = DEFAULT_ACTIVITY_LIMIT,
  resourceId,
}: Readonly<ActivityTabProps>) {
  const [range, setRange] = useUrlParam<ActivityRange>('range', '7d', {
    validate: (v) => (RANGE_IDS as readonly string[]).includes(v),
  });
  const [kinds, setKinds] = useUrlParam<Set<string>>('kind', new Set<string>(), {
    parse: (s) => new Set(s.split(',').filter(Boolean)),
    serialize: (set) => (set?.size ? [...set].join(',') : ''),
    validate: (v) => v instanceof Set,
  });
  const [sev, setSev] = useUrlParam<'any' | ActivitySeverity>('sev', 'any', {
    validate: (v) =>
      ['any', 'info', 'warning', 'error', 'critical'].includes(v),
  });
  const [q, setQ] = useUrlParam<string>('q', '');
  const [actor, setActor] = useUrlParam<string>('actor', 'any');
  const [selectedId, setSelectedId] = useState<string>(events[0]?.id ?? '');
  const [initialNowMs] = useState(() => Date.now());

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

  const kindParam = useMemo(
    () => (kinds.size ? [...kinds].join(',') : undefined),
    [kinds],
  );
  const searchParam = q.trim() || undefined;

  useEffect(() => {
    onQueryChange?.({
      range,
      kind: kindParam,
      sev: sev === 'any' ? undefined : sev,
      actor: actor === 'any' ? undefined : actor,
      q: searchParam,
      resourceId,
      limit,
    });
  }, [
    actor,
    kindParam,
    limit,
    onQueryChange,
    q,
    range,
    resourceId,
    searchParam,
    sev,
  ]);

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
  const backendTotal = total ?? filtered.length;
  const loadedCount = filtered.length;
  const activeFilterCount =
    (range === 'all' ? 0 : 1) +
    (kinds.size ? 1 : 0) +
    (sev === 'any' ? 0 : 1) +
    (actor === 'any' ? 0 : 1) +
    (q.trim() ? 1 : 0) +
    (resourceId ? 1 : 0);

  const displayNowMs = nowMs ?? initialNowMs;

  const grouped = useMemo(() => {
    if (!groupByDay) return [['', filtered] as const];
    const acc = new Map<string, ActivityEvent[]>();
    filtered.forEach((e) => {
      const day = activityDayLabel(e, displayNowMs);
      const bucket = acc.get(day) ?? [];
      bucket.push(e);
      acc.set(day, bucket);
    });
    return Array.from(acc.entries());
  }, [filtered, groupByDay, displayNowMs]);

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
                  ? 'Polling /activity every 30s'
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
                affordance exists to wipe rows.
              */}
            </div>
          </div>
        </div>

        <div className="activity-stats">
          <span className="stat">
            <b>{backendTotal}</b> matching events
          </span>
          <span className="dot-sep">·</span>
          <span className="stat">
            <b>{loadedCount}</b> loaded
          </span>
          <span className="dot-sep">·</span>
          <span className="stat">
            <b>{limit}</b> page limit
          </span>
          <span className="dot-sep">·</span>
          <span className="stat">
            <b>{activeFilterCount}</b> active filters
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
                  event={e}
                  relativeLabel={activityRelativeLabel(e, displayNowMs)}
                  folder={(e.meta?.folder as string) || folderLabel}
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
        event={selected}
        relativeLabel={selected ? activityRelativeLabel(selected, displayNowMs) : ''}
        folder={(selected?.meta?.folder as string) || folderLabel}
        onPushToast={onPushToast}
        onNavigate={onNavigate}
      />

    </div>
  );
}
