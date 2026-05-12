/**
 * Activity feed types — audit trail across source lifecycle, tag mutations,
 * retrievals, pipeline events, auth, settings.
 *
 * Contract template for backend phase 1: `GET /activity?range=&kind=&sev=&actor=&q=`
 * returns `{ items: ActivityEvent[], total }`. CSV export is client-side.
 */

import type { IconName } from '../components/Icon';

export type ActivityKind =
  | 'retrieval'
  | 'tag-mutation'
  | 'source-uploaded'
  | 'source-ready'
  | 'source-failed'
  | 'pipeline-warning'
  | 'auth'
  | 'settings';

export type ActivitySeverity = 'info' | 'warning' | 'error' | 'critical';

export type ActivityRange = '24h' | '7d' | '30d' | 'all';

export type ActivityTargetType =
  | 'query'
  | 'source'
  | 'bulk'
  | 'session'
  | 'workspace';

export interface ActivityActor {
  user: string;
  role: string;
}

export interface ActivityTarget {
  type: ActivityTargetType | string;
  label: string;
  id?: string;
}

export interface ActivityEvent {
  id: string;
  ts: string;
  rel: string;
  day: string;
  kind: ActivityKind;
  sev: ActivitySeverity;
  actor: ActivityActor;
  target: ActivityTarget;
  summary: string;
  meta: Record<string, unknown>;
}

export interface ActivityKindMeta {
  label: string;
  icon: IconName;
  color: string;
}

export const ACTIVITY_KIND_META: Record<ActivityKind, ActivityKindMeta> = {
  retrieval: { label: 'Retrieval', icon: 'search', color: 'var(--twin-accent)' },
  'tag-mutation': { label: 'Tag mutation', icon: 'tags', color: 'var(--twin-accent)' },
  'source-uploaded': {
    label: 'Source uploaded',
    icon: 'cloud-upload',
    color: 'var(--color-text-secondary)',
  },
  'source-ready': {
    label: 'Source ready',
    icon: 'circle-check',
    color: 'var(--twin-green-700)',
  },
  'source-failed': {
    label: 'Source failed',
    icon: 'alert-triangle',
    color: 'var(--twin-red-vivid)',
  },
  'pipeline-warning': {
    label: 'Pipeline',
    icon: 'alert-triangle',
    color: 'var(--twin-amber-vivid)',
  },
  auth: { label: 'Auth', icon: 'lock', color: 'var(--color-text-secondary)' },
  settings: {
    label: 'Settings',
    icon: 'settings',
    color: 'var(--color-text-secondary)',
  },
};

export const ACTIVITY_RANGES: readonly { id: ActivityRange; label: string }[] = [
  { id: '24h', label: '24h' },
  { id: '7d', label: '7d' },
  { id: '30d', label: '30d' },
  { id: 'all', label: 'All' },
];

export const ACTIVITY_RANGE_MS: Record<Exclude<ActivityRange, 'all'>, number> = {
  '24h': 86_400_000,
  '7d': 7 * 86_400_000,
  '30d': 30 * 86_400_000,
};
