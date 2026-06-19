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
  | 'doc-retagged'
  | 'doc-approved'
  | 'doc-rejected'
  | 'doc-deleted'
  | 'classification-rejected'
  | 'source-uploaded'
  | 'source-ready'
  | 'source-failed'
  | 'pipeline-warning'
  | 'graph-entity-edited'
  | 'graph-relation-edited'
  | 'api-key-created'
  | 'api-key-revoked'
  | 'auth'
  | 'settings';

export type ActivitySeverity = 'info' | 'warning' | 'error' | 'critical';

export type ActivityRange = '24h' | '7d' | '30d' | 'all';

export type ActivityTargetType =
  | 'query'
  | 'source'
  | 'bulk'
  | 'session'
  | 'folder';

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
  'doc-retagged': {
    label: 'Document retagged',
    icon: 'tags',
    color: 'var(--twin-accent)',
  },
  'doc-approved': {
    label: 'Document approved',
    icon: 'circle-check',
    color: 'var(--twin-green-700)',
  },
  'doc-rejected': {
    label: 'Document rejected',
    icon: 'alert-triangle',
    color: 'var(--twin-amber-vivid)',
  },
  'doc-deleted': {
    label: 'Document deleted',
    icon: 'trash',
    color: 'var(--color-text-secondary)',
  },
  'classification-rejected': {
    label: 'Classification rejected',
    icon: 'alert-triangle',
    color: 'var(--twin-red-vivid)',
  },
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
  'graph-entity-edited': {
    label: 'Graph entity edited',
    icon: 'circle-dot',
    color: 'var(--twin-accent)',
  },
  'graph-relation-edited': {
    label: 'Graph relation edited',
    icon: 'link',
    color: 'var(--twin-accent)',
  },
  'api-key-created': {
    label: 'API key created',
    icon: 'lock',
    color: 'var(--twin-green-700)',
  },
  'api-key-revoked': {
    label: 'API key revoked',
    icon: 'lock',
    color: 'var(--twin-amber-vivid)',
  },
  auth: { label: 'Auth', icon: 'lock', color: 'var(--color-text-secondary)' },
  settings: {
    label: 'Settings',
    icon: 'settings',
    color: 'var(--color-text-secondary)',
  },
};

/**
 * Safe meta lookup. The backend may emit kinds the UI map does not
 * enumerate (e.g. dynamic settings sub-kinds via `body.kind`), so a raw
 * `ACTIVITY_KIND_META[kind]` can be `undefined` and crash the whole feed
 * when a row dereferences `.color`/`.icon`. Never crash on an unknown kind:
 * fall back to a neutral, self-describing meta.
 */
export function resolveKindMeta(kind: string): ActivityKindMeta {
  const known = ACTIVITY_KIND_META[kind as ActivityKind];
  if (known) return known;
  const label = kind
    ? kind.replace(/[-_]/g, ' ').replace(/^./, (c) => c.toUpperCase())
    : 'Activity';
  return { label, icon: 'activity', color: 'var(--color-text-secondary)' };
}

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
