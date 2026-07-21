/**
 * Topbar-related types.
 *
 * These interfaces double as the contract that the backend phase-1 endpoints
 * will need to honor: `/folders`, `/notifications`, etc. (Q2 of the plan:
 * "interfaces TS deviennent la spec d'API que phase 1 doit respecter").
 */

export type Theme = 'light' | 'dark';

export type FolderVisibility = 'private' | 'internal' | 'public';

export type FolderRole = 'admin' | 'admin / steward' | 'steward' | 'reader' | 'owner';

export interface Folder {
  id: string;
  kb: string;
  visibility: FolderVisibility;
  sources: number;
  role: FolderRole;
  current: boolean;
}

export type NotificationKind =
  | 'tag-mutation'
  | 'source-ready'
  | 'source-failed'
  | 'pipeline-warning'
  | 'source-uploaded'
  | 'retrieval'
  | 'procedure-review'
  | 'info';

export interface Notification {
  id: string;
  kind: NotificationKind;
  title: string;
  tagname?: string;
  suffix?: string;
  sub?: string;
  rel: string;
  read: boolean;
}

export interface Tab {
  id: string;
  label: string;
}

// Canonical nav order (doctrine product) — Documents · Tags · Retrieval · Graph
// · Activity · Settings. API is NOT a top-level tab — it lives inside Settings
// as the "API" section (see SettingsTab). Topbar reordering / re-introducing
// "API" here is a regression.
export const DEFAULT_TABS: readonly Tab[] = [
  { id: 'documents', label: 'Documents' },
  { id: 'tags', label: 'Tags' },
  { id: 'retrieval', label: 'Retrieval' },
  { id: 'graph', label: 'Graph' },
  { id: 'activity', label: 'Activity' },
  { id: 'settings', label: 'Settings' },
];
