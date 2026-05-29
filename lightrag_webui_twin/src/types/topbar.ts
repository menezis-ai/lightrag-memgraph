/**
 * Topbar-related types.
 *
 * These interfaces double as the contract that the backend phase-1 endpoints
 * will need to honor: `/workspaces`, `/notifications`, etc. (Q2 of the plan:
 * "interfaces TS deviennent la spec d'API que phase 1 doit respecter").
 */

export type Theme = 'light' | 'dark';

export type WorkspaceVisibility = 'private' | 'internal' | 'public';

export type WorkspaceRole = 'admin' | 'admin / steward' | 'steward' | 'reader' | 'owner';

export interface Workspace {
  id: string;
  kb: string;
  visibility: WorkspaceVisibility;
  sources: number;
  role: WorkspaceRole;
  current: boolean;
}

export type NotificationKind =
  | 'tag-mutation'
  | 'source-ready'
  | 'source-failed'
  | 'pipeline-warning'
  | 'source-uploaded'
  | 'retrieval'
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

export const DEFAULT_TABS: readonly Tab[] = [
  { id: 'documents', label: 'Documents' },
  { id: 'retrieval', label: 'Retrieval' },
  { id: 'tags', label: 'Tags' },
  { id: 'activity', label: 'Activity' },
  { id: 'graph', label: 'Graph' },
  { id: 'api', label: 'API' },
  { id: 'settings', label: 'Settings' },
];
