import type { TagCurrentUser } from '../types/tag';

// Fallback identity when no auth backend resolves a user (open-access /
// LightRAG-parity deployments). Matches the backend's anonymous actor label.
export const CURRENT_USER: TagCurrentUser = {
  name: 'operator@twin.local',
  palier: 3,
  role: 'admin / steward',
};

export type DocumentsStatusFilterKey =
  | 'all'
  | 'completed'
  | 'processing'
  | 'pending'
  | 'failed';

export const DOCUMENTS_STATUS_FILTERS = [
  'all',
  'completed',
  'processing',
  'pending',
  'failed',
] as const;

export const DOCUMENTS_STATUS_TO_API: Record<
  Exclude<DocumentsStatusFilterKey, 'all'>,
  string
> = {
  completed: 'PROCESSED',
  processing: 'PROCESSING',
  pending: 'PENDING',
  failed: 'FAILED',
};
