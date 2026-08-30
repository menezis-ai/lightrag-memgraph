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
