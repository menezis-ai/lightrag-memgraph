/**
 * StatusBadge — small pill rendering a TagStatus.
 *
 * Ported from the inline `StatusBadge` in Desktop/UI/tags.jsx.
 * Exposed at module scope so other components (e.g. DocumentsTab in future
 * sprints) can re-use the same pill styling without duplicating the map.
 */

import type { TagStatus } from '../types/tag';

export type StatusBadgeSize = 'sm' | 'md';

interface BadgeMeta {
  label: string;
  cls: string;
}

const MAP: Record<TagStatus, BadgeMeta> = {
  active: { label: 'Active', cls: 'status-active' },
  'pending-promotion': { label: 'Pending', cls: 'status-pending' },
  'pending-review': { label: 'Pending', cls: 'status-pending' },
  deprecated: { label: 'Deprecated', cls: 'status-deprecated' },
  rejected: { label: 'Rejected', cls: 'status-rejected' },
};

export interface StatusBadgeProps {
  status: TagStatus;
  size?: StatusBadgeSize;
}

export function StatusBadge({ status, size = 'sm' }: StatusBadgeProps) {
  const m = MAP[status] ?? { label: String(status), cls: 'status-active' };
  return (
    <span className={`status-badge ${m.cls} ${size}`} data-status={status}>
      {m.label}
    </span>
  );
}
