/**
 * Disable-ingestion-controls helper.
 *
 * Returns ``true`` when the operator MUST be prevented from triggering
 * a new ingestion call (Add source, Re-process failed, …) because the
 * Memgraph instance is at quota.
 *
 * Defaults to ``false`` while the snapshot is loading or when the
 * backend reports ``ok`` / ``warning`` — only ``blocked`` disables.
 */

import { useInstanceQuota } from '../api/queries';

export function useIngestionDisabled(): boolean {
  const { data } = useInstanceQuota();
  return data?.status === 'blocked';
}
