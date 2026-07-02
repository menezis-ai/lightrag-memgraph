/**
 * Instance storage quota snapshot.
 *
 * Mirrors ``server/quota.py:snapshot``. ``configured == false`` means
 * the deployment runs without ``MEMGRAPH_MEMORY_LIMIT`` set — the UI
 * then hides the banner entirely instead of rendering ``? / ? GiB``.
 */
export type QuotaStatus = 'ok' | 'warning' | 'blocked';

export interface QuotaSnapshot {
  used_bytes: number | null;
  limit_bytes: number | null;
  used_pct: number | null;
  status: QuotaStatus;
  warn_threshold: number;
  configured: boolean;
}
