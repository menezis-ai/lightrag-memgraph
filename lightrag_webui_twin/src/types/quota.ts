/**
 * Instance storage quota snapshot.
 *
 * Mirrors ``server/quota.py:snapshot``. ``configured == false`` means
 * neither Memgraph 3.12 nor the fallback environment exposed a usable
 * memory limit, so the UI hides the banner instead of rendering ``? / ? GiB``.
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
