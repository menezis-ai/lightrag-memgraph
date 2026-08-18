/**
 * Instance storage quota banner.
 *
 * Rendered at the top of Documents tab and inside the Pipeline popover.
 * Three states (mirrored on the backend ``server/quota.py``):
 *
 *   - ``ok``        → banner hidden
 *   - ``warning``   → amber, informational ("Storage at 87 % …")
 *   - ``blocked``   → red, fail-stop ("Memgraph instance quota
 *                    reached — ingestion disabled until space freed")
 *
 * When the backend reports ``configured == false`` (no limit reported by
 * Memgraph and no fallback env), the banner is unconditionally hidden.
 */

import { useInstanceQuota } from '../api/queries';
import type { QuotaSnapshot } from '../types/quota';

export type QuotaTone = 'compact' | 'block';

export interface QuotaBannerProps {
  /** ``compact`` for inline use in popovers / smaller surfaces;
   *  ``block`` for the top-of-tab full-width variant. */
  tone?: QuotaTone;
}

function formatGiB(bytes: number | null): string {
  if (bytes === null) return '?';
  return `${(bytes / (1024 ** 3)).toFixed(2)} GiB`;
}

function formatPct(pct: number | null): string {
  if (pct === null) return '?';
  return `${Math.round(pct * 100)}%`;
}

function bannerCopy(snap: QuotaSnapshot): { headline: string; subline?: string } {
  const used = formatGiB(snap.used_bytes);
  const limit = formatGiB(snap.limit_bytes);
  const pct = formatPct(snap.used_pct);
  if (snap.status === 'blocked') {
    return {
      headline: 'Memgraph instance quota reached — ingestion disabled until space is freed',
      subline: `${used} / ${limit} (${pct}). Delete documents or raise the binding Memgraph memory/license limit to recover.`,
    };
  }
  return {
    headline: `Storage at ${pct} (${used} / ${limit})`,
    subline: `Ingestion will be blocked when usage reaches 100%. Free space before then to avoid an outage.`,
  };
}

export function QuotaBanner({ tone = 'block' }: QuotaBannerProps = {}) {
  const { data } = useInstanceQuota();
  if (!data) return null;
  if (!data.configured) return null;
  if (data.status === 'ok') return null;

  const copy = bannerCopy(data);
  const cls = `quota-banner quota-banner-${tone} quota-${data.status}`;
  const role = data.status === 'blocked' ? 'alert' : 'status';
  const live = data.status === 'blocked' ? 'assertive' : 'polite';

  return (
    <div
      className={cls}
      role={role}
      aria-live={live}
      data-testid={`quota-banner-${data.status}`}
    >
      <strong>{copy.headline}</strong>
      {copy.subline && <span className="quota-banner-sub">{copy.subline}</span>}
    </div>
  );
}
