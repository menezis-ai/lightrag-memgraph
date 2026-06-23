/**
 * SystemStatusIndicator — sys-pill in the topbar showing worst-of (LightRAG,
 * Twin) health, with a sys-popover detail panel on click.
 *
 * Polls `/health` (LightRAG-native) and `/twin/api/health` (overlay) every
 * 30s. The displayed status is the worst of the two:
 *   - both ok       → ok       → "All systems" pill, green dot
 *   - one degraded  → degraded → "Degraded"    pill (warn variant)
 *   - any down      → down     → "Outage"      pill (error variant)
 *
 * Click opens a popover listing each surface with its label + detail.
 * Failures fall back to "down" so a flaky overlay still surfaces visually.
 */

import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../api/resources';

type Status = 'ok' | 'degraded' | 'down';

const RANK: Record<Status, number> = { ok: 0, degraded: 1, down: 2 };

function worst(a: Status, b: Status): Status {
  return RANK[a] >= RANK[b] ? a : b;
}

interface OverallMeta {
  dot: 'ok' | 'warn' | 'error';
  pillVariant: '' | 'sys-pill-warn' | 'sys-pill-error';
  label: string;
  title: string;
}

const OVERALL_META: Record<Status, OverallMeta> = {
  ok: { dot: 'ok', pillVariant: '', label: 'All systems', title: 'All systems operational' },
  degraded: { dot: 'warn', pillVariant: 'sys-pill-warn', label: 'Degraded', title: 'Some systems degraded' },
  down: { dot: 'error', pillVariant: 'sys-pill-error', label: 'Outage', title: 'Service outage' },
};

// Map per-surface status → the CSS modifier suffix expected by .sys-dot-* and
// .sys-check-status.sys-*. The legacy CSS uses "warn"/"error" (not "degraded"/
// "down") on the dot variants, so we translate.
const PER_SURFACE_DOT: Record<Status, 'ok' | 'warn' | 'error'> = {
  ok: 'ok',
  degraded: 'warn',
  down: 'error',
};

const PER_SURFACE_DETAIL: Record<Status, string> = {
  ok: 'operational',
  degraded: 'degraded',
  down: 'unreachable',
};

export interface SystemStatusIndicatorProps {
  pollMs?: number;
}

export function SystemStatusIndicator({
  pollMs = 30_000,
}: Readonly<SystemStatusIndicatorProps>) {
  const lightrag = useQuery({
    queryKey: ['lightrag-health'] as const,
    queryFn: async () => {
      try {
        return await api.health();
      } catch {
        return { status: 'down' as Status };
      }
    },
    refetchInterval: pollMs,
    refetchOnWindowFocus: false,
  });
  const twin = useQuery({
    queryKey: ['twin-health'] as const,
    queryFn: async () => {
      try {
        return await api.twinHealth();
      } catch {
        return { status: 'down' as Status };
      }
    },
    refetchInterval: pollMs,
    refetchOnWindowFocus: false,
  });

  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, []);

  const lightragStatus = (lightrag.data?.status ?? 'down') as Status;
  const twinStatus = (twin.data?.status ?? 'down') as Status;
  const overall = worst(lightragStatus, twinStatus);
  const meta = OVERALL_META[overall];

  const pillClass = meta.pillVariant ? `sys-pill ${meta.pillVariant}` : 'sys-pill';

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        type="button"
        className={pillClass}
        title={meta.title}
        aria-label={`System status: ${meta.label}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        data-testid="topbar-status-indicator"
        data-status={overall}
      >
        <span
          className={`sys-dot sys-dot-${meta.dot}`}
          aria-hidden
          data-status={overall}
        />
        <span className="sys-pill-label">{meta.label}</span>
      </button>
      {open && (
        <dialog open className="sys-popover" aria-label="System status">
          <div className="sys-popover-h">
            <span className="sys-popover-title">
              <span className={`sys-dot sys-dot-${meta.dot}`} aria-hidden />
              {meta.title}
            </span>
          </div>
          <ul className="sys-popover-checks">
            <li>
              <span
                className={`sys-dot sys-dot-${PER_SURFACE_DOT[lightragStatus]}`}
                aria-hidden
              />
              <span className="sys-check-label">LightRAG</span>
              <span
                className={`sys-check-status sys-${PER_SURFACE_DOT[lightragStatus]}`}
                data-testid="status-lightrag"
                data-status={lightragStatus}
              >
                {PER_SURFACE_DETAIL[lightragStatus]}
              </span>
            </li>
            <li>
              <span
                className={`sys-dot sys-dot-${PER_SURFACE_DOT[twinStatus]}`}
                aria-hidden
              />
              <span className="sys-check-label">Twin overlay</span>
              <span
                className={`sys-check-status sys-${PER_SURFACE_DOT[twinStatus]}`}
                data-testid="status-twin"
                data-status={twinStatus}
              >
                {PER_SURFACE_DETAIL[twinStatus]}
              </span>
            </li>
          </ul>
        </dialog>
      )}
    </div>
  );
}
