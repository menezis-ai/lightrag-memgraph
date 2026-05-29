/**
 * SystemStatusIndicator — small dot showing worst-of (LightRAG, Twin) health.
 *
 * Polls `/health` (LightRAG-native) and `/twin/api/health` (overlay) every
 * 30s. The displayed status is the worst of the two:
 *   - both ok       → ok
 *   - one degraded  → degraded
 *   - any down      → down
 *
 * Click opens a tiny popover listing the per-surface status. Failures fall
 * back to "down" so a flaky overlay still surfaces visually.
 */

import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../api/resources';
import { Icon } from '../Icon';

type Status = 'ok' | 'degraded' | 'down';

const RANK: Record<Status, number> = { ok: 0, degraded: 1, down: 2 };

function worst(a: Status, b: Status): Status {
  return RANK[a] >= RANK[b] ? a : b;
}

const DOT_COLOR: Record<Status, string> = {
  ok: 'var(--twin-green-700, #2F7A40)',
  degraded: 'var(--twin-amber-vivid, #9C7000)',
  down: 'var(--twin-red-vivid, #B03030)',
};

export interface SystemStatusIndicatorProps {
  pollMs?: number;
}

export function SystemStatusIndicator({
  pollMs = 30_000,
}: SystemStatusIndicatorProps) {
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
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  const lightragStatus = (lightrag.data?.status ?? 'down') as Status;
  const twinStatus = (twin.data?.status ?? 'down') as Status;
  const overall = worst(lightragStatus, twinStatus);

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        type="button"
        className="icon-btn"
        aria-label={`System status: ${overall}`}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        data-testid="topbar-status-indicator"
      >
        <span
          aria-hidden
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: DOT_COLOR[overall],
          }}
          data-status={overall}
        />
      </button>
      {open && (
        <div
          className="notif-popover"
          role="dialog"
          aria-label="System status"
          style={{ width: 260 }}
        >
          <header className="notif-h">
            <span className="notif-title">System status</span>
          </header>
          <ul className="notif-list">
            <li className="notif-item">
              <span className="notif-ico">
                <Icon name="circle-check" size={14} />
              </span>
              <div className="notif-body">
                <div className="notif-line1">
                  <span className="notif-t">LightRAG</span>
                  <span
                    className="mono"
                    data-testid="status-lightrag"
                    data-status={lightragStatus}
                  >
                    {lightragStatus}
                  </span>
                </div>
              </div>
            </li>
            <li className="notif-item">
              <span className="notif-ico">
                <Icon name="circle-check" size={14} />
              </span>
              <div className="notif-body">
                <div className="notif-line1">
                  <span className="notif-t">Twin overlay</span>
                  <span
                    className="mono"
                    data-testid="status-twin"
                    data-status={twinStatus}
                  >
                    {twinStatus}
                  </span>
                </div>
              </div>
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
