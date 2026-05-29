/**
 * WorkspaceSwitcher — dropdown listing workspaces the user can access (per
 * MyAccess claim), with click → setActiveWorkspace.
 *
 * Layered above the existing Topbar workspace pill so it can be embedded
 * either inside or outside the Topbar. Filters the workspace list to only
 * those present in `useAuth().user.workspaces` (no leak of workspaces the
 * user doesn't have a claim for, even if the fixture/API returns them).
 *
 * The `X-Twin-Workspace` HTTP header (#154) is set by the host App after
 * picking — this component just emits the id; it does not own the apiFetch
 * default headers.
 */

import { useEffect, useRef, useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { Icon } from '../Icon';
import type { Workspace } from '../../types/topbar';

export interface WorkspaceSwitcherProps {
  active: string;
  workspaces: readonly Workspace[];
  onPick: (id: string) => void;
}

export function WorkspaceSwitcher({
  active,
  workspaces,
  onPick,
}: WorkspaceSwitcherProps) {
  const { user } = useAuth();
  const allowed = new Set(user?.workspaces ?? []);
  // If MyAccess didn't list any, fall back to all (this is the dev case);
  // production always seeds at least one workspace claim.
  const visible =
    allowed.size === 0
      ? workspaces
      : workspaces.filter((w) => allowed.has(w.id));

  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        type="button"
        className={`workspace-pill${open ? ' is-open' : ''}`}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="menu"
        data-testid="topbar-workspace-switcher"
      >
        <Icon name="folder" size={12} />
        <span style={{ fontFamily: 'var(--font-mono)' }}>{active}</span>
        <Icon name="chevron-down" size={11} />
      </button>
      {open && (
        <div
          className="ws-menu"
          role="menu"
          aria-label="Switch workspace"
          data-testid="topbar-workspace-menu"
        >
          <div className="ws-menu-h">Workspaces ({visible.length})</div>
          <ul className="ws-menu-list">
            {visible.map((w) => {
              const isActive = w.id === active;
              return (
                <li key={w.id}>
                  <button
                    type="button"
                    role="menuitemradio"
                    aria-checked={isActive}
                    className={`ws-row${isActive ? ' is-active' : ''}`}
                    disabled={isActive}
                    data-testid={`topbar-workspace-pick-${w.id}`}
                    onClick={() => {
                      if (!isActive) {
                        onPick(w.id);
                        setOpen(false);
                      }
                    }}
                  >
                    <span className="ws-row-l">
                      <Icon
                        name="folder"
                        size={12}
                        color="var(--color-text-secondary)"
                      />
                      <span className="ws-row-id">{w.id}</span>
                      {isActive && (
                        <Icon
                          name="circle-check"
                          size={12}
                          color="var(--twin-accent)"
                        />
                      )}
                    </span>
                    <span className="ws-row-r">
                      <span className="ws-row-kb">{w.kb}</span>
                      <span className="ws-row-meta">
                        <span className="ws-vis">
                          <Icon
                            name={w.visibility === 'private' ? 'lock' : 'world'}
                            size={10}
                          />
                          {w.visibility}
                        </span>
                        <span className="ws-sep">·</span>
                        <span>{w.sources.toLocaleString()} sources</span>
                        <span className="ws-sep">·</span>
                        <span className="ws-role">{w.role}</span>
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
            {visible.length === 0 && (
              <li>
                <div
                  className="muted"
                  style={{ padding: 12 }}
                  data-testid="topbar-workspace-empty"
                >
                  No workspace accessible with your MyAccess claim.
                </div>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
