/**
 * WorkspaceSwitcher — transitional wrapper for the Twin folder selector.
 *
 * The class/test ids keep their historical `workspace` name for now, but
 * visible copy and HTTP semantics now use Twin "folders".
 *
 * The `X-Twin-Folder` HTTP header is set by the host App after
 * picking — this component just emits the id; it does not own the apiFetch
 * default headers.
 */

import { useEffect, useRef, useState } from 'react';
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
  // MyAccess gates the parent BNP/LGP workspace. Folders are scoped inside
  // that KB, so dev/prod both show the configured folder list by default.
  const visible = workspaces;

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
          aria-label="Switch folder"
          data-testid="topbar-workspace-menu"
        >
          <div className="ws-menu-h">Folders ({visible.length})</div>
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
                  No folder available for this KB. Please contact Twincore Team
                </div>
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
