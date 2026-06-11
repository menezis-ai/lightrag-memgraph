/**
 * FolderSwitcher — Twin folder selector.
 *
 * The `X-Twin-Folder` HTTP header is set by the host App after
 * picking — this component just emits the id; it does not own the apiFetch
 * default headers.
 */

import { useEffect, useRef, useState } from 'react';
import { Icon } from '../Icon';
import type { Folder } from '../../types/topbar';

export interface FolderSwitcherProps {
  active: string;
  folders: readonly Folder[];
  onPick: (id: string) => void;
}

export function FolderSwitcher({
  active,
  folders,
  onPick,
}: FolderSwitcherProps) {
  // The corporate IdP gates the parent KB. Folders are scoped inside
  // that KB, so dev/prod both show the configured folder list by default.
  const visible = folders;

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
        className={`folder-pill${open ? ' is-open' : ''}`}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="menu"
        data-testid="topbar-folder-switcher"
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
          data-testid="topbar-folder-menu"
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
                    data-testid={`topbar-folder-pick-${w.id}`}
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
                  data-testid="topbar-folder-empty"
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
