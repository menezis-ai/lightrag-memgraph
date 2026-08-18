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
  onManageFolders?: () => void;
}

export function FolderSwitcher({
  active,
  folders,
  onPick,
  onManageFolders,
}: Readonly<FolderSwitcherProps>) {
  // The corporate IdP gates the parent KB. Folders are scoped inside
  // that KB, so dev/prod both show the configured folder list by default.
  const visible = folders;

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

  return (
    <div ref={ref} className="topbar-popover-anchor">
      <button
        type="button"
        className={`folder-pill${open ? ' is-open' : ''}`}
        onClick={() => setOpen((o) => !o)}
        title="Switch folder"
        aria-expanded={open}
        aria-haspopup="menu"
        data-testid="topbar-folder-switcher"
      >
        <Icon name="folder" size={12} />
        <span className="topbar-mono">{active}</span>
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
                  className="muted topbar-empty-state"
                  data-testid="topbar-folder-empty"
                >
                  No folder is provisioned for this knowledge base. Ask your
                  platform administrator to provision one.
                </div>
              </li>
            )}
          </ul>
          {onManageFolders && (
            <div className="ws-menu-f">
              <button
                type="button"
                className="link-btn"
                onClick={() => {
                  setOpen(false);
                  onManageFolders();
                }}
              >
                Manage folders →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
