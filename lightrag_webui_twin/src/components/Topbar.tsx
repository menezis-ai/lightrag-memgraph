/**
 * Topbar — logo, tabs, workspace switcher, notifications, theme toggle.
 *
 * Ported from Desktop/UI/topbar.jsx. Key changes vs the proto:
 *   - Workspaces and notifications are *injected* via props instead of
 *     read from window.MOCK_WORKSPACES / window.MOCK_NOTIFICATIONS, so the
 *     component is fully testable with fixtures and ready to wire to a
 *     real fetcher later.
 *   - WorkspaceMenu and NotificationsPopover stay internal to this file
 *     (private sub-components, not exported) — they're tightly coupled to
 *     the popover open/close state owned by Topbar.
 *   - Click-outside + Escape handling preserved 1-for-1.
 */

import { useEffect, useRef, useState } from 'react';
import { Icon, type IconName } from './Icon';
import {
  DEFAULT_TABS,
  type Notification,
  type NotificationKind,
  type Tab,
  type Theme,
  type Workspace,
} from '../types/topbar';

export interface TopbarProps {
  /** Currently active tab id. */
  tab: string;
  onTab: (id: string) => void;
  theme: Theme;
  onTheme: () => void;
  /** Current workspace id (matches Workspace.id). */
  workspace: string;
  kbName: string;
  onSwitchWorkspace: (w: Workspace) => void;
  workspaces?: readonly Workspace[];
  notifications?: readonly Notification[];
  unreadCount?: number;
  onMarkAllRead?: () => void;
  onClearNotifications?: () => void;
  onOpenActivity?: () => void;
  tabs?: readonly Tab[];
}

export function Topbar({
  tab,
  onTab,
  theme,
  onTheme,
  workspace,
  kbName,
  onSwitchWorkspace,
  workspaces = [],
  notifications = [],
  unreadCount = 0,
  onMarkAllRead,
  onClearNotifications,
  onOpenActivity,
  tabs,
}: TopbarProps) {
  const TABS = tabs ?? DEFAULT_TABS;
  const [wsOpen, setWsOpen] = useState(false);
  const [notifOpen, setNotifOpen] = useState(false);
  const wsRef = useRef<HTMLDivElement>(null);
  const notifRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (wsRef.current && !wsRef.current.contains(target)) setWsOpen(false);
      if (notifRef.current && !notifRef.current.contains(target))
        setNotifOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setWsOpen(false);
        setNotifOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, []);

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">TR</div>
        <span className="brand-name">Twin</span>
        <span className="brand-sep">|</span>
        <span className="brand-kb" title="TWIN_KB_DISPLAY_NAME (env)">
          {kbName}
        </span>
      </div>
      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`tab${tab === t.id ? ' active' : ''}`}
            onClick={() => onTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <div className="topbar-right">
        <div ref={wsRef} style={{ position: 'relative' }}>
          <button
            type="button"
            className={`workspace-pill${wsOpen ? ' is-open' : ''}`}
            onClick={() => {
              setWsOpen((o) => !o);
              setNotifOpen(false);
            }}
            title="Switch workspace"
            aria-expanded={wsOpen}
            aria-haspopup="menu"
          >
            <Icon name="folder" size={12} />
            <span style={{ fontFamily: 'var(--font-mono)' }}>{workspace}</span>
            <Icon name="chevron-down" size={11} />
          </button>
          {wsOpen && (
            <WorkspaceMenu
              current={workspace}
              workspaces={workspaces}
              onPick={(ws) => {
                setWsOpen(false);
                onSwitchWorkspace(ws);
              }}
              onClose={() => setWsOpen(false)}
            />
          )}
        </div>
        <div ref={notifRef} style={{ position: 'relative' }}>
          <button
            type="button"
            className="icon-btn"
            aria-label={`Notifications${unreadCount ? `, ${unreadCount} unread` : ''}`}
            aria-expanded={notifOpen}
            onClick={() => {
              setNotifOpen((o) => !o);
              setWsOpen(false);
            }}
          >
            <Icon name="bell" size={16} />
            {unreadCount > 0 && (
              <span className="notif-badge">
                {unreadCount > 9 ? '9+' : unreadCount}
              </span>
            )}
          </button>
          {notifOpen && (
            <NotificationsPopover
              notifications={notifications}
              onMarkAllRead={onMarkAllRead}
              onClear={onClearNotifications}
              onClose={() => setNotifOpen(false)}
              onOpenActivity={onOpenActivity}
            />
          )}
        </div>
        <button
          type="button"
          className="icon-btn"
          aria-label="Theme"
          onClick={onTheme}
        >
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={16} />
        </button>
      </div>
    </header>
  );
}

interface WorkspaceMenuProps {
  current: string;
  workspaces: readonly Workspace[];
  onPick: (w: Workspace) => void;
  onClose: () => void;
}

function WorkspaceMenu({
  current,
  workspaces,
  onPick,
  onClose,
}: WorkspaceMenuProps) {
  return (
    <div className="ws-menu" role="menu" aria-label="Switch workspace">
      <div className="ws-menu-h">Workspaces</div>
      <ul className="ws-menu-list">
        {workspaces.map((w) => {
          const active = w.id === current;
          return (
            <li key={w.id}>
              <button
                type="button"
                role="menuitemradio"
                aria-checked={active}
                className={`ws-row${active ? ' is-active' : ''}`}
                onClick={() => !active && onPick(w)}
                disabled={active}
              >
                <span className="ws-row-l">
                  <Icon
                    name="folder"
                    size={12}
                    color="var(--color-text-secondary)"
                  />
                  <span className="ws-row-id">{w.id}</span>
                  {active && (
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
      </ul>
      <div className="ws-menu-f">
        <button type="button" className="link-btn" onClick={onClose}>
          Manage workspaces →
        </button>
      </div>
    </div>
  );
}

const NOTIF_ICON: Record<NotificationKind, { name: IconName; color: string }> = {
  'tag-mutation': { name: 'tags', color: 'var(--twin-accent)' },
  'source-ready': {
    name: 'circle-check',
    color: 'var(--twin-green-700, #2F7A40)',
  },
  'source-failed': {
    name: 'alert-triangle',
    color: 'var(--twin-red-vivid, #B03030)',
  },
  'pipeline-warning': {
    name: 'alert-triangle',
    color: 'var(--twin-amber-vivid, #9C7000)',
  },
  'source-uploaded': {
    name: 'cloud-upload',
    color: 'var(--color-text-secondary)',
  },
  retrieval: { name: 'search', color: 'var(--twin-accent)' },
  info: { name: 'info-circle', color: 'var(--color-text-secondary)' },
};

interface NotificationsPopoverProps {
  notifications: readonly Notification[];
  onMarkAllRead?: () => void;
  onClear?: () => void;
  onClose: () => void;
  onOpenActivity?: () => void;
}

function NotificationsPopover({
  notifications,
  onMarkAllRead,
  onClear,
  onClose,
  onOpenActivity,
}: NotificationsPopoverProps) {
  const unread = notifications.filter((n) => !n.read).length;

  return (
    <div className="notif-popover" role="dialog" aria-label="Notifications">
      <header className="notif-h">
        <span className="notif-title">Notifications</span>
        <span className="notif-count">
          {unread > 0 ? (
            <>
              <b>{unread}</b> unread · {notifications.length} total
            </>
          ) : (
            <>{notifications.length} total</>
          )}
        </span>
        <div className="notif-h-actions">
          {unread > 0 && (
            <button
              type="button"
              className="link-btn small"
              onClick={onMarkAllRead}
            >
              Mark all read
            </button>
          )}
          <button
            type="button"
            className="icon-btn small"
            onClick={onClose}
            aria-label="Close"
          >
            <Icon name="x" size={12} />
          </button>
        </div>
      </header>
      {notifications.length === 0 ? (
        <div className="notif-empty">
          <Icon name="bell" size={20} color="var(--color-text-tertiary)" />
          <div>You're all caught up.</div>
        </div>
      ) : (
        <ul className="notif-list">
          {notifications.map((n) => {
            const ico = NOTIF_ICON[n.kind] ?? NOTIF_ICON.info;
            return (
              <li
                key={n.id}
                className={`notif-item${n.read ? '' : ' is-unread'}`}
              >
                <span className="notif-ico" style={{ color: ico.color }}>
                  <Icon name={ico.name} size={14} />
                </span>
                <div className="notif-body">
                  <div className="notif-line1">
                    <span className="notif-t">{n.title}</span>
                    {n.tagname && (
                      <code className="notif-tagname">{n.tagname}</code>
                    )}
                    {n.suffix && (
                      <span className="notif-suffix">{n.suffix}</span>
                    )}
                  </div>
                  {n.sub && <div className="notif-sub">{n.sub}</div>}
                  <div className="notif-rel">{n.rel}</div>
                </div>
                {!n.read && <span className="notif-dot" aria-label="unread" />}
              </li>
            );
          })}
        </ul>
      )}
      <footer className="notif-f">
        <button
          type="button"
          className="link-btn"
          onClick={() => {
            onClose();
            onOpenActivity?.();
          }}
        >
          View full activity log →
        </button>
        {notifications.length > 0 && (
          <button type="button" className="link-btn danger" onClick={onClear}>
            Clear all
          </button>
        )}
      </footer>
    </div>
  );
}
