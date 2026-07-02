/**
 * TodoBell — refresh-on-poll variant of the notifications dropdown.
 *
 * Reuses the same backend (`/twin/api/notifications`) as the existing
 * NotificationsPopover but configures a 20s refetchInterval so a Steward
 * doesn't miss tag-approval requests sitting in their queue. Badge color
 * scales with unread count (1-9 = accent, 10+ = warn).
 *
 * This is INTENTIONALLY separate from the existing Topbar notif bell — the
 * existing one shows the local toast/notification queue (read-when-displayed
 * pattern); TodoBell focuses on actionable items the user must clear.
 */

import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../../api/resources';
import { Icon } from '../Icon';
import type { Notification } from '../../types/topbar';

export interface TodoBellProps {
  pollMs?: number;
  onOpenActivity?: () => void;
}

const ACTIONABLE_KINDS = new Set<Notification['kind']>([
  'tag-mutation',
  'source-failed',
  'pipeline-warning',
]);

function todoBellLabel(todoCount: number): string {
  if (todoCount === 0) return 'To-do';
  return `To-do, ${todoCount} pending`;
}

export function TodoBell({ pollMs = 20_000, onOpenActivity }: Readonly<TodoBellProps>) {
  const { data: notifications = [] } = useQuery({
    queryKey: ['todo-bell-notifications'] as const,
    queryFn: () => api.listNotifications(),
    refetchInterval: pollMs,
    refetchOnWindowFocus: false,
  });

  const todos = notifications.filter(
    (n) => !n.read && ACTIONABLE_KINDS.has(n.kind),
  );
  const todoCount = todos.length;

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
        className="icon-btn"
        aria-label={todoBellLabel(todoCount)}
        aria-expanded={open}
        data-testid="topbar-todo-bell"
        onClick={() => setOpen((o) => !o)}
      >
        <Icon name="bell" size={16} />
        {todoCount > 0 && (
          <span
            className="notif-badge"
            data-testid="topbar-todo-count"
            style={
              todoCount >= 10
                ? {
                    background: 'var(--twin-amber-vivid, #9C7000)',
                  }
                : undefined
            }
          >
            {todoCount > 9 ? '9+' : todoCount}
          </span>
        )}
      </button>
      {open && (
        <dialog
          open
          className="notif-popover"
          aria-label="To-do list"
          style={{ width: 320 }}
        >
          <header className="notif-h">
            <span className="notif-title">To-do</span>
            <span className="notif-count">
              {todoCount > 0 ? (
                <>
                  <b>{todoCount}</b> actionable
                </>
              ) : (
                'all caught up'
              )}
            </span>
          </header>
          {todos.length === 0 ? (
            <div className="notif-empty">
              <Icon name="bell" size={20} color="var(--color-text-tertiary)" />
              <div>You're all caught up.</div>
            </div>
          ) : (
            <ul className="notif-list">
              {todos.map((n) => (
                <li
                  key={n.id}
                  className="notif-item is-unread"
                  data-testid={`topbar-todo-item-${n.id}`}
                >
                  <span className="notif-ico">
                    <Icon name="alert-triangle" size={14} />
                  </span>
                  <div className="notif-body">
                    <div className="notif-line1">
                      <span className="notif-t">{n.title}</span>
                      {n.suffix && (
                        <span className="notif-suffix">{n.suffix}</span>
                      )}
                    </div>
                    {n.sub && <div className="notif-sub">{n.sub}</div>}
                  </div>
                </li>
              ))}
            </ul>
          )}
          <footer className="notif-f">
            <button
              type="button"
              className="link-btn"
              onClick={() => {
                setOpen(false);
                onOpenActivity?.();
              }}
            >
              Open activity →
            </button>
          </footer>
        </dialog>
      )}
    </div>
  );
}
