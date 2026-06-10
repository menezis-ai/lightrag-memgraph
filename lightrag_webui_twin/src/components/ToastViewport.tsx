/**
 * Toast viewport — stacked runtime notifications with screen-reader
 * announcements.
 *
 * Ported from Desktop/UI/modals.jsx (ToastViewport + ToastCard). Behaviors:
 *   - Renders the last TOAST_MAX_VISIBLE toasts; older are collapsed under
 *     a "+N more" dismiss button.
 *   - Each toast is announced exactly once per `(id, kind)` pair via two
 *     ARIA live regions (polite for non-error, assertive for error).
 *   - Done toasts may carry an `undo` payload; click -> onUndo(toast).
 *   - Error toasts get a manual dismiss; propagating ones auto-resolve
 *     when the host updates them to done/error.
 *
 * The component is purely presentational — the timer & lifecycle live
 * in the host (App / RetagModal / AddSourceModal) so they own the
 * truth about what each toast means.
 */

import { useEffect, useRef, useState } from 'react';
import { Icon } from './Icon';
import { TOAST_MAX_VISIBLE, type Toast } from '../types/toast';

export interface ToastViewportProps {
  toasts: readonly Toast[];
  onUndo: (toast: Toast) => void;
  onDismiss: (toast: Toast) => void;
}

export function ToastViewport({ toasts, onUndo, onDismiss }: ToastViewportProps) {
  const visible = toasts.slice(-TOAST_MAX_VISIBLE);
  const hidden = toasts.length - visible.length;

  const announcedRef = useRef<Set<string>>(new Set());
  const [politeMsg, setPoliteMsg] = useState('');
  const [assertiveMsg, setAssertiveMsg] = useState('');

  useEffect(() => {
    let newPolite = '';
    let newAssertive = '';

    toasts.forEach((t) => {
      const key = `${t.id}:${t.kind}`;
      if (announcedRef.current.has(key)) return;
      announcedRef.current.add(key);
      const parts = [t.title, t.tagname, t.titleSuffix].filter(Boolean).join(' ');
      const msg = t.sub ? `${parts}. ${t.sub}` : parts;
      if (t.kind === 'error') newAssertive += `${msg}. `;
      else newPolite += `${msg}. `;
    });

    // Suffix forces re-announce when text is identical.
    if (newAssertive) setAssertiveMsg(`${newAssertive} · ${Date.now()}`);
    if (newPolite) setPoliteMsg(`${newPolite} · ${Date.now()}`);

    // GC announced IDs that no longer have any matching toast.
    const live = new Set(
      toasts.flatMap((t) => [
        `${t.id}:propagating`,
        `${t.id}:done`,
        `${t.id}:error`,
      ]),
    );
    announcedRef.current.forEach((k) => {
      if (!live.has(k)) announcedRef.current.delete(k);
    });
  }, [toasts]);

  return (
    <>
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {politeMsg.replace(/ · \d+$/, '')}
      </div>
      <div className="sr-only" role="alert" aria-live="assertive" aria-atomic="true">
        {assertiveMsg.replace(/ · \d+$/, '')}
      </div>

      <div className="toast-viewport" aria-label="Notifications">
        {hidden > 0 && (
          <button
            type="button"
            className="toast-stack-more"
            onClick={() => toasts.slice(0, hidden).forEach((t) => onDismiss(t))}
            aria-label={`${hidden} older notifications, click to dismiss`}
            title="Dismiss older"
          >
            +{hidden} more
          </button>
        )}
        {visible.map((t) => (
          <ToastCard
            key={t.id}
            toast={t}
            onUndo={onUndo}
            onDismiss={onDismiss}
          />
        ))}
      </div>
    </>
  );
}

interface ToastCardProps {
  toast: Toast;
  onUndo: (toast: Toast) => void;
  onDismiss: (toast: Toast) => void;
}

function ToastCard({ toast, onUndo, onDismiss }: ToastCardProps) {
  const kind = toast.kind;
  return (
    <div className={kind === 'error' ? 'toast error' : 'toast'}>
      <span className={`icon ${kind}`}>
        {kind === 'propagating' && <Icon name="loader-2" size={18} />}
        {kind === 'done' && <Icon name="circle-check" size={18} />}
        {kind === 'error' && <Icon name="alert-triangle" size={18} />}
      </span>
      <div className="body">
        <div className="title">
          {toast.title}
          {toast.tagname && <span className="tagname">{toast.tagname}</span>}
          {toast.titleSuffix && <span>{toast.titleSuffix}</span>}
        </div>
        {toast.sub && <div className="sub">{toast.sub}</div>}
      </div>
      {kind === 'done' && toast.undo !== undefined && (
        <button type="button" className="undo" onClick={() => onUndo(toast)}>
          Undo
        </button>
      )}
      <button
        type="button"
        className="icon-btn"
        onClick={() => onDismiss(toast)}
        aria-label="Dismiss"
      >
        <Icon name="x" size={14} />
      </button>
      {kind === 'done' && toast.undo !== undefined && (
        <span className="undo-progress" />
      )}
    </div>
  );
}
