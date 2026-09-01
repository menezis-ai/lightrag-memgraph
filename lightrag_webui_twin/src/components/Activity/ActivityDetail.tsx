import { useState } from 'react';
import { resolveKindMeta, type ActivityEvent } from '../../types/activity';
import type { Toast } from '../../types/toast';
import { Icon } from '../Icon';

export interface ActivityDetailProps {
  event: ActivityEvent | null;
  relativeLabel: string;
  folder: string;
  onPushToast?: (toast: Omit<Toast, 'id'>) => void;
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
}

/** Inspector and context-sensitive actions for the selected audit event. */
export function ActivityDetail({
  event,
  relativeLabel,
  folder,
  onPushToast,
  onNavigate,
}: Readonly<ActivityDetailProps>) {
  const [copied, setCopied] = useState(false);
  if (!event) {
    return (
      <aside className="activity-detail">
        <div className="empty-state">
          <div className="title">Select an event</div>
        </div>
      </aside>
    );
  }
  const meta = resolveKindMeta(event.kind);
  const copyId = () => {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      void navigator.clipboard.writeText(event.id);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <aside className="activity-detail">
      <div className="detail-head">
        <div className="detail-kind" style={{ color: meta.color }}>
          <Icon name={meta.icon} size={14} color={meta.color} />
          {meta.label}
          {event.sev !== 'info' && (
            <span className={`sev-badge sev-${event.sev}`}>{event.sev}</span>
          )}
        </div>
        <h3>{event.target.label}</h3>
        <div className="detail-summary">{event.summary}</div>
      </div>

      <div className="detail-grid">
        <div className="kv">
          <span>Event ID</span>
          <button type="button" className="copyable" onClick={copyId} title="Copy">
            {event.id} {copied ? '✓' : ''}
          </button>
        </div>
        <div className="kv"><span>Timestamp</span><code>{event.ts}</code></div>
        <div className="kv"><span>Folder</span><code data-testid="activity-detail-folder">{folder}</code></div>
        <div className="kv"><span>Relative</span><span>{relativeLabel}</span></div>
        <div className="kv">
          <span>Actor</span>
          <span>{event.actor.user} <em>({event.actor.role})</em></span>
        </div>
        <div className="kv">
          <span>Target</span>
          <span>{event.target.type} · {event.target.label}</span>
        </div>
        <div className="kv">
          <span>Severity</span>
          <span className={`sev-text sev-${event.sev}`}>{event.sev}</span>
        </div>
      </div>

      <div className="detail-section">
        <div className="detail-section-h">Metadata</div>
        <pre className="detail-meta">{JSON.stringify(event.meta, null, 2)}</pre>
      </div>

      <div className="detail-actions">
        {event.kind === 'source-failed' && (
          <button
            className="primary-btn"
            onClick={() =>
              onPushToast?.({
                kind: 'propagating',
                title: 'Re-processing failed sources',
                sub: `${event.target.label} · POST /documents/reprocess_failed`,
              })
            }
          >
            <Icon name="refresh" size={12} /> Replay ingestion
          </button>
        )}
        {event.target.type === 'source' && (
          <button
            className="ghost-btn"
            onClick={() =>
              onNavigate?.(
                'documents',
                event.target.label ? { q: event.target.label } : undefined,
              )
            }
          >
            <Icon name="arrow-right" size={12} /> Open source
          </button>
        )}
        {event.target.type === 'query' && (
          <button
            className="ghost-btn"
            onClick={() => {
              const params: Record<string, string> = {};
              if (event.target.label) params.q = event.target.label;
              const eventMeta = event.meta as { mode?: string };
              if (eventMeta?.mode) params.mode = eventMeta.mode;
              onNavigate?.('retrieval', params);
            }}
          >
            <Icon name="arrow-right" size={12} /> Re-run query
          </button>
        )}
        <button className="ghost-btn" onClick={copyId}>
          <Icon name="external-link" size={12} /> Copy payload
        </button>
      </div>
    </aside>
  );
}
