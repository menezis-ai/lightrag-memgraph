import { resolveKindMeta, type ActivityEvent } from '../../types/activity';
import { Icon } from '../Icon';

export interface ActivityRowProps {
  event: ActivityEvent;
  relativeLabel: string;
  folder: string;
  selected: boolean;
  onClick: () => void;
}

/** Compact timeline projection for one immutable audit event. */
export function ActivityRow({
  event,
  relativeLabel,
  folder,
  selected,
  onClick,
}: Readonly<ActivityRowProps>) {
  const meta = resolveKindMeta(event.kind);
  return (
    <button
      className={`activity-row ${selected ? 'is-selected' : ''} sev-${event.sev}`}
      onClick={onClick}
      aria-current={selected ? 'true' : undefined}
    >
      <span className="row-time">{relativeLabel}</span>
      <span className="row-rail" style={{ background: meta.color }} />
      <span className="row-icon" style={{ color: meta.color }}>
        <Icon name={meta.icon} size={14} />
      </span>
      <span className="row-body">
        <span className="row-line1">
          <span className="row-actor">{event.actor.user}</span>
          <span className="row-kind">{meta.label}</span>
          <span
            className="row-folder"
            title={`Folder: ${folder}`}
            data-testid="activity-row-folder"
          >
            <Icon name="folder" size={10} />
            {folder}
          </span>
          <span className="row-target">{event.target.label}</span>
        </span>
        <span className="row-summary">{event.summary}</span>
      </span>
      {event.sev !== 'info' && (
        <span className={`sev-badge sev-${event.sev}`}>{event.sev}</span>
      )}
    </button>
  );
}
