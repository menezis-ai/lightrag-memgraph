import type { ActivityEvent, ActivityRange } from '../../types/activity';

function escapeCsv(value: unknown): string {
  if (value === null || value === undefined) return '';
  let serialized: string;
  if (typeof value === 'string') serialized = value;
  else if (typeof value === 'number' || typeof value === 'boolean') {
    serialized = value.toString();
  } else {
    serialized = JSON.stringify(value) ?? '';
  }
  return /[",\n]/.test(serialized)
    ? `"${serialized.replaceAll('"', '""')}"`
    : serialized;
}

/** Flatten an ActivityEvent list to a CSV blob and trigger a download. */
export function exportActivityCsv(
  rows: readonly ActivityEvent[],
  range: ActivityRange,
): void {
  const columns = [
    'id',
    'ts',
    'kind',
    'sev',
    'actor',
    'role',
    'target_type',
    'target_label',
    'summary',
    'meta',
  ];
  const lines = [columns.join(',')];
  for (const event of rows) {
    lines.push(
      [
        event.id,
        event.ts,
        event.kind,
        event.sev,
        event.actor.user,
        event.actor.role,
        event.target.type,
        event.target.label,
        event.summary,
        event.meta,
      ]
        .map(escapeCsv)
        .join(','),
    );
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  const stamp = new Date().toISOString().slice(0, 10);
  anchor.href = url;
  anchor.download = `twin-rag-activity-${range}-${stamp}.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  setTimeout(() => {
    URL.revokeObjectURL(url);
    anchor.remove();
  }, 0);
}
