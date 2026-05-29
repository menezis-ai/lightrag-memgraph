/**
 * ISO timestamp → human-readable relative duration ("2h ago", "5d ago").
 *
 * Used in tabs where the proto rendered hand-written strings like "2h ago".
 * Since the LightRAG-aligned Document carries `updated_at` as ISO 8601, we
 * format at the edge instead of polluting the data model with display strings.
 *
 * Pass `now` explicitly in tests so the rendered output is deterministic.
 */

const MS = {
  second: 1000,
  minute: 60_000,
  hour: 3_600_000,
  day: 86_400_000,
};

export function relativeTime(iso: string, now: number = Date.now()): string {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return iso;
  const delta = Math.max(0, now - then);
  if (delta < MS.minute) return 'just now';
  if (delta < MS.hour) return `${Math.floor(delta / MS.minute)}m ago`;
  if (delta < MS.day) return `${Math.floor(delta / MS.hour)}h ago`;
  return `${Math.floor(delta / MS.day)}d ago`;
}
