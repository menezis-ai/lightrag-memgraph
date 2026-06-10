/**
 * ClassPill — sensitivity-class badge rendered next to a document's name.
 *
 * Visual contract (CSS lives in `polish.css`):
 *   span.class-pill.class-{c1|c2|c3|c4|unknown}  · text = class_id
 *
 * Tonal scale — restrained for C1 (routine), amber for C2 (frequent in a
 * bank), red for C3, alarm-red for C4. UNKNOWN uses a striped amber pattern
 * to signal "needs reviewer attention" without escalating to red.
 *
 * The pill is SILENT when the document carries no `classification` field
 * (or the legacy string `"internal"` / `"public"` — those don't deserve a
 * visual marker on every row). The Python pre-insert hook writes the
 * structured `ClassificationResult` shape; legacy seed strings render
 * nothing here.
 */

import {
  getClassId,
  getClassName,
  isStructured,
  type ClassificationValue,
} from '../types/classification';

export interface ClassPillProps {
  cls: ClassificationValue;
  /** Optional id used to scope data-testid for stable test selection. */
  docId?: string;
}

export function ClassPill({ cls, docId }: ClassPillProps) {
  // Render only when we have a structured classification payload. Legacy
  // string-based classifications (the maquette baseline) stay invisible —
  // the DocDetailPanel's "View raw" notice still surfaces them when needed.
  if (!isStructured(cls)) return null;

  const id = getClassId(cls);
  const label = getClassName(cls);
  const klass = `class-pill class-${String(id).toLowerCase()}`;
  const setDate = cls.set_date ? ` · applied ${cls.set_date.slice(0, 10)}` : '';

  return (
    <span
      className={klass}
      title={`${label}${setDate}`}
      aria-label={`Classification: ${label}`}
      data-testid={docId ? `class-pill-${docId}` : 'class-pill'}
      data-class-id={id}
    >
      {id}
    </span>
  );
}
