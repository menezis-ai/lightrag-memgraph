/**
 * ClassPill — MIP sensitivity shield rendered next to a document's name.
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
  getMipDisplayName,
  getMipTone,
  isStructured,
  type ClassificationValue,
} from '../types/classification';
import { Icon } from './Icon';

export interface ClassPillProps {
  cls: ClassificationValue;
  /** Optional id used to scope data-testid for stable test selection. */
  docId?: string;
}

export function ClassPill({ cls, docId }: Readonly<ClassPillProps>) {
  // Render only when we have a structured classification payload. Legacy
  // string-based classifications stay invisible — the DocDetailPanel's
  // "View raw" notice still surfaces them when needed.
  if (!isStructured(cls)) return null;

  const id = getClassId(cls);
  const tone = getMipTone(id);
  if (tone === 'unclassified') return null;
  const displayName = getMipDisplayName(id);
  const rawLabel = getClassName(cls);
  const label = rawLabel && rawLabel !== id ? `${displayName} · ${rawLabel}` : displayName;
  const klass = `class-pill class-${tone}`;
  const setDate = cls.set_date ? ` · applied ${cls.set_date.slice(0, 10)}` : '';

  return (
    <button
      type="button"
      className={klass}
      title={`${label}${setDate}`}
      aria-label={`Classification: ${label}`}
      data-testid={docId ? `class-pill-${docId}` : 'class-pill'}
      data-class-id={id}
      data-class-tone={tone}
      data-tooltip={`${label}${setDate}`}
    >
      <Icon name="shield" size={15} strokeWidth={1.8} />
      <span className="class-pill-label">{displayName}</span>
    </button>
  );
}
