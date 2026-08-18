/**
 * ClassPill — sensitivity shield rendered next to a document's name.
 *
 * Product decision 2026-08-04 (QA DOC-V4-001): every document shows its
 * confidentiality level, the two-level public/interne model included. The
 * pill therefore renders for:
 *   1. a structured MIP `ClassificationResult` (any mapped class);
 *   2. a legacy string classification (`"internal"`, `"public"`, …);
 *   3. the document's `visibility` field, as fallback when no
 *      classification was extracted.
 * It stays silent only when none of the three carries a level.
 */

import {
  getClassId,
  getClassName,
  getMipDisplayName,
  getMipTone,
  isStructured,
  type ClassificationValue,
  type MipTone,
} from '../types/classification';
import { Icon } from './Icon';

export interface ClassPillProps {
  cls: ClassificationValue;
  /** Two-level fallback (document visibility) when no classification was
   *  extracted. Lets un-labelled docs still surface public/interne. */
  visibility?: string;
  /** Optional id used to scope data-testid for stable test selection. */
  docId?: string;
}

interface PillSource {
  id: string;
  tone: MipTone;
  /** Short text inside the pill. */
  displayName: string;
  /** Long text for tooltip / aria. */
  label: string;
  setDate: string;
}

function fromLevelString(raw: string): PillSource | null {
  const id = raw.trim();
  if (!id) return null;
  const tone = getMipTone(id);
  // A ladder level renders its canonical name; anything else (e.g. legacy
  // "restricted") keeps the raw string so the operator sees what is stored.
  const displayName = tone === 'unknown' ? id : getMipDisplayName(id);
  return { id, tone, displayName, label: displayName, setDate: '' };
}

function resolvePillSource(
  cls: ClassificationValue,
  visibility: string | undefined,
): PillSource | null {
  if (isStructured(cls)) {
    const id = getClassId(cls);
    const tone = getMipTone(id);
    if (tone !== 'unclassified') {
      const displayName = getMipDisplayName(id);
      const rawLabel = getClassName(cls);
      const label =
        rawLabel && rawLabel !== id ? `${displayName} · ${rawLabel}` : displayName;
      const setDate = cls.set_date
        ? ` · applied ${cls.set_date.slice(0, 10)}`
        : '';
      return { id, tone, displayName, label, setDate };
    }
    // class_id null = nothing extracted — fall through to visibility.
  } else if (typeof cls === 'string') {
    const source = fromLevelString(cls);
    if (source) return source;
  }
  return visibility ? fromLevelString(visibility) : null;
}

export function ClassPill({ cls, visibility, docId }: Readonly<ClassPillProps>) {
  const source = resolvePillSource(cls, visibility);
  if (!source) return null;

  const { id, tone, displayName, label, setDate } = source;
  const klass = `class-pill class-${tone}`;

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
