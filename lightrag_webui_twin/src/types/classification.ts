/**
 * Microsoft Information Protection (MIP / AIP) classification types.
 *
 * Shape mirrors `ClassificationResult.as_dict()` from the Python module
 * `src/twindb_lightrag_memgraph/classification.py` (PR #157). The pre-insert
 * hook writes this exact payload into `DocStatus.metadata.classification`
 * during ingestion; the WebUI reads it back to:
 *
 *   - render a small pill on DocumentsTab rows + PendingDocsSection cards
 *     (`<ClassPill cls={doc.metadata.classification} />`)
 *   - gate the chunks tab and the "View raw" notice in `DocDetailPanel`
 *
 * Important: a document may carry either:
 *   - a STRING (legacy shape: `"internal" | "restricted"`), OR
 *   - a structured `ClassificationResult` (new MIP-extracted shape).
 *
 * The `isStructured()` / `getClassId()` helpers below let consumers handle
 * both transparently — they're the canonical accessors. Don't dereference
 * `doc.metadata.classification` directly.
 */

/**
 * Tenant-mapped class identifier.
 *
 * New MIP mappings use the business names below. The legacy C1-C4 ladder is
 * still accepted because older fixtures and persisted documents may carry it:
 * C1=Public, C2=Internal, C3=Confidential, C4=Secret.
 * `UNKNOWN` is returned when the MIP label has no mapping in the tenant
 * table — fail-closed, downstream treats it as above any ceiling.
 */
export type ClassId =
  | 'Public'
  | 'Internal'
  | 'Confidential'
  | 'Secret'
  | 'Private'
  | 'C1'
  | 'C2'
  | 'C3'
  | 'C4'
  | 'UNKNOWN';

/** Format the source file was detected in. */
export type ClassificationSource = 'ooxml' | 'ole' | 'pdf' | 'email' | 'unknown';

/**
 * Reason emitted when `class_id` resolution failed. `null` means a clean
 * detection. The full enumeration is documented in the Python module's
 * `ClassificationResult` docstring.
 */
export type ClassificationReason =
  | null
  | 'no-custom-props'
  | 'no-msip-label'
  | 'unknown-label-guid'
  | 'unsupported-extension'
  | 'olefile-missing'
  | 'pikepdf-missing'
  | (string & {});

/**
 * Full MIP classification payload, matches Python `ClassificationResult.as_dict()`.
 */
export interface ClassificationResult {
  class_id: ClassId | null;
  /** Human-readable label name (e.g. "C2 Confidentiel"). */
  class_name: string | null;
  /** Normalized lowercase braceless GUID. */
  label_guid: string | null;
  /** Raw `MSIP_Label_<GUID>_Name` value as written by the producer. */
  raw_name: string | null;
  /** ISO timestamp the label was applied. */
  set_date: string | null;
  /** "Standard" | "Privileged" per Microsoft's spec. */
  method: string | null;
  source_format: ClassificationSource;
  reason: ClassificationReason;
  /** Free-form per-label MSIP fields kept for audit trace. */
  meta: Record<string, unknown>;
}

/**
 * The shape Twin actually persists in `DocStatus.metadata.classification`.
 * Includes the legacy string for back-compat with un-ingested seeds.
 */
export type ClassificationValue = string | ClassificationResult | undefined;

/**
 * Discriminator: returns true when the value is the new structured payload
 * (vs the legacy string).
 */
export function isStructured(
  cls: ClassificationValue,
): cls is ClassificationResult {
  return typeof cls === 'object' && cls !== null && 'class_id' in cls;
}

/**
 * Resolve a single class identifier from either shape, defaulting to
 * `'UNCLASSIFIED'` when no classification is present. Use this when you need
 * a single string for display or comparison.
 */
export function getClassId(cls: ClassificationValue): string {
  if (cls === undefined || cls === null) return 'UNCLASSIFIED';
  if (typeof cls === 'string') return cls;
  return cls.class_id ?? 'UNCLASSIFIED';
}

/**
 * Resolve a human-readable display name. Falls back to the class id when
 * no friendlier label is available.
 */
export function getClassName(cls: ClassificationValue): string {
  if (cls === undefined || cls === null) return 'unclassified';
  if (typeof cls === 'string') return cls;
  return cls.class_name ?? cls.raw_name ?? cls.class_id ?? 'unclassified';
}

export type MipTone =
  | 'public'
  | 'internal'
  | 'confidential'
  | 'secret'
  | 'private'
  | 'unknown'
  | 'unclassified';

const CLASS_ORDER: Record<string, number> = {
  public: 0,
  internal: 1,
  private: 2,
  confidential: 3,
  secret: 4,
};

const LEGACY_TO_TONE: Record<string, MipTone> = {
  c1: 'public',
  c2: 'internal',
  c3: 'confidential',
  c4: 'secret',
};

const MIP_LABELS: Record<MipTone, string> = {
  public: 'Public',
  internal: 'Internal',
  confidential: 'Confidential',
  secret: 'Secret',
  private: 'Private',
  unknown: 'Unknown',
  unclassified: 'Unclassified',
};

export function getMipTone(classId: string | null | undefined): MipTone {
  if (classId === undefined || classId === null || classId === '') {
    return 'unclassified';
  }
  const normalized = String(classId).trim().toLowerCase();
  return LEGACY_TO_TONE[normalized] ?? (
    normalized in MIP_LABELS ? (normalized as MipTone) : 'unknown'
  );
}

export function getMipDisplayName(classId: string | null | undefined): string {
  return MIP_LABELS[getMipTone(classId)];
}

/**
 * Returns true when `class_id` strictly outranks `threshold` on the tenant
 * ladder. Unknown / non-ladder ids are treated as ABOVE — fail-closed.
 * Missing ids are not a MIP classification and are treated as not above.
 * Mirrors the Python `is_above()` semantics in `classification.py`.
 */
export function isAbove(classId: string | null | undefined, threshold: string): boolean {
  const classTone = getMipTone(classId);
  const thresholdTone = getMipTone(threshold);
  if (classTone === 'unclassified') return false;
  const classRank = CLASS_ORDER[classTone];
  const thresholdRank = CLASS_ORDER[thresholdTone];
  if (classRank === undefined) return true;
  if (thresholdRank === undefined) {
    throw new Error(`threshold "${threshold}" not in MIP classification ladder`);
  }
  return classRank > thresholdRank;
}

/**
 * Convenience: "is this above what an internal-cleared operator can see?"
 * Aligns with the WebUI gate (`metadata.classification.class_id > 'C2'`).
 *
 * Accepts the legacy string shape too — `'internal'`/`'public'` are below
 * the gate; anything else (incl. `'restricted'`, `'confidential'`) is above.
 */
export function isAboveInternal(cls: ClassificationValue): boolean {
  if (isStructured(cls)) {
    return isAbove(cls.class_id, 'Internal');
  }
  if (typeof cls === 'string') {
    const c = cls.toLowerCase();
    return c !== 'internal' && c !== 'public';
  }
  return false;
}
