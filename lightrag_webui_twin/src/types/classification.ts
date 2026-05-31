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
 *   - a STRING (legacy maquette shape: `"internal" | "restricted"`), OR
 *   - a structured `ClassificationResult` (new MIP-extracted shape).
 *
 * The `isStructured()` / `getClassId()` helpers below let consumers handle
 * both transparently — they're the canonical accessors. Don't dereference
 * `doc.metadata.classification` directly.
 */

/**
 * Tenant-mapped class identifier (BNP ladder = C1 / C2 / C3 / C4).
 * `UNKNOWN` is returned when the MIP label has no mapping in the tenant
 * table — fail-closed, downstream treats it as above any ceiling.
 */
export type ClassId = 'C1' | 'C2' | 'C3' | 'C4' | 'UNKNOWN';

/** Format the source file was detected in. */
export type ClassificationSource = 'ooxml' | 'ole' | 'pdf' | 'unknown';

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
  class_id: ClassId;
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
 * Includes the legacy maquette string for back-compat with un-ingested seeds.
 */
export type ClassificationValue = string | ClassificationResult | undefined;

/**
 * Discriminator: returns true when the value is the new structured payload
 * (vs the legacy maquette string).
 */
export function isStructured(
  cls: ClassificationValue,
): cls is ClassificationResult {
  return typeof cls === 'object' && cls !== null && 'class_id' in cls;
}

/**
 * Resolve a single class identifier from either shape, defaulting to
 * `'UNKNOWN'` when no classification is present. Use this when you need
 * a single string for display or comparison.
 */
export function getClassId(cls: ClassificationValue): ClassId | string {
  if (cls === undefined || cls === null) return 'UNKNOWN';
  if (typeof cls === 'string') return cls;
  return cls.class_id;
}

/**
 * Resolve a human-readable display name. Falls back to the class id when
 * no friendlier label is available.
 */
export function getClassName(cls: ClassificationValue): string {
  if (cls === undefined || cls === null) return 'unclassified';
  if (typeof cls === 'string') return cls;
  return cls.class_name ?? cls.raw_name ?? cls.class_id;
}

const BNP_LADDER: readonly string[] = ['C1', 'C2', 'C3', 'C4'];

/**
 * Returns true when `class_id` strictly outranks `threshold` on the BNP
 * ladder. Unknown / non-ladder ids are treated as ABOVE — fail-closed.
 * Mirrors the Python `is_above()` semantics in `classification.py`.
 */
export function isAbove(classId: string, threshold: string): boolean {
  const ci = BNP_LADDER.indexOf(classId);
  const ti = BNP_LADDER.indexOf(threshold);
  if (ci === -1) return true;
  if (ti === -1) {
    throw new Error(`threshold "${threshold}" not in BNP_LADDER`);
  }
  return ci > ti;
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
    return isAbove(cls.class_id, 'C2');
  }
  if (typeof cls === 'string') {
    const c = cls.toLowerCase();
    return c !== 'internal' && c !== 'public';
  }
  // Missing classification → treat as above (fail-closed).
  return true;
}
