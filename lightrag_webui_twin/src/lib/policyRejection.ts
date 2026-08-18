/**
 * Policy-rejection detection — distinguishes a deterministic ingestion
 * VERDICT from a transient indexing FAILURE.
 *
 * Why: both land as status FAILED, but they call for opposite operator
 * actions. A policy rejection (image with no usable content, document
 * over the classification ceiling) will produce the same outcome on
 * every retry — the observed anti-pattern is an operator hammering
 * "retry failed" on a rejected logo (OVH maquette, 2026-07-27). A
 * transient failure (vision endpoint down, timeout) is legitimately
 * retryable and must keep the plain failure affordances.
 *
 * The anchors below are Twin-owned fixed strings, not LightRAG's:
 * - `Image ingestion refused` — content_summary set by
 *   `patches/registry.py::_report_error_document` for every image
 *   refusal; `error_msg` then carries the reason whose prefix separates
 *   policy verdicts from environment errors (`_vision.py`).
 * - `[content withheld: classification` — content_summary set by
 *   `_classification_hook.py::_failed_status_for_rejection` (PIPE-6b
 *   redaction) for MIP over-classification rejections.
 * If you reword either string backend-side, update this module and its
 * test in the same commit.
 */

import type { Document } from '../types/document';

export type PolicyRejectionKind = 'vision' | 'classification';

const VISION_REFUSED_SUMMARY = 'Image ingestion refused';
const CLASSIFICATION_SUMMARY_PREFIX = '[content withheld: classification';

/** Deterministic verdict prefixes from `_vision.py`. Environment failures
 * (`vision-timeout:`, `vision-error:`, `vision-llm-error:`,
 * `vision-input-error:`) are deliberately absent: those are retryable. */
const POLICY_REASON_PREFIXES = [
  'vision-prefilter:',
  'vision-size-limit:',
  'image-dropped:',
] as const;

type RejectionFields = Pick<Document, 'status' | 'content_summary' | 'error_msg'>;

export function policyRejectionKind(
  doc: RejectionFields,
): PolicyRejectionKind | null {
  if (doc.status !== 'FAILED') return null;
  const summary = (doc.content_summary ?? '').trim();
  if (summary.startsWith(CLASSIFICATION_SUMMARY_PREFIX)) return 'classification';
  if (summary === VISION_REFUSED_SUMMARY) {
    const reason = (doc.error_msg ?? '').trim();
    if (POLICY_REASON_PREFIXES.some((p) => reason.startsWith(p))) {
      return 'vision';
    }
  }
  return null;
}

export function isPolicyRejected(doc: RejectionFields): boolean {
  return policyRejectionKind(doc) !== null;
}

export function policyRejectionGuidance(kind: PolicyRejectionKind): string {
  if (kind === 'classification') {
    return (
      'Rejected by the classification policy: its sensitivity label exceeds ' +
      'the allowed ceiling. Retrying will not change the verdict — delete ' +
      'this document, or upload a compliant version.'
    );
  }
  return (
    'Rejected by the ingestion policy: this image carries no usable content ' +
    'for the knowledge base. Retrying will not change the verdict — delete ' +
    'this document, or upload a more informative file.'
  );
}
