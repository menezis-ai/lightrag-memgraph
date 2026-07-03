/**
 * Canonical document-status vocabulary — TypeScript twin of the Python
 * `src/twindb_lightrag_memgraph/server/status_vocab.py`.
 *
 * Audit 2026-07-02 (`docs/audits/ingestion-reindex/audit-2026-07-02.md`,
 * finding DUP-1): the status vocabulary was spelled in three places on the
 * wire (native shim lowercase, twin route UPPERCASE, seed legacy
 * `completed`/`processing`/`failed`) and the dual-reads that papered over the
 * drift leaked into prod code (`DocumentsTab.statusCountsFor`,
 * `resources.normalizeDocumentStatus`). This module centralizes those reads.
 *
 * ZERO wire-contract change: no field is renamed (DUP-9 — a contract rename
 * touches 36+ files), no casing is altered. The helpers below reproduce the
 * exact historical behaviour of the call sites they replaced.
 */

import type { DocumentStatus } from '../types/document';

/** The four statuses every wire surface understood historically (UPPERCASE
 *  UI casing — `DocumentStatus` in `types/document.ts` stays the type-level
 *  source of truth; this is its runtime mirror). */
export const DOC_STATUSES = [
  'PENDING',
  'PROCESSING',
  'PROCESSED',
  'FAILED',
] as const satisfies readonly DocumentStatus[];

const ALLOWED_DOC_STATUS: ReadonlySet<string> = new Set(DOC_STATUSES);

/**
 * LightRAG 1.5.x pipeline states (`lightrag/base.py` on 1.5.4:
 * PENDING → PARSING → ANALYZING → PROCESSING → PROCESSED|FAILED, plus
 * PREPROCESSED). Never emitted by the pinned BNP runtime (1.4.9.11).
 *
 * Documented coercion: they normalize to `'PENDING'` — the same fallback the
 * Python `MemgraphDocStatusStorage._deserialize_status` applies to unknown
 * statuses. Deliberately NOT added to `DOC_STATUSES`: widening the UI enum
 * is a contract change (see audit PIPE-13 before revisiting).
 */
export const LIGHTRAG_15X_STATUSES = [
  'PARSING',
  'ANALYZING',
  'PREPROCESSED',
] as const;

/**
 * Normalize a wire status to the canonical UPPERCASE `DocumentStatus`.
 *
 * LightRAG's `DocStatus.value` is lowercase (`'pending'`, `'processing'`,
 * `'processed'`, `'failed'`) while the UI type and every consumer expect
 * uppercase. Normalizing at ingress keeps mapping/counters working whichever
 * end of the contract shifts. Unknown values — including the 1.5.x statuses
 * above — fall back to `'PENDING'` (same as the Python
 * `MemgraphDocStatusStorage._deserialize_status`).
 *
 * (Moved verbatim from `api/resources.ts`; behaviour unchanged.)
 */
export function normalizeDocumentStatus(raw: unknown): DocumentStatus {
  const s = typeof raw === 'string' ? raw.toUpperCase() : '';
  return (ALLOWED_DOC_STATUS.has(s) ? s : 'PENDING') as DocumentStatus;
}

/**
 * Dual-cased read of a `status_counts` bucket.
 *
 * The native shim emits lowercase keys, the twin route UPPERCASE; consumers
 * must read both until the backends unify (audit DUP-1 residual). Lowercase
 * wins when both are present — exact historical order of the inline
 * `statusCounts.processed ?? statusCounts.PROCESSED ?? 0` reads this
 * replaces (`DocumentsTab.statusCountsFor`).
 */
export function statusCountFor(
  counts: Record<string, number>,
  status: DocumentStatus,
): number {
  return counts[status.toLowerCase()] ?? counts[status] ?? 0;
}

/**
 * Terminal-state check for `GET /documents/track_status/{id}` rows.
 *
 * Dual-cased on purpose: LightRAG's native track_status emits lowercase
 * `DocStatus.value` strings while Twin surfaces speak UPPERCASE (audit
 * DUP-1). Exact-set semantics moved verbatim from
 * `useDocumentActions.TERMINAL_TRACK_STATUSES` — only these four spellings
 * match (behaviour-identical migration; mixed case is NOT accepted).
 */
const TERMINAL_TRACK_STATUSES: ReadonlySet<string> = new Set([
  'processed',
  'PROCESSED',
  'failed',
  'FAILED',
]);

export function isTerminalTrackStatus(status: string): boolean {
  return TERMINAL_TRACK_STATUSES.has(status);
}

/**
 * "Did this track row finish successfully?" — the historical
 * `doc.status.toLowerCase() === 'processed'` comparison from
 * `useDocumentActions.processedDocIdsIfTerminal`, centralized.
 */
export function isProcessedTrackStatus(status: string): boolean {
  return status.toLowerCase() === 'processed';
}
