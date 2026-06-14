/**
 * Document-related types.
 *
 * Aligned on LightRAG `DocStatus` (lightrag/base.py, lightrag/api/routers/document.py)
 * since the sprint Étape 0 (2026-05-29). Twin overlay adds `tags`, `folder`,
 * `review` on top of the native LightRAG shape, served by
 * `/twin/api/documents/{id}/metadata`.
 *
 * Routes:
 *   GET    /documents                                        -> { items: Document[], total }
 *   POST   /documents                                        -> upload (LightRAG native)
 *   GET    /documents/{id}/chunks                            -> chunks[]
 *   POST   /documents/{id}/scan                              -> ack-only compat no-op (audit C7;
 *                                                                no UI consumer — the WebUI
 *                                                                routes all per-doc re-process
 *                                                                intent through
 *                                                                /documents/reprocess_failed)
 *   GET    /twin/api/documents/{id}/metadata                 -> overlay fields
 *   POST   /twin/api/documents/{id}/approve                  -> review approve
 *   POST   /twin/api/documents/{id}/reject                   -> review reject
 *   PATCH  /twin/api/documents/{id}/tags                     -> overlay tags
 */

import type { FolderVisibility } from './topbar';

export type DocumentType = 'file' | 'confluence' | 'sharepoint' | 'url';

/** LightRAG-native status enum (uppercase, mirrors DocStatus.status). */
export type DocumentStatus = 'PENDING' | 'PROCESSING' | 'PROCESSED' | 'FAILED';

/**
 * UI-only display status. Strict superset of DocumentStatus that adds
 * `DELETING` for the optimistic state between a bulk-delete call and
 * the server-side cascade completing. Never travels over the wire.
 */
export type DocumentDisplayStatus = DocumentStatus | 'DELETING';

export type ReviewState = 'pending-review' | 'approved' | 'rejected' | 'modified';

/**
 * Payload describing an upstream-change event for a live source (Confluence /
 * SharePoint) that requires re-validation. Present when `review.state === 'modified'`.
 * Drives the "Modified source" pending card variant.
 */
export interface DocumentReviewUpdate {
  /** Author of the upstream edit. */
  requested_by: string;
  /** Human-readable relative time of the edit ("2h ago"). */
  edited_rel: string;
  /** ISO date the change was detected by Twin. */
  detected_at: string;
  /** Chunks currently indexed (pre-revalidation). */
  chunks_indexed: number;
  /** One-paragraph diff description (LLM-generated or rule-based). */
  summary_diff: string;
}

export interface DocumentReview {
  state: ReviewState;
  requested_by?: string;
  requested_at?: string;
  justification?: string;
  /** Set when state === 'modified' — describes the upstream change. */
  update?: DocumentReviewUpdate;
}

/**
 * LightRAG-aligned Document shape.
 *
 * snake_case field names match the LightRAG API JSON exactly so there is no
 * serialization mapping at the boundary. `type` and `visibility` are Twin
 * overlay UI hints, not present on LightRAG's native DocStatus.
 */
export interface Document {
  /** LightRAG DocStatus primary key. */
  doc_id: string;
  /** Upload batch correlation id; null for legacy or single uploads. */
  track_id: string | null;
  /** Original source identifier (file name, URL, Confluence path...). */
  file_path: string;
  /** Indexed text preview populated by LightRAG ingestion; not an abstractive summary. */
  content_summary: string;
  /** Total content length in chars. */
  content_length: number;
  status: DocumentStatus;
  /**
   * UI-only optimistic flag set by the bulk-delete mutation between
   * the request and the server-side cascade completing. Never
   * serialized back to the API. When true the row renders the
   * DELETING badge and the user keeps seeing the doc until the
   * subsequent refetch removes it.
   */
  _deleting?: boolean;
  /**
   * UI-only row inserted after an upload request so operators see a
   * stable pending source before LightRAG exposes the generated doc_id.
   */
  _optimisticUpload?: boolean;
  /** Final chunk count; null while still PROCESSING. */
  chunks_count: number | null;
  created_at: string;
  updated_at: string;
  error_msg: string | null;
  metadata: Record<string, unknown>;

  // --- Twin overlay (served by /twin/api/documents/{id}/metadata) ---
  /** UI-only hint for source icon; derived from file_path/metadata. */
  type: DocumentType;
  tags: string[];
  folder: string;
  visibility: FolderVisibility;
  review?: DocumentReview;
  /**
   * Post-extraction text used by LightRAG for retrieval (NOT the original
   * binary). Populated lazily and exposed via the Read source modal so a
   * reviewer can audit what the indexer ingested before approving. Returned
   * by `GET /twin/api/documents/{id}/extracted-text`; absent in list views.
   */
  extracted_text?: string;
}
