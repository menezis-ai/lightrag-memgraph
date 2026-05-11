/**
 * Document-related types.
 *
 * These mirror the shape used by the proto's MOCK_DOCUMENTS and double as the
 * spec the backend phase-1 endpoints will need to honor:
 *   GET    /documents              -> Document[]
 *   POST   /documents              -> Document
 *   PATCH  /documents/{id}/tags    -> Document   (delta-style)
 *   DELETE /documents/{id}/tags    -> Document
 *
 * Keep this in sync with the backend's pydantic models when phase 1 lands.
 */

import type { WorkspaceVisibility } from './topbar';

export type DocumentType = 'file' | 'confluence' | 'sharepoint' | 'url';

export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Document {
  id: string;
  type: DocumentType;
  /** File name, URL, or path identifier of the source. */
  source: string;
  /** Short human-readable summary of the content. */
  summary: string;
  tags: string[];
  status: DocumentStatus;
  /** Number of chunks the document was split into. */
  chunks: number;
  /** Human-readable relative timestamp (e.g. "2h ago"). */
  updated: string;
  visibility: WorkspaceVisibility;
  /** Workspace id this document belongs to. */
  workspace: string;
}
