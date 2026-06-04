/**
 * Typed per-resource fetchers — one function per backend endpoint.
 *
 * Two surfaces:
 *   - `lightragApi` : native LightRAG endpoints (no path prefix beyond the
 *     LightRAG mount root). These hit endpoints LightRAG already ships:
 *     /documents, /documents/{id}/chunks, /documents/{id}/scan, /query,
 *     /health, /pipeline_status, /openapi.
 *   - `twinApi` : Twin overlay endpoints, served by our FastAPI sub-app
 *     mounted via `register(mount_server=True)`. All paths share the
 *     `/twin/api/` prefix: /tags, /audit-events, /activity, /notifications,
 *     /workspaces, /graph/*, /auth/logout, /documents/{id}/metadata,
 *     /documents/{id}/approve, /documents/{id}/reject.
 *
 * The single `api` export aggregates both for convenience; queries.ts hooks
 * stay on `api.xxx()` without caring about which surface owns the endpoint.
 */

import {
  ApiError,
  apiFetch,
  buildApiHeaders,
  buildApiUrl,
  type ApiRequestInit,
} from './client';
import type { ActivityEvent } from '../types/activity';
import type { OpenApiGroup } from '../types/api';
import type { Document } from '../types/document';
import type {
  GraphEntity,
  GraphEntityPatch,
  GraphRelation,
  GraphRelationPatch,
} from '../types/graph';
import type { Notification, Workspace } from '../types/topbar';
import type { TagCategory, TagEntry } from '../types/tag';
import type { ThesaurusEntry } from '../types/thesaurus';

const TWIN = '/twin/api';

export interface ListEnvelope<T> {
  items: readonly T[];
  total: number;
}

export interface DocumentsQuery {
  status?: string;
  q?: string;
  tag?: string;
  workspace?: string;
  cursor?: string;
}

export interface DocumentChunk {
  chunk_id: string;
  order: number;
  text: string;
  /** Truncated preview when classification > internal (Louis compliance). */
  redacted?: boolean;
}

// ============================================================================
// LightRAG-native endpoints (NO /twin/api prefix)
// ============================================================================

export const lightragApi = {
  listDocuments: (q: DocumentsQuery = {}, init?: ApiRequestInit) =>
    apiFetch<ListEnvelope<Document>>('/documents', { ...init, query: { ...q } }),
  listDocumentChunks: (docId: string, init?: ApiRequestInit) =>
    apiFetch<readonly DocumentChunk[]>(
      `/documents/${encodeURIComponent(docId)}/chunks`,
      init,
    ),
  /**
   * Resolve all DocStatus rows associated with an ingestion track_id
   * (the id returned by /documents/upload). Used to discover the
   * generated doc_id once processing completes, so initial tags from
   * the AddSource modal can be applied via bulk-retag.
   */
  trackStatus: (trackId: string, init?: ApiRequestInit) =>
    apiFetch<{
      track_id: string;
      documents: readonly {
        id: string;
        status: string;
        file_path: string;
      }[];
      total_count: number;
      status_summary: Record<string, number>;
    }>(`/documents/track_status/${encodeURIComponent(trackId)}`, init),
  scanDocument: (docId: string, init?: ApiRequestInit) =>
    apiFetch<{ ok: true }>(`/documents/${encodeURIComponent(docId)}/scan`, {
      ...init,
      method: 'POST',
    }),
  /**
   * Trigger re-processing of all FAILED docs in the workspace via
   * LightRAG's native batch endpoint. There is no per-doc-by-id
   * reprocess in LightRAG 1.4.9.11; the DocDetailPanel "Re-process"
   * button surfaces this to operators as "retry failed batch" when
   * the targeted doc is FAILED, and as a clear no-op explanation
   * otherwise (see App.tsx onReprocess).
   */
  reprocessFailedDocuments: (init?: ApiRequestInit) =>
    apiFetch<{ status: string; message?: string; failed_count?: number }>(
      `/documents/reprocess_failed`,
      { ...init, method: 'POST' },
    ),
  /**
   * Upload one file to LightRAG native /documents/upload (multipart).
   *
   * apiFetch is JSON-only, so this bypass uses fetch directly. The
   * URL, auth, cookie, and workspace header pattern matches apiFetch.
   * Returns the InsertResponse shape:
   *   { status: 'success'|'duplicated', message: string, track_id: string }
   * On 4xx/5xx, throws an ApiError so the host can show a real toast.
   */
  uploadDocument: async (
    file: File,
    init?: { signal?: AbortSignal },
  ): Promise<{ status: string; message: string; track_id: string }> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(buildApiUrl('/documents/upload'), {
      method: 'POST',
      headers: buildApiHeaders(),
      body: formData,
      signal: init?.signal,
      credentials: 'include',
    });
    const text = await res.text();
    let body: unknown = text;
    try {
      body = JSON.parse(text);
    } catch {
      /* keep as text */
    }
    if (!res.ok) {
      throw new ApiError(
        `POST /documents/upload → ${res.status} ${res.statusText}`,
        res.status,
        body,
      );
    }
    return body as { status: string; message: string; track_id: string };
  },
  deleteDocument: (docId: string, init?: ApiRequestInit) =>
    apiFetch<{ ok: true }>(`/documents/${encodeURIComponent(docId)}`, {
      ...init,
      method: 'DELETE',
    }),
  health: (init?: ApiRequestInit) =>
    apiFetch<{ status: 'ok' | 'degraded' | 'down'; version?: string }>(
      '/health',
      init,
    ),
  pipelineStatus: (init?: ApiRequestInit) =>
    apiFetch<{
      busy: boolean;
      job_count: number;
      latest_message: string | null;
    }>('/pipeline_status', init),
  getOpenApi: (init?: ApiRequestInit) =>
    apiFetch<{ groups: readonly OpenApiGroup[]; version: string }>(
      '/openapi',
      init,
    ),
  /**
   * Issue a retrieval query against LightRAG native POST /query.
   * Returns the synthesized response string. Sources are not exposed by
   * the native endpoint in this version — callers should display the
   * response text and treat sources as empty until streaming + structured
   * context land. The endpoint accepts the standard LightRAG query body
   * (`query`, `mode`, `top_k`, `max_total_tokens`, optional `only_need_context`,
   * `only_need_prompt`).
   */
  query: (
    body: {
      query: string;
      mode?: string;
      top_k?: number;
      max_total_tokens?: number;
      only_need_context?: boolean;
      only_need_prompt?: boolean;
    },
    init?: ApiRequestInit,
  ) =>
    apiFetch<{ response: string }>('/query', {
      ...init,
      method: 'POST',
      body,
    }),
};

// ============================================================================
// Twin overlay endpoints (/twin/api/* prefix)
// ============================================================================

export const twinApi = {
  // Workspaces / notifications
  listWorkspaces: (init?: ApiRequestInit) =>
    apiFetch<readonly Workspace[]>(`${TWIN}/workspaces`, init),
  listNotifications: (init?: ApiRequestInit) =>
    apiFetch<readonly Notification[]>(`${TWIN}/notifications`, init),
  markAllNotificationsRead: (init?: ApiRequestInit) =>
    apiFetch<{ ok: true }>(`${TWIN}/notifications/read-all`, {
      ...init,
      method: 'POST',
    }),
  clearNotifications: (init?: ApiRequestInit) =>
    apiFetch<{ ok: true }>(`${TWIN}/notifications`, {
      ...init,
      method: 'DELETE',
    }),

  // Health (Twin overlay component health, e.g. Memgraph reachability)
  health: (init?: ApiRequestInit) =>
    apiFetch<{ status: 'ok' | 'degraded' | 'down' }>(`${TWIN}/health`, init),

  // Thesaurus + tag governance
  listThesaurus: (init?: ApiRequestInit) =>
    apiFetch<readonly ThesaurusEntry[]>(`${TWIN}/thesaurus`, init),
  listTags: (init?: ApiRequestInit) =>
    apiFetch<readonly TagEntry[]>(`${TWIN}/tags`, init),
  listTagCategories: (init?: ApiRequestInit) =>
    apiFetch<readonly TagCategory[]>(`${TWIN}/tags/categories`, init),

  /**
   * Download the canonical taxonomy template (governance JSON).
   * Server returns the JSON with ``Content-Disposition: attachment``
   * so a plain anchor + ``download`` attribute would also work, but
   * routing through ``apiFetch`` keeps auth header propagation
   * consistent when the host turns on JWT-cookie auth.
   */
  downloadCategoriesTemplate: (init?: ApiRequestInit) =>
    apiFetch<readonly TagCategory[]>(
      `${TWIN}/tags/categories/template`,
      init,
    ),

  /**
   * Mirror a JSON taxonomy into the workspace's categories store.
   * Server-side validation is strict (matches the template schema);
   * a 400 maps to ``ApiError`` with the validation message as ``body``.
   */
  importCategories: (
    body: readonly { id: string; label: string; color: string }[],
    init?: ApiRequestInit,
  ) =>
    apiFetch<{ ok: boolean }>(`${TWIN}/tags/categories/_import`, {
      ...init,
      method: 'POST',
      body,
    }),

  requestTag: (
    body: {
      tag: string;
      def: string;
      category: string;
      aliases?: readonly string[];
      justification?: string;
      actor?: string;
    },
    init?: ApiRequestInit,
  ) => apiFetch<TagEntry>(`${TWIN}/tags`, { ...init, method: 'POST', body }),
  approveTag: (name: string, actor?: string, init?: ApiRequestInit) =>
    apiFetch<TagEntry>(`${TWIN}/tags/${encodeURIComponent(name)}/approve`, {
      ...init,
      method: 'POST',
      body: { actor },
    }),
  rejectTag: (
    name: string,
    body: { reason: string; actor?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<TagEntry>(`${TWIN}/tags/${encodeURIComponent(name)}/reject`, {
      ...init,
      method: 'POST',
      body,
    }),
  editTag: (
    name: string,
    body: {
      def?: string;
      category?: string;
      aliases?: readonly string[];
      deprecates?: readonly string[];
      actor?: string;
    },
    init?: ApiRequestInit,
  ) =>
    apiFetch<TagEntry>(`${TWIN}/tags/${encodeURIComponent(name)}`, {
      ...init,
      method: 'PATCH',
      body,
    }),
  deprecateTag: (
    name: string,
    body: { reason?: string; actor?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<TagEntry>(`${TWIN}/tags/${encodeURIComponent(name)}/deprecate`, {
      ...init,
      method: 'POST',
      body,
    }),
  updateTagSynonyms: (
    name: string,
    body: { aliases: readonly string[]; actor?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<TagEntry>(`${TWIN}/tags/${encodeURIComponent(name)}/synonyms`, {
      ...init,
      method: 'POST',
      body,
    }),
  deleteTag: (
    name: string,
    body: { strategy?: 'migrate' | 'untag'; to?: string; actor?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<{ ok: boolean }>(`${TWIN}/tags/${encodeURIComponent(name)}`, {
      ...init,
      method: 'DELETE',
      body,
    }),

  // Activity audit feed (a.k.a. audit-events)
  listActivity: (
    q: {
      range?: string;
      kind?: string;
      sev?: string;
      actor?: string;
      q?: string;
      resourceId?: string;
    } = {},
    init?: ApiRequestInit,
  ) => {
    const params: Record<string, string | undefined> = { ...q };
    if (q.resourceId) {
      params['resource.id'] = q.resourceId;
      delete params.resourceId;
    }
    return apiFetch<ListEnvelope<ActivityEvent> & { nowMs?: number }>(
      `${TWIN}/activity`,
      { ...init, query: params },
    );
  },

  // Document overlay (metadata, approve, reject, multi delete)
  getDocumentMetadata: (docId: string, init?: ApiRequestInit) =>
    apiFetch<{
      tags: readonly string[];
      workspace: string;
      review?: Document['review'];
    }>(`${TWIN}/documents/${encodeURIComponent(docId)}/metadata`, init),
  approveDocument: (
    docId: string,
    body: { actor?: string; edits?: Partial<Document> } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<Document>(
      `${TWIN}/documents/${encodeURIComponent(docId)}/approve`,
      { ...init, method: 'POST', body },
    ),
  rejectDocument: (
    docId: string,
    body: { reason: string; actor?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<Document>(
      `${TWIN}/documents/${encodeURIComponent(docId)}/reject`,
      { ...init, method: 'POST', body },
    ),
  bulkDeleteDocuments: (
    body: { doc_ids: readonly string[]; actor?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<{ deleted: number }>(`${TWIN}/documents/bulk-delete`, {
      ...init,
      method: 'POST',
      body,
    }),

  /**
   * Persist a tag mutation (single or bulk) on a list of documents.
   * Doctrine: a tag is a Memgraph node attribute on DocStatus_{workspace}.
   * The server applies set semantics — adds first, removes second — and
   * emits one activity event per doc (kind="doc-retagged").
   * 404s for unknown doc_ids come back in the `failed` array, not as
   * a top-level ApiError, so partial success is the common case.
   */
  bulkRetagDocuments: (
    body: {
      targets: readonly string[];
      adds: readonly string[];
      removes: readonly string[];
      actor?: string;
    },
    init?: ApiRequestInit,
  ) =>
    apiFetch<{ updated: number; failed: readonly string[] }>(
      `${TWIN}/documents/_bulk-retag`,
      { ...init, method: 'POST', body },
    ),

  // Auth
  logout: (init?: ApiRequestInit) =>
    apiFetch<{ ok: true }>(`${TWIN}/auth/logout`, { ...init, method: 'POST' }),

  // Knowledge graph teaser
  listGraphEntities: (
    q: { workspace?: string; type?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<readonly GraphEntity[]>(`${TWIN}/graph/entities`, {
      ...init,
      query: { ...q },
    }),
  listGraphRelations: (
    q: { workspace?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<readonly GraphRelation[]>(`${TWIN}/graph/relations`, {
      ...init,
      query: { ...q },
    }),
  updateGraphEntity: (
    id: string,
    patch: GraphEntityPatch,
    init?: ApiRequestInit,
  ) =>
    apiFetch<GraphEntity>(`${TWIN}/graph/entities/${encodeURIComponent(id)}`, {
      ...init,
      method: 'PATCH',
      body: patch,
    }),
  updateGraphRelation: (
    id: string,
    patch: GraphRelationPatch,
    init?: ApiRequestInit,
  ) =>
    apiFetch<GraphRelation>(`${TWIN}/graph/relations/${encodeURIComponent(id)}`, {
      ...init,
      method: 'PATCH',
      body: patch,
    }),
  createGraphEntity: (
    body: {
      name: string;
      type: GraphEntity['type'];
      summary?: string;
      tags?: readonly string[];
      properties?: Record<string, string>;
    },
    init?: ApiRequestInit,
  ) =>
    apiFetch<GraphEntity>(`${TWIN}/graph/entities`, {
      ...init,
      method: 'POST',
      body,
    }),
  deleteGraphEntity: (id: string, init?: ApiRequestInit) =>
    apiFetch<void>(`${TWIN}/graph/entities/${encodeURIComponent(id)}`, {
      ...init,
      method: 'DELETE',
    }),
  createGraphRelation: (
    body: {
      source: string;
      target: string;
      label: string;
      strength?: number;
      properties?: Record<string, string>;
    },
    init?: ApiRequestInit,
  ) =>
    apiFetch<GraphRelation>(`${TWIN}/graph/relations`, {
      ...init,
      method: 'POST',
      body,
    }),
  deleteGraphRelation: (id: string, init?: ApiRequestInit) =>
    apiFetch<void>(`${TWIN}/graph/relations/${encodeURIComponent(id)}`, {
      ...init,
      method: 'DELETE',
    }),
};

// ============================================================================
// Aggregated facade — preserves the existing `api.xxx()` call surface so the
// query hooks (`queries.ts`) and App.tsx don't need to know which surface
// (lightragApi vs twinApi) owns each endpoint.
// ============================================================================

export const api = {
  // LightRAG-native
  listDocuments: lightragApi.listDocuments,
  listDocumentChunks: lightragApi.listDocumentChunks,
  scanDocument: lightragApi.scanDocument,
  reprocessFailedDocuments: lightragApi.reprocessFailedDocuments,
  trackStatus: lightragApi.trackStatus,
  uploadDocument: lightragApi.uploadDocument,
  deleteDocument: lightragApi.deleteDocument,
  health: lightragApi.health,
  pipelineStatus: lightragApi.pipelineStatus,
  getOpenApi: lightragApi.getOpenApi,
  query: lightragApi.query,

  // Twin overlay
  listWorkspaces: twinApi.listWorkspaces,
  listNotifications: twinApi.listNotifications,
  markAllNotificationsRead: twinApi.markAllNotificationsRead,
  clearNotifications: twinApi.clearNotifications,
  twinHealth: twinApi.health,
  listThesaurus: twinApi.listThesaurus,
  listTags: twinApi.listTags,
  listTagCategories: twinApi.listTagCategories,
  downloadCategoriesTemplate: twinApi.downloadCategoriesTemplate,
  importCategories: twinApi.importCategories,
  requestTag: twinApi.requestTag,
  approveTag: twinApi.approveTag,
  rejectTag: twinApi.rejectTag,
  editTag: twinApi.editTag,
  deprecateTag: twinApi.deprecateTag,
  updateTagSynonyms: twinApi.updateTagSynonyms,
  deleteTag: twinApi.deleteTag,
  listActivity: twinApi.listActivity,
  getDocumentMetadata: twinApi.getDocumentMetadata,
  approveDocument: twinApi.approveDocument,
  rejectDocument: twinApi.rejectDocument,
  bulkDeleteDocuments: twinApi.bulkDeleteDocuments,
  bulkRetagDocuments: twinApi.bulkRetagDocuments,
  logout: twinApi.logout,
  listGraphEntities: twinApi.listGraphEntities,
  listGraphRelations: twinApi.listGraphRelations,
  updateGraphEntity: twinApi.updateGraphEntity,
  updateGraphRelation: twinApi.updateGraphRelation,
  createGraphEntity: twinApi.createGraphEntity,
  deleteGraphEntity: twinApi.deleteGraphEntity,
  createGraphRelation: twinApi.createGraphRelation,
  deleteGraphRelation: twinApi.deleteGraphRelation,
};

export type ApiClient = typeof api;
