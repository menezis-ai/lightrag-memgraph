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

export interface TwinQueryRequest {
  query: string;
  mode?: string;
  top_k?: number;
  chunk_top_k?: number;
  max_total_tokens?: number;
  only_need_context?: boolean;
  only_need_prompt?: boolean;
  history_turns?: number;
  user_prompt?: string;
  enable_rerank?: boolean;
  tag_filter?: {
    all?: readonly string[];
    any?: readonly string[];
  };
}

export interface TwinQuerySource {
  n: number;
  type: string;
  name: string;
  meta?: string | null;
  score: number;
  doc_id?: string | null;
  chunk_id?: string | null;
}

export interface TwinQueryResponse {
  response: string;
  sources: readonly TwinQuerySource[];
}

function parseMaybeJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

// ============================================================================
// LightRAG-native endpoints (NO /twin/api prefix)
// ============================================================================

/** LightRAG's DocStatus.value is lowercase (`'pending'`, `'processing'`,
 *  `'processed'`, `'failed'`), but our `DocumentStatus` type and every
 *  consumer in this codebase expects uppercase. Normalize at ingress so
 *  the UI mapping/counters work regardless of which end of the contract
 *  shifts later. Unknown values fall back to `PENDING` (same as the
 *  Python `MemgraphDocStatusStorage._deserialize_status` does). */
const ALLOWED_DOC_STATUS = new Set([
  'PENDING',
  'PROCESSING',
  'PROCESSED',
  'FAILED',
]);
function normalizeDocumentStatus(raw: unknown): Document['status'] {
  const s = String(raw ?? '').toUpperCase();
  return (
    ALLOWED_DOC_STATUS.has(s) ? s : 'PENDING'
  ) as Document['status'];
}

export const lightragApi = {
  listDocuments: (q: DocumentsQuery = {}, init?: ApiRequestInit) =>
    apiFetch<ListEnvelope<Document>>('/documents', { ...init, query: { ...q } })
      .then((env) => ({
        ...env,
        items: env.items.map((d) => ({
          ...d,
          status: normalizeDocumentStatus(d.status),
        })),
      })),
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
};

// ============================================================================
// Twin overlay endpoints (/twin/api/* prefix)
// ============================================================================

export const twinApi = {
  // Workspaces / notifications
  listWorkspaces: (init?: ApiRequestInit) =>
    apiFetch<readonly Workspace[]>(`${TWIN}/workspaces`, init),

  // Spaces (M12 Admin CRUD — runtime additions on top of the env seed)
  listSpaces: (init?: ApiRequestInit) =>
    apiFetch<readonly Workspace[]>(`${TWIN}/spaces`, init),

  /**
   * Issue a retrieval query against the Twin overlay POST /query.
   * Returns the synthesised response string AND a structured sources
   * list (top-k chunks the retrieval grounded on, with file paths and
   * scores) so the chat surface can render citations. The endpoint
   * accepts the standard LightRAG query body.
   */
  query: (body: TwinQueryRequest, init?: ApiRequestInit) =>
    apiFetch<TwinQueryResponse>(`${TWIN}/query`, {
      ...init,
      method: 'POST',
      body,
    }),
  queryStream: async (
    body: TwinQueryRequest,
    onChunk: (chunk: string) => void,
    init?: ApiRequestInit,
  ): Promise<TwinQueryResponse> => {
    const res = await fetch(buildApiUrl(`${TWIN}/query/stream`), {
      method: 'POST',
      headers: buildApiHeaders(init, { json: true }),
      body: JSON.stringify(body),
      signal: init?.signal,
      credentials: 'include',
    });

    if (!res.ok) {
      const text = await res.text();
      throw new ApiError(
        `POST ${TWIN}/query/stream → ${res.status} ${res.statusText}`,
        res.status,
        parseMaybeJson(text),
      );
    }

    if (!res.body) {
      return { response: '', sources: [] };
    }

    // Wire format: NDJSON. One JSON object per line:
    //   {"type":"token","value":"<chunk text>"}
    //   {"type":"sources","value":[<RetrievalSource>, ...]}
    // Token events stream the LLM answer (call onChunk for live UI);
    // a single sources event arrives last with the structured panel data.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let response = '';
    let sources: TwinQuerySource[] = [];
    let buffer = '';

    const consumeLine = (line: string) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let event: { type?: string; value?: unknown };
      try {
        event = JSON.parse(trimmed) as { type?: string; value?: unknown };
      } catch {
        return; // ignore malformed line, keep streaming
      }
      if (event.type === 'token' && typeof event.value === 'string') {
        response += event.value;
        onChunk(event.value);
      } else if (event.type === 'sources' && Array.isArray(event.value)) {
        sources = event.value as TwinQuerySource[];
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        consumeLine(buffer.slice(0, nl));
        buffer = buffer.slice(nl + 1);
      }
    }
    buffer += decoder.decode();
    if (buffer) consumeLine(buffer);
    return { response, sources };
  },
  createSpace: (
    body: { id: string; label: string; kind?: string; description?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<Workspace>(`${TWIN}/spaces`, {
      ...init,
      method: 'POST',
      body,
    }),
  updateSpace: (
    id: string,
    patch: { label?: string; kind?: string; description?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<Workspace>(`${TWIN}/spaces/${encodeURIComponent(id)}`, {
      ...init,
      method: 'PATCH',
      body: patch,
    }),
  deleteSpace: (id: string, init?: ApiRequestInit) =>
    apiFetch<void>(`${TWIN}/spaces/${encodeURIComponent(id)}`, {
      ...init,
      method: 'DELETE',
    }),
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
      limit?: number;
      resourceId?: string;
    } = {},
    init?: ApiRequestInit,
  ) => {
    const params: Record<string, string | number | undefined> = { ...q };
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
  query: twinApi.query,
  queryStream: twinApi.queryStream,

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
  listSpaces: twinApi.listSpaces,
  createSpace: twinApi.createSpace,
  updateSpace: twinApi.updateSpace,
  deleteSpace: twinApi.deleteSpace,
};

export type ApiClient = typeof api;
