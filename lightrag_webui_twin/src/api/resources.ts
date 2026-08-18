/**
 * Typed per-resource fetchers — one function per backend endpoint.
 *
 * Two surfaces:
 *   - `lightragApi` : native LightRAG endpoints (no path prefix beyond the
 *     LightRAG mount root). These hit endpoints LightRAG already ships:
 *     /documents, /documents/{id}/chunks, /query,
 *     /query/data, /health, /pipeline_status, /openapi.
 *   - `twinApi` : Twin overlay endpoints, served by our FastAPI sub-app
 *     mounted via `register(mount_server=True)`. All paths share the
 *     `/twin/api/` prefix: /tags, /audit-events, /activity, /notifications,
 *     /folders, /graph/*, /auth/logout, /documents/{id}/metadata,
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
import type { ApiKeyCreated, ApiKeyPublic } from '../types/apiKey';
import type {
  VisionSettings,
  VisionSettingsPublic,
} from '../types/visionSettings';
import type { QuotaSnapshot } from '../types/quota';
import type { AboutResponse } from '../types/systemInfo';
import type {
  ProcedureBundle,
  ProcedureBundleSummary,
} from '../types/procedure';
import type { QueryMode } from '../types/retrieval';
import type { Folder, Notification } from '../types/topbar';
import type { TagCategory, TagEntry } from '../types/tag';
import type { ThesaurusEntry } from '../types/thesaurus';
import { normalizeDocumentStatus } from '../lib/docStatus';

const TWIN = '/twin/api';

export interface ListEnvelope<T> {
  items: readonly T[];
  total: number;
  page?: number;
  page_size?: number;
  status_counts?: Record<string, number> | null;
  /** Opaque cursor for the next page, or null/absent on the last page. */
  next_cursor?: string | null;
}

export interface ActivityQuery {
  range?: string;
  kind?: string;
  sev?: string;
  actor?: string;
  q?: string;
  limit?: number;
  resourceId?: string;
}

export interface DocumentsQuery {
  status?: string;
  q?: string;
  tag?: string;
  folder?: string;
  cursor?: string;
}

export interface DocumentChunk {
  chunk_id: string;
  order: number;
  text: string;
}

export interface DocumentFoldersResponse {
  doc_id: string;
  folders: readonly string[];
}

export interface RemoveDocumentFolderResponse {
  ok: true;
  doc_id: string;
  removed_folder: string;
  physically_deleted: boolean;
  remaining_folders: readonly string[];
}

/**
 * Receipt returned by the review endpoints. The backend approve/reject
 * routes return `{doc_id, review}` — NOT the full Document
 * (`webui/router.py` approve_document / reject_document). The review payload
 * is what was persisted into `DocStatus.metadata.review`.
 */
export interface DocumentReviewReceipt {
  doc_id: string;
  review: NonNullable<Document['review']>;
}

/**
 * Operator-set MIP sensitivity classification (BNP C1/C2 only). Empty =
 * "no MIP" (let the embedded label / backend default decide). C3/C4 are not
 * operator-selectable at upload: C3 is query-restricted and C4 is rejected.
 */
export type UploadClassification = 'C1' | 'C2';

/**
 * Operator-forced ingestion profile. Rides as the `X-Twin-Doc-Type` header;
 * omitted entirely for auto-detect (the backend seam then decides from the
 * document layout). Anything else is rejected 400 by the backend.
 */
export type UploadDocType = 'procedure' | 'standard';

export interface UploadDocumentOptions {
  signal?: AbortSignal;
  classification?: UploadClassification;
  docType?: UploadDocType;
}

export interface UploadDocumentInput {
  file: File;
  classification?: UploadClassification;
  docType?: UploadDocType;
}

export interface UploadDocumentResponse {
  status: string;
  message: string;
  track_id: string;
  /** Present when upload selection shared an already-ingested document. */
  doc_id?: string;
}

interface ResolveUploadResponse {
  action: 'upload' | 'shared' | 'already_present';
  message?: string;
  track_id?: string;
  doc_id?: string;
}

interface RawDocumentChunk {
  chunk_id: string;
  order?: number;
  chunk_order_index?: number;
  text?: string;
  content?: string;
}

export interface TwinQueryRequest {
  query: string;
  actor?: string;
  mode?: QueryMode;
  top_k?: number;
  chunk_top_k?: number;
  max_total_tokens?: number;
  only_need_context?: boolean;
  history_turns?: number;
  conversation_history?: readonly {
    role: 'user' | 'assistant';
    content: string;
  }[];
  enable_rerank?: boolean;
  min_score?: number;
  tag_filter?: {
    all?: readonly string[];
    any?: readonly string[];
  };
  doc_filter?: {
    all?: readonly string[];
    any?: readonly string[];
  };
  fallback_to_mix?: boolean;
}

/** Wire twin of the backend ``TwinSourceAnchor`` model (offsets only). */
export interface TwinQuerySourceAnchor {
  start: number;
  end: number;
  paragraph_idx: number;
  paragraph_count: number;
  confidence: number;
  method: string;
}

export interface TwinQuerySource {
  n: number;
  type: string;
  name: string;
  meta?: string | null;
  /** Real retrieval metric, when the backend can expose one. */
  score?: number | null;
  /** Explicit grounding provenance; absent on legacy backends. */
  retrieval_origin?: 'vector' | 'graph' | null;
  doc_id?: string | null;
  chunk_id?: string | null;
  /** Optional intra-chunk paragraph anchor; null on legacy backends. */
  anchor?: TwinQuerySourceAnchor | null;
}

/**
 * TR-RET-02: ``answer_status`` mirrors LightRAG's no-context detection
 * (the ``[no-context]`` marker the backend strips before sending the
 * response). ``insufficient_information`` means the retrieval pipeline
 * had nothing useful — the React port hides the Sources panel and
 * shows a discrete cue instead of pretending the listed chunks back
 * the (fail) answer.
 */
export type TwinAnswerStatus =
  | 'grounded'
  | 'insufficient_information'
  | 'source_projection_failed'
  | 'no_retrieval'
  | 'query_failed';

export interface TwinQueryResponse {
  response: string;
  sources: readonly TwinQuerySource[];
  /** Deployment-provided synthesis model name. */
  model?: string;
  /** Backend default is ``"grounded"``. Optional in the TS contract
   *  so a legacy backend without the field still parses cleanly. */
  answer_status?: TwinAnswerStatus;
}

export type TwinQueryProgressStage = 'retrieval' | 'generation' | 'sources';

export interface TwinQueryDataResponse {
  status: string;
  message: string;
  data: {
    entities?: readonly Record<string, unknown>[];
    relationships?: readonly Record<string, unknown>[];
    chunks?: readonly Record<string, unknown>[];
    references?: readonly Record<string, unknown>[];
    [key: string]: unknown;
  };
  metadata: Record<string, unknown>;
}

export interface AuthStatusResponse {
  auth_enabled: boolean;
  authenticated: boolean;
  user?: string | null;
  expires_at?: string | null;
  login_required: boolean;
}

export interface PipelineStatusResponse {
  busy: boolean;
  job_count: number;
  job_name?: string | null;
  latest_message: string | null;
  history_messages: readonly string[];
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

function parseMaybeJson(text: string): unknown {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function isLikelyJwtToken(token: string): boolean {
  const parts = token.split('.');
  if (parts.length !== 3) return false;
  return parts.every((part) => part.length > 0);
}

// ============================================================================
// LightRAG-native endpoints (NO /twin/api prefix)
// ============================================================================

// Status normalization at ingress (lowercase LightRAG wire → UPPERCASE UI
// enum, unknown → PENDING) lives in the shared vocabulary module — audit
// 2026-07-02, DUP-1. Behaviour is unchanged; see lib/docStatus.ts.

export const lightragApi = {
  authStatus: (init?: ApiRequestInit) =>
    apiFetch<AuthStatusResponse>('/auth-status', init),
  login: (
    body: { username: string; password: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<LoginResponse>('/login', {
      ...init,
      method: 'POST',
      body,
    }),
  logoutLocal: (init?: ApiRequestInit) =>
    apiFetch<{ ok: true }>('/logout', {
      ...init,
      method: 'POST',
    }),
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
    apiFetch<readonly RawDocumentChunk[]>(
      `/documents/${encodeURIComponent(docId)}/chunks`,
      init,
    ).then((chunks) =>
      chunks.map((chunk, index) => ({
        chunk_id: chunk.chunk_id,
        order: chunk.order ?? chunk.chunk_order_index ?? index,
        text: chunk.text ?? chunk.content ?? '',
      })),
    ),
  /**
   * Resolve all DocStatus rows associated with an ingestion track_id
   * (the id returned by /documents/upload). Used to discover the
   * generated doc_id once processing completes, so initial tags from
   * the AddSource modal can be applied via bulk-retag.
   *
   * NOTE: track lookup is workspace-GLOBAL, not folder-scoped (a track_id is an
   * opaque per-upload handle). Do not treat it as a cloisonnement surface;
   * downstream writes still enforce active-folder membership. See
   * docstatus_impl.get_docs_by_track_id.
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
  /**
   * Trigger re-processing of FAILED docs via LightRAG's native batch endpoint.
   *
   * WARNING: this reprocesses LightRAG's GLOBAL failed queue for the whole
   * workspace — it is NOT folder-scoped. The native pipeline
   * (apipeline_process_enqueue_documents → get_docs_by_statuses) has no folder
   * filter, so a "retry failed batch" click can re-enqueue failed/pending docs
   * from other folders too. A folder-scoped reprocess would need a dedicated
   * Twin route + test (it touches the LightRAG pipeline contract) — follow-up,
   * deliberately out of scope here.
   *
   * There is no per-doc-by-id reprocess in LightRAG 1.4.9.11; the DocDetailPanel
   * "Re-process" button surfaces this as "retry failed batch" when the targeted
   * doc is FAILED, and as a clear no-op explanation otherwise (see App.tsx
   * onReprocess).
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
   * URL, auth, cookie, and folder header pattern matches apiFetch.
   * Returns the InsertResponse shape:
   *   { status: 'success'|'duplicated', message: string, track_id: string }
   * On 4xx/5xx, throws an ApiError so the host can show a real toast.
   */
  uploadDocument: async (
    file: File,
    init?: UploadDocumentOptions,
  ): Promise<UploadDocumentResponse> => {
    const resolveExisting = () =>
      apiFetch<ResolveUploadResponse>(`${TWIN}/documents/resolve-upload`, {
        method: 'POST',
        body: { file_name: file.name },
        signal: init?.signal,
      });
    const asResolvedUpload = (
      resolved: ResolveUploadResponse,
    ): UploadDocumentResponse | null => {
      if (resolved.action === 'upload') return null;
      if (
        resolved.action !== 'shared' &&
        resolved.action !== 'already_present'
      ) {
        throw new ApiError(
          'POST /twin/api/documents/resolve-upload returned an invalid action',
          502,
          resolved,
        );
      }
      return {
        status: resolved.action === 'shared' ? 'shared' : 'duplicated',
        message: resolved.message ?? '',
        track_id: resolved.track_id ?? '',
        ...(resolved.doc_id ? { doc_id: resolved.doc_id } : {}),
      };
    };

    const resolvedBeforeUpload = asResolvedUpload(await resolveExisting());
    if (resolvedBeforeUpload) return resolvedBeforeUpload;

    const formData = new FormData();
    formData.append('file', file);
    const headers = buildApiHeaders();
    const authorization = headers.Authorization ?? '';
    const bearerPrefix = 'Bearer ';
    const bearer = authorization.startsWith(bearerPrefix)
      ? authorization.slice(bearerPrefix.length).trim()
      : undefined;
    if (bearer && !headers['X-API-Key'] && !isLikelyJwtToken(bearer)) {
      // Native LightRAG upload routes validate LIGHTRAG_API_KEY via X-API-Key,
      // while the Twin overlay accepts the same value as Authorization: Bearer.
      // Do not send both for opaque static keys: LightRAG validates OAuth Bearer
      // first and rejects the static API key value as an invalid token before
      // checking X-API-Key. JWT sessions are different: keep Authorization so
      // local/IdP login uploads do not turn into 401s on the multipart path.
      headers['X-API-Key'] = bearer;
      delete headers.Authorization;
    }
    const res = await fetch(buildApiUrl('/documents/upload'), {
      method: 'POST',
      // Operator classification rides as an HTTP header (X-Twin-Classification),
      // NOT a multipart field — the backend reads it per-upload and applies it
      // as a floor-raising value. Omitted entirely when unset.
      headers: {
        ...headers,
        ...(init?.classification
          ? { 'X-Twin-Classification': init.classification }
          : {}),
        // Operator-forced ingestion profile (procedure|standard). Omitted for
        // auto-detect so the backend seam keeps its layout-based detection.
        ...(init?.docType ? { 'X-Twin-Doc-Type': init.docType } : {}),
      },
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
      // Close the preflight/upload race: another request may have created the
      // canonical source after our first resolution but before LightRAG's
      // strict name check. Re-resolve only for a collision response; unrelated
      // upload failures retain their original detail.
      const detail =
        body && typeof body === 'object' && 'detail' in body
          ? (body as { detail?: unknown }).detail
          : undefined;
      const isNameCollision =
        res.status === 409 &&
        typeof detail === 'string' &&
        (detail.includes('Document storage already contains') ||
          detail.includes('Input directory already contains'));
      if (isNameCollision) {
        const resolvedAfterCollision = asResolvedUpload(await resolveExisting());
        if (resolvedAfterCollision) return resolvedAfterCollision;
      }
      throw new ApiError(
        `POST /documents/upload → ${res.status} ${res.statusText}`,
        res.status,
        body,
      );
    }
    return body as UploadDocumentResponse;
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
    apiFetch<PipelineStatusResponse>('/pipeline_status', init),
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
  // Folders (runtime additions on top of the env seed)
  listFolders: (init?: ApiRequestInit) =>
    apiFetch<readonly Folder[]>(`${TWIN}/folders`, init),

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
  /**
   * Structured retrieval data endpoint. This mirrors LightRAG's native
   * /query/data through the Twin prefix so folder headers and tag_filter
   * stay on the same governed surface as the chat endpoints.
   */
  queryData: (body: TwinQueryRequest, init?: ApiRequestInit) =>
    apiFetch<TwinQueryDataResponse>(`${TWIN}/query/data`, {
      ...init,
      method: 'POST',
      body,
    }),
  queryStream: async (
    body: TwinQueryRequest,
    onChunk: (chunk: string) => void,
    init?: ApiRequestInit,
    onStage?: (stage: TwinQueryProgressStage) => void,
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
    //   {"type":"status","value":"grounded"|"insufficient_information"
    //                            |"source_projection_failed"|"no_retrieval"
    //                            |"query_failed"}
    //   {"type":"meta","value":{"model":"<llm model>"}}
    //   {"type":"sources","value":[<RetrievalSource>, ...]}
    // Token events stream the LLM answer (call onChunk for live UI);
    // the status event arrives exactly once before the final sources
    // event so the host can decide whether to render the panel; a
    // single sources event arrives last with the structured panel data.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let response = '';
    let sources: TwinQuerySource[] = [];
    let answerStatus: TwinAnswerStatus = 'grounded';
    let model: string | undefined;
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
      } else if (
        event.type === 'stage' &&
        (event.value === 'retrieval' ||
          event.value === 'generation' ||
          event.value === 'sources')
      ) {
        onStage?.(event.value);
      } else if (
        event.type === 'meta' &&
        typeof event.value === 'object' &&
        event.value !== null &&
        typeof (event.value as { model?: unknown }).model === 'string'
      ) {
        model = (event.value as { model: string }).model;
      } else if (
        event.type === 'status' &&
        (event.value === 'grounded' ||
          event.value === 'insufficient_information' ||
          event.value === 'source_projection_failed' ||
          event.value === 'no_retrieval' ||
          event.value === 'query_failed')
      ) {
        answerStatus = event.value;
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
    return { response, sources, answer_status: answerStatus, model };
  },
  createFolder: (
    body: { id: string; label: string; kind?: string; description?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<Folder>(`${TWIN}/folders`, {
      ...init,
      method: 'POST',
      body,
    }),
  updateFolder: (
    id: string,
    patch: { label?: string; kind?: string; description?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<Folder>(`${TWIN}/folders/${encodeURIComponent(id)}`, {
      ...init,
      method: 'PATCH',
      body: patch,
    }),
  deleteFolder: (id: string, init?: ApiRequestInit) =>
    apiFetch<void>(`${TWIN}/folders/${encodeURIComponent(id)}`, {
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

  // ── Instance storage quota (Memgraph memory limit) ───────────────
  // Public read — banner needs to render even for anonymous browsing
  // surfaces. Polled every 30s by the WebUI; cheap server-side.
  getQuotaSnapshot: (init?: ApiRequestInit) =>
    apiFetch<QuotaSnapshot>(`${TWIN}/quota`, init),

  // ── About / system identity card (Settings → About) ───────────────
  // Two-tier payload: versions for any caller the backend serves, deployment
  // shape (Memgraph, Python, storage topology) only for admins.
  getAbout: (init?: ApiRequestInit) =>
    apiFetch<AboutResponse>(`${TWIN}/system/about`, init),

  // ── API key management ───────────────────────────────────────────
  // Per-operator keys minted via Settings → API keys. Distinct from
  // the static LIGHTRAG_API_KEY (infra root, never exposed via UI).
  listApiKeys: (init?: ApiRequestInit) =>
    apiFetch<readonly ApiKeyPublic[]>(`${TWIN}/settings/api-keys`, init),
  createApiKey: (body: { name: string }, init?: ApiRequestInit) =>
    apiFetch<ApiKeyCreated>(`${TWIN}/settings/api-keys`, {
      ...init,
      method: 'POST',
      body,
    }),
  revokeApiKey: (id: string, init?: ApiRequestInit) =>
    apiFetch<ApiKeyPublic>(
      `${TWIN}/settings/api-keys/${encodeURIComponent(id)}`,
      {
        ...init,
        method: 'DELETE',
      },
    ),

  // ── Vision ingestion settings ────────────────────────────────────
  // The two operator-tunable curation knobs of the image-ingestion
  // pipeline (RapidOCR pre-filter + post-LLM class drop). GET is open
  // to any authenticated operator; PUT is admin-gated server-side.
  getVisionSettings: (init?: ApiRequestInit) =>
    apiFetch<VisionSettingsPublic>(`${TWIN}/settings/vision`, init),
  updateVisionSettings: (body: VisionSettings, init?: ApiRequestInit) =>
    apiFetch<VisionSettingsPublic>(`${TWIN}/settings/vision`, {
      ...init,
      method: 'PUT',
      body,
    }),

  // Tag governance. listThesaurus is legacy compatibility only; new UI
  // surfaces must use listTags as the canonical runtime catalog.
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
   * Mirror a JSON taxonomy into the folder categories store.
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
      long_description?: string;
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
      tag?: string;
      def?: string;
      long_description?: string;
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
  suggestTagEdit: (
    name: string,
    body: {
      def?: string;
      long_description?: string;
      category?: string;
      aliases?: readonly string[];
      justification?: string;
      actor?: string;
    },
    init?: ApiRequestInit,
  ) =>
    apiFetch<TagEntry>(
      `${TWIN}/tags/${encodeURIComponent(name)}/suggest-edit`,
      {
        ...init,
        method: 'POST',
        body,
      },
    ),
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
  reactivateTag: (
    name: string,
    body: { actor?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<TagEntry>(`${TWIN}/tags/${encodeURIComponent(name)}/reactivate`, {
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
    q: ActivityQuery = {},
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

  // Document overlay (metadata, approve, reject, folder membership, multi delete)
  listDocumentFolders: (docId: string, init?: ApiRequestInit) =>
    apiFetch<DocumentFoldersResponse>(
      `${TWIN}/documents/${encodeURIComponent(docId)}/folders`,
      init,
    ),
  addDocumentToFolder: (
    docId: string,
    folderId: string,
    init?: ApiRequestInit,
  ) =>
    apiFetch<DocumentFoldersResponse>(
      `${TWIN}/documents/${encodeURIComponent(docId)}/folders`,
      { ...init, method: 'POST', body: { folder_id: folderId } },
    ),
  removeDocumentFromFolder: (
    docId: string,
    folderId: string,
    init?: ApiRequestInit,
  ) =>
    apiFetch<RemoveDocumentFolderResponse>(
      `${TWIN}/documents/${encodeURIComponent(docId)}/folders/${encodeURIComponent(folderId)}`,
      { ...init, method: 'DELETE' },
    ),
  getDocumentMetadata: (docId: string, init?: ApiRequestInit) =>
    apiFetch<{
      tags: readonly string[];
      tags_source?: 'tagged_with';
      tags_status?: 'ok' | 'unavailable';
      folder: string;
      review?: Document['review'];
    }>(`${TWIN}/documents/${encodeURIComponent(docId)}/metadata`, init),
  approveDocument: (
    docId: string,
    body: { actor?: string; edits?: Partial<Document> } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<DocumentReviewReceipt>(
      `${TWIN}/documents/${encodeURIComponent(docId)}/approve`,
      { ...init, method: 'POST', body },
    ),
  rejectDocument: (
    docId: string,
    body: { reason: string; actor?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<DocumentReviewReceipt>(
      `${TWIN}/documents/${encodeURIComponent(docId)}/reject`,
      { ...init, method: 'POST', body },
    ),
  /**
   * `busy` lists ids the backend deferred because the ingestion pipeline
   * was held by a processing job — those documents are untouched and the
   * same call succeeds after the pipeline drains (the host auto-retries).
   * An all-deferred batch is a 423 ApiError instead (nothing deleted).
   */
  bulkDeleteDocuments: (
    body: { doc_ids: readonly string[]; actor?: string },
    init?: ApiRequestInit,
  ) =>
    apiFetch<{
      deleted: number;
      failed: readonly string[];
      busy?: readonly string[];
    }>(`${TWIN}/documents/bulk-delete`, {
      ...init,
      method: 'POST',
      body,
    }),

  /**
   * Persist a tag mutation (single or bulk) on a list of documents.
   * Doctrine: a tag is a Memgraph node linked to DocStatus in the active folder.
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
  recordSourceUploaded: (
    body: {
      source: string;
      track_id?: string;
      status?: string;
      actor?: string;
    },
    init?: ApiRequestInit,
  ) =>
    apiFetch<{ ok: true }>(`${TWIN}/documents/uploads/activity`, {
      ...init,
      method: 'POST',
      body,
    }),

  // ── Procedure approval bundles (BNP procedure-PDF profile) ───────
  // Parked procedure documents awaiting human review. The list is
  // folder-bound (any authenticated operator); detail + decisions are
  // admin-gated server-side — a 403 surfaces through the mutation error.
  // A degraded bundle store returns 503 with a recovery hint.
  listProcedures: (
    q: { state?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<readonly ProcedureBundleSummary[]>(`${TWIN}/procedures`, {
      ...init,
      query: { ...q },
    }),
  getProcedureBundle: (bundleId: string, init?: ApiRequestInit) =>
    apiFetch<ProcedureBundle>(
      `${TWIN}/procedures/${encodeURIComponent(bundleId)}`,
      init,
    ),
  approveProcedure: (
    bundleId: string,
    body: { folder?: string | null } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<ProcedureBundle>(
      `${TWIN}/procedures/${encodeURIComponent(bundleId)}/approve`,
      { ...init, method: 'POST', body },
    ),
  rejectProcedure: (
    bundleId: string,
    body: { comment?: string | null } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<ProcedureBundle>(
      `${TWIN}/procedures/${encodeURIComponent(bundleId)}/reject`,
      { ...init, method: 'POST', body },
    ),
  retryProcedure: (bundleId: string, init?: ApiRequestInit) =>
    apiFetch<ProcedureBundle>(
      `${TWIN}/procedures/${encodeURIComponent(bundleId)}/retry`,
      { ...init, method: 'POST' },
    ),
  rerouteProcedureStandard: (
    bundleId: string,
    body: { folder?: string | null } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<ProcedureBundle>(
      `${TWIN}/procedures/${encodeURIComponent(bundleId)}/reroute-standard`,
      { ...init, method: 'POST', body },
    ),

  // Knowledge graph teaser
  listGraphEntities: (
    q: { folder?: string; type?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<readonly GraphEntity[]>(`${TWIN}/graph/entities`, {
      ...init,
      query: { ...q },
    }),
  listGraphRelations: (
    q: { folder?: string } = {},
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
  authStatus: lightragApi.authStatus,
  login: lightragApi.login,
  logoutLocal: lightragApi.logoutLocal,
  listDocuments: lightragApi.listDocuments,
  listDocumentChunks: lightragApi.listDocumentChunks,
  reprocessFailedDocuments: lightragApi.reprocessFailedDocuments,
  trackStatus: lightragApi.trackStatus,
  uploadDocument: lightragApi.uploadDocument,
  deleteDocument: (docId: string, init?: ApiRequestInit) =>
    twinApi.bulkDeleteDocuments({ doc_ids: [docId] }, init).then((res) => {
      // bulk-delete reports per-doc failures in {deleted, failed} (HTTP 207 for
      // partial success). Surface a single-doc failure as a thrown error so the
      // mutation's optimistic rollback fires and the toast is honest, instead
      // of a false "Document removed" on {deleted: 0, failed: [docId]}.
      if (res.deleted !== 1 || (res.failed ?? []).includes(docId)) {
        throw new Error(`Delete failed for ${docId}`);
      }
      return { ok: true as const };
    }),
  health: lightragApi.health,
  pipelineStatus: lightragApi.pipelineStatus,
  getOpenApi: lightragApi.getOpenApi,
  query: twinApi.query,
  queryData: twinApi.queryData,
  queryStream: twinApi.queryStream,

  // Twin overlay
  listFolders: twinApi.listFolders,
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
  suggestTagEdit: twinApi.suggestTagEdit,
  deprecateTag: twinApi.deprecateTag,
  reactivateTag: twinApi.reactivateTag,
  updateTagSynonyms: twinApi.updateTagSynonyms,
  deleteTag: twinApi.deleteTag,
  listActivity: twinApi.listActivity,
  getDocumentMetadata: twinApi.getDocumentMetadata,
  listDocumentFolders: twinApi.listDocumentFolders,
  addDocumentToFolder: twinApi.addDocumentToFolder,
  removeDocumentFromFolder: twinApi.removeDocumentFromFolder,
  approveDocument: twinApi.approveDocument,
  rejectDocument: twinApi.rejectDocument,
  bulkDeleteDocuments: twinApi.bulkDeleteDocuments,
  bulkRetagDocuments: twinApi.bulkRetagDocuments,
  logout: twinApi.logout,
  recordSourceUploaded: twinApi.recordSourceUploaded,
  listProcedures: twinApi.listProcedures,
  getProcedureBundle: twinApi.getProcedureBundle,
  approveProcedure: twinApi.approveProcedure,
  rejectProcedure: twinApi.rejectProcedure,
  retryProcedure: twinApi.retryProcedure,
  rerouteProcedureStandard: twinApi.rerouteProcedureStandard,
  listGraphEntities: twinApi.listGraphEntities,
  listGraphRelations: twinApi.listGraphRelations,
  updateGraphEntity: twinApi.updateGraphEntity,
  updateGraphRelation: twinApi.updateGraphRelation,
  createGraphEntity: twinApi.createGraphEntity,
  deleteGraphEntity: twinApi.deleteGraphEntity,
  createGraphRelation: twinApi.createGraphRelation,
  deleteGraphRelation: twinApi.deleteGraphRelation,
  createFolder: twinApi.createFolder,
  updateFolder: twinApi.updateFolder,
  deleteFolder: twinApi.deleteFolder,
  listApiKeys: twinApi.listApiKeys,
  createApiKey: twinApi.createApiKey,
  revokeApiKey: twinApi.revokeApiKey,
  getVisionSettings: twinApi.getVisionSettings,
  updateVisionSettings: twinApi.updateVisionSettings,
  getQuotaSnapshot: twinApi.getQuotaSnapshot,
  getAbout: twinApi.getAbout,
};

export type ApiClient = typeof api;
