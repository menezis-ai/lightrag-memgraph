/**
 * TanStack Query hooks — one hook per WebUI tab's data dependency.
 *
 * All hooks return the raw fetch payload + the usual `{ isLoading, isError,
 * error }` triplet. Stale time defaults to 60s so tab switches don't refetch
 * needlessly. Mutation hooks live next to their resource (notifications) and
 * use `queryClient.invalidateQueries(['notifications'])` to refresh.
 */

import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from '@tanstack/react-query';
import { getActiveFolder } from './client';
import { api } from './resources';
import { parseOpenApiSpec } from './openapi-parser';
import type { Document } from '../types/document';
import type { GraphEntity, GraphRelation } from '../types/graph';
import type { UploadDocumentInput } from './resources';

const DEFAULTS = { staleTime: 60_000 } as const;
const DOCUMENTS_REFETCH_INTERVAL_MS = 2_000;
const PIPELINE_REFETCH_INTERVAL_MS = 2_000;
const DEFAULT_UPLOAD_CONCURRENCY = 4;
type QueryGate = { enabled?: boolean; folderKey?: string | null };
type UploadBatchItem = UploadDocumentInput | File;

function gateOptions(options: QueryGate): Pick<QueryGate, 'enabled'> {
  return { enabled: options.enabled };
}

function folderScope(
  options: QueryGate,
  requestedFolder?: string | null,
): string {
  return options.folderKey ?? requestedFolder ?? getActiveFolder() ?? 'default';
}

function graphEntitiesKey(folder = getActiveFolder() ?? 'default') {
  return ['graph-entities', folder] as const;
}

function graphRelationsKey(folder = getActiveFolder() ?? 'default') {
  return ['graph-relations', folder] as const;
}

function normalizeUploadInput(item: UploadBatchItem): UploadDocumentInput {
  return item instanceof File ? { file: item } : item;
}

async function mapSettledWithConcurrency<T, R>(
  items: readonly T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<PromiseSettledResult<R>[]> {
  const results = new Array<PromiseSettledResult<R>>(items.length);
  let next = 0;
  const workerCount = Math.min(Math.max(1, concurrency), items.length);

  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      while (next < items.length) {
        const index = next;
        next += 1;
        try {
          results[index] = {
            status: 'fulfilled',
            value: await fn(items[index], index),
          };
        } catch (reason) {
          results[index] = { status: 'rejected', reason };
        }
      }
    }),
  );

  return results;
}

export function useDocuments(
  query: {
    status?: string;
    q?: string;
    tag?: string;
    folder?: string;
  } = {},
  options: QueryGate = {},
) {
  const scope = folderScope(options, query.folder);
  // Infinite (cursor) pagination: the documents list can be arbitrarily large
  // (BNP-scale KBs + RAG 1.5). The backend returns one page + a `next_cursor`;
  // the operator pulls more via "Load more" in DocumentsTab. No hard cap.
  return useInfiniteQuery({
    queryKey: ['documents', scope, query] as const,
    queryFn: ({ pageParam, signal }) =>
      api.listDocuments(
        { ...query, cursor: pageParam as string | undefined },
        { signal },
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    ...DEFAULTS,
    staleTime: 0,
    refetchInterval: DOCUMENTS_REFETCH_INTERVAL_MS,
    refetchIntervalInBackground: true,
    refetchOnReconnect: 'always',
    refetchOnWindowFocus: 'always',
    ...gateOptions(options),
  });
}

export function usePipelineStatus(options: QueryGate = {}) {
  const scope = folderScope(options);
  return useQuery({
    queryKey: ['pipeline_status', scope] as const,
    queryFn: ({ signal }) => api.pipelineStatus({ signal }),
    ...DEFAULTS,
    staleTime: 0,
    refetchInterval: PIPELINE_REFETCH_INTERVAL_MS,
    refetchIntervalInBackground: true,
    ...gateOptions(options),
  });
}

// Folders — admin CRUD on top of the env seed.
export function useFolders(options: QueryGate = {}) {
  return useQuery({
    queryKey: ['folders'] as const,
    queryFn: ({ signal }) => api.listFolders({ signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useCreateFolder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Parameters<typeof api.createFolder>[0]) =>
      api.createFolder(body),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['folders'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

export function useUpdateFolder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      patch,
    }: { id: string; patch: Parameters<typeof api.updateFolder>[1] }) =>
      api.updateFolder(id, patch),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['folders'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

export function useDeleteFolder() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteFolder(id),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['folders'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

// Instance storage quota — polled every 30s so the banner reflects
// Memgraph pressure without forcing a manual refresh.
export function useInstanceQuota(options: QueryGate = {}) {
  return useQuery({
    queryKey: ['quota'] as const,
    queryFn: ({ signal }) => api.getQuotaSnapshot({ signal }),
    refetchInterval: 30_000,
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

// API keys — per-operator, distinct from the static LIGHTRAG_API_KEY.
export function useApiKeys(options: QueryGate = {}) {
  return useQuery({
    queryKey: ['api-keys'] as const,
    queryFn: ({ signal }) => api.listApiKeys({ signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useCreateApiKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Parameters<typeof api.createApiKey>[0]) =>
      api.createApiKey(body),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['api-keys'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

export function useRevokeApiKey() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.revokeApiKey(id),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['api-keys'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

export function useNotifications(options: QueryGate = {}) {
  const scope = folderScope(options);
  return useQuery({
    queryKey: ['notifications', scope] as const,
    queryFn: ({ signal }) => api.listNotifications({ signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useMarkAllNotificationsRead() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.markAllNotificationsRead(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['notifications'] }),
  });
}

export function useClearNotifications() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.clearNotifications(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['notifications'] }),
  });
}

export function useThesaurus(options: QueryGate = {}) {
  // Legacy compatibility for /thesaurus. Runtime tag pickers should use
  // useTags(), which is the canonical catalog surface.
  const scope = folderScope(options);
  return useQuery({
    queryKey: ['thesaurus', scope] as const,
    queryFn: ({ signal }) => api.listThesaurus({ signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useTags(options: QueryGate = {}) {
  const scope = folderScope(options);
  return useQuery({
    queryKey: ['tags', scope] as const,
    queryFn: ({ signal }) => api.listTags({ signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useTagCategories(options: QueryGate = {}) {
  const scope = folderScope(options);
  return useQuery({
    queryKey: ['tag-categories', scope] as const,
    queryFn: ({ signal }) => api.listTagCategories({ signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useActivity(
  query: {
    range?: string;
    kind?: string;
    sev?: string;
    actor?: string;
    q?: string;
    limit?: number;
  } = {},
  options: QueryGate = {},
) {
  const scope = folderScope(options);
  return useQuery({
    queryKey: ['activity', scope, query] as const,
    queryFn: ({ signal }) => api.listActivity(query, { signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useOpenApi(options: QueryGate = {}) {
  // Hit the FastAPI-auto `/openapi.json` directly so the Twin ApiTab is
  // ISO with the LightRAG WebUI by construction — any route added to
  // the host app (LightRAG native + Twin overlay via `include_router`)
  // appears here automatically without a manual catalog. See
  // `openapi-parser.ts` for the reshape into `OpenApiGroup[]`.
  return useQuery({
    queryKey: ['openapi'] as const,
    queryFn: async ({ signal }) => {
      const resp = await fetch('/openapi.json', { signal });
      if (!resp.ok) {
        throw new Error(
          `OpenAPI fetch failed: ${resp.status} ${resp.statusText}`,
        );
      }
      return parseOpenApiSpec(await resp.json());
    },
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useGraphEntities(options: QueryGate = {}) {
  const scope = folderScope(options);
  return useQuery({
    queryKey: graphEntitiesKey(scope),
    queryFn: ({ signal }) => api.listGraphEntities({}, { signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

export function useGraphRelations(options: QueryGate = {}) {
  const scope = folderScope(options);
  return useQuery({
    queryKey: graphRelationsKey(scope),
    queryFn: ({ signal }) => api.listGraphRelations({}, { signal }),
    ...DEFAULTS,
    ...gateOptions(options),
  });
}

// ─── Graph entity / relation editing — optimistic + rollback. ───────────
// onMutate snapshots the previous list, patches in place, then rolls back
// on error. The MSW handlers (and the future backend) persist to Memgraph,
// so the entity_definition / relation_label change survives a refresh.

export function useUpdateGraphEntity() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      patch,
    }: { id: string; patch: Parameters<typeof api.updateGraphEntity>[1] }) =>
      api.updateGraphEntity(id, patch),
    onMutate: async ({ id, patch }) => {
      const key = graphEntitiesKey();
      await qc.cancelQueries({ queryKey: key });
      const prev = qc.getQueryData<readonly GraphEntity[]>(key);
      if (prev) {
        qc.setQueryData<readonly GraphEntity[]>(
          key,
          prev.map((e) => (e.id === id ? { ...e, ...patch } : e)),
        );
      }
      return { prev };
    },
    onError: (_err, _vars, ctx) => {
      if (ctx?.prev) qc.setQueryData(graphEntitiesKey(), ctx.prev);
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['graph-entities'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

export function useUpdateGraphRelation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      patch,
    }: { id: string; patch: Parameters<typeof api.updateGraphRelation>[1] }) =>
      api.updateGraphRelation(id, patch),
    onMutate: async ({ id, patch }) => {
      const key = graphRelationsKey();
      await qc.cancelQueries({ queryKey: key });
      const prev = qc.getQueryData<readonly GraphRelation[]>(key);
      if (prev) {
        qc.setQueryData<readonly GraphRelation[]>(
          key,
          prev.map((r) => (r.id === id ? { ...r, ...patch } : r)),
        );
      }
      return { prev };
    },
    onError: (_err, _vars, ctx) => {
      if (ctx?.prev) qc.setQueryData(graphRelationsKey(), ctx.prev);
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['graph-relations'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

/**
 * Graph lifecycle mutations (M12 batch 3 backend wiring).
 *
 * Doctrine: no optimistic mutation on create/delete — the new
 * server-minted id (entity id, relation id) needs the real response
 * before the UI can address the row. We just invalidate the relevant
 * queries on settle so the next refetch picks up the change. The
 * caller wraps the call to surface the toast.
 */
export function useCreateGraphEntity() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Parameters<typeof api.createGraphEntity>[0]) =>
      api.createGraphEntity(body),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['graph-entities'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

export function useDeleteGraphEntity() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteGraphEntity(id),
    onMutate: async (id) => {
      // Optimistically prune the entity + any incident relation so
      // the canvas reflows immediately. Roll back on error.
      const entitiesKey = graphEntitiesKey();
      const relationsKey = graphRelationsKey();
      await qc.cancelQueries({ queryKey: entitiesKey });
      await qc.cancelQueries({ queryKey: relationsKey });
      const prevEntities =
        qc.getQueryData<readonly GraphEntity[]>(entitiesKey);
      const prevRelations =
        qc.getQueryData<readonly GraphRelation[]>(relationsKey);
      if (prevEntities) {
        qc.setQueryData<readonly GraphEntity[]>(
          entitiesKey,
          prevEntities.filter((e) => e.id !== id),
        );
      }
      if (prevRelations) {
        qc.setQueryData<readonly GraphRelation[]>(
          relationsKey,
          prevRelations.filter((r) => r.source !== id && r.target !== id),
        );
      }
      return { prevEntities, prevRelations };
    },
    onError: (_err, _vars, ctx) => {
      if (ctx?.prevEntities)
        qc.setQueryData(graphEntitiesKey(), ctx.prevEntities);
      if (ctx?.prevRelations)
        qc.setQueryData(graphRelationsKey(), ctx.prevRelations);
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['graph-entities'] });
      void qc.invalidateQueries({ queryKey: ['graph-relations'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

export function useCreateGraphRelation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Parameters<typeof api.createGraphRelation>[0]) =>
      api.createGraphRelation(body),
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['graph-relations'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

export function useDeleteGraphRelation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteGraphRelation(id),
    onMutate: async (id) => {
      const key = graphRelationsKey();
      await qc.cancelQueries({ queryKey: key });
      const prev = qc.getQueryData<readonly GraphRelation[]>(key);
      if (prev) {
        qc.setQueryData<readonly GraphRelation[]>(
          key,
          prev.filter((r) => r.id !== id),
        );
      }
      return { prev };
    },
    onError: (_err, _vars, ctx) => {
      if (ctx?.prev) qc.setQueryData(graphRelationsKey(), ctx.prev);
    },
    onSettled: () => {
      void qc.invalidateQueries({ queryKey: ['graph-relations'] });
      void qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

// Helper: unwrap a ListEnvelope into just the items array. Returns an empty
// readonly array while loading so tab props stay non-null. Errors are surfaced
// upstream — components are free to render their existing empty states.
export function unwrap<T>(data: { items: readonly T[] } | undefined): readonly T[] {
  return data?.items ?? [];
}

// ---------------------------------------------------------------------------
// Tag mutations (S4c slice 2)
//
// Every mutation invalidates ['tags'], ['activity'], and ['notifications'] —
// the WebUI tab queries refetch on next access, so the user sees the new
// tag state + the synthesized audit event + the unread notification badge
// without any manual refresh.
// ---------------------------------------------------------------------------

function invalidateTagSideEffects(qc: ReturnType<typeof useQueryClient>): void {
  qc.invalidateQueries({ queryKey: ['tags'] });
  qc.invalidateQueries({ queryKey: ['activity'] });
  qc.invalidateQueries({ queryKey: ['notifications'] });
  // Delete (untag/migrate), rename (edit), and approve all change the
  // tag-set displayed on documents — the backend cascades
  // [:TAGGED_WITH] edges, but the DocumentsTab keeps showing stale
  // chips until ['documents'] is refetched. Forgetting this here was
  // the cause of the 2026-06-07 "untag doesn't untag" report.
  qc.invalidateQueries({ queryKey: ['documents'] });
}

// The documents query is an infinite (cursor) query, so its cache is
// `{ pages: ListEnvelope<Document>[], pageParams }`, NOT a bare
// `{ items }`. Optimistic updates must map across every page's items —
// touching `old.items` directly silently no-ops (undefined) and was the
// cause of the RC-1 "mutation doesn't survive reload" regression.
type DocsPage = { items: readonly Document[]; [key: string]: unknown };
type InfiniteDocs = { pages: DocsPage[]; pageParams: unknown[] };

function mapDocumentsPages(
  old: InfiniteDocs | undefined,
  mapItems: (items: readonly Document[]) => readonly Document[],
): InfiniteDocs | undefined {
  if (!old?.pages) return old;
  return {
    ...old,
    pages: old.pages.map((p) => ({ ...p, items: mapItems(p.items) })),
  };
}

function applyDocumentTags(
  items: readonly Document[],
  targets: readonly string[],
  adds: readonly string[],
  removes: readonly string[],
): readonly Document[] {
  const targetSet = new Set(targets);
  const removeSet = new Set(removes);
  return items.map((doc) => {
    if (!targetSet.has(doc.doc_id)) return doc;
    const nextTags = Array.from(
      new Set([...doc.tags.filter((tag) => !removeSet.has(tag)), ...adds]),
    );
    return { ...doc, tags: nextTags };
  });
}

export function useRequestTag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Parameters<typeof api.requestTag>[0]) =>
      api.requestTag(body),
    onSuccess: () => invalidateTagSideEffects(qc),
  });
}

export function useApproveTag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, actor }: { name: string; actor?: string }) =>
      api.approveTag(name, actor),
    onSuccess: () => invalidateTagSideEffects(qc),
  });
}

export function useRejectTag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      reason,
      actor,
    }: {
      name: string;
      reason: string;
      actor?: string;
    }) => api.rejectTag(name, { reason, actor }),
    onSuccess: () => invalidateTagSideEffects(qc),
  });
}

export function useEditTag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      ...body
    }: { name: string } & Parameters<typeof api.editTag>[1]) =>
      api.editTag(name, body),
    onSuccess: () => invalidateTagSideEffects(qc),
  });
}

export function useDeprecateTag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      ...body
    }: { name: string } & Parameters<typeof api.deprecateTag>[1]) =>
      api.deprecateTag(name, body),
    onSuccess: () => invalidateTagSideEffects(qc),
  });
}

export function useUpdateTagSynonyms() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      aliases,
      actor,
    }: {
      name: string;
      aliases: readonly string[];
      actor?: string;
    }) => api.updateTagSynonyms(name, { aliases, actor }),
    onSuccess: () => invalidateTagSideEffects(qc),
  });
}

export function useDeleteTag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      ...body
    }: { name: string } & Parameters<typeof api.deleteTag>[1]) =>
      api.deleteTag(name, body),
    onSuccess: () => invalidateTagSideEffects(qc),
  });
}

/**
 * Delete one document via the shimmed DELETE /documents/{id} route.
 * The shim calls ``rag.adelete_by_doc_id`` which cascades to entities,
 * relations, chunks, vector embeddings — full removal from Memgraph.
 *
 * Doctrine: every UI-visible mutation persists.
 */
/**
 * Mark the targeted documents with the UI-only `_deleting` flag so the
 * row shows the "DELETING" badge instead of disappearing while the
 * server-side cascade runs. Returns the previous cache snapshot so
 * `onError` can roll back.
 */
function flagDocsAsDeleting(
  qc: ReturnType<typeof useQueryClient>,
  ids: readonly string[],
): { previous: ReadonlyArray<[unknown, unknown]> } {
  const idSet = new Set(ids);
  const previous = qc.getQueriesData<InfiniteDocs>({ queryKey: ['documents'] });
  qc.setQueriesData<InfiniteDocs>({ queryKey: ['documents'] }, (old) =>
    mapDocumentsPages(old, (items) =>
      items.map((d) => (idSet.has(d.doc_id) ? { ...d, _deleting: true } : d)),
    ),
  );
  return { previous };
}

function invalidateDeleteSideEffects(qc: ReturnType<typeof useQueryClient>) {
  // The bulk-delete cascade nukes documents, graph entities, graph
  // relations, and emits an audit event. Stale-cache on any of these
  // produces the "I deleted the doc but nothing changed in the Graph
  // tab" symptom (2026-06-08 prod report). Keep all four in sync.
  qc.invalidateQueries({ queryKey: ['documents'] });
  qc.invalidateQueries({ queryKey: ['graph-entities'] });
  qc.invalidateQueries({ queryKey: ['graph-relations'] });
  qc.invalidateQueries({ queryKey: ['activity'] });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (docId: string) => api.deleteDocument(docId),
    onMutate: async (docId) => {
      await qc.cancelQueries({ queryKey: ['documents'] });
      return flagDocsAsDeleting(qc, [docId]);
    },
    onError: (_err, _docId, ctx) => {
      ctx?.previous.forEach(([key, data]) =>
        qc.setQueryData(key as readonly unknown[], data),
      );
    },
    onSettled: () => invalidateDeleteSideEffects(qc),
  });
}

export function useBulkDeleteDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Parameters<typeof api.bulkDeleteDocuments>[0]) =>
      api.bulkDeleteDocuments(body),
    onMutate: async (body) => {
      await qc.cancelQueries({ queryKey: ['documents'] });
      return flagDocsAsDeleting(qc, body.doc_ids);
    },
    onError: (_err, _body, ctx) => {
      ctx?.previous.forEach(([key, data]) =>
        qc.setQueryData(key as readonly unknown[], data),
      );
    },
    onSettled: () => invalidateDeleteSideEffects(qc),
  });
}

/**
 * Upload one file to LightRAG (multipart). Returns the InsertResponse with
 * status and track_id. Use `useUploadDocumentsBatch` for multi-file drops so
 * concurrency and cache invalidation stay bounded.
 */
export function useUploadDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => api.uploadDocument(file),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] });
      qc.invalidateQueries({ queryKey: ['pipeline_status'] });
    },
  });
}

export function useUploadDocumentsBatch() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (uploads: readonly UploadBatchItem[]) =>
      mapSettledWithConcurrency(
        uploads,
        DEFAULT_UPLOAD_CONCURRENCY,
        (item) => {
          const upload = normalizeUploadInput(item);
          return api.uploadDocument(upload.file, {
            classification: upload.classification,
            ragEngine: upload.ragEngine,
          });
        },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] });
      qc.invalidateQueries({ queryKey: ['pipeline_status'] });
    },
  });
}

/**
 * Persist a tag mutation on N documents (single doc = N=1).
 *
 * Doctrine: a tag is a Memgraph node attribute on
 * DocStatus_{workspace} storage label. Optimistic UI is no longer acceptable —
 * every retag MUST hit the backend so a refresh shows the new
 * state. On success: invalidate ['documents'] (cards refresh),
 * ['activity'] (audit feed picks up the doc-retagged events),
 * ['notifications'] (the operator's bell badge increments).
 *
 * The server may return ``failed: [doc_id, ...]`` for stale UI
 * selections (doc deleted between the user opening the modal and
 * submitting). Callers should surface this in a toast so the
 * operator knows the action didn't fully apply.
 */
export function useBulkRetagDocuments() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Parameters<typeof api.bulkRetagDocuments>[0]) =>
      api.bulkRetagDocuments(body),
    onMutate: async (body) => {
      await qc.cancelQueries({ queryKey: ['documents'] });
      const previousDocuments = qc.getQueriesData<InfiniteDocs>({
        queryKey: ['documents'],
      });
      qc.setQueriesData<InfiniteDocs>({ queryKey: ['documents'] }, (old) =>
        mapDocumentsPages(old, (items) =>
          applyDocumentTags(items, body.targets, body.adds, body.removes),
        ),
      );
      return { previousDocuments };
    },
    onError: (_err, _body, ctx) => {
      ctx?.previousDocuments.forEach(([queryKey, data]) => {
        qc.setQueryData(queryKey, data);
      });
    },
    onSettled: async () => {
      await Promise.all([
        qc.invalidateQueries({ queryKey: ['documents'] }),
        qc.invalidateQueries({ queryKey: ['tags'] }),
        qc.invalidateQueries({ queryKey: ['activity'] }),
        qc.invalidateQueries({ queryKey: ['notifications'] }),
      ]);
    },
  });
}

/**
 * Import a JSON taxonomy via POST /tags/categories/_import. Server-side
 * validation matches docs/templates/twin-categories.schema.json — a
 * 400 surfaces as ApiError with the validation message. On success,
 * we invalidate both ['tag-categories'] (sidebar refreshes) and
 * ['tags'] (the existing tags' category labels may now point at
 * a renamed/removed category that the UI should re-render against).
 */
export function useImportCategories() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Parameters<typeof api.importCategories>[0]) =>
      api.importCategories(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tag-categories'] });
      qc.invalidateQueries({ queryKey: ['tags'] });
      qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

// Helper: split a useDocuments() result into shape the DocumentsTab expects
// (an array). DocumentsTab currently takes `docs: readonly Document[]`.
export function asDocuments(
  data: { items: readonly Document[] } | undefined,
): readonly Document[] {
  return data?.items ?? [];
}
