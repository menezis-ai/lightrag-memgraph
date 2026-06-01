/**
 * TanStack Query hooks — one hook per WebUI tab's data dependency.
 *
 * All hooks return the raw fetch payload + the usual `{ isLoading, isError,
 * error }` triplet. Stale time defaults to 60s so tab switches don't refetch
 * needlessly. Mutation hooks live next to their resource (notifications) and
 * use `queryClient.invalidateQueries(['notifications'])` to refresh.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from './resources';
import type { Document } from '../types/document';

const DEFAULTS = { staleTime: 60_000 } as const;

export function useDocuments(query: { status?: string; q?: string; tag?: string } = {}) {
  return useQuery({
    queryKey: ['documents', query] as const,
    queryFn: ({ signal }) => api.listDocuments(query, { signal }),
    ...DEFAULTS,
  });
}

export function useWorkspaces() {
  return useQuery({
    queryKey: ['workspaces'] as const,
    queryFn: ({ signal }) => api.listWorkspaces({ signal }),
    ...DEFAULTS,
  });
}

export function useNotifications() {
  return useQuery({
    queryKey: ['notifications'] as const,
    queryFn: ({ signal }) => api.listNotifications({ signal }),
    ...DEFAULTS,
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

export function useThesaurus() {
  return useQuery({
    queryKey: ['thesaurus'] as const,
    queryFn: ({ signal }) => api.listThesaurus({ signal }),
    ...DEFAULTS,
  });
}

export function useTags() {
  return useQuery({
    queryKey: ['tags'] as const,
    queryFn: ({ signal }) => api.listTags({ signal }),
    ...DEFAULTS,
  });
}

export function useTagCategories() {
  return useQuery({
    queryKey: ['tag-categories'] as const,
    queryFn: ({ signal }) => api.listTagCategories({ signal }),
    ...DEFAULTS,
  });
}

export function useActivity(
  query: { range?: string; kind?: string; sev?: string; actor?: string; q?: string } = {},
) {
  return useQuery({
    queryKey: ['activity', query] as const,
    queryFn: ({ signal }) => api.listActivity(query, { signal }),
    ...DEFAULTS,
  });
}

export function useOpenApi() {
  return useQuery({
    queryKey: ['openapi'] as const,
    queryFn: ({ signal }) => api.getOpenApi({ signal }),
    ...DEFAULTS,
  });
}

export function useGraphEntities() {
  return useQuery({
    queryKey: ['graph-entities'] as const,
    queryFn: ({ signal }) => api.listGraphEntities({}, { signal }),
    ...DEFAULTS,
  });
}

export function useGraphRelations() {
  return useQuery({
    queryKey: ['graph-relations'] as const,
    queryFn: ({ signal }) => api.listGraphRelations({}, { signal }),
    ...DEFAULTS,
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
 * Doctrine: every UI-visible mutation persists. Bulk-delete is
 * implemented in the host (Promise.allSettled over N of these) to
 * keep the server endpoint surface narrow.
 */
export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (docId: string) => api.deleteDocument(docId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] });
      qc.invalidateQueries({ queryKey: ['activity'] });
    },
  });
}

/**
 * Upload one file to LightRAG (multipart). Returns the InsertResponse
 * with status and track_id. Callers chain N of these for bulk upload
 * and invalidate the documents query once they all settle.
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

/**
 * Persist a tag mutation on N documents (single doc = N=1).
 *
 * Doctrine: a tag is a Memgraph node attribute on
 * DocStatus_{workspace}. Optimistic UI is no longer acceptable —
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
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['documents'] });
      qc.invalidateQueries({ queryKey: ['activity'] });
      qc.invalidateQueries({ queryKey: ['notifications'] });
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
