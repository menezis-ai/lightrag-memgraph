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

// Helper: split a useDocuments() result into shape the DocumentsTab expects
// (an array). DocumentsTab currently takes `docs: readonly Document[]`.
export function asDocuments(
  data: { items: readonly Document[] } | undefined,
): readonly Document[] {
  return data?.items ?? [];
}
