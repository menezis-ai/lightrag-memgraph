/**
 * Typed per-resource fetchers — one function per backend phase-1 endpoint.
 *
 * Each function signature mirrors the corresponding fixture shape so the
 * backend contract is self-documenting from the WebUI side. Listing endpoints
 * return `{ items, total }` envelopes so pagination + filtering can grow
 * without breaking callers.
 */

import { apiFetch, type ApiRequestInit } from './client';
import type { ActivityEvent } from '../types/activity';
import type { OpenApiGroup } from '../types/api';
import type { Document } from '../types/document';
import type { GraphEntity, GraphRelation } from '../types/graph';
import type { Notification, Workspace } from '../types/topbar';
import type { TagCategory, TagEntry } from '../types/tag';
import type { ThesaurusEntry } from '../types/thesaurus';

export interface ListEnvelope<T> {
  items: readonly T[];
  total: number;
}

export interface DocumentsQuery {
  status?: string;
  q?: string;
  tag?: string;
  cursor?: string;
}

export const api = {
  // Documents
  listDocuments: (q: DocumentsQuery = {}, init?: ApiRequestInit) =>
    apiFetch<ListEnvelope<Document>>('/documents', { ...init, query: { ...q } }),

  // Workspaces / notifications / topbar
  listWorkspaces: (init?: ApiRequestInit) =>
    apiFetch<readonly Workspace[]>('/workspaces', init),
  listNotifications: (init?: ApiRequestInit) =>
    apiFetch<readonly Notification[]>('/notifications', init),
  markAllNotificationsRead: (init?: ApiRequestInit) =>
    apiFetch<{ ok: true }>('/notifications/read-all', { ...init, method: 'POST' }),
  clearNotifications: (init?: ApiRequestInit) =>
    apiFetch<{ ok: true }>('/notifications', { ...init, method: 'DELETE' }),

  // Thesaurus + tag governance
  listThesaurus: (init?: ApiRequestInit) =>
    apiFetch<readonly ThesaurusEntry[]>('/thesaurus', init),
  listTags: (init?: ApiRequestInit) => apiFetch<readonly TagEntry[]>('/tags', init),
  listTagCategories: (init?: ApiRequestInit) =>
    apiFetch<readonly TagCategory[]>('/tags/categories', init),

  // Activity audit feed
  listActivity: (
    q: { range?: string; kind?: string; sev?: string; actor?: string; q?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<ListEnvelope<ActivityEvent> & { nowMs?: number }>('/activity', {
      ...init,
      query: { ...q },
    }),

  // OpenAPI surface (proxied from the underlying LightRAG server)
  getOpenApi: (init?: ApiRequestInit) =>
    apiFetch<{ groups: readonly OpenApiGroup[]; version: string }>('/openapi', init),

  // Knowledge graph teaser
  listGraphEntities: (
    q: { workspace?: string; type?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<readonly GraphEntity[]>('/graph/entities', { ...init, query: { ...q } }),
  listGraphRelations: (
    q: { workspace?: string } = {},
    init?: ApiRequestInit,
  ) =>
    apiFetch<readonly GraphRelation[]>('/graph/relations', { ...init, query: { ...q } }),
};

export type ApiClient = typeof api;
