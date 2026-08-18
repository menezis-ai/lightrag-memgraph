import type { Dispatch, SetStateAction } from 'react';
import { setActiveFolder } from '../api/client';
import type { Document } from '../types/document';
import {
  DOCUMENTS_STATUS_FILTERS,
  type DocumentsStatusFilterKey,
} from './appConstants';
import { queryClient } from './queryClient';
import { FOLDER_STORAGE_KEY, writeUiPreference } from './uiPreferences';

export interface DetailRequest {
  doc?: string;
  source?: string;
  chunk?: string;
  /** Paragraph-anchor offsets for the requested chunk (URL param strings). */
  anchorStart?: string;
  anchorEnd?: string;
}

interface UseAppNavigationOptions {
  setClearedNotificationIds: Dispatch<SetStateAction<ReadonlySet<string>>>;
  setDetailChunkId: Dispatch<SetStateAction<string | null>>;
  setDetailDoc: Dispatch<SetStateAction<Document | null>>;
  setDetailRequest: Dispatch<SetStateAction<DetailRequest | null>>;
  setDocumentsSearch?: (value: string) => void;
  setDocumentsSourceFilters?: (value: readonly string[]) => void;
  setDocumentsStatusFilter?: (value: DocumentsStatusFilterKey) => void;
  setDocumentsTagFilters?: (value: readonly string[]) => void;
  setFolderState: Dispatch<SetStateAction<string>>;
  setReadNotificationIds: Dispatch<SetStateAction<ReadonlySet<string>>>;
  setReadSourceDoc: Dispatch<SetStateAction<Document | null>>;
  setRetagBulk: Dispatch<SetStateAction<readonly Document[] | null>>;
  setRetagDoc: Dispatch<SetStateAction<Document | null>>;
  setTab: Dispatch<SetStateAction<string>>;
}

function splitCsvParam(value: string | undefined): readonly string[] {
  return (value ?? '')
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);
}

function documentsStatusFromParam(
  value: string | undefined,
): DocumentsStatusFilterKey {
  return DOCUMENTS_STATUS_FILTERS.includes(value as DocumentsStatusFilterKey)
    ? (value as DocumentsStatusFilterKey)
    : 'all';
}

export function useAppNavigation({
  setClearedNotificationIds,
  setDetailChunkId,
  setDetailDoc,
  setDetailRequest,
  setDocumentsSearch,
  setDocumentsSourceFilters,
  setDocumentsStatusFilter,
  setDocumentsTagFilters,
  setFolderState,
  setReadNotificationIds,
  setReadSourceDoc,
  setRetagBulk,
  setRetagDoc,
  setTab,
}: UseAppNavigationOptions) {
  const onNavigate = (nextTab: string, params?: Record<string, string>) => {
    const search = new URLSearchParams(globalThis.location.search);
    Array.from(search.keys()).forEach((key) => search.delete(key));
    if (params) {
      Object.entries(params).forEach(([key, value]) => search.set(key, value));
    }
    if (nextTab === 'documents' && (params?.doc || params?.source)) {
      setDetailDoc(null);
      setDetailChunkId(null);
      setDetailRequest({
        doc: params.doc,
        source: params.source,
        chunk: params.chunk,
        anchorStart: params.astart,
        anchorEnd: params.aend,
      });
    } else {
      setDetailRequest(null);
    }
    if (nextTab === 'documents') {
      setDocumentsStatusFilter?.(documentsStatusFromParam(params?.status));
      setDocumentsSearch?.(params?.q ?? '');
      setDocumentsTagFilters?.(splitCsvParam(params?.tag));
      setDocumentsSourceFilters?.(
        splitCsvParam(params?.source ?? params?.doc),
      );
    }
    const qs = search.toString();
    globalThis.history.replaceState(
      null,
      '',
      globalThis.location.pathname + (qs ? '?' + qs : ''),
    );
    setTab(nextTab);
  };

  const onSwitchFolder = (nextFolder: string) => {
    globalThis.history.replaceState(null, '', globalThis.location.pathname);
    setActiveFolder(nextFolder);
    writeUiPreference(FOLDER_STORAGE_KEY, nextFolder);
    setFolderState(nextFolder);
    setReadNotificationIds(new Set());
    setClearedNotificationIds(new Set());
    setDetailDoc(null);
    setDetailChunkId(null);
    setDetailRequest(null);
    setReadSourceDoc(null);
    setRetagDoc(null);
    setRetagBulk(null);
    setDocumentsStatusFilter?.('all');
    setDocumentsSearch?.('');
    setDocumentsTagFilters?.([]);
    setDocumentsSourceFilters?.([]);
    void Promise.all([
      queryClient.invalidateQueries({ queryKey: ['documents'] }),
      queryClient.invalidateQueries({ queryKey: ['pipeline_status'] }),
      queryClient.invalidateQueries({ queryKey: ['tags'] }),
      queryClient.invalidateQueries({ queryKey: ['tag-categories'] }),
      queryClient.invalidateQueries({ queryKey: ['activity'] }),
      queryClient.invalidateQueries({ queryKey: ['notifications'] }),
      queryClient.invalidateQueries({ queryKey: ['graph-entities'] }),
      queryClient.invalidateQueries({ queryKey: ['graph-relations'] }),
    ]);
  };

  return { onNavigate, onSwitchFolder };
}
