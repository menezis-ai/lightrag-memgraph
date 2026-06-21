import type { Dispatch, SetStateAction } from 'react';
import { setActiveFolder } from '../api/client';
import type { Document } from '../types/document';
import { queryClient } from './queryClient';
import { FOLDER_STORAGE_KEY, writeUiPreference } from './uiPreferences';

export interface DetailRequest {
  doc?: string;
  source?: string;
  chunk?: string;
}

interface UseAppNavigationOptions {
  setClearedNotificationIds: Dispatch<SetStateAction<ReadonlySet<string>>>;
  setDetailChunkId: Dispatch<SetStateAction<string | null>>;
  setDetailDoc: Dispatch<SetStateAction<Document | null>>;
  setDetailRequest: Dispatch<SetStateAction<DetailRequest | null>>;
  setFolderState: Dispatch<SetStateAction<string>>;
  setReadNotificationIds: Dispatch<SetStateAction<ReadonlySet<string>>>;
  setReadSourceDoc: Dispatch<SetStateAction<Document | null>>;
  setRetagBulk: Dispatch<SetStateAction<readonly Document[] | null>>;
  setRetagDoc: Dispatch<SetStateAction<Document | null>>;
  setTab: Dispatch<SetStateAction<string>>;
}

export function useAppNavigation({
  setClearedNotificationIds,
  setDetailChunkId,
  setDetailDoc,
  setDetailRequest,
  setFolderState,
  setReadNotificationIds,
  setReadSourceDoc,
  setRetagBulk,
  setRetagDoc,
  setTab,
}: UseAppNavigationOptions) {
  const onNavigate = (nextTab: string, params?: Record<string, string>) => {
    const search = new URLSearchParams(window.location.search);
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
      });
    } else {
      setDetailRequest(null);
    }
    const qs = search.toString();
    window.history.replaceState(
      null,
      '',
      window.location.pathname + (qs ? '?' + qs : ''),
    );
    setTab(nextTab);
  };

  const onSwitchFolder = (nextFolder: string) => {
    window.history.replaceState(null, '', window.location.pathname);
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
