/**
 * App shell — wires Topbar + DocumentsTab + RetrievalTab + Toast + Modals
 * against the local fixtures so `bun dev` shows the full Twin WebUI shell.
 *
 * In sprint 4 the fixtures will be swapped out for real fetchers against
 * LightRAG's `/documents`, `/retrieval`, `/notifications` endpoints (gated
 * by the backend phase-1 contract).
 */

import { useEffect, useState } from 'react';
import { ActivityTab } from './components/ActivityTab';
import { AddSourceModal, type AddSourceAction } from './components/AddSourceModal';
import { ApiTab } from './components/ApiTab';
import { DocumentsTab } from './components/DocumentsTab';
import { GraphTab } from './components/GraphTab';
import { RetagModal, type RetagAction } from './components/RetagModal';
import { RetrievalTab } from './components/RetrievalTab';
import { ToastViewport } from './components/ToastViewport';
import { Topbar } from './components/Topbar';
import {
  ACTIVITY_FIXTURES,
  ACTIVITY_NOW_MS,
  ANSWER_TOKENS_FIXTURE,
  API_BASE_URL,
  API_SERVERS,
  API_VERSION,
  DOCUMENT_FIXTURES,
  FORMAT_CATEGORY_FIXTURES,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
  NOTIFICATION_FIXTURES,
  OPENAPI_GROUPS,
  RETRIEVAL_SOURCES_FIXTURE,
  THESAURUS_FIXTURES,
  WORKSPACE_FIXTURES,
  makeSampleThreads,
} from './fixtures';
import type { Document } from './types/document';
import type { Theme } from './types/topbar';
import type { Toast } from './types/toast';

function App() {
  const [tab, setTab] = useState('documents');
  const [theme, setTheme] = useState<Theme>('light');
  const [workspace, setWorkspace] = useState('cib');
  const [notifications, setNotifications] = useState([...NOTIFICATION_FIXTURES]);
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Modal state
  const [addOpen, setAddOpen] = useState(false);
  const [retagDoc, setRetagDoc] = useState<Document | null>(null);
  const [retagBulk, setRetagBulk] = useState<readonly Document[] | null>(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const unreadCount = notifications.filter((n) => !n.read).length;
  const kbName = WORKSPACE_FIXTURES.find((w) => w.id === workspace)?.kb ?? '';

  const pushToast = (t: Omit<Toast, 'id'>) => {
    const id = `tst_${Date.now()}_${Math.random().toString(16).slice(2, 6)}`;
    setToasts((ts) => [...ts, { id, ...t }]);
  };

  const onAddToast = (title: string, sub?: string) =>
    pushToast({ kind: 'done', title, sub });

  const onRetagSubmit = (action: RetagAction) => {
    const verb = action.adds.length > 0 ? 'applied' : 'removed';
    const sample = action.adds[0] ?? action.removes[0];
    pushToast({
      kind: 'done',
      title: 'Tag',
      tagname: sample,
      titleSuffix: action.bulk
        ? `${verb} to ${action.targets.length} sources`
        : verb,
      sub: action.primary.source,
      undo: { adds: action.adds, removes: action.removes },
    });
  };

  const onAddSourceSubmit = (action: AddSourceAction) => {
    pushToast({
      kind: 'done',
      title: 'Sources queued for ingestion',
      sub: `${action.readyCount} added${action.tags.length ? ' · tags: ' + action.tags.join(', ') : ''}`,
    });
  };

  const onNavigate = (nextTab: string, params?: Record<string, string>) => {
    const search = new URLSearchParams(window.location.search);
    Array.from(search.keys()).forEach((k) => search.delete(k));
    if (params) {
      Object.entries(params).forEach(([k, v]) => search.set(k, v));
    }
    const qs = search.toString();
    window.history.replaceState(
      null,
      '',
      window.location.pathname + (qs ? '?' + qs : ''),
    );
    setTab(nextTab);
  };

  return (
    <>
      <Topbar
        tab={tab}
        onTab={setTab}
        theme={theme}
        onTheme={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
        workspace={workspace}
        kbName={kbName}
        onSwitchWorkspace={(w) => setWorkspace(w.id)}
        workspaces={WORKSPACE_FIXTURES}
        notifications={notifications}
        unreadCount={unreadCount}
        onMarkAllRead={() =>
          setNotifications((ns) => ns.map((n) => ({ ...n, read: true })))
        }
        onClearNotifications={() => setNotifications([])}
      />
      <main>
        {tab === 'documents' && (
          <DocumentsTab
            docs={DOCUMENT_FIXTURES}
            thesaurus={THESAURUS_FIXTURES}
            onOpenAdd={() => setAddOpen(true)}
            onOpenRetag={(d) => setRetagDoc(d)}
            onOpenBulkRetag={(ds) => setRetagBulk(ds)}
            onAddToast={onAddToast}
          />
        )}
        {tab === 'retrieval' && (
          <RetrievalTab
            thesaurus={THESAURUS_FIXTURES}
            answerTokens={ANSWER_TOKENS_FIXTURE}
            answerSources={RETRIEVAL_SOURCES_FIXTURE}
            initialThreads={makeSampleThreads()}
          />
        )}
        {tab === 'activity' && (
          <ActivityTab
            events={ACTIVITY_FIXTURES}
            nowMs={ACTIVITY_NOW_MS}
            onPushToast={pushToast}
            onNavigate={onNavigate}
          />
        )}
        {tab === 'graph' && (
          <GraphTab
            entities={GRAPH_ENTITY_FIXTURES}
            relations={GRAPH_RELATION_FIXTURES}
            onNavigate={onNavigate}
          />
        )}
        {tab === 'api' && (
          <ApiTab
            apiVersion={API_VERSION}
            groups={OPENAPI_GROUPS}
            servers={API_SERVERS}
            baseUrl={API_BASE_URL}
          />
        )}
        {tab === 'tags' && (
          <div className="p-6 text-sm text-text-secondary">
            Tab "tags" — coming in S3 slice 2.
          </div>
        )}
      </main>

      <AddSourceModal
        open={addOpen}
        thesaurus={THESAURUS_FIXTURES}
        formatCategories={FORMAT_CATEGORY_FIXTURES}
        onClose={() => setAddOpen(false)}
        onSubmit={onAddSourceSubmit}
      />
      <RetagModal
        open={retagDoc !== null || retagBulk !== null}
        doc={retagDoc}
        docs={retagBulk ?? undefined}
        thesaurus={THESAURUS_FIXTURES}
        onClose={() => {
          setRetagDoc(null);
          setRetagBulk(null);
        }}
        onSubmit={onRetagSubmit}
      />
      <ToastViewport
        toasts={toasts}
        onUndo={(t) =>
          setToasts((ts) => ts.filter((x) => x.id !== t.id))
        }
        onDismiss={(t) =>
          setToasts((ts) => ts.filter((x) => x.id !== t.id))
        }
      />
    </>
  );
}

export default App;
