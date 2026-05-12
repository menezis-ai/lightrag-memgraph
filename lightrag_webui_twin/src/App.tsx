/**
 * App shell — wires Topbar + tabs + modals against the TanStack Query layer.
 *
 * Data flow (S4a):
 *   - Each resource has a typed query hook (`useDocuments`, `useTags`, ...)
 *     that hits `/documents`, `/tags`, etc. via `apiFetch`.
 *   - In dev, MSW intercepts those fetches and answers from the fixtures —
 *     the contract template lives in `src/fixtures/`.
 *   - Each `useQuery` is seeded with `initialData = FIXTURE` so the first
 *     paint is instant. Fetched data replaces the fixture as soon as the
 *     query resolves (background revalidation pattern).
 *   - Components keep their prop-driven signature so unit tests pass arrays
 *     directly without a QueryClient wrapper.
 *
 * Env switches:
 *   - VITE_USE_MSW=false   → skip the MSW worker; fetches hit VITE_API_BASE_URL.
 *   - VITE_API_BASE_URL=…  → real backend origin.
 *   - VITE_AUTH_TOKEN=…    → bearer attached on every fetch.
 */

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { ActivityTab } from './components/ActivityTab';
import { AddSourceModal, type AddSourceAction } from './components/AddSourceModal';
import { ApiTab } from './components/ApiTab';
import { DocumentsTab } from './components/DocumentsTab';
import { GraphTab } from './components/GraphTab';
import { RetagModal, type RetagAction } from './components/RetagModal';
import { RetrievalTab } from './components/RetrievalTab';
import { TagsTab, type TagApproveAction } from './components/TagsTab';
import type { TagActionCommit } from './components/TagActionModal';
import { ToastViewport } from './components/ToastViewport';
import { Topbar } from './components/Topbar';
import {
  TweakSection,
  TweakSlider,
  TweakToggle,
  TweaksPanel,
  useTweaks,
} from './components/TweaksPanel';
import {
  useActivity,
  useDocuments,
  useGraphEntities,
  useGraphRelations,
  useNotifications,
  useOpenApi,
  useTagCategories,
  useTags,
  useThesaurus,
  useWorkspaces,
} from './api/queries';
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
  TAG_CATEGORY_FIXTURES,
  TAG_FIXTURES,
  THESAURUS_FIXTURES,
  WORKSPACE_FIXTURES,
  makeSampleThreads,
} from './fixtures';
import type { Document } from './types/document';
import type { TagCurrentUser } from './types/tag';
import type { Theme } from './types/topbar';
import type { Toast } from './types/toast';

const CURRENT_USER: TagCurrentUser = {
  name: 'claire.benoit',
  palier: 3,
  role: 'admin / steward',
};

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function AppShell() {
  const [tab, setTab] = useState('documents');
  const [theme, setTheme] = useState<Theme>('light');
  const [workspace, setWorkspace] = useState('cib');
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [tweaksOpen, setTweaksOpen] = useState(false);
  const [tweaks, setTweak] = useTweaks({
    density: 'regular' as 'compact' | 'regular' | 'comfy',
    fontSize: 14,
    liveActivity: true,
  });

  // Modal state
  const [addOpen, setAddOpen] = useState(false);
  const [retagDoc, setRetagDoc] = useState<Document | null>(null);
  const [retagBulk, setRetagBulk] = useState<readonly Document[] | null>(null);

  // Data — every tab is backed by a query, seeded with the corresponding
  // fixture so first paint is instant even if the worker is still booting.
  const docs = useDocuments();
  const workspaces = useWorkspaces();
  const notificationsQ = useNotifications();
  const thesaurus = useThesaurus();
  const tags = useTags();
  const tagCategories = useTagCategories();
  const activity = useActivity();
  const openApi = useOpenApi();
  const graphEntities = useGraphEntities();
  const graphRelations = useGraphRelations();

  // Notifications carry mutable client state (read/cleared) on top of the
  // query data, so we mirror them locally and use the query result as the
  // source of truth on first load + refetch.
  const [notifications, setNotifications] = useState([...NOTIFICATION_FIXTURES]);
  useEffect(() => {
    if (notificationsQ.data) setNotifications([...notificationsQ.data]);
  }, [notificationsQ.data]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const unreadCount = notifications.filter((n) => !n.read).length;
  const workspaceList = workspaces.data ?? WORKSPACE_FIXTURES;
  const kbName = workspaceList.find((w) => w.id === workspace)?.kb ?? '';

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

  const onTagApprove = (action: TagApproveAction) => {
    pushToast({
      kind: 'done',
      title: 'Tag',
      tagname: action.tag.tag,
      titleSuffix: 'approved',
      sub: 'Added to thesaurus · tag.approved emitted',
      undo: { tag: action.tag.tag },
    });
  };

  const onTagCommit = (commit: TagActionCommit) => {
    const tagname = commit.tag?.tag ?? commit.name ?? '';
    const verbMap: Record<TagActionCommit['kind'], string> = {
      edit: 'definition updated',
      suggest: 'edit suggested',
      synonyms: 'synonyms updated',
      deprecate: 'deprecated',
      delete:
        commit.migrate?.strategy === 'migrate'
          ? `migrated to ${commit.migrate.to ?? ''}`
          : 'deleted (docs untagged)',
      reject: 'rejected',
      'edit-approve': 'approved (edited)',
      request: 'requested for review',
    };
    const subMap: Record<TagActionCommit['kind'], string> = {
      edit: 'tag.edited emitted to Activity',
      suggest: 'Awaiting palier-3 review',
      synonyms: 'Query rewriting refreshed at gateway',
      deprecate: `${commit.tag?.sources_count ?? 0} docs flagged · tag.deprecated emitted`,
      delete: `${commit.tag?.sources_count ?? 0} docs updated · tag.deleted emitted`,
      reject: commit.reason ?? 'tag.rejected emitted · author notified',
      'edit-approve': 'Added to thesaurus · tag.approved emitted',
      request: 'Queued for palier-3 approval · tag.request_new emitted',
    };
    pushToast({
      kind: 'done',
      title: 'Tag',
      tagname,
      titleSuffix: verbMap[commit.kind],
      sub: subMap[commit.kind],
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

  // Resolved props — fall back to the local fixtures while the first fetch is
  // in flight so the UI never shows an empty shell on cold start.
  const docList = docs.data?.items ?? DOCUMENT_FIXTURES;
  const thesaurusList = thesaurus.data ?? THESAURUS_FIXTURES;
  const tagList = tags.data ?? TAG_FIXTURES;
  const tagCategoryList = tagCategories.data ?? TAG_CATEGORY_FIXTURES;
  const activityEvents = activity.data?.items ?? ACTIVITY_FIXTURES;
  const activityNow = activity.data?.nowMs ?? ACTIVITY_NOW_MS;
  const openApiGroups = openApi.data?.groups ?? OPENAPI_GROUPS;
  const apiVersion = openApi.data?.version ?? API_VERSION;
  const graphEntityList = graphEntities.data ?? GRAPH_ENTITY_FIXTURES;
  const graphRelationList = graphRelations.data ?? GRAPH_RELATION_FIXTURES;

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
        workspaces={workspaceList}
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
            docs={docList}
            thesaurus={thesaurusList}
            onOpenAdd={() => setAddOpen(true)}
            onOpenRetag={(d) => setRetagDoc(d)}
            onOpenBulkRetag={(ds) => setRetagBulk(ds)}
            onAddToast={onAddToast}
          />
        )}
        {tab === 'retrieval' && (
          <RetrievalTab
            thesaurus={thesaurusList}
            answerTokens={ANSWER_TOKENS_FIXTURE}
            answerSources={RETRIEVAL_SOURCES_FIXTURE}
            initialThreads={makeSampleThreads()}
          />
        )}
        {tab === 'activity' && (
          <ActivityTab
            events={activityEvents}
            nowMs={activityNow}
            density={tweaks.density === 'compact' ? 'compact' : 'comfortable'}
            live={tweaks.liveActivity}
            onPushToast={pushToast}
            onNavigate={onNavigate}
          />
        )}
        {tab === 'graph' && (
          <GraphTab
            entities={graphEntityList}
            relations={graphRelationList}
            onNavigate={onNavigate}
          />
        )}
        {tab === 'api' && (
          <ApiTab
            apiVersion={apiVersion}
            groups={openApiGroups}
            servers={API_SERVERS}
            baseUrl={API_BASE_URL}
          />
        )}
        {tab === 'tags' && (
          <TagsTab
            tags={tagList}
            categories={tagCategoryList}
            currentUser={CURRENT_USER}
            onApprove={onTagApprove}
            onCommit={onTagCommit}
            onNavigate={onNavigate}
          />
        )}
      </main>

      <AddSourceModal
        open={addOpen}
        thesaurus={thesaurusList}
        formatCategories={FORMAT_CATEGORY_FIXTURES}
        onClose={() => setAddOpen(false)}
        onSubmit={onAddSourceSubmit}
      />
      <RetagModal
        open={retagDoc !== null || retagBulk !== null}
        doc={retagDoc}
        docs={retagBulk ?? undefined}
        thesaurus={thesaurusList}
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
      <button
        type="button"
        className="tweaks-fab"
        onClick={() => setTweaksOpen((o) => !o)}
        aria-label="Open dev tweaks"
        title="Dev tweaks (T)"
        style={{
          position: 'fixed',
          right: 12,
          top: 60,
          zIndex: 2147483645,
          width: 28,
          height: 28,
          borderRadius: 6,
          border: '0.5px solid var(--color-border-secondary, #ddd)',
          background: 'var(--color-background-primary, #fff)',
          color: 'var(--color-text-secondary, #555)',
          cursor: 'pointer',
          fontSize: 14,
        }}
      >
        ⚙
      </button>
      <TweaksPanel open={tweaksOpen} onClose={() => setTweaksOpen(false)} title="Twin tweaks">
        <TweakSection label="Activity feed">
          <TweakToggle
            label="Live polling"
            value={tweaks.liveActivity}
            onChange={(v) => setTweak('liveActivity', v)}
          />
        </TweakSection>
        <TweakSection label="Display">
          <TweakSlider
            label="Font size"
            value={tweaks.fontSize}
            min={11}
            max={18}
            unit="px"
            onChange={(v) => setTweak('fontSize', v)}
          />
        </TweakSection>
      </TweaksPanel>
    </>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
    </QueryClientProvider>
  );
}

export default App;
