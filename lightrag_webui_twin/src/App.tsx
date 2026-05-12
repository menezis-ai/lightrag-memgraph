/**
 * App shell — minimal scaffold for sprint 1.
 *
 * Wires the Topbar with fixture data so the dev server (`bun dev`) shows a
 * live preview of the proto's header. Tabs, popovers, theme toggle all work
 * against in-memory state — no network calls yet, those land in sprint 4
 * when phase-1 backend is ready.
 */

import { useState, useEffect } from 'react';
import { Topbar } from './components/Topbar';
import { TagChip } from './components/TagChip';
import { DOCUMENT_FIXTURES, NOTIFICATION_FIXTURES, WORKSPACE_FIXTURES } from './fixtures';
import type { Theme } from './types/topbar';

function App() {
  const [tab, setTab] = useState('documents');
  const [theme, setTheme] = useState<Theme>('light');
  const [workspace, setWorkspace] = useState('cib');
  const [notifications, setNotifications] = useState([...NOTIFICATION_FIXTURES]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const unreadCount = notifications.filter((n) => !n.read).length;
  const kbName = WORKSPACE_FIXTURES.find((w) => w.id === workspace)?.kb ?? '';

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
      <main className="p-6">
        <p className="text-sm text-text-secondary">
          Sprint 1 scaffold — active tab: <code className="font-mono">{tab}</code> · workspace:{' '}
          <code className="font-mono">{workspace}</code> · theme:{' '}
          <code className="font-mono">{theme}</code>
        </p>
        <h2 className="mt-4 text-xl font-medium">Tag chips (sample)</h2>
        <div className="mt-2 flex flex-wrap gap-2">
          {DOCUMENT_FIXTURES.flatMap((d) => d.tags)
            .filter((t, i, arr) => arr.indexOf(t) === i)
            .map((t) => (
              <TagChip key={t} tag={t} />
            ))}
        </div>
      </main>
    </>
  );
}

export default App;
