// Top bar — logo, tabs, workspace switcher, notifications popover, theme

const DEFAULT_TABS = [
  { id: "documents", label: "Documents" },
  { id: "retrieval", label: "Retrieval" },
  { id: "tags", label: "Tags" },
  { id: "activity", label: "Activity" },
  { id: "api", label: "API" }
];

window.TopBar = function TopBar({
  tab, onTab, theme, onTheme,
  workspace, kbName, onSwitchWorkspace,
  notifications = [], unreadCount = 0,
  onMarkAllRead, onClearNotifications,
  tabs,
  systemStatus, computedStatus
}) {
  const TABS = tabs || DEFAULT_TABS;
  const [wsOpen, setWsOpen] = React.useState(false);
  const [notifOpen, setNotifOpen] = React.useState(false);
  const wsRef = React.useRef(null);
  const notifRef = React.useRef(null);

  React.useEffect(() => {
    const onDown = (e) => {
      if (wsRef.current && !wsRef.current.contains(e.target)) setWsOpen(false);
      if (notifRef.current && !notifRef.current.contains(e.target)) setNotifOpen(false);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { setWsOpen(false); setNotifOpen(false); }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark">TR</div>
        <span className="brand-name">Twin</span>
        <span className="brand-sep">|</span>
        <span className="brand-kb" title="TWIN_KB_DISPLAY_NAME (env)">{kbName}</span>
      </div>
      <nav className="tabs">
        {TABS.map(t => (
          <button
            key={t.id}
            className={`tab${tab === t.id ? " active" : ""}`}
            onClick={() => onTab(t.id)}
          >{t.label}</button>
        ))}
      </nav>
      <div className="topbar-right">
        {systemStatus && computedStatus && window.SystemStatusIndicator && (
          <window.SystemStatusIndicator status={systemStatus} computed={computedStatus} />
        )}
        <MyAccessPill />
        <div ref={wsRef} style={{ position: "relative" }}>
          <button
            className={`workspace-pill${wsOpen ? " is-open" : ""}`}
            onClick={() => { setWsOpen(o => !o); setNotifOpen(false); }}
            title="Switch workspace"
            aria-expanded={wsOpen}
            aria-haspopup="menu"
          >
            <Icon name="folder" size={12} />
            <span style={{ fontFamily: "var(--font-mono)" }}>{workspace}</span>
            <Icon name="chevron-down" size={11} />
          </button>
          {wsOpen && (
            <WorkspaceMenu
              current={workspace}
              onPick={(ws) => { setWsOpen(false); onSwitchWorkspace(ws); }}
              onClose={() => setWsOpen(false)}
            />
          )}
        </div>
        <div ref={notifRef} style={{ position: "relative" }}>
          <button
            className="icon-btn"
            aria-label={`Notifications${unreadCount ? `, ${unreadCount} unread` : ""}`}
            aria-expanded={notifOpen}
            onClick={() => { setNotifOpen(o => !o); setWsOpen(false); }}
          >
            <Icon name="bell" size={16} />
            {unreadCount > 0 && <span className="notif-badge">{unreadCount > 9 ? "9+" : unreadCount}</span>}
          </button>
          {notifOpen && (
            <NotificationsPopover
              notifications={notifications}
              onMarkAllRead={onMarkAllRead}
              onClear={onClearNotifications}
              onClose={() => setNotifOpen(false)}
            />
          )}
        </div>
        <button className="icon-btn" aria-label="Theme" onClick={onTheme}>
          <Icon name={theme === "dark" ? "sun" : "moon"} size={16} />
        </button>
        {window.twinDb && (
          <button
            className="icon-btn"
            aria-label="Reset demo data (sqlite)"
            title="Reset demo data — wipes the IndexedDB SQLite snapshot and reloads"
            onClick={() => {
              if (window.confirm("Reset the demo SQLite database? All approvals / rejections / mutations made during this session will be lost and the seeded fixture restored.")) {
                window.twinDb.reset();
              }
            }}
          >
            <Icon name="trash" size={14} />
          </button>
        )}
      </div>
    </header>
  );
};

// MyAccess identity pill — positions the BNP-mandated heritance pattern
// (transcription Vihn 2026-05-20 + Manu/Kore.ai 2026-05-21). Twin must
// inherit from MyAccess, not duplicate. The pill surfaces the cache
// state of the bridge; the actual backend implementation comes once
// MyAccess team has shared the auth contract (OIDC / SAML / REST).
function MyAccessPill() {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);
  const [syncing, setSyncing] = React.useState(false);
  const [lastSync, setLastSync] = React.useState("2026-05-21 16:24 UTC");

  React.useEffect(() => {
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, []);

  const forceSync = () => {
    setSyncing(true);
    setTimeout(() => {
      setSyncing(false);
      const now = new Date();
      const stamp = now.toISOString().replace("T", " ").slice(0, 16) + " UTC";
      setLastSync(stamp);
    }, 1200);
  };

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        className={`myaccess-pill${open ? " is-open" : ""}`}
        onClick={() => setOpen(o => !o)}
        title="Identity inherited from MyAccess (BNP referential master)"
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        <span className="myaccess-dot" />
        <span className="myaccess-label">MyAccess</span>
      </button>
      {open && (
        <div className="myaccess-popover" role="dialog" aria-label="MyAccess identity bridge">
          <header className="myaccess-h">
            <div className="myaccess-title">
              <Icon name="lock" size={12} color="var(--twin-accent)" />
              Identity bridge · MyAccess
            </div>
            <span className="myaccess-status">cached</span>
          </header>
          <dl className="myaccess-kv">
            <dt>Source</dt>
            <dd className="mono-meta">MyAccess (BNP)</dd>
            <dt>Current user</dt>
            <dd className="mono-meta">claire.benoit@bnpparibas.com</dd>
            <dt>Entity (AUID)</dt>
            <dd className="mono-meta">cib · AUID-7f4e</dd>
            <dt>Last sync</dt>
            <dd className="mono-meta">{lastSync}</dd>
            <dt>Refresh cycle</dt>
            <dd className="mono-meta">every 5 min · pull · fail-closed if &gt;1h stale</dd>
          </dl>
          <p className="myaccess-note">
            Twin <b>inherits</b> the entity-level access decision from MyAccess
            (ITGP workflow, owner-validated). The Reader / Contributor /
            Steward palier is local to Twin — set by Twin admins, not BNP
            referential. Cache lives in the same Memgraph graph as the
            knowledge, so the audit trail is requestable in one place.
          </p>
          <div className="myaccess-actions">
            <button
              className={"link-btn" + (syncing ? " is-busy" : "")}
              onClick={forceSync}
              disabled={syncing}
            >
              {syncing ? <><Icon name="loader-2" size={11} /> Syncing…</> : <><Icon name="refresh" size={11} /> Force sync now</>}
            </button>
            <span className="myaccess-preview-pill">preview</span>
          </div>
        </div>
      )}
    </div>
  );
}

function WorkspaceMenu({ current, onPick, onClose }) {
  const list = window.MOCK_WORKSPACES || [];
  return (
    <div className="ws-menu" role="menu" aria-label="Switch workspace">
      <div className="ws-menu-h">Workspaces</div>
      <ul className="ws-menu-list">
        {list.map(w => {
          const active = w.id === current;
          return (
            <li key={w.id}>
              <button
                role="menuitemradio"
                aria-checked={active}
                className={`ws-row${active ? " is-active" : ""}`}
                onClick={() => !active && onPick(w)}
                disabled={active}
              >
                <span className="ws-row-l">
                  <Icon name="folder" size={12} color="var(--color-text-secondary)" />
                  <span className="ws-row-id">{w.id}</span>
                  {active && <Icon name="circle-check" size={12} color="var(--twin-accent)" />}
                </span>
                <span className="ws-row-r">
                  <span className="ws-row-kb">{w.kb}</span>
                  <span className="ws-row-meta">
                    <span className="ws-vis">
                      <Icon name={w.visibility === "private" ? "lock" : "world"} size={10} />
                      {w.visibility}
                    </span>
                    <span className="ws-sep">·</span>
                    <span>{w.sources.toLocaleString()} sources</span>
                    <span className="ws-sep">·</span>
                    <span className="ws-role">{w.role}</span>
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      <div className="ws-menu-f">
        <button className="link-btn" onClick={onClose}>Manage workspaces →</button>
      </div>
    </div>
  );
}

const NOTIF_ICON = {
  "tag-mutation":     { name: "tags",           color: "var(--twin-accent)" },
  "tag-request":      { name: "tags",           color: "var(--twin-amber-vivid, #9C7000)" },
  "doc-review":       { name: "alert-triangle", color: "var(--twin-amber-vivid, #9C7000)" },
  "source-ready":     { name: "circle-check",   color: "var(--twin-green-700, #2F7A40)" },
  "source-failed":    { name: "alert-triangle", color: "var(--twin-red-vivid, #B03030)" },
  "pipeline-warning": { name: "alert-triangle", color: "var(--twin-amber-vivid, #9C7000)" },
  "source-uploaded":  { name: "cloud-upload",   color: "var(--color-text-secondary)" },
  "retrieval":        { name: "search",         color: "var(--twin-accent)" },
  "info":             { name: "info-circle",    color: "var(--color-text-secondary)" }
};

function NotificationsPopover({ notifications, onMarkAllRead, onClear, onClose }) {
  const unread = notifications.filter(n => !n.read).length;
  const goActivity = () => {
    onClose();
    const p = new URLSearchParams(window.location.search);
    p.set("tab", "activity");
    window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
    window.dispatchEvent(new PopStateEvent("popstate"));
  };
  return (
    <div className="notif-popover" role="dialog" aria-label="Notifications">
      <header className="notif-h">
        <span className="notif-title">Notifications</span>
        <span className="notif-count">
          {unread > 0
            ? <><b>{unread}</b> unread · {notifications.length} total</>
            : <>{notifications.length} total</>
          }
        </span>
        <div className="notif-h-actions">
          {unread > 0 && (
            <button className="link-btn small" onClick={onMarkAllRead}>Mark all read</button>
          )}
          <button className="icon-btn small" onClick={onClose} aria-label="Close">
            <Icon name="x" size={12} />
          </button>
        </div>
      </header>
      {notifications.length === 0 ? (
        <div className="notif-empty">
          <Icon name="bell" size={20} color="var(--color-text-tertiary)" />
          <div>You're all caught up.</div>
        </div>
      ) : (
        <ul className="notif-list">
          {notifications.map(n => {
            const ico = NOTIF_ICON[n.kind] || NOTIF_ICON.info;
            return (
              <li key={n.id} className={`notif-item${n.read ? "" : " is-unread"}`}>
                <span className="notif-ico" style={{ color: ico.color }}>
                  <Icon name={ico.name} size={14} />
                </span>
                <div className="notif-body">
                  <div className="notif-line1">
                    <span className="notif-t">{n.title}</span>
                    {n.tagname && <code className="notif-tagname">{n.tagname}</code>}
                    {n.suffix && <span className="notif-suffix">{n.suffix}</span>}
                  </div>
                  {n.sub && <div className="notif-sub">{n.sub}</div>}
                  <div className="notif-rel">{n.rel}</div>
                </div>
                {!n.read && <span className="notif-dot" aria-label="unread" />}
              </li>
            );
          })}
        </ul>
      )}
      <footer className="notif-f">
        <button className="link-btn" onClick={goActivity}>View full activity log →</button>
        {notifications.length > 0 && (
          <button className="link-btn danger" onClick={onClear}>Clear all</button>
        )}
      </footer>
    </div>
  );
}
