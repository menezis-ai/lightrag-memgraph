// Twin RAG WebUI fork — main app
const { useState, useEffect, useRef, useCallback } = React;

// Deployment-time config — populated from Helm env vars in real deploy
const ENV_CONFIG = {
  KB_NAME: "CIB KB",           // env: TWIN_KB_DISPLAY_NAME
  WORKSPACE: "cib",            // env: TWIN_WORKSPACE_DEFAULT
  VISIBILITY: "private",       // env: TWIN_INSTANCE_VISIBILITY
  GRAPH_TAB_ENABLED: true      // env: TWIN_GRAPH_TAB_ENABLED — show Knowledge Graph tab
};

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "light",
  "tierAccent": "twin-rag",
  "activityDensity": "comfortable",
  "activityLive": true,
  "activityGroupByDay": true,
  "graphTabOverride": "env",
  "emptyWorkspace": false
}/*EDITMODE-END*/;

const TIER_ACCENTS = {
  "twin": { accent: "#569C6A", hover: "#3D7A4E", soft: "#EBF5ED", softText: "#3D7A4E", softBorder: "#B5DAB8" },
  "twin-rag": { accent: "#3871B4", hover: "#2D5A8E", soft: "#E6EFF8", softText: "#2D5A8E", softBorder: "#B5D4F4" },
  "twin-graph": { accent: "#8A5C0E", hover: "#6B4708", soft: "#FEF3CD", softText: "#8A5C0E", softBorder: "#E8C97D" },
  "twin-graph-plus": { accent: "#1E2D3D", hover: "#0F1822", soft: "#D4DCE5", softText: "#1E2D3D", softBorder: "#A8B5C2" }
};

function App() {
  const [t, setTweak] = window.useTweaks
    ? window.useTweaks(TWEAK_DEFAULTS)
    : [TWEAK_DEFAULTS, () => {}];

  // Graph tab visibility — env flag, optionally overridden by Tweaks (for demo).
  const graphEnabled = (() => {
    if (t.graphTabOverride === "on") return true;
    if (t.graphTabOverride === "off") return false;
    return !!ENV_CONFIG.GRAPH_TAB_ENABLED;
  })();
  const VALID_TABS = ["documents","retrieval","tags","activity","api","settings", ...(graphEnabled ? ["kg"] : [])];
  const TAB_LIST = [
    { id: "documents", label: "Documents" },
    { id: "retrieval", label: "Retrieval" },
    ...(graphEnabled ? [{ id: "kg", label: "Knowledge Graph" }] : []),
    { id: "tags", label: "Tags" },
    { id: "activity", label: "Activity" },
    { id: "api", label: "API" },
    { id: "settings", label: "Settings" }
  ];

  const [tab, setTab] = useState(() => {
    const p = new URLSearchParams(window.location.search);
    const t = p.get("tab");
    return VALID_TABS.includes(t) ? t : "documents";
  });
  // If the graph tab gets disabled while it's selected, bounce back to documents.
  useEffect(() => {
    if (!VALID_TABS.includes(tab)) setTab("documents");
  }, [graphEnabled]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    p.set("tab", tab);
    const url = window.location.pathname + "?" + p.toString();
    window.history.replaceState(null, "", url);
  }, [tab]);
  useEffect(() => {
    const onPop = () => {
      const p = new URLSearchParams(window.location.search);
      const t = p.get("tab");
      if (VALID_TABS.includes(t)) setTab(t);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [graphEnabled]); // eslint-disable-line react-hooks/exhaustive-deps
  const [docs, setDocs] = useState(window.MOCK_DOCUMENTS);
  // sql.js (WASM SQLite) persistence — hydrate docs from IndexedDB on boot,
  // snapshot back on every mutation. The Approve/Reject buttons in the
  // pending-review queue now produce REAL state changes that survive a
  // page reload — required for a credible demo. See db.jsx for schema.
  const [dbReady, setDbReady] = useState(false);
  useEffect(() => {
    if (!window.twinDb) return;
    window.twinDb.boot().then(() => {
      const persisted = window.twinDb.getAll("docs");
      if (persisted) setDocs(persisted);
      setDbReady(true);
    }).catch(err => {
      console.error("twinDb boot failed:", err);
      setDbReady(true); // proceed without persistence rather than block the UI
    });
  }, []);
  useEffect(() => {
    if (dbReady && window.twinDb) window.twinDb.replaceAll("docs", docs);
  }, [docs, dbReady]);
  // Doc mutator surfaced to children so approve/reject actually move the
  // document out of the pending queue (was toast-only in the proto).
  const mutateDoc = (id, patch) => {
    setDocs(arr => arr.map(d => {
      if (d.id !== id) return d;
      const next = { ...d, ...patch };
      if (patch.review !== undefined) {
        next.review = { ...(d.review || {}), ...patch.review };
      }
      return next;
    }));
    if (window.twinDb) {
      window.twinDb.logMutation("docs", patch.review ? `review.${patch.review.state || "update"}` : "update", id, patch);
    }
  };
  // Empty-workspace demo mode — swaps docs out for an empty array so the
  // first-run / onboarding flow can be shown without losing the full mock.
  const effectiveDocs = t.emptyWorkspace ? [] : docs;
  const loadDemoData = () => {
    setTweak("emptyWorkspace", false);
    addSimpleToast("Demo data loaded", `${window.MOCK_DOCUMENTS.length} sources · ${window.MOCK_TAGS_FULL.length} tags · fixture restored`);
  };
  const [addOpen, setAddOpen] = useState(false);
  // Global system status (gateway / quota / providers / read-only).
  const [systemStatus, setSystemStatus] = useState(window.INITIAL_SYSTEM_STATUS);
  const computedStatus = window.computeSystemStatus(systemStatus);
  const sysCtx = {
    status: systemStatus,
    effectiveReadOnly: computedStatus.effectiveReadOnly,
    readOnlyReason: computedStatus.readOnlyReason
  };
  const setStatusKey = (k, v) => setSystemStatus(s => ({ ...s, [k]: v }));
  // Unified retag target — { mode: "single", doc } | { mode: "bulk", docs } | null.
  // A union avoids two parallel <RetagModal> mounts that could fight over the focus trap.
  const [retagTarget, setRetagTarget] = useState(null);
  const openRetagSingle = (d) => setRetagTarget({ mode: "single", doc: d });
  const openRetagBulk = (arr) => setRetagTarget({ mode: "bulk", docs: arr });
  const closeRetag = () => setRetagTarget(null);
  // Network retry queue — visible to user via the retry banner.
  const [retryQueue, setRetryQueue] = useState([]);
  const retryIdRef = useRef(0);
  const enqueueRetry = useCallback((op) => {
    const id = ++retryIdRef.current;
    const item = { id, status: "failed", attempts: 1, createdAt: Date.now(), ...op };
    setRetryQueue(q => [item, ...q]);
    return id;
  }, []);
  const retryOne = useCallback((id) => {
    setRetryQueue(q => q.map(it => it.id === id ? { ...it, status: "retrying", attempts: it.attempts + 1 } : it));
    // Mock: 65% succeed on retry; otherwise back to failed.
    setTimeout(() => {
      setRetryQueue(q => q.map(it => {
        if (it.id !== id) return it;
        return Math.random() < 0.65
          ? { ...it, status: "ok" }
          : { ...it, status: "failed", lastError: "Provider returned 503 after 3 attempts" };
      }));
      // Auto-clear successes after a beat.
      setTimeout(() => setRetryQueue(q => q.filter(it => it.id !== id || it.status !== "ok")), 1400);
    }, 1200);
  }, []);
  const retryAll = useCallback(() => {
    setRetryQueue(q => q.filter(it => it.status !== "ok").map(it => ({ ...it })));
    // Stagger the kicks so the UI shows progress, not a flash.
    setRetryQueue(curr => {
      curr.filter(it => it.status === "failed").forEach((it, i) => setTimeout(() => retryOne(it.id), i * 220));
      return curr;
    });
  }, [retryOne]);
  const dismissRetry = useCallback((id) => setRetryQueue(q => q.filter(it => it.id !== id)), []);
  const [toasts, setToasts] = useState([]);
  const toastId = useRef(0);

  // Workspace + notifications state (topbar)
  const [workspace, setWorkspace] = useState(ENV_CONFIG.WORKSPACE);
  const [kbName, setKbName] = useState(ENV_CONFIG.KB_NAME);
  const [notifications, setNotifications] = useState(() => window.MOCK_NOTIFICATIONS || []);
  const unreadCount = notifications.filter(n => !n.read).length;

  const switchWorkspace = (ws) => {
    const prev = workspace;
    setWorkspace(ws.id);
    setKbName(ws.kb);
    addToast({
      kind: "done",
      title: "Workspace switched",
      sub: `${prev} → ${ws.id} · ${ws.kb}`,
      undo: false
    });
  };
  const markAllRead = () => setNotifications(ns => ns.map(n => ({ ...n, read: true })));
  const clearNotifications = () => setNotifications([]);

  // Onboarding state (welcome + checklist)
  const [onboard, setOnboard] = window.useOnboarding ? window.useOnboarding() : [null, () => {}];
  // Auto-complete `ingestDone` once at least one source reaches "completed".
  useEffect(() => {
    if (!onboard) return;
    if (onboard.tasks.addSource && !onboard.tasks.ingestDone && effectiveDocs.some(d => d.status === "completed")) {
      const t = setTimeout(() => window.twinCompleteTask && window.twinCompleteTask("ingestDone"), 1400);
      return () => clearTimeout(t);
    }
  }, [onboard && onboard.tasks.addSource, onboard && onboard.tasks.ingestDone, effectiveDocs.length]);
  const jumpToTask = (task) => {
    if (task.id === "addSource") { setTab("documents"); setAddOpen(true); return; }
    if (task.tab) setTab(task.tab);
    if (task.sec) {
      setTimeout(() => {
        const p = new URLSearchParams(window.location.search);
        p.set("tab", task.tab); p.set("sec", task.sec);
        window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
        window.dispatchEvent(new PopStateEvent("popstate"));
      }, 40);
    }
  };

  // Apply theme + tier accent globally
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", t.theme);
  }, [t.theme]);

  useEffect(() => {
    const ta = TIER_ACCENTS[t.tierAccent] || TIER_ACCENTS["twin-rag"];
    const root = document.documentElement;
    root.style.setProperty("--twin-accent", ta.accent);
    root.style.setProperty("--twin-accent-hover", ta.hover);
    root.style.setProperty("--twin-accent-soft-bg", ta.soft);
    root.style.setProperty("--twin-accent-soft-text", ta.softText);
    root.style.setProperty("--twin-accent-soft-border", ta.softBorder);
  }, [t.tierAccent]);

  // Toast machinery
  const addToast = useCallback((toast) => {
    const id = ++toastId.current;
    const item = { ...toast, id };
    setToasts(ts => [...ts, item]);

    if (toast.kind === "propagating" && toast.autoDone) {
      setTimeout(() => {
        setToasts(ts => ts.map(x => x.id === id ? {
          ...x,
          kind: "done",
          title: toast.autoDone.title || x.title,
          tagname: toast.autoDone.tagname !== undefined ? toast.autoDone.tagname : x.tagname,
          titleSuffix: toast.autoDone.titleSuffix,
          sub: toast.autoDone.sub,
          undo: toast.autoDone.undo,
          undoDocId: toast.autoDone.undoDocId,
          startedAt: Date.now()
        } : x));
        // Auto dismiss after 5s
        setTimeout(() => {
          setToasts(ts => ts.filter(x => x.id !== id));
        }, 5000);
      }, 1800);
    }
    return id;
  }, []);

  const addSimpleToast = (title, sub) => addToast({ kind: "done", title, sub, undo: false });

  const undoToast = (toast) => {
    setToasts(ts => ts.filter(x => x.id !== toast.id));
    addSimpleToast("Change undone", toast.sub || "");
  };

  const dismissToast = (toast) => setToasts(ts => ts.filter(x => x.id !== toast.id));

  // Render
  return (
    <div className="app">
      <TopBar
        tab={tab}
        onTab={setTab}
        tabs={TAB_LIST}
        theme={t.theme}
        onTheme={() => setTweak("theme", t.theme === "dark" ? "light" : "dark")}
        kbName={kbName}
        workspace={workspace}
        onSwitchWorkspace={switchWorkspace}
        notifications={notifications}
        unreadCount={unreadCount}
        onMarkAllRead={markAllRead}
        onClearNotifications={clearNotifications}
        systemStatus={systemStatus}
        computedStatus={computedStatus}
      />

      <RetryQueueBanner
        queue={retryQueue}
        onRetry={retryOne}
        onRetryAll={retryAll}
        onDismiss={dismissRetry}
      />

      {window.SystemStatusBanner && (
        <window.SystemStatusBanner
          banners={computedStatus.banners}
          onDismissQuota={() => setStatusKey("dismissedQuotaWarn", true)}
        />
      )}

      <main style={{ flex: 1, overflow: "hidden", display: "flex", position: "relative" }}>
        <window.SystemStatusContext.Provider value={sysCtx}>
        <div className="tab-pane" key={tab}>
          {tab === "documents" && (
            <DocumentsTab
              docs={effectiveDocs}
              mutateDoc={mutateDoc}
              isEmptyWorkspace={t.emptyWorkspace}
              onOpenAdd={() => setAddOpen(true)}
              onOpenRetag={openRetagSingle}
              onOpenBulkRetag={openRetagBulk}
              onAddToast={addSimpleToast}
              onLoadDemo={loadDemoData}
              onEnqueueRetry={enqueueRetry}
            />
          )}
          {tab === "retrieval" && <RetrievalTab />}
          {tab === "api" && <ApiTab />}
          {tab === "activity" && (
            <ActivityTab
              density={t.activityDensity || "comfortable"}
              live={t.activityLive !== false}
              groupByDay={t.activityGroupByDay !== false}
              onPushToast={addToast}
            />
          )}
          {tab === "tags" && <TagsTab onPushToast={addToast} />}
          {tab === "kg" && graphEnabled && window.GraphTab && <window.GraphTab />}
          {tab === "settings" && window.SettingsTab && (
            <window.SettingsTab
              workspace={workspace}
              kbName={kbName}
              onPushToast={addToast}
            />
          )}
        </div>
        </window.SystemStatusContext.Provider>
      </main>

      <AddSourceModal
        open={addOpen}
        onClose={() => setAddOpen(false)}
        onAddToast={addToast}
      />
      <RetagModal
        open={!!retagTarget}
        doc={retagTarget && retagTarget.mode === "single" ? retagTarget.doc : null}
        docs={retagTarget && retagTarget.mode === "bulk" ? retagTarget.docs : null}
        onClose={closeRetag}
        onAddToast={addToast}
      />

      <ToastViewport toasts={toasts} onUndo={undoToast} onDismiss={dismissToast} />

      {window.OnboardingWelcome && onboard && (
        <window.OnboardingWelcome
          open={!onboard.welcomed && !onboard.dismissed}
          onClose={() => setOnboard(s => ({ ...s, welcomed: true }))}
          kbName={kbName}
          userName={window.MOCK_CURRENT_USER ? window.MOCK_CURRENT_USER.name : null}
        />
      )}
      {window.OnboardingChecklist && onboard && onboard.welcomed && (
        <window.OnboardingChecklist
          state={onboard}
          set={setOnboard}
          onJump={jumpToTask}
        />
      )}

      {window.TweaksPanel && (
        <window.TweaksPanel title="Tweaks">
          <window.TweakSection label="Appearance">
            <window.TweakRadio
              label="Theme"
              value={t.theme}
              onChange={v => setTweak("theme", v)}
              options={[{ value: "light", label: "Light" }, { value: "dark", label: "Dark" }]}
            />
            <window.TweakSelect
              label="Tier accent"
              value={t.tierAccent}
              onChange={v => setTweak("tierAccent", v)}
              options={[
                { value: "twin", label: "Twin (green)" },
                { value: "twin-rag", label: "Twin RAG (blue)" },
                { value: "twin-graph", label: "Twin Graph (amber)" },
                { value: "twin-graph-plus", label: "Twin Graph+ (slate)" }
              ]}
            />
          </window.TweakSection>
          <window.TweakSection label="Activity tab">
            <window.TweakRadio
              label="Density"
              value={t.activityDensity}
              onChange={v => setTweak("activityDensity", v)}
              options={[
                { value: "comfortable", label: "Comfortable" },
                { value: "compact", label: "Compact" }
              ]}
            />
            <window.TweakToggle
              label="Live polling"
              value={t.activityLive}
              onChange={v => setTweak("activityLive", v)}
            />
            <window.TweakToggle
              label="Group by day"
              value={t.activityGroupByDay}
              onChange={v => setTweak("activityGroupByDay", v)}
            />
          </window.TweakSection>
          <window.TweakSection label="Onboarding (demo)">
            <window.TweakSelect
              label="Force state"
              value="auto"
              onChange={v => window.applyOnboardingPreset && window.applyOnboardingPreset(v)}
              options={[
                { value: "auto", label: "Auto (localStorage)" },
                { value: "welcome", label: "First visit — show welcome" },
                { value: "mid", label: "Mid-flow — 2/6 done" },
                { value: "done", label: "Complete — 6/6" },
                { value: "off", label: "Dismissed" }
              ]}
            />
            <window.TweakToggle
              label="Empty workspace"
              value={t.emptyWorkspace}
              onChange={v => setTweak("emptyWorkspace", v)}
            />
            <div className="tweak-hint">Empty mode hides mock docs so the empty-state CTA + onboarding can be shown.</div>
          </window.TweakSection>
          <window.TweakSection label="System status (demo)">
            <window.TweakRadio
              label="Gateway"
              value={systemStatus.gateway}
              onChange={v => setStatusKey("gateway", v)}
              options={[
                { value: "ok", label: "OK" },
                { value: "degraded", label: "Degraded" },
                { value: "down", label: "Down" }
              ]}
            />
            <window.TweakSlider
              label="LLM quota %"
              value={systemStatus.llmQuotaPercent}
              onChange={v => { setStatusKey("llmQuotaPercent", v); setStatusKey("dismissedQuotaWarn", false); }}
              min={0} max={100} step={1}
            />
            <window.TweakRadio
              label="Embedder"
              value={systemStatus.embedderStatus}
              onChange={v => setStatusKey("embedderStatus", v)}
              options={[
                { value: "ok", label: "OK" },
                { value: "rate-limited", label: "Rate-limited" },
                { value: "degraded", label: "Degraded" }
              ]}
            />
            <window.TweakToggle
              label="Manual read-only"
              value={systemStatus.manualReadOnly}
              onChange={v => setStatusKey("manualReadOnly", v)}
            />
            <window.TweakButton label="Reset to healthy" onClick={() => setSystemStatus(window.INITIAL_SYSTEM_STATUS)} />
          </window.TweakSection>
          <window.TweakSection label="Feature flags">
            <window.TweakRadio
              label="Knowledge Graph tab"
              value={t.graphTabOverride || "env"}
              onChange={v => setTweak("graphTabOverride", v)}
              options={[
                { value: "env", label: "Env" },
                { value: "on", label: "On" },
                { value: "off", label: "Off" }
              ]}
            />
            <div className="tweak-hint">
              Env: <code>TWIN_GRAPH_TAB_ENABLED = {String(ENV_CONFIG.GRAPH_TAB_ENABLED)}</code>
            </div>
          </window.TweakSection>
          <window.TweakSection label="Demo">
            <window.TweakButton label="Open Add source modal" onClick={() => setAddOpen(true)} />
            <window.TweakButton label="Open Retag modal" onClick={() => openRetagSingle(window.MOCK_DOCUMENTS[0])} />
            <window.TweakButton label="Simulate network error" onClick={() => {
              const fixtures = [
                { kind: "embed",    label: "Embed batch · 18 chunks",          target: "swift-iso20022-migration.pdf",      provider: "openai · text-embedding-3-large", error: "503 Service Unavailable" },
                { kind: "ingest",   label: "Ingest source",                    target: "cib-incidents-2026-Q1-postmortems", provider: "pipeline · stage:extract",        error: "ReadTimeout(60s) on chunk 78/124" },
                { kind: "retrieval", label: "Query · hybrid · top_k=60",       target: "How to restart Oracle on RHEL 9?",  provider: "llm · gpt-4o-mini",                error: "429 Rate limited" }
              ];
              enqueueRetry(fixtures[Math.floor(Math.random() * fixtures.length)]);
            }} />
            <window.TweakButton label="Trigger demo toast" onClick={() => addToast({
              kind: "propagating",
              title: "Propagating tag",
              tagname: "rman",
              sub: "418 chunks · ~2 seconds",
              autoDone: { title: "Tag", tagname: "rman", titleSuffix: "applied", sub: "oracle-restart-procedure.pdf", undo: true }
            })} />
            <window.TweakButton label="Fire 5-toast burst" onClick={() => {
              const variants = [
                { kind: "propagating", title: "Propagating tag", tagname: "oracle", sub: "204 chunks", autoDone: { title: "Tag", tagname: "oracle", titleSuffix: "applied", sub: "oracle-restart.pdf", undo: true } },
                { kind: "done", title: "Source scan complete", sub: "12 sources · 2 changed" },
                { kind: "error", title: "Embedder timeout", sub: "Chunk 17/42 · retry queued" },
                { kind: "propagating", title: "Re-processing", tagname: "Trino_Connector.pdf", sub: "ETA 4 min", autoDone: { title: "Re-processed", titleSuffix: "Trino_Connector.pdf", sub: "47 chunks · 3.2s", undo: false } },
                { kind: "done", title: "Workspace switched", sub: "cib-core → cib-edge" }
              ];
              variants.forEach((v, i) => setTimeout(() => addToast(v), i * 260));
            }} />
          </window.TweakSection>
        </window.TweaksPanel>
      )}
    </div>
  );
}

// Retry queue banner — sits under the topbar, shows failed network ops with retry.
function RetryQueueBanner({ queue, onRetry, onRetryAll, onDismiss }) {
  const [collapsed, setCollapsed] = useState(false);
  const failedCount = queue.filter(it => it.status === "failed").length;
  const retryingCount = queue.filter(it => it.status === "retrying").length;
  if (queue.length === 0) return null;
  return (
    <div className={`retry-banner${collapsed ? " is-collapsed" : ""}`} role="alert" aria-live="polite">
      <div className="retry-banner-h">
        <button
          className="retry-banner-toggle"
          onClick={() => setCollapsed(c => !c)}
          aria-expanded={!collapsed}
        >
          <Icon name={collapsed ? "chevron-right" : "chevron-down"} size={11} />
          <span className="retry-banner-ico"><Icon name="alert-triangle" size={13} /></span>
          <span className="retry-banner-title">
            <b>{failedCount}</b> network operation{failedCount === 1 ? "" : "s"} failed
            {retryingCount > 0 && <span className="retry-banner-sub"> · {retryingCount} retrying…</span>}
          </span>
        </button>
        <div className="retry-banner-actions">
          {failedCount > 0 && (
            <button className="retry-banner-btn primary" onClick={onRetryAll}>
              <Icon name="refresh" size={11} /> Retry all
            </button>
          )}
          <button
            className="retry-banner-btn ghost"
            onClick={() => queue.forEach(it => onDismiss(it.id))}
            title="Dismiss all"
          >Dismiss</button>
        </div>
      </div>
      {!collapsed && (
        <ul className="retry-list">
          {queue.map(it => (
            <li key={it.id} className={`retry-row is-${it.status}`}>
              <span className="retry-kind">{it.kind}</span>
              <span className="retry-body">
                <span className="retry-label">{it.label}</span>
                <span className="retry-target" title={it.target}>{it.target}</span>
                <span className="retry-meta">
                  <span>{it.provider}</span>
                  <span className="retry-sep">·</span>
                  <span>attempt {it.attempts}</span>
                  {it.lastError && <><span className="retry-sep">·</span><span className="retry-err">{it.lastError}</span></>}
                </span>
              </span>
              <span className="retry-status">
                {it.status === "retrying" && (
                  <><span className="retry-spinner" /><span>retrying…</span></>
                )}
                {it.status === "ok" && (
                  <><Icon name="circle-check" size={12} color="var(--twin-green-700, #2F7A40)" /><span>ok</span></>
                )}
                {it.status === "failed" && (
                  <span className="retry-err-pill">{it.error || "failed"}</span>
                )}
              </span>
              <span className="retry-actions">
                {it.status === "failed" && (
                  <button className="retry-banner-btn small" onClick={() => onRetry(it.id)}>
                    <Icon name="refresh" size={10} /> Retry
                  </button>
                )}
                <button className="retry-banner-btn small ghost" onClick={() => onDismiss(it.id)} aria-label="Dismiss">
                  <Icon name="x" size={11} />
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
