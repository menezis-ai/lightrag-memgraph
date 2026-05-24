// External sync connections — Confluence, SharePoint, URL feeds.
// Rendered as a Settings section. Manages OAuth lifecycle (connect, expire, reconnect),
// per-connection schedule, and surfaces sync history.

const { useState: _useStateC, useMemo: _useMemoC, useRef: _useRefC } = React;

const KIND_META = {
  confluence: { label: "Confluence", icon: "brand-confluence", color: "#1F8A7A", oauth: true },
  sharepoint: { label: "SharePoint", icon: "cloud",            color: "#5A7FB4", oauth: true },
  url:        { label: "URL feed",   icon: "link",             color: "#8A98A8", oauth: false }
};

const STATUS_PRESENT = {
  "ok":            { label: "Connected",     cls: "ok",   icon: "circle-check" },
  "syncing":       { label: "Syncing…",       cls: "info", icon: "refresh" },
  "token-expired": { label: "Token expired", cls: "warn", icon: "lock" },
  "sync-failed":   { label: "Sync failed",   cls: "err",  icon: "alert-triangle" },
  "disconnected":  { label: "Disconnected",  cls: "neutral", icon: "x" }
};

window.ConnectionsSection = function ConnectionsSection({ canEdit, onPushToast }) {
  const [conns, setConns] = _useStateC(window.MOCK_CONNECTIONS || []);
  const [history, setHistory] = _useStateC(window.MOCK_SYNC_HISTORY || []);
  const [connectOpen, setConnectOpen] = _useStateC(false);
  const [reconnect, setReconnect] = _useStateC(null);
  const [syncing, setSyncing] = _useStateC(() => new Set());

  const counts = _useMemoC(() => ({
    total: conns.length,
    ok: conns.filter(c => c.status === "ok").length,
    warn: conns.filter(c => c.status === "token-expired").length,
    err: conns.filter(c => c.status === "sync-failed").length
  }), [conns]);

  const runSync = (c) => {
    setSyncing(s => { const n = new Set(s); n.add(c.id); return n; });
    onPushToast && onPushToast({
      id: "sync-" + Date.now(),
      kind: "propagating",
      title: "Syncing",
      tagname: c.name,
      sub: `${c.kind === "confluence" ? "Confluence space " + c.space_key : c.kind === "sharepoint" ? "SharePoint site " + c.site_id : "URL " + c.url}`,
      autoDone: {
        title: "Sync",
        tagname: c.name,
        titleSuffix: "completed",
        sub: "0 added · 4 changed · 0 deleted · 6.2s",
        undo: false
      }
    });
    setTimeout(() => {
      setSyncing(s => { const n = new Set(s); n.delete(c.id); return n; });
      setConns(cs => cs.map(x => x.id === c.id ? {
        ...x,
        last_sync_at: new Date().toISOString(),
        last_sync_duration_ms: 6230,
        status: x.status === "ok" ? "ok" : x.status
      } : x));
      setHistory(h => [{
        id: "syn_now_" + Date.now(),
        conn_id: c.id,
        at: new Date().toISOString(),
        outcome: "ok",
        summary: "0 added · 4 changed · 0 deleted",
        duration_ms: 6230
      }, ...h].slice(0, 20));
    }, 2200);
  };

  const disconnect = (c) => {
    setConns(cs => cs.filter(x => x.id !== c.id));
    onPushToast && onPushToast({ id: "disc-" + Date.now(), title: "Connection removed", sub: `${KIND_META[c.kind].label} · ${c.name}`, undo: true });
  };

  const reconnectDone = (c) => {
    setConns(cs => cs.map(x => x.id === c.id ? {
      ...x,
      status: "ok",
      health: "ok",
      error: undefined,
      next_sync_at: new Date(Date.now() + 6 * 3600e3).toISOString()
    } : x));
    setReconnect(null);
    onPushToast && onPushToast({ id: "rec-" + Date.now(), title: "Reconnected", sub: `${KIND_META[c.kind].label} · ${c.name} · sync resumed`, undo: false });
  };

  const addConn = (payload) => {
    const id = "conn_new_" + Math.random().toString(16).slice(2, 6);
    const c = {
      id,
      kind: payload.kind,
      name: payload.name,
      url: payload.url,
      status: "ok",
      health: "ok",
      sources_tracked: 0,
      last_sync_at: null,
      last_sync_duration_ms: null,
      next_sync_at: new Date(Date.now() + 5 * 60e3).toISOString(),
      schedule: payload.schedule,
      oauth_account: payload.kind === "url" ? null : "svc-twin-sync@bnpparibas.com",
      scopes: payload.kind === "confluence" ? ["read:pages"] : payload.kind === "sharepoint" ? ["Sites.Read.All"] : [],
      pages_added_7d: 0, pages_changed_7d: 0, pages_deleted_7d: 0,
      default_tags: payload.default_tags,
      visibility: "private",
      connected_at: new Date().toISOString().slice(0, 10),
      connected_by: window.MOCK_CURRENT_USER ? window.MOCK_CURRENT_USER.email : "—"
    };
    setConns(cs => [c, ...cs]);
    setConnectOpen(false);
    onPushToast && onPushToast({ id: "conn-" + Date.now(), title: "Connection added", sub: `${KIND_META[c.kind].label} · ${c.name} · first sync in 5 min`, undo: false });
  };

  return (
    <window.SettingsBody
      title="Connections"
      sub="External sources synced on a schedule. Confluence + SharePoint use a delegated service account; URL feeds poll anonymously over HTTPS."
    >
      <div className="settings-card">
        <div className="settings-card-h">
          <h3>{counts.total} connection{counts.total !== 1 ? "s" : ""}</h3>
          <span className="settings-card-sub">
            <b className="conn-stat-ok">{counts.ok}</b> healthy
            {counts.warn > 0 && <> · <b className="conn-stat-warn">{counts.warn}</b> need re-auth</>}
            {counts.err > 0 && <> · <b className="conn-stat-err">{counts.err}</b> failing</>}
          </span>
          {canEdit && (
            <button className="primary-btn small" onClick={() => setConnectOpen(o => !o)}>
              <Icon name={connectOpen ? "x" : "plus"} size={11} /> {connectOpen ? "Cancel" : "Connect source"}
            </button>
          )}
        </div>

        {connectOpen && (
          <ConnectForm onCancel={() => setConnectOpen(false)} onSubmit={addConn} />
        )}

        <ul className="conn-list">
          {conns.map(c => (
            <ConnectionCard
              key={c.id}
              c={c}
              meta={KIND_META[c.kind]}
              syncing={syncing.has(c.id)}
              canEdit={canEdit}
              onSync={() => runSync(c)}
              onReconnect={() => setReconnect(c)}
              onDisconnect={() => disconnect(c)}
            />
          ))}
          {conns.length === 0 && (
            <li className="conn-empty">
              <Icon name="link" size={20} color="var(--color-text-tertiary)" />
              <div>No external sources connected yet.</div>
              <div className="muted-sm">Connect a Confluence space, SharePoint site, or URL feed to scope retrieval to live, refreshable content.</div>
            </li>
          )}
        </ul>
      </div>

      {history.length > 0 && (
        <div className="settings-card">
          <div className="settings-card-h">
            <h3>Recent syncs</h3>
            <span className="settings-card-sub">Last {Math.min(history.length, 8)} runs across all connections.</span>
          </div>
          <ul className="sync-history">
            {history.slice(0, 8).map(h => {
              const c = conns.find(x => x.id === h.conn_id);
              const m = c ? KIND_META[c.kind] : KIND_META.url;
              return (
                <li key={h.id} className={`sync-history-row outcome-${h.outcome}`}>
                  <span className={`sync-dot ${h.outcome}`} />
                  <span className="sync-h-kind" style={{ color: m.color }}>
                    <Icon name={m.icon} size={11} />
                  </span>
                  <span className="sync-h-name">{c ? c.name : "—"}</span>
                  <span className="sync-h-summary">{h.summary}</span>
                  <span className="sync-h-time mono-meta" title={h.at}>
                    {window.relTimeShort ? window.relTimeShort(h.at) : h.at}
                  </span>
                  <span className="sync-h-dur mono-meta">{(h.duration_ms / 1000).toFixed(1)}s</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {reconnect && (
        <ReconnectDialog
          conn={reconnect}
          meta={KIND_META[reconnect.kind]}
          onClose={() => setReconnect(null)}
          onDone={() => reconnectDone(reconnect)}
        />
      )}
    </window.SettingsBody>
  );
};

// ─── Per-connection card ─────────────────────────────────────────────────
function ConnectionCard({ c, meta, syncing, canEdit, onSync, onReconnect, onDisconnect }) {
  const status = STATUS_PRESENT[syncing ? "syncing" : c.status];
  const lastSyncRel = c.last_sync_at && window.relTimeShort ? window.relTimeShort(c.last_sync_at) : "never";
  const nextRel = c.next_sync_at && window.relTimeShort ? window.relTimeShort(c.next_sync_at) : "—";
  return (
    <li className={`conn-card status-${status.cls}`}>
      <div className="conn-card-head">
        <div className="conn-icon" style={{ color: meta.color }}>
          <Icon name={meta.icon} size={18} />
        </div>
        <div className="conn-titles">
          <div className="conn-title-line">
            <span className="conn-name">{c.name}</span>
            <span className="conn-kind">{meta.label}</span>
          </div>
          <a className="conn-url mono-meta" href={c.url} target="_blank" rel="noopener noreferrer" onClick={e => e.stopPropagation()}>
            {c.url}
          </a>
        </div>
        <div className={`conn-status-pill ${status.cls}`}>
          <span className={`sync-dot ${status.cls}`} />
          {syncing && <span className="conn-syncing-ring" />}
          <span>{status.label}</span>
        </div>
      </div>

      {c.error && (
        <div className={`conn-error ${c.status === "token-expired" ? "warn" : "err"}`}>
          <Icon name="alert-triangle" size={12} />
          <span>{c.error}</span>
        </div>
      )}

      <dl className="conn-meta">
        <div><dt>Sources tracked</dt><dd>{c.sources_tracked.toLocaleString()}</dd></div>
        <div><dt>Last sync</dt><dd>{lastSyncRel}</dd></div>
        <div><dt>Next scheduled</dt><dd>{c.status === "ok" ? nextRel : <span className="muted-sm">paused</span>}</dd></div>
        <div><dt>Schedule</dt><dd className="mono-meta">{c.schedule}</dd></div>
        {c.oauth_account && (
          <div><dt>Auth</dt><dd className="mono-meta">{c.oauth_account}</dd></div>
        )}
        <div><dt>Connected by</dt><dd className="mono-meta">{c.connected_by} · {c.connected_at}</dd></div>
      </dl>

      {c.status === "ok" && (c.pages_added_7d + c.pages_changed_7d + c.pages_deleted_7d > 0) && (
        <div className="conn-delta">
          <span className="conn-delta-lbl">Last 7 days</span>
          <span className="conn-delta-stat added">+{c.pages_added_7d}</span>
          <span className="conn-delta-stat changed">~{c.pages_changed_7d}</span>
          <span className="conn-delta-stat deleted">−{c.pages_deleted_7d}</span>
        </div>
      )}

      {c.default_tags && c.default_tags.length > 0 && (
        <div className="conn-tags">
          <span className="muted-sm">Default tags:</span>
          {c.default_tags.map(t => <TagChip key={t} tag={t} />)}
        </div>
      )}

      <div className="conn-actions">
        {c.status === "token-expired" ? (
          <button className="primary-btn small" onClick={onReconnect}>
            <Icon name="refresh" size={11} /> Reconnect
          </button>
        ) : c.status === "sync-failed" ? (
          <button className="primary-btn small" onClick={onSync} disabled={syncing}>
            <Icon name="refresh" size={11} /> Retry sync
          </button>
        ) : (
          <button className="ghost-btn small" onClick={onSync} disabled={syncing}>
            <Icon name="refresh" size={11} /> {syncing ? "Syncing…" : "Sync now"}
          </button>
        )}
        <button className="ghost-btn small">
          <Icon name="settings" size={11} /> Configure
        </button>
        {canEdit && (
          <button className="ghost-btn small danger" onClick={onDisconnect} style={{ marginLeft: "auto" }}>
            <Icon name="trash" size={11} /> Disconnect
          </button>
        )}
      </div>
    </li>
  );
}

// ─── Add new connection (mock OAuth flow) ────────────────────────────────
function ConnectForm({ onCancel, onSubmit }) {
  const [kind, setKind] = _useStateC("confluence");
  const [url, setUrl] = _useStateC("");
  const [name, setName] = _useStateC("");
  const [schedule, setSchedule] = _useStateC("every 12h");
  const [tags, setTags] = _useStateC([]);
  const [tagInput, setTagInput] = _useStateC("");
  const [step, setStep] = _useStateC("config"); // config → oauth → done

  const placeholderUrl = {
    confluence: "https://confluence.bnp/spaces/<KEY>",
    sharepoint: "https://erwin-labs.sharepoint.com/sites/<site>",
    url: "https://docs.example.com/api/v2"
  }[kind];

  const proceed = () => {
    if (!url.trim() || !name.trim()) return;
    if (kind === "url") {
      // No OAuth — submit immediately.
      onSubmit({ kind, url: url.trim(), name: name.trim(), schedule, default_tags: tags });
      return;
    }
    setStep("oauth");
    setTimeout(() => {
      onSubmit({ kind, url: url.trim(), name: name.trim(), schedule, default_tags: tags });
    }, 1600);
  };

  const addTag = () => {
    const v = tagInput.trim().toLowerCase();
    if (v && !tags.includes(v)) setTags([...tags, v]);
    setTagInput("");
  };
  const removeTag = (t) => setTags(tags.filter(x => x !== t));

  if (step === "oauth") {
    const meta = KIND_META[kind];
    return (
      <div className="connect-oauth">
        <div className="oauth-shell">
          <div className="oauth-h">
            <Icon name={meta.icon} size={18} color={meta.color} />
            <span>Redirecting to {meta.label}…</span>
          </div>
          <div className="oauth-body">
            <p>Authorize <code>svc-twin-sync@bnpparibas.com</code> to read content from this {kind === "confluence" ? "space" : "site"}.</p>
            <div className="oauth-scopes">
              {(kind === "confluence" ? ["read:pages", "read:attachments"] : ["Sites.Read.All", "Files.Read.All"]).map(s => (
                <code key={s} className="scope-chip tiny">{s}</code>
              ))}
            </div>
            <div className="oauth-spinner-row">
              <span className="retry-spinner" />
              <span className="muted-sm">Waiting for redirect callback…</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="connect-form">
      <label className="field-label">Source kind</label>
      <div className="connect-kind-tabs">
        {Object.entries(KIND_META).map(([k, m]) => (
          <button
            key={k}
            className={`connect-kind-tab ${kind === k ? "is-active" : ""}`}
            onClick={() => setKind(k)}
          >
            <Icon name={m.icon} size={12} color={kind === k ? m.color : "var(--color-text-tertiary)"} />
            <span>{m.label}</span>
            {m.oauth && <span className="connect-kind-badge">OAuth</span>}
          </button>
        ))}
      </div>

      <label className="field-label" style={{ marginTop: 12 }}>URL</label>
      <input
        type="text"
        className="text-input"
        placeholder={placeholderUrl}
        value={url}
        onChange={e => setUrl(e.target.value)}
      />

      <label className="field-label" style={{ marginTop: 10 }}>Display name <span className="hint">— shown in source lists</span></label>
      <input
        type="text"
        className="text-input"
        placeholder={kind === "confluence" ? "e.g. CIB Runbooks" : kind === "sharepoint" ? "e.g. Incidents site" : "e.g. Oracle docs"}
        value={name}
        onChange={e => setName(e.target.value)}
      />

      <div className="connect-row">
        <div style={{ flex: 1 }}>
          <label className="field-label">Schedule</label>
          <select className="mini-select" value={schedule} onChange={e => setSchedule(e.target.value)}>
            <option value="every 1h">every 1h</option>
            <option value="every 6h">every 6h</option>
            <option value="every 12h">every 12h</option>
            <option value="daily">daily</option>
            <option value="manual">manual only</option>
          </select>
        </div>
        <div style={{ flex: 1 }}>
          <label className="field-label">Default tags <span className="hint">applied to all synced sources</span></label>
          <div className="tag-chips" style={{ marginTop: 4 }}>
            {tags.map(t => <TagChip key={t} tag={t} removable onRemove={removeTag} />)}
            <input
              className="conn-tag-input"
              value={tagInput}
              onChange={e => setTagInput(e.target.value.toLowerCase())}
              onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); addTag(); } }}
              placeholder="Add tag…"
            />
          </div>
        </div>
      </div>

      <div className="connect-actions">
        <button className="ghost-btn" onClick={onCancel}>Cancel</button>
        <button
          className="primary-btn"
          disabled={!url.trim() || !name.trim()}
          onClick={proceed}
        >
          {kind === "url" ? "Add feed" : <>Continue to {KIND_META[kind].label} <Icon name="arrow-right" size={11} /></>}
        </button>
      </div>
    </div>
  );
}

// ─── Reconnect modal (expired token) ─────────────────────────────────────
function ReconnectDialog({ conn, meta, onClose, onDone }) {
  const ref = _useRefC(null);
  window.useModalA11y && window.useModalA11y({ open: true, onClose, ref });
  const [step, setStep] = _useStateC("idle"); // idle → authorizing → done

  const start = () => {
    setStep("authorizing");
    setTimeout(() => {
      setStep("done");
      setTimeout(onDone, 500);
    }, 1500);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal small" role="dialog" aria-modal="true" aria-labelledby="rec-title" ref={ref} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ flex: 1 }}>
            <h2 id="rec-title">Reconnect {meta.label}</h2>
            <div className="ctx">
              <Icon name={meta.icon} size={13} color={meta.color} />
              <span style={{ fontFamily: "var(--font-mono)" }}>{conn.name}</span>
            </div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" size={18} /></button>
        </div>
        <div className="modal-body">
          {step === "idle" && (
            <>
              <div className="impact-box warning">
                <Icon name="alert-triangle" size={13} color="var(--twin-amber-vivid, #9C7000)" />
                <span>{conn.error || "Refresh token expired. Sync paused until a new authorization grant is obtained."}</span>
              </div>
              <p style={{ fontSize: 12, color: "var(--color-text-secondary)", lineHeight: 1.55 }}>
                Re-authorizing creates a new refresh token bound to the existing service account. No data is re-fetched —
                only credentials are rotated. The next scheduled sync resumes automatically.
              </p>
              <dl className="settings-kv">
                <dt>Service account</dt><dd className="mono-meta">{conn.oauth_account}</dd>
                <dt>Scopes</dt><dd>
                  {conn.scopes.map(s => <code key={s} className="scope-chip tiny" style={{ marginRight: 4 }}>{s}</code>)}
                </dd>
                <dt>Last successful sync</dt><dd className="mono-meta">{conn.last_sync_at}</dd>
              </dl>
            </>
          )}
          {step === "authorizing" && (
            <div className="oauth-spinner-row" style={{ padding: "30px 0", justifyContent: "center" }}>
              <span className="retry-spinner" />
              <span>Awaiting authorization callback from {meta.label}…</span>
            </div>
          )}
          {step === "done" && (
            <div className="oauth-spinner-row" style={{ padding: "30px 0", justifyContent: "center", color: "var(--twin-green-700, #2F7A40)" }}>
              <Icon name="circle-check" size={16} color="var(--twin-green-700, #2F7A40)" />
              <span>New refresh token issued · sync resuming…</span>
            </div>
          )}
        </div>
        {step === "idle" && (
          <div className="modal-footer">
            <button className="ghost-btn" onClick={onClose}>Cancel</button>
            <button className="primary-btn" onClick={start} style={{ marginLeft: "auto" }}>
              <Icon name="lock" size={12} /> Re-authorize on {meta.label}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
