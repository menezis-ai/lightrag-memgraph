// Activity tab — split timeline (left) + event detail (right)
const { useState, useEffect, useMemo } = React;

// CSV export helper — flattens MOCK_ACTIVITY rows (incl. nested meta) for spreadsheet triage.
function exportActivityCsv(rows, range) {
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = typeof v === "object" ? JSON.stringify(v) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const cols = ["id","ts","kind","sev","actor","role","target_type","target_label","summary","meta"];
  const lines = [cols.join(",")];
  rows.forEach(e => {
    lines.push([
      e.id, e.ts, e.kind, e.sev,
      e.actor.user, e.actor.role,
      e.target.type, e.target.label,
      e.summary, e.meta
    ].map(esc).join(","));
  });
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stamp = new Date().toISOString().slice(0, 10);
  a.href = url;
  a.download = `twin-rag-activity-${range}-${stamp}.csv`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
}
// Mixed audit: source lifecycle, tag mutations, retrievals, pipeline errors, auth, settings.

const KIND_META = {
  "retrieval":         { label: "Retrieval",       icon: "search",        color: "var(--twin-accent)" },
  "tag-mutation":      { label: "Tag mutation",    icon: "tags",          color: "var(--twin-accent)" },
  "doc-review":        { label: "Doc review",      icon: "circle-check",  color: "var(--twin-accent)" },
  "source-uploaded":   { label: "Source uploaded", icon: "cloud-upload",  color: "var(--color-text-secondary)" },
  "source-ready":      { label: "Source ready",    icon: "circle-check",  color: "var(--twin-green-700)" },
  "source-failed":     { label: "Source failed",   icon: "alert-triangle",color: "var(--twin-red-vivid)" },
  "pipeline-warning":  { label: "Pipeline",        icon: "alert-triangle",color: "var(--twin-amber-vivid)" },
  "auth":              { label: "Auth",            icon: "lock",          color: "var(--color-text-secondary)" },
  "settings":          { label: "Settings",        icon: "settings",      color: "var(--color-text-secondary)" },
  // External sync sources (Connections section in Settings)
  "confluence":        { label: "Confluence sync", icon: "brand-confluence", color: "#1F8A7A" },
  "sharepoint":        { label: "SharePoint sync", icon: "cloud",         color: "#5A7FB4" },
  "url":               { label: "URL feed sync",   icon: "link",          color: "var(--color-text-secondary)" }
};
// Fallback so an unknown kind doesn't crash the whole tab — defensive
// guard against future MOCK_ACTIVITY additions that forget to register
// here.
const KIND_FALLBACK = { label: "Event", icon: "circle-dot", color: "var(--color-text-secondary)" };
const RANGES = [
  { id: "24h", label: "24h" },
  { id: "7d",  label: "7d"  },
  { id: "30d", label: "30d" },
  { id: "all", label: "All" }
];

window.ActivityTab = function ActivityTab({ density = "comfortable", live = true, groupByDay = true, onPushToast }) {
  const [range, setRange] = window.useUrlParam("range", "7d", {
    validate: v => ["24h","7d","30d","all"].includes(v)
  });
  const RANGE_MS = { "24h": 864e5, "7d": 7 * 864e5, "30d": 30 * 864e5 };
  // Pinned "now" for the demo so the mock fixture stays in-range. In prod this is Date.now().
  const NOW_MS = Date.parse("2026-05-11T10:00:00Z");
  const [kinds, setKinds] = window.useUrlParam("kind", new Set(), {
    parse: s => new Set(s.split(",").filter(Boolean)),
    serialize: set => set && set.size ? [...set].join(",") : "",
    validate: v => v instanceof Set
  });
  const [sev, setSev] = window.useUrlParam("sev", "any", {
    validate: v => ["any","info","warn","error","critical"].includes(v)
  });
  const [q, setQ] = window.useUrlParam("q", "");
  const [actor, setActor] = window.useUrlParam("actor", "any");
  const [selectedId, setSelectedId] = useState(window.MOCK_ACTIVITY[0].id);
  const [pendingCount, setPendingCount] = useState(0);
  const [clearOpen, setClearOpen] = useState(false);
  const [clearConfirm, setClearConfirm] = useState("");
  const clearModalRef = React.useRef(null);
  window.useModalA11y && window.useModalA11y({ open: clearOpen, onClose: () => setClearOpen(false), ref: clearModalRef });

  // Simulated live polling
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => setPendingCount(c => c + 1), 9000);
    return () => clearInterval(t);
  }, [live]);

  const actors = useMemo(() => {
    const s = new Set(window.MOCK_ACTIVITY.map(e => e.actor.user));
    return ["any", ...s];
  }, []);

  const toggleKind = (k) => {
    const next = new Set(kinds);
    if (next.has(k)) next.delete(k); else next.add(k);
    setKinds(next);
  };

  const filtered = window.MOCK_ACTIVITY.filter(e => {
    if (range !== "all") {
      const cutoff = NOW_MS - (RANGE_MS[range] || RANGE_MS["7d"]);
      const ts = Date.parse(e.ts);
      if (!Number.isNaN(ts) && ts < cutoff) return false;
    }
    if (kinds.size && !kinds.has(e.kind)) return false;
    if (sev !== "any" && e.sev !== sev) return false;
    if (actor !== "any" && e.actor.user !== actor) return false;
    if (q.trim()) {
      const needle = q.trim().toLowerCase();
      const hay = (e.summary + " " + e.target.label + " " + e.actor.user + " " + e.id).toLowerCase();
      if (!hay.includes(needle)) return false;
    }
    return true;
  });

  const selected = filtered.find(e => e.id === selectedId) || filtered[0];

  // Group by day if requested
  const grouped = groupByDay
    ? filtered.reduce((acc, e) => { (acc[e.day] = acc[e.day] || []).push(e); return acc; }, {})
    : { "": filtered };

  return (
    <div className={"activity " + (density === "compact" ? "is-compact" : "")}>
      <div className="activity-main">
        <div className="activity-header">
          <h1>Activity</h1>
          <div className="activity-sub">
            <span>Audit trail · workspace <code>cib</code></span>
            <span className="dot-sep">·</span>
            <span className={"activity-live " + (live ? "is-on" : "is-paused")} title={live ? "Polling /activity every 9s" : "Polling disabled — new events will not surface until re-enabled in Tweaks"}>
              <span className="live-dot" /> {live ? "Live polling" : "Polling paused"}
            </span>
          </div>
        </div>

        <div className="activity-filters">
          <div className="seg-range">
            {RANGES.map(r => (
              <button
                key={r.id}
                className={"seg " + (range === r.id ? "is-active" : "")}
                onClick={() => setRange(r.id)}
              >{r.label}</button>
            ))}
          </div>

          <div className="activity-kinds">
            {Object.entries(KIND_META).map(([k, m]) => {
              const active = kinds.size === 0 || kinds.has(k);
              const explicit = kinds.has(k);
              return (
                <button
                  key={k}
                  className={"kind-pill " + (explicit ? "is-explicit" : active ? "is-dim" : "is-off")}
                  onClick={() => toggleKind(k)}
                  title={m.label}
                >
                  <Icon name={m.icon} size={11} color={m.color} />
                  {m.label}
                </button>
              );
            })}
          </div>

          <div className="activity-secondary">
            <select className="mini-select" value={sev} onChange={e => setSev(e.target.value)}>
              <option value="any">All severities</option>
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
            </select>

            <select className="mini-select" value={actor} onChange={e => setActor(e.target.value)}>
              {actors.map(a => <option key={a} value={a}>{a === "any" ? "All actors" : a}</option>)}
            </select>

            <div className="activity-search">
              <Icon name="search" size={13} color="var(--color-text-tertiary)" />
              <input
                type="text"
                value={q}
                onChange={e => setQ(e.target.value)}
                placeholder="Search summary, target, event ID…"
              />
              {q && (
                <button className="x" onClick={() => setQ("")} aria-label="Clear">
                  <Icon name="x" size={11} color="var(--color-text-tertiary)" />
                </button>
              )}
            </div>

            <div className="activity-actions">
              <button
                className="ghost-btn"
                onClick={() => exportActivityCsv(filtered, range)}
                title={`Download ${filtered.length} event${filtered.length === 1 ? "" : "s"} as CSV`}
              >
                <Icon name="external-link" size={12} />
                Export
              </button>
              <button className="ghost-btn danger" onClick={() => setClearOpen(true)} title="Palier 3 only — purges events past their retention window">
                <Icon name="trash" size={12} />
                Clear
              </button>
            </div>
          </div>
        </div>

        {pendingCount > 0 && (
          <button
            className="activity-pending"
            onClick={() => setPendingCount(0)}
          >
            <span className="pending-dot" />
            {pendingCount} new event{pendingCount > 1 ? "s" : ""} since you opened this view — click to refresh
          </button>
        )}

        <div className="activity-stats">
          <span className="stat"><b>{filtered.length}</b> events</span>
          <span className="dot-sep">·</span>
          <span className="stat"><b>{filtered.filter(e => e.sev === "error").length}</b> errors</span>
          <span className="dot-sep">·</span>
          <span className="stat"><b>{filtered.filter(e => e.sev === "warning").length}</b> warnings</span>
          <span className="dot-sep">·</span>
          <span className="stat"><b>{filtered.filter(e => e.kind === "retrieval").length}</b> retrievals</span>
        </div>

        <div className="activity-timeline">
          {Object.entries(grouped).map(([day, evts]) => (
            <div key={day} className="day-group">
              {day && (
                <div className="day-h">
                  <span>{day}</span>
                  <span className="day-line" />
                  <span className="day-count">{evts.length}</span>
                </div>
              )}
              {evts.map(e => (
                <ActivityRow
                  key={e.id}
                  e={e}
                  selected={selected && selected.id === e.id}
                  onClick={() => setSelectedId(e.id)}
                />
              ))}
            </div>
          ))}
          {!filtered.length && (
            <div className="empty-state" style={{ padding: 60 }}>
              <Icon name="activity" size={24} color="var(--color-text-tertiary)" />
              <div className="title">No events match the current filter</div>
              <button className="suggestion" onClick={() => { setKinds(new Set()); setSev("any"); setActor("any"); setQ(""); }}>Clear filters</button>
            </div>
          )}
        </div>
      </div>

      <ActivityDetail e={selected} onPushToast={onPushToast} />

      {clearOpen && (
        <div className="modal-bg" onClick={() => setClearOpen(false)}>
          <div
            ref={clearModalRef}
            className="modal"
            style={{ width: 480 }}
            role="dialog"
            aria-modal="true"
            aria-labelledby="clear-title"
            tabIndex={-1}
            onClick={e => e.stopPropagation()}
          >
            <div className="modal-h">
              <h3 id="clear-title">Clear activity events</h3>
              <div className="modal-h-sub">Palier 3 · admin action</div>
              <button className="modal-x" onClick={() => setClearOpen(false)} aria-label="Close dialog"><Icon name="x" size={14} /></button>
            </div>
            <div className="modal-body">
              <div className="impact-box danger">
                <Icon name="alert-triangle" size={13} color="var(--twin-red-vivid)" />
                <span>
                  Purges events <b>past their retention window</b> only. Events still within retention (e.g. <code>system.policy_violation</code> kept for 7 years) are untouched. Action is recorded as <code>admin.clear</code> in this log itself.
                </span>
              </div>
              <div className="retention-grid">
                <div><span>Source mgmt</span><code>90d</code></div>
                <div><span>Tag mgmt</span><code>90d</code></div>
                <div><span>Retrieval</span><code>30d</code></div>
                <div><span>Admin</span><code>1y</code></div>
                <div><span>Auth</span><code>1y</code></div>
                <div><span>Policy / System</span><code>7y</code></div>
              </div>
              <label className="field-label">Type <code>CLEAR</code> to confirm</label>
              <input className="text-input" value={clearConfirm} onChange={e => setClearConfirm(e.target.value)} placeholder="CLEAR" autoFocus />
            </div>
            <div className="modal-footer">
              <button className="ghost-btn" onClick={() => setClearOpen(false)}>Cancel</button>
              <button
                className="primary-btn danger"
                disabled={clearConfirm !== "CLEAR"}
                onClick={() => { setClearOpen(false); setClearConfirm(""); onPushToast && onPushToast({ id: "clr-" + Date.now(), title: "Activity", titleSuffix: "events past retention purged", sub: "admin.clear emitted · 1,247 events removed", undo: false }); }}
              >Purge expired events</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

function ActivityRow({ e, selected, onClick }) {
  const m = KIND_META[e.kind] || KIND_FALLBACK;
  return (
    <button className={"activity-row " + (selected ? "is-selected" : "") + " sev-" + e.sev} onClick={onClick}>
      <span className="row-time">{e.rel}</span>
      <span className="row-rail" style={{ background: m.color }} />
      <span className="row-icon" style={{ color: m.color }}>
        <Icon name={m.icon} size={14} />
      </span>
      <span className="row-body">
        <span className="row-line1">
          <span className="row-actor">{e.actor.user}</span>
          <span className="row-kind">{m.label}</span>
          <span className="row-target">{e.target.label}</span>
        </span>
        <span className="row-summary">{e.summary}</span>
      </span>
      {e.sev !== "info" && <span className={"sev-badge sev-" + e.sev}>{e.sev}</span>}
    </button>
  );
}

function ActivityDetail({ e, onPushToast }) {
  if (!e) return <aside className="activity-detail"><div className="empty-state"><div className="title">Select an event</div></div></aside>;
  const m = KIND_META[e.kind] || KIND_FALLBACK;
  const [copied, setCopied] = useState(false);
  const copyId = () => {
    navigator.clipboard && navigator.clipboard.writeText(e.id);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <aside className="activity-detail">
      <div className="detail-head">
        <div className="detail-kind" style={{ color: m.color }}>
          <Icon name={m.icon} size={14} color={m.color} />
          {m.label}
          {e.sev !== "info" && <span className={"sev-badge sev-" + e.sev}>{e.sev}</span>}
        </div>
        <h3>{e.target.label}</h3>
        <div className="detail-summary">{e.summary}</div>
      </div>

      <div className="detail-grid">
        <div className="kv"><span>Event ID</span><code className="copyable" onClick={copyId} title="Copy">{e.id} {copied ? "✓" : ""}</code></div>
        <div className="kv"><span>Timestamp</span><code>{e.ts}</code></div>
        <div className="kv"><span>Relative</span><span>{e.rel}</span></div>
        <div className="kv"><span>Actor</span><span>{e.actor.user} <em>({e.actor.role})</em></span></div>
        <div className="kv"><span>Target</span><span>{e.target.type} · {e.target.label}</span></div>
        <div className="kv"><span>Severity</span><span className={"sev-text sev-" + e.sev}>{e.sev}</span></div>
      </div>

      <div className="detail-section">
        <div className="detail-section-h">Metadata</div>
        <pre className="detail-meta">{JSON.stringify(e.meta, null, 2)}</pre>
      </div>

      <div className="detail-actions">
        {e.kind === "source-failed" && (
          <button
            className="primary-btn"
            onClick={() => onPushToast && onPushToast({
              id: "replay-" + Date.now(),
              kind: "propagating",
              title: "Replay queued",
              sub: `${e.target.label} · POST /documents/scan?retry=${e.target.id || "source"} · worker picking up`,
              autoDone: { title: "Source", titleSuffix: "re-ingested", sub: `${e.target.label} · chunks re-embedded`, undo: false }
            })}
          ><Icon name="refresh" size={12} /> Replay ingestion</button>
        )}
        {e.target.type === "source" && (
          <button
            className="ghost-btn"
            onClick={() => {
              const p = new URLSearchParams();
              p.set("tab", "documents");
              if (e.target.label) p.set("q", e.target.label);
              window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
              window.dispatchEvent(new PopStateEvent("popstate"));
            }}
          ><Icon name="arrow-right" size={12} /> Open source</button>
        )}
        {e.target.type === "query" && (
          <button
            className="ghost-btn"
            onClick={() => {
              const p = new URLSearchParams();
              p.set("tab", "retrieval");
              if (e.target.label) p.set("q", e.target.label);
              if (e.meta && e.meta.mode) p.set("mode", e.meta.mode);
              window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
              window.dispatchEvent(new PopStateEvent("popstate"));
            }}
          ><Icon name="arrow-right" size={12} /> Re-run query</button>
        )}
        <button className="ghost-btn" onClick={copyId}>
          <Icon name="external-link" size={12} /> Copy payload
        </button>
      </div>
    </aside>
  );
}
