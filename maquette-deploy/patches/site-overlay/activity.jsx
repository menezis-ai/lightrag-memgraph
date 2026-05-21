// Activity tab — split timeline (left) + event detail (right)
const { useState, useEffect, useMemo, useRef } = React;

// Retention windows by event-kind bucket (days). Mirrors the table the
// Clear modal shows. Used to compute the preview list of events that
// will actually be purged (audit #45 — operator currently confirms
// without seeing what's about to be deleted).
const RETENTION_DAYS = {
  "source-uploaded": 90, "source-ready": 90, "source-failed": 90,
  "confluence": 90, "sharepoint": 90, "url": 90,
  "tag-mutation": 90, "doc-review": 90,
  "retrieval": 30,
  "settings": 365, "auth": 365,
  "pipeline-warning": 2555 // 7y — policy / system
};

// Filter buckets: condense the 12 event kinds into 5 readable groups.
// Sub-kinds remain accessible via the "Advanced" dropdown next to the
// bucket pills. Each KIND_META key MUST belong to exactly one bucket.
const ACTIVITY_BUCKETS = [
  { id: "sources",   label: "Sources",   icon: "cloud-upload",
    kinds: ["source-uploaded", "source-ready", "source-failed", "confluence", "sharepoint", "url"] },
  { id: "tags",      label: "Tags",      icon: "tags",
    kinds: ["tag-mutation", "doc-review"] },
  { id: "retrieval", label: "Retrieval", icon: "search",
    kinds: ["retrieval"] },
  { id: "auth",      label: "Auth",      icon: "lock",
    kinds: ["auth"] },
  { id: "system",    label: "System",    icon: "settings",
    kinds: ["pipeline-warning", "settings"] }
];

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

// Map a backend mutation row → activity event shape so the Activity tab
// can render live persistence events alongside the seeded MOCK_ACTIVITY.
function _mutationToActivity(m) {
  if (!m || !m.ts) return null;
  const ts = m.ts;
  const day = ts.slice(0, 10);
  const ago = (() => {
    const ms = Date.now() - Date.parse(ts);
    if (ms < 60_000) return "now";
    if (ms < 3_600_000) return Math.round(ms / 60_000) + "m ago";
    if (ms < 86_400_000) return Math.round(ms / 3_600_000) + "h ago";
    return Math.round(ms / 86_400_000) + "d ago";
  })();
  const p = m.payload || {};
  let kind = m.kind;
  let summary = `${m.action} ${m.kind}`;
  let sev = "info";
  let actor = "system";
  if (m.kind === "docs" && p.review) {
    kind = "doc-review";
    if (p.review.state === "approved") summary = `Approved · ${m.target_id} entered active retrieval set`;
    else if (p.review.state === "rejected") { summary = `Rejected · ${m.target_id} · ${p.review.reason || "no reason"}`; sev = "warning"; }
    else summary = `Review state → ${p.review.state}`;
    actor = p.review.reviewed_by || "system";
  } else if (m.kind === "state" && m.action === "reset") {
    kind = "settings";
    summary = "Demo state reset · SQLite reseeded from JSON fixtures";
  } else if (m.kind === "tags") {
    kind = "tag-mutation";
    summary = `Tag ${m.target_id} · ${m.action}`;
    actor = (p.review && p.review.reviewed_by) || "system";
  }
  return {
    id: `mut-${m.id}`,
    kind,
    ts,
    day,
    rel: ago,
    sev,
    actor: { user: actor, role: "Steward" },
    target: { type: m.kind, label: m.target_id || "—" },
    summary,
    meta: { ...(p || {}), mutation_id: m.id }
  };
}

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
  // Live mutations from the FastAPI backend (`/api/mutations`) prepended
  // to the seeded MOCK_ACTIVITY fixture. Gives the Activity tab a real
  // audit trail spine that grows every time a steward approves a doc.
  // Killer for the Manu demo — Kore.ai admitted their audit is "in the
  // logs, not in the interface" (transcription 2026-05-21).
  const [liveMutations, setLiveMutations] = useState([]);
  useEffect(() => {
    if (!window.twinDb) return;
    let cancelled = false;
    const load = () => {
      fetch("/api/mutations?limit=100")
        .then(r => r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status)))
        .then(rows => { if (!cancelled) setLiveMutations(rows.map(_mutationToActivity).filter(Boolean)); })
        .catch(() => {});
    };
    load();
    const t = setInterval(load, 15000); // refresh every 15s
    return () => { cancelled = true; clearInterval(t); };
  }, []);
  const fullActivity = useMemo(
    () => [...liveMutations, ...(window.MOCK_ACTIVITY || [])],
    [liveMutations]
  );
  const [selectedId, setSelectedId] = useState(window.MOCK_ACTIVITY[0].id);
  const [pendingCount, setPendingCount] = useState(0);
  const [clearOpen, setClearOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const advancedRef = useRef(null);
  useEffect(() => {
    if (!advancedOpen) return;
    const onDown = (e) => { if (advancedRef.current && !advancedRef.current.contains(e.target)) setAdvancedOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setAdvancedOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [advancedOpen]);
  const ALL_BUCKET_KINDS = useMemo(() => ACTIVITY_BUCKETS.flatMap(b => b.kinds), []);
  const bucketState = (b) => {
    if (kinds.size === 0) return "on";
    const hasAll = b.kinds.every(k => kinds.has(k));
    const hasSome = b.kinds.some(k => kinds.has(k));
    if (hasAll) return "on";
    if (hasSome) return "partial";
    return "off";
  };
  const toggleBucket = (b) => {
    const next = new Set(kinds);
    const st = bucketState(b);
    if (kinds.size === 0) {
      // Default "all on" → restrict to every bucket *except* this one
      ACTIVITY_BUCKETS.forEach(bb => { if (bb.id !== b.id) bb.kinds.forEach(k => next.add(k)); });
    } else if (st === "off") {
      b.kinds.forEach(k => next.add(k));
    } else {
      // on or partial → remove this bucket's kinds
      b.kinds.forEach(k => next.delete(k));
    }
    // Collapse to the canonical "all on" state when every kind is present
    if (ALL_BUCKET_KINDS.every(k => next.has(k))) next.clear();
    setKinds(next);
  };
  const toggleSubKind = (k) => {
    const seed = kinds.size === 0 ? new Set(ALL_BUCKET_KINDS) : new Set(kinds);
    if (seed.has(k)) seed.delete(k); else seed.add(k);
    if (ALL_BUCKET_KINDS.every(kk => seed.has(kk))) seed.clear();
    setKinds(seed);
  };
  const [clearConfirm, setClearConfirm] = useState("");
  const clearModalRef = React.useRef(null);
  window.useModalA11y && window.useModalA11y({ open: clearOpen, onClose: () => setClearOpen(false), ref: clearModalRef });

  // Simulated live polling. Interval bumped 9s → 30s and the visual
  // indicator (below) only surfaces once 3+ events have queued, so the
  // demo doesn't have a "+1 every 9s" pulse in the corner of the eye
  // (audit feedback — was reading as visual chatter).
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => setPendingCount(c => c + 1), 30000);
    return () => clearInterval(t);
  }, [live]);

  const actors = useMemo(() => {
    const s = new Set(fullActivity.map(e => e.actor.user));
    return ["any", ...s];
  }, []);

  const toggleKind = (k) => {
    const next = new Set(kinds);
    if (next.has(k)) next.delete(k); else next.add(k);
    setKinds(next);
  };

  const filtered = fullActivity.filter(e => {
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

          <div className="activity-buckets" ref={advancedRef}>
            {ACTIVITY_BUCKETS.map(b => {
              const st = bucketState(b);
              return (
                <button
                  key={b.id}
                  className={"bucket-pill is-" + st}
                  onClick={() => toggleBucket(b)}
                  title={`${b.label} (${b.kinds.length} kind${b.kinds.length > 1 ? "s" : ""})`}
                  aria-pressed={st !== "off"}
                >
                  <Icon name={b.icon} size={11} />
                  {b.label}
                  {st === "partial" && <span className="bucket-partial-dot" aria-label="partial" />}
                </button>
              );
            })}
            <button
              className={"bucket-pill is-advanced" + (advancedOpen ? " is-open" : "")}
              onClick={() => setAdvancedOpen(o => !o)}
              aria-expanded={advancedOpen}
              aria-haspopup="dialog"
              title="Filter by individual event kind"
            >
              Advanced
              <Icon name="chevron-down" size={10} />
            </button>
            {advancedOpen && (
              <div className="bucket-advanced-popover" role="dialog" aria-label="Filter by event kind">
                <div className="bucket-advanced-h">Filter by individual kind</div>
                {ACTIVITY_BUCKETS.map(b => (
                  <div key={b.id} className="bucket-advanced-group">
                    <div className="bucket-advanced-group-h">{b.label}</div>
                    <ul>
                      {b.kinds.map(k => {
                        const m = KIND_META[k] || KIND_FALLBACK;
                        const on = kinds.size === 0 || kinds.has(k);
                        return (
                          <li key={k}>
                            <label className="bucket-advanced-row">
                              <input type="checkbox" checked={on} onChange={() => toggleSubKind(k)} />
                              <Icon name={m.icon} size={11} color={m.color} />
                              <span>{m.label}</span>
                            </label>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                ))}
                <div className="bucket-advanced-f">
                  <button className="link-btn" onClick={() => setKinds(new Set())}>Reset (all on)</button>
                </div>
              </div>
            )}
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

        {pendingCount >= 3 && (
          <button
            className="activity-pending"
            onClick={() => setPendingCount(0)}
          >
            <span className="pending-dot" />
            {pendingCount} new events since you opened this view — click to refresh
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
              <div className="modal-h-sub">Steward · admin action</div>
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
              {(() => {
                // Compute the purge preview against the full MOCK_ACTIVITY
                // (not the filtered view — the action is global). Demo NOW_MS
                // is pinned above so the fixture's ages stay deterministic.
                const purge = (window.MOCK_ACTIVITY || []).filter(e => {
                  const days = RETENTION_DAYS[e.kind];
                  if (!days) return false;
                  const ts = Date.parse(e.ts);
                  if (!Number.isFinite(ts)) return false;
                  return (NOW_MS - ts) / 86400000 > days;
                });
                if (purge.length === 0) {
                  return (
                    <div className="purge-preview is-empty">
                      <Icon name="info-circle" size={11} />
                      <span>No events are currently past their retention window — nothing to purge.</span>
                    </div>
                  );
                }
                return (
                  <div className="purge-preview">
                    <div className="purge-preview-h">
                      Will purge <b>{purge.length.toLocaleString()}</b> event{purge.length > 1 ? "s" : ""} past retention
                    </div>
                    <ul className="purge-preview-list">
                      {purge.slice(0, 5).map(e => {
                        const m = KIND_META[e.kind] || KIND_FALLBACK;
                        return (
                          <li key={e.id}>
                            <Icon name={m.icon} size={10} color={m.color} />
                            <span className="pp-kind">{m.label}</span>
                            <span className="pp-target">{e.target.label}</span>
                            <span className="pp-when">{e.day}</span>
                          </li>
                        );
                      })}
                      {purge.length > 5 && (
                        <li className="pp-more">+{(purge.length - 5).toLocaleString()} more</li>
                      )}
                    </ul>
                  </div>
                );
              })()}
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
