// System status — global health banner + topbar indicator + read-only propagation.
// Centralizes 4 degraded modes:
//   - gateway-down  (red, persistent, forces read-only)
//   - read-only     (slate, persistent, no writes allowed)
//   - quota-warn    (amber, dismissible, soft warning)
//   - provider-deg  (amber, persistent until status flips)

const { useState: _useStateSys, useEffect: _useEffectSys, useContext: _useContextSys, createContext: _createContextSys, useRef: _useRefSys } = React;

// Context lets any descendant query the read-only state without prop drilling.
// `effectiveReadOnly` is true if any of: explicit readOnly toggle, gateway down,
// or LLM quota fully exhausted.
window.SystemStatusContext = _createContextSys({
  status: null,
  effectiveReadOnly: false,
  readOnlyReason: ""
});

window.useReadOnly = function useReadOnly() {
  const ctx = _useContextSys(window.SystemStatusContext);
  return ctx;
};

// Initial healthy state. App.jsx owns this; Tweaks panel mutates it.
window.INITIAL_SYSTEM_STATUS = {
  gateway: "ok",                            // ok | degraded | down
  gatewayLastSuccessAt: Date.now(),
  llmQuotaPercent: 24,                      // 0..100; >=85 warn, ===100 exhausted
  llmQuotaResetAt: "2026-06-01T00:00:00Z",
  embedderStatus: "ok",                     // ok | rate-limited | degraded
  rerankerStatus: "ok",
  manualReadOnly: false,                    // operator-toggled
  dismissedQuotaWarn: false,
  sessionExpiresAt: null                    // ISO; banner appears <5min before
};

// Compute derived flags + the active banner stack.
window.computeSystemStatus = function computeSystemStatus(s) {
  const banners = [];
  let effectiveReadOnly = false;
  let readOnlyReason = "";

  if (s.gateway === "down") {
    effectiveReadOnly = true;
    readOnlyReason = "Gateway unreachable";
    banners.push({
      kind: "gateway-down",
      severity: "error",
      title: "Gateway unreachable",
      body: `Retrying every 5s · last successful contact ${window.relTimeShort ? window.relTimeShort(s.gatewayLastSuccessAt) : "—"}`,
      meta: `cib-kb.twin.internal · twin-gateway-7d6b8f@${s.gatewayLastSuccessAt}`
    });
  } else if (s.gateway === "degraded") {
    banners.push({
      kind: "gateway-degraded",
      severity: "warn",
      title: "Gateway degraded",
      body: "Elevated latency detected (p95 > 2s). Retrieval may stall.",
      meta: "p95 2840ms · error rate 1.2%"
    });
  }

  if (s.llmQuotaPercent >= 100) {
    effectiveReadOnly = true;
    readOnlyReason = "LLM quota exhausted";
    banners.push({
      kind: "quota-exhausted",
      severity: "error",
      title: "LLM monthly quota exhausted",
      body: `Retrieval and synthesis halted. Quota resets ${s.llmQuotaResetAt.slice(0, 10)}.`,
      meta: "Raise the cap in Settings → Providers (Steward only)"
    });
  } else if (s.llmQuotaPercent >= 85 && !s.dismissedQuotaWarn) {
    banners.push({
      kind: "quota-warn",
      severity: "warn",
      title: `LLM quota at ${s.llmQuotaPercent}%`,
      body: `Estimated ${Math.max(1, Math.round((100 - s.llmQuotaPercent) / 4))} days remaining at current pace.`,
      meta: `Resets ${s.llmQuotaResetAt.slice(0, 10)}`,
      dismissible: true
    });
  }

  if (s.manualReadOnly && !effectiveReadOnly) {
    effectiveReadOnly = true;
    readOnlyReason = "Manual read-only mode";
    banners.push({
      kind: "read-only",
      severity: "info",
      title: "Read-only mode",
      body: "Writes are disabled by operator. Existing data remains queryable.",
      meta: "Toggle off in Tweaks → System status (demo)"
    });
  }

  if (s.embedderStatus === "rate-limited") {
    banners.push({
      kind: "provider-deg",
      severity: "warn",
      title: "Embedder rate-limited",
      body: "Ingestion queue holding · automatic backoff active.",
      meta: "openai · text-embedding-3-large · 429 since 14:02"
    });
  } else if (s.embedderStatus === "degraded") {
    banners.push({
      kind: "provider-deg",
      severity: "warn",
      title: "Embedder degraded",
      body: "Elevated error rate. New ingestions queued, existing chunks unaffected.",
      meta: "openai · 5xx rate 8% over 15min"
    });
  }

  if (s.sessionExpiresAt) {
    const minutesLeft = Math.round((Date.parse(s.sessionExpiresAt) - Date.now()) / 60000);
    if (minutesLeft <= 5 && minutesLeft > 0) {
      banners.push({
        kind: "session-soon",
        severity: "warn",
        title: `Session expires in ${minutesLeft}min`,
        body: "Save your work. We'll redirect to the IDP for a silent refresh.",
        meta: "keycloak · twin-cib"
      });
    }
  }

  // Overall health: most-severe banner wins.
  let health = "ok";
  if (banners.some(b => b.severity === "error")) health = "error";
  else if (banners.some(b => b.severity === "warn")) health = "warn";

  return { banners, effectiveReadOnly, readOnlyReason, health };
};

// Compact relative-time helper used in banners. Declared on window so other files can reuse.
window.relTimeShort = function relTimeShort(ts) {
  if (!ts) return "—";
  const epoch = typeof ts === "number" ? ts : Date.parse(ts);
  if (!Number.isFinite(epoch)) return "—";
  const d = Math.max(0, Date.now() - epoch);
  if (d < 60_000) return Math.round(d / 1000) + "s ago";
  if (d < 3_600_000) return Math.round(d / 60_000) + "m ago";
  if (d < 86_400_000) return Math.round(d / 3_600_000) + "h ago";
  return Math.round(d / 86_400_000) + "d ago";
};

// ─── Banner stack ────────────────────────────────────────────────────────
window.SystemStatusBanner = function SystemStatusBanner({ banners, onDismissQuota }) {
  if (!banners || banners.length === 0) return null;
  return (
    <div className="sys-banner-stack" role="region" aria-label="System status">
      {banners.map(b => (
        <div key={b.kind} className={`sys-banner sys-${b.severity} k-${b.kind}`} role={b.severity === "error" ? "alert" : "status"}>
          <span className="sys-banner-ico">
            <Icon
              name={b.severity === "error" ? "alert-triangle" : b.severity === "warn" ? "alert-triangle" : "info-circle"}
              size={14}
            />
          </span>
          <div className="sys-banner-body">
            <div className="sys-banner-line1">
              <span className="sys-banner-title">{b.title}</span>
              <span className="sys-banner-sub">{b.body}</span>
            </div>
            {b.meta && <div className="sys-banner-meta">{b.meta}</div>}
          </div>
          <div className="sys-banner-actions">
            {b.kind === "gateway-down" && (
              <button className="sys-banner-btn"><Icon name="refresh" size={11} /> Retry now</button>
            )}
            {b.kind === "session-soon" && (
              <button className="sys-banner-btn primary"><Icon name="refresh" size={11} /> Refresh session</button>
            )}
            {b.kind === "quota-exhausted" && (
              <button className="sys-banner-btn" onClick={() => {
                const p = new URLSearchParams(window.location.search);
                p.set("tab", "settings"); p.set("sec", "providers");
                window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
                window.dispatchEvent(new PopStateEvent("popstate"));
              }}><Icon name="arrow-right" size={11} /> Open Settings</button>
            )}
            {b.dismissible && (
              <button className="sys-banner-btn ghost" onClick={() => onDismissQuota && onDismissQuota()} aria-label="Dismiss">
                <Icon name="x" size={11} />
              </button>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

// ─── Topbar status indicator + popover ───────────────────────────────────
window.SystemStatusIndicator = function SystemStatusIndicator({ status, computed }) {
  const [open, setOpen] = _useStateSys(false);
  const ref = _useRefSys(null);
  _useEffectSys(() => {
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, []);

  const label = computed.health === "ok" ? "All systems operational"
              : computed.health === "warn" ? `${computed.banners.length} warning${computed.banners.length > 1 ? "s" : ""}`
              : "System incident";

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        className={`sys-pill sys-pill-${computed.health}`}
        onClick={() => setOpen(o => !o)}
        title={label}
        aria-expanded={open}
        aria-haspopup="dialog"
      >
        <span className={`sys-dot sys-dot-${computed.health}`} />
        <span className="sys-pill-label">{computed.health === "ok" ? "All systems" : label}</span>
      </button>
      {open && (
        <div className="sys-popover" role="dialog" aria-label="System status">
          <header className="sys-popover-h">
            <div className="sys-popover-title">
              <span className={`sys-dot sys-dot-${computed.health}`} />
              <span>{label}</span>
            </div>
            <span className="sys-popover-sub">Last check {window.relTimeShort(status.gatewayLastSuccessAt)}</span>
          </header>
          <ul className="sys-popover-checks">
            <SysCheck label="Gateway"    status={status.gateway} okLabel="reachable" />
            <SysCheck label="LLM"        status={status.llmQuotaPercent >= 100 ? "down" : status.llmQuotaPercent >= 85 ? "degraded" : "ok"}
                       okLabel={`quota ${status.llmQuotaPercent}%`}
                       degradedLabel={`quota ${status.llmQuotaPercent}% — high`}
                       downLabel={`quota exhausted`} />
            <SysCheck label="Embedder"   status={status.embedderStatus} okLabel="responsive" degradedLabel="degraded" />
            <SysCheck label="Reranker"   status={status.rerankerStatus} okLabel="responsive" />
            <SysCheck label="Memgraph"   status="ok" okLabel="lag 0.4s" />
            <SysCheck label="Indexer"    status="ok" okLabel="31 chk/s" />
          </ul>
          {computed.effectiveReadOnly && (
            <div className="sys-popover-readonly">
              <Icon name="lock" size={11} />
              <span><b>Read-only mode active</b> — {computed.readOnlyReason}</span>
            </div>
          )}
          <footer className="sys-popover-f">
            <a
              href="?tab=activity"
              className="link-btn"
              onClick={e => {
                e.preventDefault();
                const p = new URLSearchParams(window.location.search);
                p.set("tab", "activity");
                window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
                window.dispatchEvent(new PopStateEvent("popstate"));
                setOpen(false);
              }}
            >View incident log →</a>
          </footer>
        </div>
      )}
    </div>
  );
};

function SysCheck({ label, status, okLabel, degradedLabel, downLabel }) {
  const norm = status === "rate-limited" ? "degraded" : status;
  const dotCls = norm === "ok" ? "ok" : norm === "down" ? "error" : "warn";
  const text = norm === "ok" ? (okLabel || "ok")
             : norm === "down" ? (downLabel || "down")
             : (degradedLabel || status);
  return (
    <li>
      <span className={`sys-dot sys-dot-${dotCls}`} />
      <span className="sys-check-label">{label}</span>
      <span className={`sys-check-status sys-${dotCls}`}>{text}</span>
    </li>
  );
}
