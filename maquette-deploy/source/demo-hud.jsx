// Demo HUD — orchestrates the 3-act scripted demo described by the user.
// Drives tab switches, prefilled queries, auto-actions in Ontology Studio,
// and shows a floating progress bar at the bottom of the screen.
//
// Hotkeys:
//   D       → toggle demo mode on/off
//   1/2/3   → jump to act
//   →       → advance to next step within an act
//   Esc     → exit demo mode

const {
  useState: _dhUseState, useEffect: _dhUseEffect, useRef: _dhUseRef, useCallback: _dhUseCallback
} = React;

// Pub/sub bus so the Ontology Studio (and other tabs) can listen for demo
// events without prop-drilling.
function makeDemoBus() {
  const listeners = {};
  return {
    on(channel, fn) {
      (listeners[channel] = listeners[channel] || []).push(fn);
      return () => {
        listeners[channel] = (listeners[channel] || []).filter(f => f !== fn);
      };
    },
    emit(channel, payload) {
      (listeners[channel] || []).forEach(fn => {
        try { fn(payload); } catch (e) { console.warn("demo bus", e); }
      });
    }
  };
}
window.__demoBus = window.__demoBus || makeDemoBus();

// ─── Script ─────────────────────────────────────────────────────────────
const DEMO_QUERY_NAIVE = "What is the procedure to restart the SRV-PARIS-01 server?";
const DEMO_QUERY_IMPACT = "If router R-CORE-02 goes down, which payment applications are impacted and is there an up-to-date DR plan?";

const ACTS = [
  {
    id: 1,
    title: "Act 1 · Classic RAG limitations",
    sub: "Architecture question scattered across 10 documents.",
    steps: [
      { label: "Go to Retrieval",   action: "goto-retrieval" },
      { label: "Prefill complex question", action: "prefill-impact" },
      { label: "Run (fragmented answer)", action: "send-naive" }
    ]
  },
  {
    id: 2,
    title: "Act 2 · Steward Expert Mode",
    sub: "The steward sculpts the ontology by hand.",
    steps: [
      { label: "Ouvrir Ontology Studio", action: "goto-studio" },
      { label: "Focus · R-CORE-02 ↔ SWIFT-Payment", action: "studio-focus" },
      { label: "Create the CRITICAL_DEPENDENCY relation", action: "studio-create-edge" }
    ]
  },
  {
    id: 3,
    title: "Act 3 · Resilience stress test",
    sub: "Same data — the graph enables simulation.",
    steps: [
      { label: "Activer l'analyse depuis R-CORE-02", action: "studio-wargame" },
      { label: "Back to Retrieval", action: "goto-retrieval-impact" },
      { label: "Render impact widget", action: "send-impact" }
    ]
  }
];

window.DemoHUD = function DemoHUD({ active, setActive, onGoto, onPrefill, onSend }) {
  const [actIdx, setActIdx] = _dhUseState(0);
  const [stepIdx, setStepIdx] = _dhUseState(-1);
  const [busy, setBusy] = _dhUseState(false);
  const [collapsed, setCollapsed] = _dhUseState(false);

  const reset = () => { setActIdx(0); setStepIdx(-1); setBusy(false); };

  // Hotkeys
  _dhUseEffect(() => {
    const onKey = (e) => {
      // Ignore key events when an input/textarea/contenteditable has focus,
      // so typing the letter "d" in the Retrieval composer doesn't fire the
      // demo toggle.
      const t = e.target;
      const isTextInput = t && (
        t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
        t.isContentEditable
      );
      if (isTextInput && !e.metaKey && !e.ctrlKey) return;

      if (e.key === "d" || e.key === "D") {
        setActive(a => !a);
        return;
      }
      if (!active) return;
      if (e.key === "Escape") { setActive(false); reset(); return; }
      if (e.key === "1") { runAct(0); }
      if (e.key === "2") { runAct(1); }
      if (e.key === "3") { runAct(2); }
      if (e.key === "ArrowRight") { runNextStep(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line
  }, [active, actIdx, stepIdx]);

  const dispatch = async (action) => {
    if (action === "goto-retrieval") {
      onGoto("retrieval");
      await wait(400);
    }
    if (action === "prefill-impact") {
      onPrefill(DEMO_QUERY_IMPACT);
      await wait(280);
    }
    if (action === "send-naive") {
      // Bundle text+mode so retrieval doesn't race state updates.
      window.__demoBus.emit("retrieval", { kind: "send", text: DEMO_QUERY_IMPACT, mode: "naive" });
    }
    if (action === "goto-studio") {
      onGoto("ontology");
      await wait(450);
    }
    if (action === "studio-focus") {
      window.__demoBus.emit("studio", { kind: "focus-pair", a: "e_router02", b: "e_swift_pay" });
      await wait(800);
    }
    if (action === "studio-create-edge") {
      window.__demoBus.emit("studio", {
        kind: "create-edge",
        source: "e_router02",
        target: "e_swift_pay",
        label: "CRITICAL_DEPENDENCY",
        status: "validated"
      });
      await wait(900);
    }
    if (action === "studio-wargame") {
      window.__demoBus.emit("studio", { kind: "set-wargame", active: true, originId: "e_router02", depth: 3 });
      await wait(700);
    }
    if (action === "goto-retrieval-impact") {
      onGoto("retrieval");
      await wait(400);
      onPrefill(DEMO_QUERY_IMPACT);
    }
    if (action === "send-impact") {
      window.__demoBus.emit("retrieval", { kind: "send", text: DEMO_QUERY_IMPACT, mode: "wargame" });
    }
  };

  const runAct = async (i) => {
    if (busy) return;
    setBusy(true);
    setActIdx(i);
    setStepIdx(-1);
    const steps = ACTS[i].steps;
    for (let s = 0; s < steps.length; s++) {
      setStepIdx(s);
      // eslint-disable-next-line no-await-in-loop
      await dispatch(steps[s].action);
      // eslint-disable-next-line no-await-in-loop
      await wait(380);
    }
    setBusy(false);
  };

  const runNextStep = async () => {
    if (busy) return;
    const next = stepIdx + 1;
    if (next < ACTS[actIdx].steps.length) {
      setBusy(true);
      setStepIdx(next);
      await dispatch(ACTS[actIdx].steps[next].action);
      setBusy(false);
    } else if (actIdx + 1 < ACTS.length) {
      runAct(actIdx + 1);
    }
  };

  if (!active) {
    return (
      <button className="demo-fab" onClick={() => setActive(true)} title="Start scripted demo (D)">
        <span className="demo-fab-dot" /> Demo
        <kbd>D</kbd>
      </button>
    );
  }

  const act = ACTS[actIdx];
  return (
    <div className={`demo-hud${collapsed ? " is-collapsed" : ""}`} role="region" aria-label="Demo HUD">
      <div className="demo-hud-bar">
        <div className="demo-hud-left">
          <span className="demo-hud-pulse" />
          <span className="demo-hud-label">DEMO</span>
          <button
            className={`demo-hud-collapse`}
            onClick={() => setCollapsed(c => !c)}
            title={collapsed ? "Expand" : "Collapse"}
            aria-label={collapsed ? "Expand" : "Collapse"}
          ><Icon name={collapsed ? "chevron-up" : "chevron-down"} size={11} /></button>
        </div>

        {!collapsed && (
          <>
            <div className="demo-hud-acts">
              {ACTS.map((a, i) => (
                <button
                  key={a.id}
                  className={`demo-hud-act${i === actIdx ? " is-on" : ""}`}
                  onClick={() => runAct(i)}
                  disabled={busy}
                >
                  <span className="demo-hud-act-n">{a.id}</span>
                  <span className="demo-hud-act-title">{a.title.split("·")[1] ? a.title.split("·")[1].trim() : a.title}</span>
                </button>
              ))}
            </div>

            <div className="demo-hud-right">
              <button className="demo-hud-btn" onClick={runNextStep} disabled={busy} title="Next step (→)">
                Next <kbd>→</kbd>
              </button>
              <button className="demo-hud-btn ghost" onClick={() => { setActive(false); reset(); }} title="Exit (Esc)">
                Exit <kbd>Esc</kbd>
              </button>
            </div>
          </>
        )}
      </div>

      {!collapsed && (
        <div className="demo-hud-script">
          <div className="demo-hud-script-row">
            <div className="demo-hud-script-text">
              <div className="demo-hud-script-act">{act.title}</div>
              <div className="demo-hud-script-sub">{act.sub}</div>
            </div>
            <div className="demo-hud-steps">
              {act.steps.map((s, i) => (
                <div key={i} className={`demo-hud-step${i === stepIdx ? " is-on" : ""}${i < stepIdx ? " is-done" : ""}`}>
                  <span className="demo-hud-step-bullet" />
                  <span>{s.label}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="demo-hud-keys">
            <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd><span>jump to act</span>
            <span className="dot-sep">·</span>
            <kbd>→</kbd><span>next step</span>
            <span className="dot-sep">·</span>
            <kbd>D</kbd><span>toggle</span>
            <span className="dot-sep">·</span>
            <kbd>Esc</kbd><span>exit</span>
          </div>
        </div>
      )}
    </div>
  );
};

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }
