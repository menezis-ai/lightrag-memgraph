// Onboarding — welcome modal + floating checklist + empty-workspace helpers.
// Tracks 6 tasks. Components broadcast completion via window.twinCompleteTask(id);
// state persists in localStorage so first-visit logic survives reloads.

const { useState: _useStateO, useEffect: _useEffectO, useRef: _useRefO } = React;

const ONBOARD_KEY = "twin-rag.onboarding.v1";

window.TWIN_ONBOARD_TASKS = [
  {
    id: "signin",
    label: "Sign in to your workspace",
    body: "Done — you're authenticated against the corporate IDP.",
    auto: true
  },
  {
    id: "addSource",
    label: "Add your first source",
    body: "Upload a file, paste a Confluence URL, or connect SharePoint.",
    cta: "Open Documents",
    tab: "documents"
  },
  {
    id: "ingestDone",
    label: "Watch the pipeline finish",
    body: "Sources land in your KB once the embedder + indexer complete.",
    cta: "Open pipeline",
    tab: "documents"
  },
  {
    id: "tag",
    label: "Apply your first tag",
    body: "Retag a source, approve a request, or add tags at upload time — all count.",
    cta: "Open Tags",
    tab: "tags"
  },
  {
    id: "query",
    label: "Ask your first question",
    body: "Citations link back to the chunks that supported the answer.",
    cta: "Open Retrieval",
    tab: "retrieval"
  },
  {
    id: "invite",
    label: "Invite a teammate",
    body: "Stewards review tag requests; readers can query immediately.",
    cta: "Open Members",
    tab: "settings",
    sec: "members"
  }
];

// ─── Storage helpers ────────────────────────────────────────────────────
function loadOnboard() {
  try {
    const raw = localStorage.getItem(ONBOARD_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) {}
  return null;
}
function saveOnboard(state) {
  try { localStorage.setItem(ONBOARD_KEY, JSON.stringify(state)); } catch (e) {}
}
function initialOnboardState() {
  return {
    welcomed: false,
    dismissed: false,
    collapsed: false,
    tasks: { signin: true, addSource: false, ingestDone: false, tag: false, query: false, invite: false }
  };
}

// Force-states for the Tweaks demo:
//   "off"    : skip onboarding entirely
//   "welcome": show welcome modal regardless of localStorage
//   "mid"    : welcomed, 2 tasks done, checklist visible
//   "done"   : all tasks complete (celebration state)
//   "auto"   : default — read from localStorage
window.applyOnboardingPreset = function applyOnboardingPreset(preset) {
  if (preset === "off") {
    saveOnboard({ ...initialOnboardState(), welcomed: true, dismissed: true });
  } else if (preset === "welcome") {
    saveOnboard(initialOnboardState());
  } else if (preset === "mid") {
    saveOnboard({ ...initialOnboardState(), welcomed: true, tasks: { signin: true, addSource: true, ingestDone: true, tag: false, query: false, invite: false } });
  } else if (preset === "done") {
    saveOnboard({ ...initialOnboardState(), welcomed: true, tasks: { signin: true, addSource: true, ingestDone: true, tag: true, query: true, invite: true } });
  } else {
    return;
  }
  window.dispatchEvent(new Event("twin-onboard-refresh"));
};

window.twinCompleteTask = function twinCompleteTask(id) {
  window.dispatchEvent(new CustomEvent("twin-onboard-complete", { detail: id }));
};

// ─── Hook used by App ───────────────────────────────────────────────────
window.useOnboarding = function useOnboarding() {
  const [state, setState] = _useStateO(() => loadOnboard() || initialOnboardState());

  _useEffectO(() => {
    const onComplete = (e) => {
      setState(s => {
        if (s.tasks[e.detail]) return s;
        const next = { ...s, tasks: { ...s.tasks, [e.detail]: true } };
        saveOnboard(next);
        return next;
      });
    };
    const onRefresh = () => {
      setState(loadOnboard() || initialOnboardState());
    };
    window.addEventListener("twin-onboard-complete", onComplete);
    window.addEventListener("twin-onboard-refresh", onRefresh);
    return () => {
      window.removeEventListener("twin-onboard-complete", onComplete);
      window.removeEventListener("twin-onboard-refresh", onRefresh);
    };
  }, []);

  const set = (patch) => setState(s => {
    const next = typeof patch === "function" ? patch(s) : { ...s, ...patch };
    saveOnboard(next);
    return next;
  });

  return [state, set];
};

// ─── Welcome modal ──────────────────────────────────────────────────────
window.OnboardingWelcome = function OnboardingWelcome({ open, onClose, kbName, userName }) {
  const ref = _useRefO(null);
  window.useModalA11y && window.useModalA11y({ open, onClose, ref });
  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="welcome-modal" role="dialog" aria-modal="true" aria-labelledby="welcome-title" ref={ref} onClick={e => e.stopPropagation()}>
        <button className="welcome-x" onClick={onClose} aria-label="Close"><Icon name="x" size={16} /></button>
        <div className="welcome-illus" aria-hidden="true">
          <svg viewBox="0 0 240 110" width="240" height="110">
            <defs>
              <linearGradient id="welc-grad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="var(--twin-accent)" />
                <stop offset="100%" stopColor="var(--twin-accent-hover)" />
              </linearGradient>
            </defs>
            <rect x="20" y="30" width="56" height="60" rx="4" fill="url(#welc-grad)" opacity="0.18" />
            <rect x="92" y="20" width="56" height="70" rx="4" fill="url(#welc-grad)" opacity="0.28" />
            <rect x="164" y="34" width="56" height="56" rx="4" fill="url(#welc-grad)" opacity="0.18" />
            <line x1="76" y1="60" x2="92" y2="55" stroke="currentColor" strokeOpacity="0.3" strokeDasharray="2 3" />
            <line x1="148" y1="55" x2="164" y2="62" stroke="currentColor" strokeOpacity="0.3" strokeDasharray="2 3" />
            <circle cx="120" cy="55" r="6" fill="var(--twin-accent)" />
            <circle cx="48" cy="60" r="3.5" fill="var(--twin-accent)" opacity="0.6" />
            <circle cx="192" cy="62" r="3.5" fill="var(--twin-accent)" opacity="0.6" />
          </svg>
        </div>
        <h2 id="welcome-title" className="welcome-title">
          Welcome to TwinRAG{userName ? `, ${userName.split(" ")[0]}` : ""}.
        </h2>
        <p className="welcome-sub">
          You're the first steward of <code>{kbName || "this workspace"}</code>.
          Six short steps and your knowledge base is ready to answer.
        </p>
        <div className="welcome-promises">
          <div className="welcome-promise">
            <span className="welcome-promise-num">1</span>
            <div>
              <div className="welcome-promise-h">Bring your sources</div>
              <p>Files, Confluence spaces, SharePoint sites, or live URL feeds — synced and tagged automatically.</p>
            </div>
          </div>
          <div className="welcome-promise">
            <span className="welcome-promise-num">2</span>
            <div>
              <div className="welcome-promise-h">Govern with tags</div>
              <p>A shared thesaurus keeps retrieval scoped, auditable, and explainable across teams.</p>
            </div>
          </div>
          <div className="welcome-promise">
            <span className="welcome-promise-num">3</span>
            <div>
              <div className="welcome-promise-h">Ask, with citations</div>
              <p>Every answer points back to the chunks it came from. No hallucinated context.</p>
            </div>
          </div>
        </div>
        <div className="welcome-actions">
          <button className="ghost-btn" onClick={onClose}>Skip — I'll explore</button>
          <button className="primary-btn welcome-cta" onClick={onClose}>
            Get started <Icon name="arrow-right" size={12} />
          </button>
        </div>
      </div>
    </div>
  );
};

// ─── Floating checklist widget ──────────────────────────────────────────
window.OnboardingChecklist = function OnboardingChecklist({ state, set, onJump }) {
  if (state.dismissed) return null;
  const tasks = window.TWIN_ONBOARD_TASKS;
  const done = tasks.filter(t => state.tasks[t.id]).length;
  const total = tasks.length;
  const pct = Math.round((done / total) * 100);
  const allDone = done === total;
  const collapsed = state.collapsed && !allDone;
  const next = tasks.find(t => !state.tasks[t.id]);

  return (
    <div className={`onboard-widget${collapsed ? " is-collapsed" : ""}${allDone ? " is-complete" : ""}`} role="region" aria-label="Onboarding checklist">
      <header className="onboard-h">
        <button
          className="onboard-toggle"
          onClick={() => set(s => ({ ...s, collapsed: !s.collapsed }))}
          aria-expanded={!collapsed}
        >
          <span className="onboard-progress-ring" aria-hidden="true">
            <svg width="22" height="22" viewBox="0 0 22 22">
              <circle cx="11" cy="11" r="9" fill="none" stroke="var(--color-border-tertiary)" strokeWidth="2" />
              <circle
                cx="11" cy="11" r="9" fill="none"
                stroke={allDone ? "var(--twin-green-700, #2F7A40)" : "var(--twin-accent)"}
                strokeWidth="2"
                strokeDasharray={`${(pct / 100) * 2 * Math.PI * 9} ${2 * Math.PI * 9}`}
                strokeLinecap="round"
                transform="rotate(-90 11 11)"
                style={{ transition: "stroke-dasharray 360ms cubic-bezier(0.22, 0.8, 0.28, 1)" }}
              />
              {allDone && (
                <path d="M7 11l3 3l5-6" fill="none" stroke="var(--twin-green-700, #2F7A40)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              )}
            </svg>
          </span>
          <div className="onboard-h-text">
            <div className="onboard-h-title">
              {allDone ? "You're all set" : "Get started"}
            </div>
            <div className="onboard-h-sub">
              {allDone
                ? "Workspace fully configured · explore at your own pace"
                : `${done} of ${total} complete${next ? ` · next: ${next.label.toLowerCase()}` : ""}`}
            </div>
          </div>
          <Icon name={collapsed ? "chevron-up" : "chevron-down"} size={12} color="var(--color-text-tertiary)" />
        </button>
        <button
          className="onboard-dismiss"
          onClick={() => set({ dismissed: true })}
          aria-label="Dismiss onboarding"
          title="Dismiss"
        >
          <Icon name="x" size={12} />
        </button>
      </header>

      {!collapsed && (
        <ul className="onboard-tasks">
          {tasks.map(t => {
            const done = state.tasks[t.id];
            const isNext = !done && t.id === (next && next.id);
            return (
              <li key={t.id} className={`onboard-task${done ? " is-done" : ""}${isNext ? " is-next" : ""}`}>
                <span className={`onboard-check${done ? " is-done" : ""}`} aria-hidden="true">
                  {done ? <Icon name="circle-check" size={14} /> : <span className="onboard-check-empty" />}
                </span>
                <div className="onboard-task-body">
                  <div className="onboard-task-label">{t.label}</div>
                  <div className="onboard-task-detail">{t.body}</div>
                </div>
                {!done && t.cta && (
                  <button
                    className="onboard-cta"
                    onClick={() => onJump && onJump(t)}
                  >
                    {t.cta} <Icon name="arrow-right" size={10} />
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {allDone && !collapsed && (
        <div className="onboard-celebrate">
          <div className="onboard-celebrate-emoji" aria-hidden="true">✦</div>
          <div className="onboard-celebrate-body">
            <div className="onboard-celebrate-title">Nice work.</div>
            <p>Share the workspace URL with your team. Stewards review tags, readers can query straight away.</p>
            <button className="ghost-btn small" onClick={() => set({ dismissed: true })}>Hide this widget</button>
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Empty-workspace card (used in Documents when 0 sources) ────────────
window.EmptyWorkspaceCard = function EmptyWorkspaceCard({ onAddSource, onLoadDemo }) {
  return (
    <div className="empty-workspace">
      <div className="empty-workspace-illus" aria-hidden="true">
        <svg width="180" height="120" viewBox="0 0 180 120">
          <defs>
            <linearGradient id="empty-grad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--twin-accent)" stopOpacity="0.2" />
              <stop offset="100%" stopColor="var(--twin-accent)" stopOpacity="0" />
            </linearGradient>
          </defs>
          <rect x="40" y="30" width="100" height="60" rx="4" fill="none" stroke="var(--color-border-secondary)" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="55" y1="48" x2="105" y2="48" stroke="var(--color-border-secondary)" strokeWidth="1" />
          <line x1="55" y1="58" x2="125" y2="58" stroke="var(--color-border-secondary)" strokeWidth="1" />
          <line x1="55" y1="68" x2="95"  y2="68" stroke="var(--color-border-secondary)" strokeWidth="1" />
          <circle cx="140" cy="34" r="14" fill="url(#empty-grad)" />
          <circle cx="140" cy="34" r="11" fill="var(--color-background-primary)" stroke="var(--twin-accent)" strokeWidth="1.5" />
          <path d="M140 28v12 M134 34h12" stroke="var(--twin-accent)" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </div>
      <h2>Your knowledge base is empty</h2>
      <p>
        Add a source to start indexing. TwinRAG accepts PDFs, Markdown, DOCX,
        Confluence spaces, SharePoint sites, and live URL feeds. First ingestion
        typically completes in under a minute.
      </p>
      <div className="empty-workspace-actions">
        <button className="primary-btn large" onClick={onAddSource}>
          <Icon name="cloud-upload" size={13} /> Add your first source
        </button>
        {onLoadDemo && (
          <button className="ghost-btn" onClick={onLoadDemo}>
            <Icon name="refresh" size={11} /> Load demo data
          </button>
        )}
      </div>
      <ul className="empty-workspace-tips">
        <li><Icon name="info-circle" size={11} /> Tags applied at upload time propagate to every chunk.</li>
        <li><Icon name="info-circle" size={11} /> Once a source is "completed", it surfaces in Retrieval immediately.</li>
        <li><Icon name="info-circle" size={11} /> Re-ingestion happens automatically when source content changes.</li>
      </ul>
    </div>
  );
};
