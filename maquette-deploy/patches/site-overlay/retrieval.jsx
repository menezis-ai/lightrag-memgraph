// Retrieval tab — conversation + citations + Parameters panel
const { useState, useRef, useEffect } = React;

function relTime(ts) {
  if (ts === null || ts === undefined || ts === "") return "";
  // Accept numeric epoch or ISO string. Anything else (or NaN parse) is treated as unknown.
  const epoch = typeof ts === "number" ? ts : Date.parse(ts);
  if (!Number.isFinite(epoch)) return "";
  const d = Date.now() - epoch;
  if (d < 0) return "now";
  if (d < 60e3) return "now";
  if (d < 3600e3) return Math.round(d / 60e3) + "m";
  if (d < 86400e3) return Math.round(d / 3600e3) + "h";
  return Math.round(d / 86400e3) + "d";
}

function parseAnswer(tokens) {
  // tokens are strings, some contain {cite:n} markers and `code` markdown
  // Returns array of {type:'text'|'cite'|'code', value}
  const out = [];
  tokens.forEach(tk => {
    const re = /\{cite:(\d+)\}|`([^`]+)`/g;
    let last = 0; let m;
    while ((m = re.exec(tk)) !== null) {
      if (m.index > last) out.push({ type: "text", value: tk.slice(last, m.index) });
      if (m[1]) out.push({ type: "cite", value: parseInt(m[1], 10) });
      else if (m[2]) out.push({ type: "code", value: m[2] });
      last = re.lastIndex;
    }
    if (last < tk.length) out.push({ type: "text", value: tk.slice(last) });
  });
  return out;
}

const QUERY_MODE_HINTS = [
  { id: "naive",  desc: "Plain vector search. Fast, no graph traversal." },
  { id: "local",  desc: "Vector + 1-hop graph neighbours. Best for entity-anchored questions." },
  { id: "global", desc: "Community summaries first. Best for broad, thematic questions." },
  { id: "hybrid", desc: "Local + global blended. Slower, broader recall." },
  { id: "mix",    desc: "Default. Adaptive blend of all modes per query." },
  { id: "bypass", desc: "Debug only. Skip retrieval, send the query straight to the LLM." }
];

function QueryModeInfo() {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <span ref={ref} className="field-label-info">
      <button
        type="button"
        className="info-btn"
        aria-label="Query mode help"
        aria-expanded={open}
        onClick={() => setOpen(o => !o)}
      >
        <Icon name="info-circle" size={12} />
      </button>
      {open && (
        <div className="tooltip query-mode-tooltip" role="tooltip">
          {QUERY_MODE_HINTS.map(h => (
            <div key={h.id} className="tooltip-row">
              <span className="cat">{h.id}</span>
              <span className="desc">{h.desc}</span>
            </div>
          ))}
        </div>
      )}
    </span>
  );
}

window.RetrievalTab = function RetrievalTab() {
  const sys = window.useReadOnly ? window.useReadOnly() : { effectiveReadOnly: false, readOnlyReason: "" };
  const ro = sys.effectiveReadOnly;
  // Params panel auto-collapses under 1500px so the conv slot gets the
  // room (proto's 240/1fr/320 grid otherwise shrinks the conversation
  // to ~600px at 1366px, wrapping every answer chunk on 4 lines).
  const [paramsOpen, setParamsOpen] = useState(() =>
    typeof window !== "undefined" ? window.innerWidth >= 1500 : true
  );
  const [query, setQuery] = useState("");
  const [threads, setThreads] = useState(() => {
    try { const raw = localStorage.getItem("twin-rag.threads"); if (raw) return JSON.parse(raw); } catch (e) {}
    return window.MOCK_THREADS || [];
  });
  const [activeThreadId, setActiveThreadId] = useState(() => (window.MOCK_THREADS && window.MOCK_THREADS[0] && window.MOCK_THREADS[0].id) || null);
  const activeThread = threads.find(t => t.id === activeThreadId);
  const convo = activeThread ? activeThread.messages : [];
  const setConvo = (updater) => {
    setThreads(ts => {
      // Auto-create a thread if none active
      let id = activeThreadId;
      let arr = ts;
      if (!id || !arr.find(t => t.id === id)) {
        id = "th_" + Math.random().toString(16).slice(2, 8);
        arr = [{ id, title: "New thread", created: Date.now(), updated: Date.now(), messages: [] }, ...arr];
        setActiveThreadId(id);
      }
      return arr.map(t => t.id !== id ? t : {
        ...t,
        updated: Date.now(),
        messages: typeof updater === "function" ? updater(t.messages) : updater,
        title: (t.messages.length === 0 && typeof updater === "function")
          ? (updater(t.messages).find(m => m.role === "user") || {}).text?.slice(0, 64) || t.title
          : t.title
      });
    });
  };
  useEffect(() => {
    try { localStorage.setItem("twin-rag.threads", JSON.stringify(threads)); } catch (e) {}
  }, [threads]);
  const newThread = () => {
    const id = "th_" + Math.random().toString(16).slice(2, 8);
    setThreads(ts => [{ id, title: "New thread", created: Date.now(), updated: Date.now(), messages: [] }, ...ts]);
    setActiveThreadId(id);
    setStreamedTokens([]); setStreaming(false);
  };
  const deleteThread = (id) => {
    setThreads(ts => {
      const next = ts.filter(t => t.id !== id);
      if (id === activeThreadId) setActiveThreadId(next[0] ? next[0].id : null);
      return next;
    });
  };
  const [streaming, setStreaming] = useState(false);
  const [streamedTokens, setStreamedTokens] = useState([]);
  const [highlightSrc, setHighlightSrc] = useState(null);
  const [tagFilters, setTagFilters] = window.useUrlArrayParam("rtag", ["rman"]);
  const [tagInput, setTagInput] = useState("");
  const [queryMode, setQueryMode] = window.useUrlParam("mode", "mix", {
    validate: v => ["naive","local","global","hybrid","mix","bypass"].includes(v)
  });
  const [topK, setTopK] = window.useUrlNumberParam("topk", 10);
  const [maxTok, setMaxTok] = window.useUrlNumberParam("maxtok", 4000);
  const [history, setHistory] = window.useUrlNumberParam("hist", 3);
  const [onlyCtx, setOnlyCtx] = useState(false);
  const [onlyPrompt, setOnlyPrompt] = useState(false);
  const convRef = useRef(null);
  const lastTurnRef = useRef(null);

  const send = (text) => {
    const q = (text === undefined ? query : text).trim();
    if (!q) return;
    window.twinCompleteTask && window.twinCompleteTask("query");
    setQuery("");
    setConvo(c => [...c, { role: "user", text: q }]);
    setStreamedTokens([]);
    setStreaming(true);

    // Simulate streaming
    const tokens = window.MOCK_ANSWER_TOKENS;
    let i = 0;
    const interval = setInterval(() => {
      i++;
      setStreamedTokens(tokens.slice(0, i));
      if (i >= tokens.length) {
        clearInterval(interval);
        setStreaming(false);
        setConvo(c => [...c, {
          role: "assistant",
          tokens: tokens,
          sources: window.MOCK_RETRIEVAL_SOURCES
        }]);
        setStreamedTokens([]);
      }
    }, 70);
  };

  useEffect(() => {
    if (convRef.current) convRef.current.scrollTop = convRef.current.scrollHeight;
  }, [streamedTokens, convo]);

  const clear = () => { setConvo([]); setStreamedTokens([]); setStreaming(false); };

  const onCiteHover = (n) => setHighlightSrc(n);
  const onCiteLeave = () => setTimeout(() => setHighlightSrc(null), 200);
  const onCiteClick = (n) => {
    setHighlightSrc(n);
    const el = document.getElementById(`src-${n}`);
    if (el) el.scrollIntoView ? null : null; // avoid scrollIntoView per guidance; do manual
    if (el && convRef.current) {
      const containerTop = convRef.current.getBoundingClientRect().top;
      const elTop = el.getBoundingClientRect().top;
      convRef.current.scrollBy({ top: elTop - containerTop - 80, behavior: "smooth" });
    }
    setTimeout(() => setHighlightSrc(null), 1400);
  };

  const removeTag = (t) => setTagFilters(tagFilters.filter(x => x !== t));
  const addTag = (t) => {
    if (t && !tagFilters.includes(t)) setTagFilters([...tagFilters, t]);
    setTagInput("");
  };
  // Jump to the Tags tab with `req=<name>` so the steward can validate
  // the new tag through the governance flow (tags.jsx auto-opens the
  // Request modal on that param). Replaces the silent fail when the
  // typed tag isn't in the thesaurus.
  const requestNewTag = (name) => {
    const p = new URLSearchParams(window.location.search);
    p.set("tab", "tags");
    p.set("req", name.trim().toLowerCase());
    window.history.pushState(null, "", window.location.pathname + "?" + p.toString());
    window.dispatchEvent(new PopStateEvent("popstate"));
  };

  const tagSugg = window.MOCK_THESAURUS
    .filter(t => !tagFilters.includes(t.tag))
    .filter(t => !tagInput || t.tag.includes(tagInput.toLowerCase()))
    .slice(0, 4);

  return (
    <div className={`retrieval has-history${paramsOpen ? "" : " is-params-collapsed"}`}>
      <aside className="history-panel">
        <div className="history-head">
          <span className="history-title">Conversations</span>
          <button className="history-new" onClick={newThread} title="New conversation">
            <Icon name="plus" size={12} /> New
          </button>
        </div>
        <ul className="history-list">
          {threads.length === 0 && (<li className="history-empty">No conversations yet</li>)}
          {threads.map(t => (
            <li
              key={t.id}
              className={"history-item" + (t.id === activeThreadId ? " is-active" : "")}
              onClick={() => setActiveThreadId(t.id)}
            >
              <div className="history-item-title" title={t.title}>{t.title}</div>
              <div className="history-item-meta">
                <span>{t.messages.filter(m => m.role === "user").length} q · {t.messages.filter(m => m.role === "assistant").length} a</span>
                <span className="sep">·</span>
                <span>{relTime(t.updated)}</span>
              </div>
              <button className="history-del" title="Delete" onClick={e => { e.stopPropagation(); deleteThread(t.id); }}>
                <Icon name="x" size={11} />
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <div className="retrieval-main">
        <div className="retrieval-conv" ref={convRef}>
          {convo.length === 0 && !streaming && (
            <div className="empty-state">
              <Icon name="search" size={28} color="var(--color-text-tertiary)" />
              <div className="title">Ask a question to retrieve from the knowledge base</div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center", marginTop: 8 }}>
                <button className="suggestion" onClick={() => send("How do I restart Oracle on RHEL 9?")}>
                  Try: "How do I restart Oracle on RHEL 9?"
                </button>
                <button className="suggestion" onClick={() => send("Common RMAN backup errors")}>
                  Try: "Common RMAN backup errors"
                </button>
                <button className="suggestion" onClick={() => send("CFT troubleshooting checklist")}>
                  Try: "CFT troubleshooting checklist"
                </button>
              </div>
            </div>
          )}
          {convo.map((m, i) => (
            <Turn
              key={i}
              msg={m}
              highlightSrc={highlightSrc}
              onCiteHover={onCiteHover}
              onCiteLeave={onCiteLeave}
              onCiteClick={onCiteClick}
            />
          ))}
          {streaming && streamedTokens.length > 0 && (
            <Turn
              streaming
              msg={{ role: "assistant", tokens: streamedTokens, sources: window.MOCK_RETRIEVAL_SOURCES }}
              highlightSrc={highlightSrc}
              onCiteHover={onCiteHover}
              onCiteLeave={onCiteLeave}
              onCiteClick={onCiteClick}
            />
          )}
        </div>
        <div className="querybar">
          <button className="btn subtle" onClick={clear}>
            <Icon name="x" size={13} /> Clear
          </button>
          <textarea
            placeholder="Type your query…"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
          />
          <button className="btn primary" onClick={() => send()} disabled={streaming || ro} title={ro ? `Disabled — ${sys.readOnlyReason}` : undefined}>
            <Icon name="send" size={13} /> Send
          </button>
        </div>
      </div>

      {paramsOpen ? (
      <aside className="params-panel">
        <div className="params-header">
          <h3>Parameters</h3>
          <p>Configure your query</p>
          <button
            className="params-collapse"
            onClick={() => setParamsOpen(false)}
            aria-label="Collapse parameters panel"
            title="Collapse panel"
          >
            <Icon name="x" size={12} />
          </button>
        </div>

        <div className="field">
          <label className="field-label">
            Query mode
            <QueryModeInfo />
          </label>
          <select value={queryMode} onChange={e => setQueryMode(e.target.value)}>
            <option value="naive">naive</option>
            <option value="local">local</option>
            <option value="global">global</option>
            <option value="hybrid">hybrid</option>
            <option value="mix">mix</option>
            <option value="bypass">bypass</option>
          </select>
        </div>

        <div className="field">
          <label className="field-label">Tag filter <span style={{ color: "var(--color-text-tertiary)", fontSize: 10 }}>— Twin</span></label>
          <div className="chip-input">
            {tagFilters.map(t => <TagChip key={t} tag={t} removable onRemove={removeTag} />)}
            <input
              value={tagInput}
              onChange={e => setTagInput(e.target.value)}
              onKeyDown={e => { if (e.key === "Enter" && tagSugg[0]) addTag(tagSugg[0].tag); }}
              placeholder={tagFilters.length ? "" : "add tag…"}
            />
          </div>
          {tagInput && tagSugg.length > 0 && (
            <div className="autocomplete" style={{ marginTop: 4 }}>
              {tagSugg.map((s, i) => (
                <div key={s.tag} className={`autocomplete-row${i === 0 ? " focus" : ""}`} onMouseDown={() => addTag(s.tag)}>
                  <div className="row1">
                    <span style={{ fontSize: 12 }}>{s.tag}</span>
                    <span className="badge">{s.category}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
          {tagInput && tagSugg.length === 0 && (
            <div className="tag-input-miss" role="status">
              <Icon name="info-circle" size={11} />
              <span>
                No tag named <code>{tagInput}</code> in the thesaurus.
                {" "}
                <button className="link-btn small" onMouseDown={() => requestNewTag(tagInput)}>
                  Request new tag →
                </button>
              </span>
            </div>
          )}
        </div>

        <div className="field">
          <label className="field-label">Top K results</label>
          <input type="number" value={topK} onChange={e => setTopK(parseInt(e.target.value || 0))} />
        </div>
        <div className="field">
          <label className="field-label">Max tokens · text unit</label>
          <input type="number" value={maxTok} onChange={e => setMaxTok(parseInt(e.target.value || 0))} />
        </div>
        <div className="field">
          <label className="field-label">History turns</label>
          <input type="number" value={history} onChange={e => setHistory(parseInt(e.target.value || 0))} />
        </div>
        <div className="toggle">
          <span className={`switch${onlyCtx ? " on" : ""}`} onClick={() => setOnlyCtx(!onlyCtx)} />
          Only need context
        </div>
        <div className="toggle">
          <span className={`switch${onlyPrompt ? " on" : ""}`} onClick={() => setOnlyPrompt(!onlyPrompt)} />
          Only need prompt
        </div>

        <div className="connected">
          <span className="dot" /> Connected
        </div>
      </aside>
      ) : (
        <button
          className="params-collapsed-rail"
          onClick={() => setParamsOpen(true)}
          aria-label="Show retrieval parameters"
          title="Show parameters"
        >
          <Icon name="settings" size={12} />
          <span>Params</span>
        </button>
      )}
    </div>
  );
};

function Turn({ msg, streaming, highlightSrc, onCiteHover, onCiteLeave, onCiteClick }) {
  if (msg.role === "user") return <div className="msg-user">{msg.text}</div>;

  const parts = parseAnswer(msg.tokens);

  return (
    <div className="msg-assistant">
      <div className="msg-text">
        {parts.map((p, i) => {
          if (p.type === "text") return <React.Fragment key={i}>{p.value}</React.Fragment>;
          if (p.type === "code") return <code key={i}>{p.value}</code>;
          if (p.type === "cite") {
            const src = msg.sources && msg.sources.find(s => s.n === p.value);
            return (
              <span key={i} className="citation-wrap">
                <button
                  className="citation"
                  onMouseEnter={() => onCiteHover(p.value)}
                  onMouseLeave={onCiteLeave}
                  onClick={() => onCiteClick(p.value)}
                  aria-label={src ? `Source ${p.value} — ${src.name} (score ${src.score.toFixed(2)})` : `Source ${p.value}`}
                >{p.value}</button>
                {src && (
                  <span className="citation-tooltip" role="tooltip">
                    <span className="ct-name">{src.name}</span>
                    <span className="ct-score">{src.score.toFixed(2)}</span>
                  </span>
                )}
              </span>
            );
          }
          return null;
        })}
        {streaming && <span className="cursor" style={{ display: "inline-block", width: 6, height: 14, background: "var(--twin-accent)", verticalAlign: "-2px", marginLeft: 2, animation: "blink 1s infinite" }} />}
      </div>
      {!streaming && msg.sources && (
        <>
          <div className="sources-header">Sources</div>
          <div className="sources-list">
            {msg.sources.map(s => (
              <div
                key={s.n}
                id={`src-${s.n}`}
                className={`source-card${highlightSrc === s.n ? " hl" : ""}`}
              >
                <span className="src-pill">{s.n}</span>
                <SourceIcon type={s.type} size={13} />
                <span className={s.type !== "file" ? "src-name mono" : "src-name"}>{s.name}</span>
                {s.meta && <span className="src-meta">{s.meta}</span>}
                <span className="src-score">{s.score.toFixed(2)}</span>
                <span className="src-ext" title="Open source"><Icon name="external-link" size={12} /></span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
