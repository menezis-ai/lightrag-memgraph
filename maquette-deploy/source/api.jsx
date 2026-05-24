// Swagger-style API screen — proxies the upstream LightRAG OpenAPI surface 1:1.
const { useState } = React;
// We don't fork the endpoint shapes; the Twin layer only injects tag/visibility
// scoping at the gateway. This screen is therefore the *vanilla* swagger,
// rendered in our chrome.

const API_VERSION = "v1.4.12/0279"; // env: LIGHTRAG_VERSION
const OPENAPI_GROUPS = [
  {
    id: "documents",
    name: "documents",
    desc: "Source ingestion, listing and lifecycle.",
    endpoints: [
      { m: "POST",   p: "/documents/upload",          s: "Upload Document" },
      { m: "POST",   p: "/documents/text",            s: "Insert Text" },
      { m: "POST",   p: "/documents/texts",           s: "Insert Texts" },
      { m: "POST",   p: "/documents/scan",            s: "Scan For New Documents" },
      { m: "GET",    p: "/documents",                 s: "List Documents" },
      { m: "GET",    p: "/documents/pipeline_status", s: "Get Pipeline Status" },
      { m: "DELETE", p: "/documents",                 s: "Clear Documents" },
      { m: "DELETE", p: "/documents/delete_document", s: "Delete Document" },
      { m: "POST",   p: "/documents/clear_cache",     s: "Clear Cache" }
    ]
  },
  {
    id: "query",
    name: "query",
    desc: "Retrieval + LLM synthesis endpoints.",
    endpoints: [
      { m: "POST", p: "/query",         s: "Query Text" },
      { m: "POST", p: "/query/stream",  s: "Query Text Stream" }
    ]
  },
  {
    id: "graph",
    name: "graph",
    desc: "Knowledge-graph CRUD and label browsing.",
    endpoints: [
      { m: "GET",  p: "/graph/label/list",     s: "Get Graph Labels" },
      { m: "GET",  p: "/graph/label/popular",  s: "Get Popular Labels" },
      { m: "GET",  p: "/graph/label/search",   s: "Search Labels" },
      { m: "GET",  p: "/graphs",               s: "Get Knowledge Graph" },
      { m: "GET",  p: "/graph/entity/exists",  s: "Check Entity Exists" },
      { m: "POST", p: "/graph/entity/edit",    s: "Update Entity" },
      { m: "POST", p: "/graph/relation/edit",  s: "Update Relation" },
      { m: "POST", p: "/graph/entity/create",  s: "Create Entity" },
      { m: "POST", p: "/graph/relation/create",s: "Create Relation" }
    ]
  },
  {
    id: "ollama",
    name: "ollama",
    desc: "Drop-in Ollama-compatible chat & generate surface.",
    endpoints: [
      { m: "GET",  p: "/api/version",  s: "Get Version" },
      { m: "GET",  p: "/api/tags",     s: "Get Tags" },
      { m: "GET",  p: "/api/ps",       s: "Get Running Models" },
      { m: "POST", p: "/api/generate", s: "Generate" },
      { m: "POST", p: "/api/chat",     s: "Chat" }
    ]
  },
  {
    id: "default",
    name: "default",
    desc: "Auth, health and root.",
    endpoints: [
      { m: "GET",  p: "/",            s: "Redirect To Webui" },
      { m: "GET",  p: "/auth-status", s: "Get Auth Status" },
      { m: "POST", p: "/login",       s: "Login" },
      { m: "GET",  p: "/health",      s: "Get system health and configuration status" }
    ]
  }
];

const METHOD_COLOR = {
  GET:    { bg: "#E6EFFA", fg: "#1B5BAE", border: "#B5D4F4" },
  POST:   { bg: "#E5F3EA", fg: "#1F7A3A", border: "#B6DDC1" },
  DELETE: { bg: "#FBE7E7", fg: "#A33030", border: "#F0B7B7" },
  PUT:    { bg: "#FCEFDE", fg: "#9C5A0E", border: "#F0CFA0" },
  PATCH:  { bg: "#E8F0EE", fg: "#15706B", border: "#A9D2CC" }
};

function MethodPill({ method }) {
  const c = METHOD_COLOR[method] || METHOD_COLOR.GET;
  return (
    <span
      style={{
        display: "inline-flex",
        justifyContent: "center",
        alignItems: "center",
        minWidth: 62,
        height: 24,
        padding: "0 8px",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: 0.4,
        color: c.fg,
        background: c.bg,
        border: `0.5px solid ${c.border}`,
        borderRadius: 4
      }}
    >
      {method}
    </span>
  );
}

function EndpointRow({ ep, secured, token }) {
  const [open, setOpen] = useState(false);
  const [tryOpen, setTryOpen] = useState(false);
  const [reqBody, setReqBody] = useState(() => requestBodyFor(ep));
  const [resp, setResp] = useState(null);
  const [running, setRunning] = useState(false);

  const execute = () => {
    setRunning(true);
    setResp(null);
    setTimeout(() => {
      const unauth = secured && !token;
      setRunning(false);
      setResp(unauth ? mockUnauthorized(ep) : mockResponseFor(ep, reqBody));
    }, 480);
  };
  const reset = () => { setReqBody(requestBodyFor(ep)); setResp(null); };
  return (
    <div className={"swagger-row " + (open ? "is-open" : "")}>
      <button className="swagger-row-head" onClick={() => setOpen(o => !o)}>
        <MethodPill method={ep.m} />
        <code className="swagger-path">{ep.p}</code>
        <span className="swagger-summary">{ep.s}</span>
        <span className="swagger-lock" title={secured ? "Requires bearer token" : "Public"}>
          <Icon name={secured ? "lock" : "lock-open"} size={13} color="var(--color-text-tertiary)" />
        </span>
        <Icon
          name="chevron-down"
          size={14}
          color="var(--color-text-tertiary)"
          style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform .15s" }}
        />
      </button>
      {open && (
        <div className="swagger-row-body">
          <div className="swagger-section">
            <div className="swagger-section-h">Parameters</div>
            {ep.m === "GET" && ep.p.includes("label") ? (
              <table className="swagger-params">
                <thead>
                  <tr><th>Name</th><th>Type</th><th>In</th><th>Description</th></tr>
                </thead>
                <tbody>
                  <tr><td><code>limit</code></td><td>integer</td><td>query</td><td>Default 50. Max 500.</td></tr>
                  <tr><td><code>q</code></td><td>string</td><td>query</td><td>Substring filter (case-insensitive).</td></tr>
                </tbody>
              </table>
            ) : (
              <div className="swagger-empty">No parameters</div>
            )}
          </div>
          <div className="swagger-section">
            <div className="swagger-section-h">Request body</div>
            {ep.m === "POST" ? (
              <pre className="swagger-code">{requestBodyFor(ep)}</pre>
            ) : (
              <div className="swagger-empty">—</div>
            )}
          </div>
          <div className="swagger-section">
            <div className="swagger-section-h">Responses</div>
            <table className="swagger-responses">
              <tbody>
                <tr><td className="code-cell ok">200</td><td>Successful Response</td></tr>
                {ep.m !== "GET" && (
                  <tr><td className="code-cell err">422</td><td>Validation Error</td></tr>
                )}
                {secured && (
                  <tr><td className="code-cell err">401</td><td>Unauthorized</td></tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="swagger-actions">
            <button
              className={"swagger-tryit" + (tryOpen ? " is-on" : "")}
              onClick={() => setTryOpen(t => !t)}
            >{tryOpen ? "Cancel" : "Try it out"}</button>
          </div>
          {tryOpen && (
            <div className="swagger-tryit-panel">
              <div className="swagger-section-h">Request <span className="swagger-curl-hint">curl preview</span></div>
              <pre className="swagger-code curl">{curlFor(ep, reqBody, token)}</pre>
              {ep.m !== "GET" && (
                <>
                  <div className="swagger-section-h">Body</div>
                  <textarea
                    className="swagger-body-edit"
                    value={reqBody}
                    onChange={e => setReqBody(e.target.value)}
                    spellCheck="false"
                  />
                </>
              )}
              <div className="swagger-tryit-actions">
                <button className="primary-btn" onClick={execute} disabled={running}>
                  {running ? <><Icon name="refresh" size={12} /> Executing…</> : <><Icon name="arrow-right" size={12} /> Execute</>}
                </button>
                <button className="ghost-btn" onClick={reset}>Reset</button>
                {secured && !token && (
                  <span className="swagger-warn"><Icon name="lock" size={12} /> Endpoint requires bearer — click Authorize</span>
                )}
              </div>
              {resp && (
                <div className="swagger-resp">
                  <div className="swagger-mock-banner" role="note">
                    <Icon name="info-circle" size={11} />
                    <span><b>MOCK</b> · static fixture from this demo bundle, not a live call to the backend</span>
                  </div>
                  <div className="swagger-resp-h">
                    <span className={"code-cell " + (resp.status < 300 ? "ok" : "err")}>{resp.status}</span>
                    <span className="swagger-resp-msg">{resp.statusText}</span>
                    <span className="swagger-sep">·</span>
                    <span className="swagger-resp-time">{resp.tookMs}ms</span>
                  </div>
                  <pre className="swagger-code resp">{resp.body}</pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function requestBodyFor(ep) {
  if (ep.p === "/query" || ep.p === "/query/stream") {
    return JSON.stringify({
      query: "How do I restart Oracle RMAN after a failed backup?",
      mode: "hybrid",
      top_k: 60,
      response_type: "Multiple Paragraphs",
      tag_filter: { all: ["rman"], any: [] }
    }, null, 2);
  }
  if (ep.p === "/documents/text") {
    return JSON.stringify({
      text: "",
      file_source: "",
      tags: ["twin"]
    }, null, 2);
  }
  if (ep.p.includes("/graph/entity/edit")) {
    return JSON.stringify({ entity_name: "", updated_data: {} }, null, 2);
  }
  return JSON.stringify({}, null, 2);
}

function curlFor(ep, body, token) {
  const base = "https://cib-kb.twin.internal";
  const lines = [`curl -X ${ep.m} '${base}${ep.p}' \\`];
  lines.push("  -H 'Accept: application/json' \\");
  if (ep.m !== "GET") lines.push("  -H 'Content-Type: application/json' \\");
  if (token) lines.push(`  -H 'Authorization: Bearer ${token.slice(0, 6)}…' \\`);
  if (ep.m !== "GET") lines.push(`  -d '${(body || "").replace(/\n\s*/g, " ")}'`);
  else lines[lines.length - 1] = lines[lines.length - 1].replace(/ \\$/, "");
  return lines.join("\n");
}

function mockUnauthorized(ep) {
  return {
    status: 401, statusText: "Unauthorized", tookMs: 12,
    body: JSON.stringify({ detail: "Missing or invalid Bearer token. Use Authorize to attach one." }, null, 2)
  };
}
function mockResponseFor(ep, body) {
  const tookMs = 120 + Math.floor(Math.random() * 380);
  if (ep.p === "/query" || ep.p === "/query/stream") {
    return {
      status: 200, statusText: "OK", tookMs,
      body: JSON.stringify({
        response: "To restart Oracle RMAN after a failed backup, first verify the recovery catalog state … [truncated]",
        sources: [
          { id: "chunk_4a12", source: "oracle-restart-procedure.pdf", score: 0.91, tags: ["rman", "oracle"] },
          { id: "chunk_88e0", source: "DBA Runbook · Backup recovery", score: 0.84, tags: ["rman"] }
        ],
        mode: "hybrid", tag_filter: { all: ["rman"] }, took_ms: tookMs
      }, null, 2)
    };
  }
  if (ep.m === "GET" && ep.p === "/documents") {
    return {
      status: 200, statusText: "OK", tookMs,
      body: JSON.stringify({ items: [{ id: "doc_001", source: "oracle-restart.pdf", status: "completed" }], total: 247 }, null, 2)
    };
  }
  if (ep.p === "/documents/upload" || ep.p === "/documents/text") {
    return {
      status: 200, statusText: "OK", tookMs,
      body: JSON.stringify({ id: "doc_" + Math.random().toString(16).slice(2, 8), status: "pending", queued_at: new Date().toISOString() }, null, 2)
    };
  }
  if (ep.p === "/health") {
    return { status: 200, statusText: "OK", tookMs, body: JSON.stringify({ status: "ok", uptime_s: 184213 }, null, 2) };
  }
  if (ep.p === "/auth-status") {
    return { status: 200, statusText: "OK", tookMs, body: JSON.stringify({ authorized: true, scopes: ["read:documents", "read:query"] }, null, 2) };
  }
  return { status: 200, statusText: "OK", tookMs, body: JSON.stringify({ ok: true }, null, 2) };
}

window.ApiTab = function ApiTab() {
  const [filter, setFilter] = useState("");
  const [server, setServer] = useState("prod");
  const [authOpen, setAuthOpen] = useState(false);
  const [token, setToken] = useState("");
  const norm = filter.trim().toLowerCase();

  const filtered = OPENAPI_GROUPS.map(g => ({
    ...g,
    endpoints: g.endpoints.filter(e =>
      !norm ||
      e.p.toLowerCase().includes(norm) ||
      e.s.toLowerCase().includes(norm) ||
      e.m.toLowerCase() === norm
    )
  })).filter(g => g.endpoints.length);

  return (
    <div className="swagger">
      <div className="swagger-topbar">
        <div className="swagger-title">
          <span className="swagger-title-main">LightRAG Server API</span>
          <span className="swagger-version">{API_VERSION}</span>
          <span className="swagger-oas">OAS 3.1</span>
        </div>
        <div className="swagger-meta">
          <code>/openapi.json</code>
          <span className="swagger-sep">·</span>
          <span>Providing API for LightRAG core, Web UI and Ollama Model Emulation</span>
        </div>
        <div className="swagger-banner">
          <Icon name="info-circle" size={14} color="var(--twin-accent)" />
          <span>
            Twin RAG fork inherits this surface unchanged. The gateway transparently
            injects <code>tag_filter</code> and <code>visibility</code> scoping from the
            current workspace — see <a href="#">HLA §4.1</a>.
          </span>
        </div>
        <div className="swagger-servers">
          <label>Servers</label>
          <select value={server} onChange={e => setServer(e.target.value)}>
            <option value="prod">https://cib-kb.twin.internal — production</option>
            <option value="stg">https://cib-kb.stg.twin.internal — staging</option>
          </select>
          <button className={"swagger-auth" + (token ? " is-on" : "")} onClick={() => setAuthOpen(true)}>
            <Icon name={token ? "circle-check" : "lock"} size={12} color={token ? "var(--twin-green-700, #2F7A40)" : "var(--color-text-secondary)"} />
            {token ? "Authorized" : "Authorize"}
          </button>
        </div>
        <div className="swagger-filter">
          <Icon name="search" size={13} color="var(--color-text-tertiary)" />
          <input
            type="text"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="Filter by path, summary or method (GET, POST…)"
          />
          {filter && (
            <button className="swagger-filter-clear" onClick={() => setFilter("")} aria-label="Clear">
              <Icon name="x" size={12} color="var(--color-text-tertiary)" />
            </button>
          )}
        </div>
      </div>

      <div className="swagger-groups">
        {filtered.map(g => <Group key={g.id} g={g} secured={g.id !== "default"} token={token} />)}
        {!filtered.length && (
          <div className="empty-state" style={{ padding: 60 }}>
            <div className="title">No endpoints match "{filter}"</div>
          </div>
        )}
      </div>
      {authOpen && (
        <AuthorizeDialog
          token={token}
          onSave={(t) => { setToken(t); setAuthOpen(false); }}
          onLogout={() => { setToken(""); setAuthOpen(false); }}
          onClose={() => setAuthOpen(false)}
        />
      )}
    </div>
  );
};

function AuthorizeDialog({ token, onSave, onLogout, onClose }) {
  const [val, setVal] = useState(token || "");
  const ref = React.useRef(null);
  window.useModalA11y && window.useModalA11y({ open: true, onClose, ref });
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal small" role="dialog" aria-modal="true" aria-labelledby="auth-title" ref={ref} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ flex: 1 }}>
            <h2 id="auth-title">Authorize</h2>
            <div className="ctx"><Icon name="lock" size={12} /> Bearer (HTTP, scheme: bearer)</div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" size={18} /></button>
        </div>
        <div className="modal-body">
          <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
            Paste your bearer token. It's attached to every request from "Try it out".
            In production this is delegated to the Twin gateway (Keycloak OIDC).
          </p>
          <label className="field-label" style={{ display: "block", marginBottom: 6 }}>Value</label>
          <input
            type="password"
            autoFocus
            value={val}
            onChange={e => setVal(e.target.value)}
            placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9…"
            style={{ width: "100%", fontFamily: "var(--font-mono)", fontSize: 12, padding: "8px 10px", borderRadius: 4, border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)" }}
          />
          <div style={{ marginTop: 14, fontSize: 11, color: "var(--color-text-tertiary)" }}>
            Scopes: <code>read:documents</code> <code>read:query</code> <code>write:documents</code> (Contributor or Steward)
          </div>
        </div>
        <div className="modal-footer">
          {token && <button className="ghost-btn" onClick={onLogout}>Logout</button>}
          <button className="ghost-btn" onClick={onClose} style={{ marginLeft: "auto" }}>Close</button>
          <button className="primary-btn" onClick={() => onSave(val.trim())} disabled={!val.trim()}>Authorize</button>
        </div>
      </div>
    </div>
  );
}

function Group({ g, secured, token }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="swagger-group">
      <button className="swagger-group-head" onClick={() => setOpen(o => !o)}>
        <span className="swagger-group-name">{g.name}</span>
        <span className="swagger-group-desc">{g.desc}</span>
        <span className="swagger-group-count">{g.endpoints.length}</span>
        <Icon
          name="chevron-down"
          size={14}
          color="var(--color-text-tertiary)"
          style={{ transform: open ? "none" : "rotate(-90deg)", transition: "transform .15s" }}
        />
      </button>
      <div className="swagger-group-line" />
      {open && (
        <div className="swagger-rows">
          {g.endpoints.map((ep, i) => <EndpointRow key={i} ep={ep} secured={secured} token={token} />)}
        </div>
      )}
    </div>
  );
}
