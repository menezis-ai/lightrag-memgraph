/**
 * ApiTab — Swagger-style endpoint browser for the LightRAG OpenAPI surface.
 *
 * Ported from Desktop/UI/api.jsx. The Twin RAG fork inherits the upstream
 * shapes 1:1; the gateway injects `tag_filter` / `visibility` scoping.
 *
 * Behavior delta vs the proto:
 *   - `groups` are injected via props (no static window globals).
 *   - "Try it out" stays mock-only (deterministic responses), no real fetch.
 *   - Authorize dialog a11y via useModalA11y.
 */

import { useRef, useState } from 'react';
import { Icon } from './Icon';
import { useModalA11y } from '../hooks/useModalA11y';
import {
  METHOD_COLOR,
  type HttpMethod,
  type MockResponse,
  type OpenApiEndpoint,
  type OpenApiGroup,
  type OpenApiServer,
} from '../types/api';

export interface ApiTabProps {
  apiVersion: string;
  groups: readonly OpenApiGroup[];
  servers: readonly OpenApiServer[];
  baseUrl: Record<string, string>;
}

export function ApiTab({ apiVersion, groups, servers, baseUrl }: ApiTabProps) {
  const [filter, setFilter] = useState('');
  const [server, setServer] = useState(servers[0]?.id ?? '');
  const [authOpen, setAuthOpen] = useState(false);
  const [token, setToken] = useState('');
  const norm = filter.trim().toLowerCase();

  const filtered = groups
    .map((g) => ({
      ...g,
      endpoints: g.endpoints.filter(
        (e) =>
          !norm ||
          e.p.toLowerCase().includes(norm) ||
          e.s.toLowerCase().includes(norm) ||
          e.m.toLowerCase() === norm,
      ),
    }))
    .filter((g) => g.endpoints.length);

  const currentBase = baseUrl[server] ?? '';

  return (
    <div className="swagger">
      <div className="swagger-topbar">
        <div className="swagger-title">
          <span className="swagger-title-main">LightRAG Server API</span>
          <span className="swagger-version">{apiVersion}</span>
          <span className="swagger-oas">OAS 3.1</span>
        </div>
        <div className="swagger-meta">
          <code>/openapi.json</code>
          <span className="swagger-sep">·</span>
          <span>
            Providing API for LightRAG core, Web UI and Ollama Model Emulation
          </span>
        </div>
        <div className="swagger-banner">
          <Icon name="info-circle" size={14} color="var(--twin-accent)" />
          <span>
            Twin RAG fork inherits this surface unchanged. The gateway transparently
            injects <code>tag_filter</code> and <code>visibility</code> scoping from
            the current space.
          </span>
        </div>
        <div className="swagger-servers">
          <label htmlFor="swagger-server-select">Servers</label>
          <select
            id="swagger-server-select"
            value={server}
            onChange={(e) => setServer(e.target.value)}
          >
            {servers.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          <button
            className={'swagger-auth' + (token ? ' is-on' : '')}
            onClick={() => setAuthOpen(true)}
          >
            <Icon
              name={token ? 'circle-check' : 'lock'}
              size={12}
              color={token ? 'var(--twin-green-700)' : 'var(--color-text-secondary)'}
            />
            {token ? 'Authorized' : 'Authorize'}
          </button>
        </div>
        <div className="swagger-filter">
          <Icon name="search" size={13} color="var(--color-text-tertiary)" />
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by path, summary or method (GET, POST…)"
            aria-label="Filter endpoints"
          />
          {filter && (
            <button
              className="swagger-filter-clear"
              onClick={() => setFilter('')}
              aria-label="Clear filter"
            >
              <Icon name="x" size={12} color="var(--color-text-tertiary)" />
            </button>
          )}
        </div>
      </div>

      <div className="swagger-groups">
        {filtered.map((g) => (
          <Group
            key={g.id}
            g={g}
            secured={g.id !== 'default'}
            token={token}
            baseUrl={currentBase}
          />
        ))}
        {!filtered.length && (
          <div className="empty-state" style={{ padding: 60 }}>
            <div className="title">No endpoints match "{filter}"</div>
          </div>
        )}
      </div>
      {authOpen && (
        <AuthorizeDialog
          token={token}
          onSave={(t) => {
            setToken(t);
            setAuthOpen(false);
          }}
          onLogout={() => {
            setToken('');
            setAuthOpen(false);
          }}
          onClose={() => setAuthOpen(false)}
        />
      )}
    </div>
  );
}

interface GroupProps {
  g: OpenApiGroup;
  secured: boolean;
  token: string;
  baseUrl: string;
}

function Group({ g, secured, token, baseUrl }: GroupProps) {
  const [open, setOpen] = useState(true);
  return (
    <div className="swagger-group" data-testid={`api-group-${g.id}`}>
      <button
        className="swagger-group-head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className="swagger-group-name">{g.name}</span>
        <span className="swagger-group-desc">{g.desc}</span>
        <span className="swagger-group-count">{g.endpoints.length}</span>
        <span
          style={{
            display: 'inline-flex',
            transform: open ? 'none' : 'rotate(-90deg)',
            transition: 'transform .15s',
          }}
        >
          <Icon
            name="chevron-down"
            size={14}
            color="var(--color-text-tertiary)"
          />
        </span>
      </button>
      <div className="swagger-group-line" />
      {open && (
        <div className="swagger-rows">
          {g.endpoints.map((ep, i) => (
            <EndpointRow
              key={`${ep.m}-${ep.p}-${i}`}
              ep={ep}
              secured={secured}
              token={token}
              baseUrl={baseUrl}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface EndpointRowProps {
  ep: OpenApiEndpoint;
  secured: boolean;
  token: string;
  baseUrl: string;
}

function EndpointRow({ ep, secured, token, baseUrl }: EndpointRowProps) {
  const [open, setOpen] = useState(false);
  const [tryOpen, setTryOpen] = useState(false);
  const [reqBody, setReqBody] = useState(() => requestBodyFor(ep));
  const [resp, setResp] = useState<MockResponse | null>(null);
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
  const reset = () => {
    setReqBody(requestBodyFor(ep));
    setResp(null);
  };

  return (
    <div
      className={'swagger-row ' + (open ? 'is-open' : '')}
      data-testid={`endpoint-${ep.m}-${ep.p}`}
    >
      <button
        className="swagger-row-head"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <MethodPill method={ep.m} />
        <code className="swagger-path">{ep.p}</code>
        <span className="swagger-summary">{ep.s}</span>
        <span
          className="swagger-lock"
          title={secured ? 'Requires bearer token' : 'Public'}
        >
          <Icon
            name={secured ? 'lock' : 'lock-open'}
            size={13}
            color="var(--color-text-tertiary)"
          />
        </span>
        <span
          style={{
            display: 'inline-flex',
            transform: open ? 'rotate(180deg)' : 'none',
            transition: 'transform .15s',
          }}
        >
          <Icon
            name="chevron-down"
            size={14}
            color="var(--color-text-tertiary)"
          />
        </span>
      </button>
      {open && (
        <div className="swagger-row-body">
          <div className="swagger-section">
            <div className="swagger-section-h">Parameters</div>
            {ep.m === 'GET' && ep.p.includes('label') ? (
              <table className="swagger-params">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>In</th>
                    <th>Description</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>
                      <code>limit</code>
                    </td>
                    <td>integer</td>
                    <td>query</td>
                    <td>Default 50. Max 500.</td>
                  </tr>
                  <tr>
                    <td>
                      <code>q</code>
                    </td>
                    <td>string</td>
                    <td>query</td>
                    <td>Substring filter (case-insensitive).</td>
                  </tr>
                </tbody>
              </table>
            ) : (
              <div className="swagger-empty">No parameters</div>
            )}
          </div>
          <div className="swagger-section">
            <div className="swagger-section-h">Request body</div>
            {ep.m === 'POST' ? (
              <pre className="swagger-code">{requestBodyFor(ep)}</pre>
            ) : (
              <div className="swagger-empty">—</div>
            )}
          </div>
          <div className="swagger-section">
            <div className="swagger-section-h">Responses</div>
            <table className="swagger-responses">
              <tbody>
                <tr>
                  <td className="code-cell ok">200</td>
                  <td>Successful Response</td>
                </tr>
                {ep.m !== 'GET' && (
                  <tr>
                    <td className="code-cell err">422</td>
                    <td>Validation Error</td>
                  </tr>
                )}
                {secured && (
                  <tr>
                    <td className="code-cell err">401</td>
                    <td>Unauthorized</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="swagger-actions">
            <button
              className={'swagger-tryit' + (tryOpen ? ' is-on' : '')}
              onClick={() => setTryOpen((t) => !t)}
            >
              {tryOpen ? 'Cancel' : 'Try it out'}
            </button>
          </div>
          {tryOpen && (
            <div className="swagger-tryit-panel">
              <div className="swagger-section-h">
                Request <span className="swagger-curl-hint">curl preview</span>
              </div>
              <pre className="swagger-code curl">
                {curlFor(ep, reqBody, token, baseUrl)}
              </pre>
              {ep.m !== 'GET' && (
                <>
                  <div className="swagger-section-h">Body</div>
                  <textarea
                    className="swagger-body-edit"
                    value={reqBody}
                    onChange={(e) => setReqBody(e.target.value)}
                    spellCheck="false"
                    aria-label="Request body"
                  />
                </>
              )}
              <div className="swagger-tryit-actions">
                <button
                  className="primary-btn"
                  onClick={execute}
                  disabled={running}
                >
                  {running ? (
                    <>
                      <Icon name="refresh" size={12} /> Executing…
                    </>
                  ) : (
                    <>
                      <Icon name="arrow-right" size={12} /> Execute
                    </>
                  )}
                </button>
                <button className="ghost-btn" onClick={reset}>
                  Reset
                </button>
                {secured && !token && (
                  <span className="swagger-warn">
                    <Icon name="lock" size={12} /> Endpoint requires bearer — click
                    Authorize
                  </span>
                )}
              </div>
              {resp && (
                <div className="swagger-resp">
                  <div className="swagger-resp-h">
                    <span
                      className={
                        'code-cell ' + (resp.status < 300 ? 'ok' : 'err')
                      }
                    >
                      {resp.status}
                    </span>
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

interface MethodPillProps {
  method: HttpMethod;
}

function MethodPill({ method }: MethodPillProps) {
  const c = METHOD_COLOR[method] ?? METHOD_COLOR.GET;
  return (
    <span
      data-method={method}
      style={{
        display: 'inline-flex',
        justifyContent: 'center',
        alignItems: 'center',
        minWidth: 62,
        height: 24,
        padding: '0 8px',
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        fontWeight: 700,
        letterSpacing: 0.4,
        color: c.fg,
        background: c.bg,
        border: `0.5px solid ${c.border}`,
        borderRadius: 4,
      }}
    >
      {method}
    </span>
  );
}

interface AuthorizeDialogProps {
  token: string;
  onSave: (t: string) => void;
  onLogout: () => void;
  onClose: () => void;
}

function AuthorizeDialog({ token, onSave, onLogout, onClose }: AuthorizeDialogProps) {
  const [val, setVal] = useState(token);
  const [revokeArmed, setRevokeArmed] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useModalA11y({ open: true, onClose, ref });
  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      data-testid="authorize-backdrop"
    >
      <div
        className="modal small"
        role="dialog"
        aria-modal="true"
        aria-labelledby="auth-title"
        ref={ref}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div style={{ flex: 1 }}>
            <h2 id="auth-title">Authorize</h2>
            <div className="ctx">
              <Icon name="lock" size={12} /> Bearer (HTTP, scheme: bearer)
            </div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <Icon name="x" size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
            Paste your bearer token. It's attached to every request from "Try it
            out". In production this is delegated to the Twin gateway (Keycloak
            OIDC).
          </p>
          <label
            className="field-label"
            htmlFor="auth-token-input"
            style={{ display: 'block', marginBottom: 6 }}
          >
            Value
          </label>
          <input
            id="auth-token-input"
            type="password"
            autoFocus
            value={val}
            onChange={(e) => {
              setVal(e.target.value);
              setRevokeArmed(false);
            }}
            placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9…"
            style={{
              width: '100%',
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              padding: '8px 10px',
              borderRadius: 4,
              border: '0.5px solid var(--color-border-secondary)',
              background: 'var(--color-background-secondary)',
              color: 'var(--color-text-primary)',
            }}
          />
          <div
            style={{
              marginTop: 14,
              fontSize: 11,
              color: 'var(--color-text-tertiary)',
            }}
          >
            Scopes: <code>read:documents</code> <code>read:query</code>{' '}
            <code>write:documents</code> (palier 2+)
          </div>
        </div>
        <div className="modal-footer">
          {token && (
            <button
              className="ghost-btn"
              onClick={() => {
                if (!revokeArmed) {
                  setRevokeArmed(true);
                  return;
                }
                onLogout();
              }}
              aria-describedby={revokeArmed ? 'auth-revoke-confirm' : undefined}
            >
              {revokeArmed ? 'Confirm revoke token' : 'Revoke token'}
            </button>
          )}
          {revokeArmed && (
            <span id="auth-revoke-confirm" className="muted" style={{ fontSize: 11 }}>
              Click again to remove the bearer token from this session.
            </span>
          )}
          <button
            className="ghost-btn"
            onClick={onClose}
            style={{ marginLeft: 'auto' }}
          >
            Close
          </button>
          <button
            className="primary-btn"
            onClick={() => onSave(val.trim())}
            disabled={!val.trim()}
          >
            Authorize
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Default request body for "Try it out" — deterministic per endpoint.
 * Exported for unit testing.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function requestBodyFor(ep: OpenApiEndpoint): string {
  if (ep.p === '/query' || ep.p === '/query/stream') {
    return JSON.stringify(
      {
        query: 'How do I restart Oracle RMAN after a failed backup?',
        mode: 'hybrid',
        top_k: 60,
        response_type: 'Multiple Paragraphs',
        tag_filter: { all: ['rman'], any: [] },
      },
      null,
      2,
    );
  }
  if (ep.p === '/documents/text') {
    return JSON.stringify(
      { text: '', file_source: '', tags: ['twin'] },
      null,
      2,
    );
  }
  if (ep.p.includes('/graph/entity/edit')) {
    return JSON.stringify({ entity_name: '', updated_data: {} }, null, 2);
  }
  return JSON.stringify({}, null, 2);
}

/** Build a curl preview for the given endpoint. Exported for testing. */
// eslint-disable-next-line react-refresh/only-export-components
export function curlFor(
  ep: OpenApiEndpoint,
  body: string,
  token: string,
  baseUrl: string,
): string {
  const lines = [`curl -X ${ep.m} '${baseUrl}${ep.p}' \\`];
  lines.push("  -H 'Accept: application/json' \\");
  if (ep.m !== 'GET') lines.push("  -H 'Content-Type: application/json' \\");
  if (token) lines.push(`  -H 'Authorization: Bearer ${token.slice(0, 6)}…' \\`);
  if (ep.m !== 'GET')
    lines.push(`  -d '${(body || '').replace(/\n\s*/g, ' ')}'`);
  else lines[lines.length - 1] = lines[lines.length - 1].replace(/ \\$/, '');
  return lines.join('\n');
}

/** Mock 401 response. Exported for testing. */
// eslint-disable-next-line react-refresh/only-export-components
export function mockUnauthorized(ep: OpenApiEndpoint): MockResponse {
  void ep;
  return {
    status: 401,
    statusText: 'Unauthorized',
    tookMs: 12,
    body: JSON.stringify(
      {
        detail:
          'Missing or invalid Bearer token. Use Authorize to attach one.',
      },
      null,
      2,
    ),
  };
}

/** Mock success response. Body shape varies by endpoint. Exported for testing. */
// eslint-disable-next-line react-refresh/only-export-components
export function mockResponseFor(
  ep: OpenApiEndpoint,
  _body: string,
  tookMsOverride?: number,
): MockResponse {
  const tookMs = tookMsOverride ?? 120 + Math.floor(Math.random() * 380);
  if (ep.p === '/query' || ep.p === '/query/stream') {
    return {
      status: 200,
      statusText: 'OK',
      tookMs,
      body: JSON.stringify(
        {
          response:
            'To restart Oracle RMAN after a failed backup, first verify the recovery catalog state … [truncated]',
          sources: [
            {
              id: 'chunk_4a12',
              source: 'oracle-restart-procedure.pdf',
              score: 0.91,
              tags: ['rman', 'oracle'],
            },
            {
              id: 'chunk_88e0',
              source: 'DBA Runbook · Backup recovery',
              score: 0.84,
              tags: ['rman'],
            },
          ],
          mode: 'hybrid',
          tag_filter: { all: ['rman'] },
          took_ms: tookMs,
        },
        null,
        2,
      ),
    };
  }
  if (ep.m === 'GET' && ep.p === '/documents') {
    return {
      status: 200,
      statusText: 'OK',
      tookMs,
      body: JSON.stringify(
        {
          items: [
            { id: 'doc_001', source: 'oracle-restart.pdf', status: 'completed' },
          ],
          total: 247,
        },
        null,
        2,
      ),
    };
  }
  if (ep.p === '/documents/upload' || ep.p === '/documents/text') {
    return {
      status: 200,
      statusText: 'OK',
      tookMs,
      body: JSON.stringify(
        {
          id: 'doc_' + Math.random().toString(16).slice(2, 8),
          status: 'pending',
          queued_at: new Date().toISOString(),
        },
        null,
        2,
      ),
    };
  }
  if (ep.p === '/health') {
    return {
      status: 200,
      statusText: 'OK',
      tookMs,
      body: JSON.stringify({ status: 'ok', uptime_s: 184_213 }, null, 2),
    };
  }
  if (ep.p === '/auth-status') {
    return {
      status: 200,
      statusText: 'OK',
      tookMs,
      body: JSON.stringify(
        { authorized: true, scopes: ['read:documents', 'read:query'] },
        null,
        2,
      ),
    };
  }
  return {
    status: 200,
    statusText: 'OK',
    tookMs,
    body: JSON.stringify({ ok: true }, null, 2),
  };
}
