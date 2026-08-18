/**
 * ApiTab — Swagger-style endpoint browser for the LightRAG OpenAPI surface.
 *
 * Ported from Desktop/UI/api.jsx. Twin overlay extends the LightRAG
 * OpenAPI surface with `/twin/api/*` routes. `tag_filter` is honored
 * server-side on Twin query routes via `TAGGED_WITH`; `/query/data`
 * also has a filtered graph-mode fallback to `mix`. Native LightRAG
 * routes pass through unchanged. The
 * previous claim about "transparent injection of `tag_filter` /
 * `visibility` scoping" was incorrect and was retracted by audit C8.
 *
 * Behavior delta vs the proto:
 *   - `groups` are injected via props (no static window globals).
 *   - "Try it out" performs a real request through the same runtime URL,
 *     bearer/session, cookie, and folder header helpers as the rest of the app.
 *   - Authorize dialog a11y via useModalA11y.
 */

import { useRef, useState } from 'react';
import { Icon } from './Icon';
import { useModalA11y } from '../hooks/useModalA11y';
import { buildApiHeaders, buildApiUrl } from '../api/client';
import {
  METHOD_COLOR,
  type HttpMethod,
  type MockResponse,
  type OpenApiEndpoint,
  type OpenApiGroup,
  type OpenApiParam,
} from '../types/api';

export interface ApiTabProps {
  apiVersion: string;
  groups: readonly OpenApiGroup[];
  /** Origin used in the curl preview. Defaults to the current browser
   *  origin — the previous prod/stg dropdown was removed as part of
   *  mock-kill F2 because the displayed hostnames (`cib-kb.twin.internal`)
   *  didn't exist. */
  baseUrl: string;
}

export function ApiTab({ apiVersion, groups, baseUrl }: Readonly<ApiTabProps>) {
  const [filter, setFilter] = useState('');
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

  const currentBase = baseUrl;

  return (
    <div className="swagger">
      <div className="swagger-topbar">
        <div className="swagger-title">
          <span className="swagger-title-main">Twin KMS API</span>
          <span className="swagger-version">{apiVersion}</span>
          <span className="swagger-oas">OAS 3.1</span>
        </div>
        <div className="swagger-meta">
          <code>/openapi.json</code>
          <span className="swagger-sep">·</span>
          <span>
            Documents, folders, tags, knowledge graph and grounded retrieval
          </span>
        </div>
        <div className="swagger-servers">
          <span className="swagger-server-current">
            <Icon name="world" size={12} /> {currentBase || '(no server)'}
          </span>
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

function Group({ g, secured, token, baseUrl }: Readonly<GroupProps>) {
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

function EndpointRow({ ep, secured, token, baseUrl }: Readonly<EndpointRowProps>) {
  const [open, setOpen] = useState(false);
  const [tryOpen, setTryOpen] = useState(false);
  const [reqBody, setReqBody] = useState(() => requestBodyFor(ep));
  const [paramValues, setParamValues] = useState<Record<string, string>>({});
  const [resp, setResp] = useState<MockResponse | null>(null);
  const [running, setRunning] = useState(false);

  // Spec-declared auth state wins; the group heuristic only covers
  // sparse specs that say nothing about security.
  const rowSecured = ep.secured ?? secured;
  const requestHasBody = endpointHasBody(ep);
  const target = resolveRequestTarget(ep, paramValues);

  const execute = async () => {
    setRunning(true);
    setResp(null);
    const start = nowForTiming();
    try {
      const r = await fetch(buildApiUrl(target.path), {
        method: ep.m,
        headers: {
          ...buildApiHeaders(
            { token: token || undefined },
            { json: requestHasBody },
          ),
          ...target.headers,
        },
        body: requestHasBody ? reqBody : undefined,
        credentials: 'include',
      });
      const text = await r.text();
      const tookMs = Math.round(nowForTiming() - start);
      setResp({
        status: r.status,
        statusText: r.statusText || (r.ok ? 'OK' : 'Error'),
        tookMs,
        body: tryPrettyJson(text),
      });
    } catch (err) {
      const tookMs = Math.round(nowForTiming() - start);
      setResp({
        status: 0,
        statusText: 'Network error',
        tookMs,
        body: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setRunning(false);
    }
  };
  const reset = () => {
    setReqBody(requestBodyFor(ep));
    setParamValues({});
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
          title={rowSecured ? 'Requires bearer token' : 'Public'}
        >
          <Icon
            name={rowSecured ? 'lock' : 'lock-open'}
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
          {ep.desc && <p className="swagger-desc">{ep.desc}</p>}
          {!!ep.params?.length && (
            <div className="swagger-section">
              <div className="swagger-section-h">Parameters</div>
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
                  {ep.params.map((p) => (
                    <tr key={`${p.in}-${p.name}`}>
                      <td>
                        <code>{p.name}</code>
                        {p.required && (
                          <span
                            className="swagger-param-required"
                            title="Required"
                          >
                            {' '}
                            *
                          </span>
                        )}
                      </td>
                      <td>{p.type || '—'}</td>
                      <td>{p.in}</td>
                      <td>
                        {p.desc}
                        {p.example !== undefined && (
                          <span className="swagger-param-example">
                            {p.desc ? ' ' : ''}Example: <code>{p.example}</code>
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {requestHasBody && (
            <div className="swagger-section">
              <div className="swagger-section-h">Request body</div>
              <pre className="swagger-code">{requestBodyFor(ep)}</pre>
            </div>
          )}
          <div className="swagger-section">
            <div className="swagger-section-h">Responses</div>
            <table className="swagger-responses">
              <tbody>
                {responsesFor(ep, rowSecured).map((r) => (
                  <tr key={r.code}>
                    <td
                      className={
                        'code-cell ' + (r.code.startsWith('2') ? 'ok' : 'err')
                      }
                    >
                      {r.code}
                    </td>
                    <td>{r.desc}</td>
                  </tr>
                ))}
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
              {(ep.params?.length ?? 0) > 0 && (
                <>
                  <div className="swagger-section-h">Parameters</div>
                  <div className="swagger-tryit-params">
                    {ep.params!.map((p) => (
                      <label
                        key={paramKey(p)}
                        className="swagger-tryit-param"
                      >
                        <span className="swagger-tryit-param-name">
                          <code>{p.name}</code>
                          <span className="swagger-tryit-param-in">{p.in}</span>
                          {p.required && (
                            <span
                              className="swagger-param-required"
                              title="Required"
                            >
                              *
                            </span>
                          )}
                        </span>
                        <input
                          value={paramValues[paramKey(p)] ?? ''}
                          onChange={(e) =>
                            setParamValues((v) => ({
                              ...v,
                              [paramKey(p)]: e.target.value,
                            }))
                          }
                          placeholder={p.example ?? ''}
                          spellCheck="false"
                          aria-label={`Parameter ${p.name}`}
                        />
                      </label>
                    ))}
                  </div>
                </>
              )}
              <div className="swagger-section-h">
                Request <span className="swagger-curl-hint">curl preview</span>
              </div>
              <pre className="swagger-code curl">
                {curlFor(ep, reqBody, token, baseUrl, target)}
              </pre>
              {requestHasBody && (
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
                  disabled={running || target.missingRequired.length > 0}
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
                {target.missingRequired.length > 0 && (
                  <span className="swagger-warn">
                    Fill the required parameter
                    {target.missingRequired.length > 1 ? 's' : ''}:{' '}
                    {target.missingRequired.join(', ')}
                  </span>
                )}
                {rowSecured && !token && (
                  <span className="swagger-warn">
                    <Icon name="lock" size={12} /> Endpoint requires bearer — click
                    Authorize
                  </span>
                )}
              </div>
              {resp && (
                <div className="swagger-resp" data-testid="swagger-response">
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

function MethodPill({ method }: Readonly<MethodPillProps>) {
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

function AuthorizeDialog({ token, onSave, onLogout, onClose }: Readonly<AuthorizeDialogProps>) {
  const [val, setVal] = useState(token);
  const [revokeArmed, setRevokeArmed] = useState(false);
  const ref = useRef<HTMLDialogElement>(null);
  useModalA11y({ open: true, onClose, ref });
  return (
    <div
      className="modal-backdrop"
    >
      <button
        type="button"
        className="modal-backdrop-dismiss"
        onClick={onClose}
        aria-label="Close authorize dialog"
        data-testid="authorize-backdrop"
      />
      <dialog
        open
        className="modal small"
        aria-modal="true"
        aria-labelledby="auth-title"
        ref={ref}
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
            Paste a bearer token (from <code>POST /login</code>) or an API key
            (Settings → API keys). It's attached to every request from "Try it
            out".
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
            Endpoints marked with a lock require authentication; admin
            endpoints additionally require an administrator identity.
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
      </dialog>
    </div>
  );
}

/**
 * Default request body for "Try it out" — deterministic per endpoint.
 * Exported for unit testing.
 */
// The OpenAPI surface exposes the LightRAG-native `/query{,/stream}`
// and the Twin overlay's prefixed `/twin/api/query{,/stream}`. Both
// share the same QueryRequest schema, so a single matcher fills the
// "Try it out" body for either form. Without this the prefixed
// variants would default to `{}` and round-trip a 422.
const QUERY_ENDPOINTS = new Set([
  '/query',
  '/query/data',
  '/query/stream',
  '/twin/api/query',
  '/twin/api/query/data',
  '/twin/api/query/stream',
]);

/** Methods that carry a request body even when a sparse spec omits
 *  `requestBody` (fallback display only). */
const METHODS_WITH_BODY = new Set<HttpMethod>(['POST', 'PUT', 'PATCH']);

/** The real OpenAPI parser always sets `hasBody`, including `false`.
 *  The method fallback exists only for old hand-written fixtures. */
function endpointHasBody(ep: OpenApiEndpoint): boolean {
  return ep.hasBody ?? METHODS_WITH_BODY.has(ep.m);
}

/** Stable form key for a parameter (name alone can collide across `in`s). */
// eslint-disable-next-line react-refresh/only-export-components -- pure helper exported for unit tests.
export function paramKey(p: OpenApiParam): string {
  return `${p.in}:${p.name}`;
}

export interface ResolvedTarget {
  /** Request path with path params substituted and query params appended. */
  path: string;
  /** Declared header parameters with a non-empty value. */
  headers: Record<string, string>;
  /** Names of required parameters still missing a value. */
  missingRequired: string[];
}

/**
 * Apply the operator-provided parameter values to the endpoint: substitute
 * `{path}` params, serialize query params, collect declared headers, and
 * report which required parameters are still empty (Execute stays disabled
 * until that list is empty). Exported for unit testing.
 */
// eslint-disable-next-line react-refresh/only-export-components -- pure helper exported for unit tests.
export function resolveRequestTarget(
  ep: OpenApiEndpoint,
  values: Record<string, string>,
): ResolvedTarget {
  let path = ep.p;
  const headers: Record<string, string> = {};
  const query = new URLSearchParams();
  const missingRequired: string[] = [];
  for (const p of ep.params ?? []) {
    const raw = (values[paramKey(p)] ?? '').trim();
    if (!raw) {
      if (p.required) missingRequired.push(p.name);
      continue;
    }
    if (p.in === 'path') {
      path = path.replaceAll(`{${p.name}}`, encodeURIComponent(raw));
    } else if (p.in === 'query') {
      query.append(p.name, raw);
    } else if (p.in === 'header') {
      headers[p.name] = raw;
    }
  }
  const qs = query.toString();
  return { path: qs ? `${path}?${qs}` : path, headers, missingRequired };
}

/**
 * Responses to display: the ones documented in the spec when present,
 * otherwise an operator-readable safe fallback (200, 422 on write, 401
 * when the group is secured). Exported for unit testing.
 */
// eslint-disable-next-line react-refresh/only-export-components -- pure helper exported for unit tests.
export function responsesFor(
  ep: OpenApiEndpoint,
  secured: boolean,
): { code: string; desc: string }[] {
  if (ep.responses?.length) {
    return ep.responses.map((r) => ({ code: r.code, desc: r.desc }));
  }
  const rows = [{ code: '200', desc: 'Request completed successfully.' }];
  if (ep.m !== 'GET') {
    rows.push({
      code: '422',
      desc: 'The request body or parameters failed validation.',
    });
  }
  if (secured) {
    rows.push({
      code: '401',
      desc: 'Authentication credentials are missing, invalid, or expired.',
    });
  }
  return rows;
}

// eslint-disable-next-line react-refresh/only-export-components -- pure helper exported for unit tests.
export function requestBodyFor(ep: OpenApiEndpoint): string {
  if (!endpointHasBody(ep)) return '';
  // The spec's own example is authoritative when the backend declares one.
  if (ep.bodyExample) return ep.bodyExample;
  if (QUERY_ENDPOINTS.has(ep.p)) {
    const dataEndpoint = ep.p.endsWith('/query/data');
    const twinEndpoint = ep.p.startsWith('/twin/api/query');
    const supportsTagFilter = twinEndpoint || dataEndpoint;
    return JSON.stringify(
      {
        query: 'Ask a question about the indexed knowledge base',
        mode: 'hybrid',
        top_k: 60,
        ...(dataEndpoint ? { chunk_top_k: 20 } : {}),
        response_type: 'Multiple Paragraphs',
        // Twin query routes honor tag_filter server-side via TAGGED_WITH.
        // Plain native routes keep the unfiltered upstream sample.
        ...(supportsTagFilter ? { tag_filter: { all: ['tag-name'], any: [] } } : {}),
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

/** Build a curl preview for the given endpoint. Exported for testing.
 *  `target` (optional) carries the parameter-resolved path and declared
 *  header values from `resolveRequestTarget`. */
// eslint-disable-next-line react-refresh/only-export-components
export function curlFor(
  ep: OpenApiEndpoint,
  body: string,
  token: string,
  baseUrl: string,
  target?: Pick<ResolvedTarget, 'path' | 'headers'>,
): string {
  const path = target?.path ?? ep.p;
  const lines = [`curl -X ${ep.m} '${baseUrl}${path}' \\`];
  lines.push("  -H 'Accept: application/json' \\");
  const requestHasBody = endpointHasBody(ep);
  if (requestHasBody) lines.push("  -H 'Content-Type: application/json' \\");
  for (const [name, value] of Object.entries(target?.headers ?? {})) {
    lines.push(`  -H '${name}: ${value}' \\`);
  }
  if (token) lines.push(`  -H 'Authorization: Bearer ${token.slice(0, 6)}…' \\`);
  if (!requestHasBody) {
    lines[lines.length - 1] = lines.at(-1)?.replace(/ \\$/, '') ?? '';
  } else {
    lines.push(`  -d '${(body || '').replaceAll(/\n\s*/g, ' ')}'`);
  }
  return lines.join('\n');
}

function nowForTiming(): number {
  return globalThis.performance === undefined
    ? Date.now()
    : globalThis.performance.now();
}

/** Best-effort prettify of a response body. Returns the raw string if
 *  it doesn't parse as JSON (HTML errors, plain text, etc.). */
function tryPrettyJson(text: string): string {
  if (!text) return '';
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}
