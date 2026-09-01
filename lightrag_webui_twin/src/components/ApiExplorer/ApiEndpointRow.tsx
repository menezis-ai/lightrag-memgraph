import { useState } from 'react';
import { buildApiHeaders, buildApiUrl, reportUnauthorizedResponse } from '../../api/client';
import { METHOD_COLOR, type HttpMethod, type MockResponse, type OpenApiEndpoint } from '../../types/api';
import { Icon } from '../Icon';
import { curlFor, endpointHasBody, paramKey, requestBodyFor, resolveRequestTarget, responsesFor } from './apiRequest';

export interface ApiEndpointRowProps {
  ep: OpenApiEndpoint;
  secured: boolean;
  token: string;
  baseUrl: string;
}

export function ApiEndpointRow({ ep, secured, token, baseUrl }: Readonly<ApiEndpointRowProps>) {
  const [open, setOpen] = useState(false);
  const [tryOpen, setTryOpen] = useState(false);
  const [reqBody, setReqBody] = useState(() => requestBodyFor(ep));
  const [paramValues, setParamValues] = useState<Record<string, string>>({});
  const [response, setResponse] = useState<MockResponse | null>(null);
  const [running, setRunning] = useState(false);
  const rowSecured = ep.secured ?? secured;
  const requestHasBody = endpointHasBody(ep);
  const target = resolveRequestTarget(ep, paramValues);

  const execute = async (): Promise<void> => {
    setRunning(true);
    setResponse(null);
    const start = nowForTiming();
    try {
      const result = await fetch(buildApiUrl(target.path), {
        method: ep.m,
        headers: {
          ...buildApiHeaders({ token: token || undefined }, { json: requestHasBody }),
          ...target.headers,
        },
        body: requestHasBody ? reqBody : undefined,
        credentials: 'include',
      });
      const text = await result.text();
      reportUnauthorizedResponse(target.path, result.status);
      setResponse({
        status: result.status,
        statusText: result.statusText || (result.ok ? 'OK' : 'Error'),
        tookMs: Math.round(nowForTiming() - start),
        body: tryPrettyJson(text),
      });
    } catch (error) {
      setResponse({
        status: 0,
        statusText: 'Network error',
        tookMs: Math.round(nowForTiming() - start),
        body: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setRunning(false);
    }
  };
  const reset = (): void => {
    setReqBody(requestBodyFor(ep));
    setParamValues({});
    setResponse(null);
  };

  return (
    <div className={'swagger-row ' + (open ? 'is-open' : '')} data-testid={`endpoint-${ep.m}-${ep.p}`}>
      <button className="swagger-row-head" onClick={() => setOpen((current) => !current)} aria-expanded={open}>
        <MethodPill method={ep.m} />
        <code className="swagger-path">{ep.p}</code>
        <span className="swagger-summary">{ep.s}</span>
        <span className="swagger-lock" title={rowSecured ? 'Requires bearer token' : 'Public'}>
          <Icon name={rowSecured ? 'lock' : 'lock-open'} size={13} color="var(--color-text-tertiary)" />
        </span>
        <span style={{ display: 'inline-flex', transform: open ? 'rotate(180deg)' : 'none', transition: 'transform .15s' }}>
          <Icon name="chevron-down" size={14} color="var(--color-text-tertiary)" />
        </span>
      </button>
      {open && (
        <div className="swagger-row-body">
          {ep.desc && <p className="swagger-desc">{ep.desc}</p>}
          {!!ep.params?.length && (
            <div className="swagger-section">
              <div className="swagger-section-h">Parameters</div>
              <table className="swagger-params">
                <thead><tr><th>Name</th><th>Type</th><th>In</th><th>Description</th></tr></thead>
                <tbody>
                  {ep.params.map((param) => (
                    <tr key={`${param.in}-${param.name}`}>
                      <td><code>{param.name}</code>{param.required && <span className="swagger-param-required" title="Required"> *</span>}</td>
                      <td>{param.type || '—'}</td><td>{param.in}</td>
                      <td>{param.desc}{param.example !== undefined && <span className="swagger-param-example">{param.desc ? ' ' : ''}Example: <code>{param.example}</code></span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {requestHasBody && <div className="swagger-section"><div className="swagger-section-h">Request body</div><pre className="swagger-code">{requestBodyFor(ep)}</pre></div>}
          <div className="swagger-section">
            <div className="swagger-section-h">Responses</div>
            <table className="swagger-responses"><tbody>
              {responsesFor(ep, rowSecured).map((row) => <tr key={row.code}><td className={'code-cell ' + (row.code.startsWith('2') ? 'ok' : 'err')}>{row.code}</td><td>{row.desc}</td></tr>)}
            </tbody></table>
          </div>
          <div className="swagger-actions"><button className={'swagger-tryit' + (tryOpen ? ' is-on' : '')} onClick={() => setTryOpen((current) => !current)}>{tryOpen ? 'Cancel' : 'Try it out'}</button></div>
          {tryOpen && (
            <div className="swagger-tryit-panel">
              {(ep.params?.length ?? 0) > 0 && <><div className="swagger-section-h">Parameters</div><div className="swagger-tryit-params">
                {ep.params!.map((param) => (
                  <label key={paramKey(param)} className="swagger-tryit-param">
                    <span className="swagger-tryit-param-name"><code>{param.name}</code><span className="swagger-tryit-param-in">{param.in}</span>{param.required && <span className="swagger-param-required" title="Required">*</span>}</span>
                    <input value={paramValues[paramKey(param)] ?? ''} onChange={(event) => setParamValues((current) => ({ ...current, [paramKey(param)]: event.target.value }))} placeholder={param.example ?? ''} spellCheck="false" aria-label={`Parameter ${param.name}`} />
                  </label>
                ))}
              </div></>}
              <div className="swagger-section-h">Request <span className="swagger-curl-hint">curl preview</span></div>
              <pre className="swagger-code curl">{curlFor(ep, reqBody, token, baseUrl, target)}</pre>
              {requestHasBody && <><div className="swagger-section-h">Body</div><textarea className="swagger-body-edit" value={reqBody} onChange={(event) => setReqBody(event.target.value)} spellCheck="false" aria-label="Request body" /></>}
              <div className="swagger-tryit-actions">
                <button className="primary-btn" onClick={() => void execute()} disabled={running || target.missingRequired.length > 0}><Icon name={running ? 'refresh' : 'arrow-right'} size={12} /> {running ? 'Executing…' : 'Execute'}</button>
                <button className="ghost-btn" onClick={reset}>Reset</button>
                {target.missingRequired.length > 0 && <span className="swagger-warn">Fill the required parameter{target.missingRequired.length > 1 ? 's' : ''}: {target.missingRequired.join(', ')}</span>}
                {rowSecured && !token && <span className="swagger-warn"><Icon name="lock" size={12} /> Endpoint requires bearer — click Authorize</span>}
              </div>
              {response && <div className="swagger-resp" data-testid="swagger-response"><div className="swagger-resp-h"><span className={'code-cell ' + (response.status < 300 ? 'ok' : 'err')}>{response.status}</span><span className="swagger-resp-msg">{response.statusText}</span><span className="swagger-sep">·</span><span className="swagger-resp-time">{response.tookMs}ms</span></div><pre className="swagger-code resp">{response.body}</pre></div>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function MethodPill({ method }: Readonly<{ method: HttpMethod }>) {
  const colors = METHOD_COLOR[method] ?? METHOD_COLOR.GET;
  return <span data-method={method} style={{ display: 'inline-flex', justifyContent: 'center', alignItems: 'center', minWidth: 62, height: 24, padding: '0 8px', fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 700, letterSpacing: 0.4, color: colors.fg, background: colors.bg, border: `0.5px solid ${colors.border}`, borderRadius: 4 }}>{method}</span>;
}

function nowForTiming(): number {
  return globalThis.performance === undefined ? Date.now() : globalThis.performance.now();
}

function tryPrettyJson(text: string): string {
  if (!text) return '';
  try { return JSON.stringify(JSON.parse(text), null, 2); } catch { return text; }
}
