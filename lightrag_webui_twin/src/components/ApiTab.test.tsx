/**
 * Unit tests for ApiTab and its helper utilities.
 *
 * Covers: groups render, filter narrows, endpoint expand, Try it out flow
 * (mock success + unauthorized), authorize dialog flow, curl/request helpers.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  ApiTab,
  curlFor,
  requestBodyFor,
  responsesFor,
  resolveRequestTarget,
} from './ApiTab';
import { setActiveFolder, setSessionAuthToken } from '../api/client';
import { API_VERSION, OPENAPI_GROUPS } from '../fixtures';

function defaultProps() {
  return {
    apiVersion: API_VERSION,
    groups: OPENAPI_GROUPS,
    // After mock-kill F2 the servers dropdown was removed — the curl
    // preview uses the current browser origin as a single source.
    baseUrl: 'http://localhost',
  };
}

describe('ApiTab — rendering', () => {
  it('renders the title, version and OAS badge', () => {
    render(<ApiTab {...defaultProps()} />);
    expect(screen.getByText('Twin KMS API')).toBeInTheDocument();
    expect(screen.getByText(API_VERSION)).toBeInTheDocument();
    expect(screen.getByText('OAS 3.1')).toBeInTheDocument();
  });

  it('renders all 5 endpoint groups by default', () => {
    render(<ApiTab {...defaultProps()} />);
    expect(screen.getByTestId('api-group-documents')).toBeInTheDocument();
    expect(screen.getByTestId('api-group-query')).toBeInTheDocument();
    expect(screen.getByTestId('api-group-graph')).toBeInTheDocument();
    expect(screen.getByTestId('api-group-ollama')).toBeInTheDocument();
    expect(screen.getByTestId('api-group-default')).toBeInTheDocument();
  });

  it('Authorize button shows "Authorize" when no token', () => {
    render(<ApiTab {...defaultProps()} />);
    expect(screen.getByRole('button', { name: /Authorize$/ })).toBeInTheDocument();
  });

  it('does not duplicate the Settings API contract inside the explorer', () => {
    render(<ApiTab {...defaultProps()} />);
    expect(screen.queryByTestId('apitab-banner')).toBeNull();
    // The previous lie must be gone — operator should not read
    // "the gateway transparently injects tag_filter / visibility"
    // anywhere in this surface.
    expect(screen.queryByText(/transparently injects/i)).toBeNull();
    expect(screen.queryByText(/visibility/i)).toBeNull();
  });
});

describe('ApiTab — filter', () => {
  it('filtering by path narrows the groups', async () => {
    render(<ApiTab {...defaultProps()} />);
    await userEvent.type(
      screen.getByLabelText('Filter endpoints'),
      '/documents',
    );
    expect(screen.getByTestId('api-group-documents')).toBeInTheDocument();
    expect(screen.queryByTestId('api-group-ollama')).toBeNull();
  });

  it('filtering by method matches case-insensitively', async () => {
    render(<ApiTab {...defaultProps()} />);
    await userEvent.type(screen.getByLabelText('Filter endpoints'), 'delete');
    // Only documents group has DELETE endpoints
    expect(screen.getByTestId('api-group-documents')).toBeInTheDocument();
    expect(screen.queryByTestId('api-group-query')).toBeNull();
  });

  it('empty filter result shows empty state', async () => {
    render(<ApiTab {...defaultProps()} />);
    await userEvent.type(
      screen.getByLabelText('Filter endpoints'),
      'zzz-no-such-endpoint',
    );
    expect(screen.getByText(/No endpoints match/)).toBeInTheDocument();
  });
});

describe('ApiTab — endpoint expand + Try it out', () => {
  it('omits empty Parameters and Request body sections', async () => {
    render(<ApiTab {...defaultProps()} />);
    const row = screen.getByTestId('endpoint-GET-/health');
    await userEvent.click(within(row).getByRole('button'));
    expect(within(row).queryByText('Parameters')).toBeNull();
    expect(within(row).queryByText('Request body')).toBeNull();
    expect(within(row).queryByText('No parameters')).toBeNull();
    expect(within(row).getByText('Responses')).toBeInTheDocument();
  });

  it('renders spec-declared description, parameters and responses', async () => {
    const groups = [
      {
        id: 'docs',
        name: 'docs',
        desc: '',
        endpoints: [
          {
            m: 'GET' as const,
            p: '/documents',
            s: 'List documents',
            desc: 'List the documents of the active folder.',
            params: [
              {
                name: 'X-Twin-Folder',
                in: 'header' as const,
                type: 'string',
                required: false,
                desc: 'Folder to scope this request to.',
                example: 'general',
              },
              {
                name: 'status',
                in: 'query' as const,
                type: 'string',
                required: true,
                desc: 'Only documents with this status.',
              },
            ],
            responses: [
              { code: '200', desc: 'The document list' },
              { code: '403', desc: 'Folder out of scope' },
            ],
          },
        ],
      },
    ];
    render(<ApiTab {...defaultProps()} groups={groups} />);
    const row = screen.getByTestId('endpoint-GET-/documents');
    await userEvent.click(within(row).getByRole('button'));
    expect(
      within(row).getByText('List the documents of the active folder.'),
    ).toBeInTheDocument();
    expect(within(row).getByText('X-Twin-Folder')).toBeInTheDocument();
    expect(within(row).getByText(/Example:/)).toBeInTheDocument();
    expect(within(row).getByText('header')).toBeInTheDocument();
    expect(within(row).queryByText('No parameters')).toBeNull();
    expect(within(row).getByText('Folder out of scope')).toBeInTheDocument();
    // Spec-declared responses replace the generic fallback rows.
    expect(within(row).queryByText('Successful Response')).toBeNull();
  });

  it('gates Execute on required parameters and applies them to the request', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response('{}', { status: 200 }));
    const groups = [
      {
        id: 'docs',
        name: 'docs',
        desc: '',
        endpoints: [
          {
            m: 'GET' as const,
            p: '/documents/{doc_id}/metadata',
            s: 'Metadata',
            params: [
              {
                name: 'doc_id',
                in: 'path' as const,
                type: 'string',
                required: true,
                desc: '',
              },
              {
                name: 'X-Twin-Folder',
                in: 'header' as const,
                type: 'string',
                required: false,
                desc: '',
              },
            ],
          },
        ],
      },
    ];
    render(<ApiTab {...defaultProps()} groups={groups} />);
    const row = screen.getByTestId('endpoint-GET-/documents/{doc_id}/metadata');
    await userEvent.click(within(row).getByRole('button'));
    await userEvent.click(within(row).getByRole('button', { name: 'Try it out' }));

    // Required path param empty → Execute disabled with a hint.
    const executeBtn = within(row).getByRole('button', { name: /Execute/ });
    expect(executeBtn).toBeDisabled();
    expect(within(row).getByText(/Fill the required parameter/)).toBeInTheDocument();

    await userEvent.type(
      within(row).getByLabelText('Parameter doc_id'),
      'doc-42',
    );
    await userEvent.type(
      within(row).getByLabelText('Parameter X-Twin-Folder'),
      'general',
    );
    expect(executeBtn).toBeEnabled();
    await userEvent.click(executeBtn);

    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    // Path param substituted, header param applied.
    expect(String(url)).toContain('/documents/doc-42/metadata');
    expect(String(url)).not.toContain('{doc_id}');
    expect((init.headers as Record<string, string>)['X-Twin-Folder']).toBe(
      'general',
    );
    fetchSpy.mockRestore();
  });

  it('prefers the spec-declared body example in the request body preview', async () => {
    const groups = [
      {
        id: 'tags',
        name: 'tags',
        desc: '',
        endpoints: [
          {
            m: 'POST' as const,
            p: '/tags',
            s: 'Request a tag',
            hasBody: true,
            bodyExample: JSON.stringify({ tag: 'from-spec' }, null, 2),
          },
        ],
      },
    ];
    render(<ApiTab {...defaultProps()} groups={groups} />);
    const row = screen.getByTestId('endpoint-POST-/tags');
    await userEvent.click(within(row).getByRole('button'));
    expect(within(row).getByText(/from-spec/)).toBeInTheDocument();
  });

  it('Try it out fires a real fetch against baseUrl + path and renders the response', async () => {
    // Mock-kill F2 — Try it out now does a real fetch, not a synthetic
    // mock. We stub `fetch` directly so the test asserts the call shape
    // and a happy-path 200 render.
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ status: 'ok' }), {
          status: 200,
          statusText: 'OK',
          headers: { 'content-type': 'application/json' },
        }),
      );
    try {
      render(<ApiTab {...defaultProps()} />);
      const row = screen.getByTestId('endpoint-GET-/health');
      await userEvent.click(within(row).getByRole('button'));
      await userEvent.click(
        within(row).getByRole('button', { name: 'Try it out' }),
      );
      await userEvent.click(within(row).getByRole('button', { name: /Execute/ }));
      await waitFor(() => {
        const resp = row.querySelector('.swagger-resp');
        expect(resp).not.toBeNull();
        expect(resp!.textContent).toMatch(/200/);
      });
      expect(fetchSpy).toHaveBeenCalledWith(
        '/health',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({ Accept: 'application/json' }),
          credentials: 'include',
        }),
      );
    } finally {
      fetchSpy.mockRestore();
    }
  });

  it('Try it out reuses the app session token and active folder header', async () => {
    setSessionAuthToken('session-token-123');
    setActiveFolder('securetransport');
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ items: [], total: 0 }), {
          status: 200,
          statusText: 'OK',
          headers: { 'content-type': 'application/json' },
        }),
      );
    try {
      render(<ApiTab {...defaultProps()} />);
      const row = screen.getByTestId('endpoint-GET-/documents');
      await userEvent.click(within(row).getByRole('button'));
      await userEvent.click(
        within(row).getByRole('button', { name: 'Try it out' }),
      );
      await userEvent.click(within(row).getByRole('button', { name: /Execute/ }));
      await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
      expect(fetchSpy).toHaveBeenCalledWith(
        '/documents',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            Authorization: 'Bearer session-token-123',
            'X-Twin-Folder': 'securetransport',
          }),
          credentials: 'include',
        }),
      );
    } finally {
      fetchSpy.mockRestore();
      setSessionAuthToken(null);
      setActiveFolder(null);
    }
  });

  it('Try it out surfaces a 401 from the real backend with WWW-Authenticate', async () => {
    const fetchSpy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Missing bearer' }), {
          status: 401,
          statusText: 'Unauthorized',
        }),
      );
    try {
      render(<ApiTab {...defaultProps()} />);
      const row = screen.getByTestId('endpoint-GET-/documents');
      await userEvent.click(within(row).getByRole('button'));
      await userEvent.click(
        within(row).getByRole('button', { name: 'Try it out' }),
      );
      await userEvent.click(within(row).getByRole('button', { name: /Execute/ }));
      await waitFor(() => {
        const resp = row.querySelector('.swagger-resp');
        expect(resp).not.toBeNull();
        expect(resp!.textContent).toMatch(/401/);
      });
      const resp = row.querySelector('.swagger-resp') as HTMLElement;
      expect(resp.textContent).toMatch(/Unauthorized/);
    } finally {
      fetchSpy.mockRestore();
    }
  });
});

describe('ApiTab — Authorize dialog', () => {
  it('clicking Authorize opens the dialog; entering a token saves and toggles state', async () => {
    render(<ApiTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('button', { name: /Authorize$/ }));
    const dialog = await screen.findByRole('dialog', { name: 'Authorize' });
    const input = within(dialog).getByLabelText('Value');
    await new Promise((r) => setTimeout(r, 60));
    (input as HTMLInputElement).focus();
    await userEvent.type(input, 'eyJtest-token-xyz');
    await userEvent.click(
      within(dialog).getByRole('button', { name: 'Authorize' }),
    );
    // Topbar button switches to Authorized state
    expect(
      screen.getByRole('button', { name: /Authorized/ }),
    ).toBeInTheDocument();
  });

  it('revoke token requires a second confirmation click', async () => {
    render(<ApiTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('button', { name: /Authorize$/ }));
    let dialog = await screen.findByRole('dialog', { name: 'Authorize' });
    const input = within(dialog).getByLabelText('Value');
    await new Promise((r) => setTimeout(r, 60));
    (input as HTMLInputElement).focus();
    await userEvent.type(input, 'eyJtest-token-xyz');
    await userEvent.click(
      within(dialog).getByRole('button', { name: 'Authorize' }),
    );
    expect(screen.getByRole('button', { name: /Authorized/ })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Authorized/ }));
    dialog = await screen.findByRole('dialog', { name: 'Authorize' });
    await userEvent.click(
      within(dialog).getByRole('button', { name: 'Revoke token' }),
    );
    expect(
      within(dialog).getByRole('button', { name: 'Confirm revoke token' }),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Authorized/ })).toBeInTheDocument();

    await userEvent.click(
      within(dialog).getByRole('button', { name: 'Confirm revoke token' }),
    );
    expect(screen.getByRole('button', { name: /Authorize$/ })).toBeInTheDocument();
  });

  it('clicking the backdrop closes the Authorize dialog', async () => {
    render(<ApiTab {...defaultProps()} />);
    await userEvent.click(screen.getByRole('button', { name: /Authorize$/ }));
    await screen.findByRole('dialog', { name: 'Authorize' });
    await userEvent.click(screen.getByTestId('authorize-backdrop'));
    expect(screen.queryByRole('dialog', { name: 'Authorize' })).toBeNull();
  });
});

describe('Helpers — resolveRequestTarget', () => {
  const ep = {
    m: 'GET' as const,
    p: '/graph/search',
    s: '',
    params: [
      {
        name: 'q',
        in: 'query' as const,
        type: 'string',
        required: true,
        desc: '',
      },
      {
        name: 'limit',
        in: 'query' as const,
        type: 'integer',
        required: false,
        desc: '',
      },
    ],
  };

  it('reports missing required params and leaves the path untouched', () => {
    const t = resolveRequestTarget(ep, {});
    expect(t.missingRequired).toEqual(['q']);
    expect(t.path).toBe('/graph/search');
  });

  it('serializes provided query params and skips empty optional ones', () => {
    const t = resolveRequestTarget(ep, { 'query:q': 'firewall' });
    expect(t.missingRequired).toEqual([]);
    expect(t.path).toBe('/graph/search?q=firewall');
  });

  it('URL-encodes substituted path params', () => {
    const t = resolveRequestTarget(
      {
        m: 'DELETE' as const,
        p: '/tags/{name}',
        s: '',
        params: [
          {
            name: 'name',
            in: 'path' as const,
            type: 'string',
            required: true,
            desc: '',
          },
        ],
      },
      { 'path:name': 'a/b' },
    );
    expect(t.path).toBe('/tags/a%2Fb');
  });
});

describe('ApiTab — per-endpoint security', () => {
  it('spec-declared public endpoints show no lock even in a secured group', async () => {
    const groups = [
      {
        id: 'auth',
        name: 'auth',
        desc: '',
        endpoints: [
          { m: 'POST' as const, p: '/login', s: 'Log in', secured: false },
          { m: 'POST' as const, p: '/x', s: 'Locked', secured: true },
        ],
      },
    ];
    render(<ApiTab {...defaultProps()} groups={groups} />);
    const loginRow = screen.getByTestId('endpoint-POST-/login');
    const lockedRow = screen.getByTestId('endpoint-POST-/x');
    expect(within(loginRow).getByTitle('Public')).toBeInTheDocument();
    expect(
      within(lockedRow).getByTitle('Requires bearer token'),
    ).toBeInTheDocument();
  });
});

describe('Helpers — requestBodyFor', () => {
  it('builds a native /query payload with hybrid mode and no Twin tag_filter', () => {
    // Plain native routes keep the upstream-shaped sample. Twin-prefixed routes
    // below advertise the overlay's server-side tag_filter support.
    const b = JSON.parse(requestBodyFor({ m: 'POST', p: '/query', s: '' }));
    expect(b.mode).toBe('hybrid');
    expect(b).not.toHaveProperty('tag_filter');
  });

  it('builds an /entity/edit body with empty entity_name and updated_data', () => {
    const b = JSON.parse(
      requestBodyFor({ m: 'POST', p: '/graph/entity/edit', s: '' }),
    );
    expect(b).toEqual({ entity_name: '', updated_data: {} });
  });

  it('falls back to empty object for unknown endpoint', () => {
    expect(requestBodyFor({ m: 'POST', p: '/whatever', s: '' })).toBe('{}');
  });

  it('also matches the Twin-prefixed /twin/api/query path with tag_filter', () => {
    // OpenAPI under the plugin / standalone topology exposes
    // /twin/api/query — without this matcher the Try-it-out body
    // defaults to `{}` and round-trips a 422 instead of a real call.
    const b = JSON.parse(
      requestBodyFor({ m: 'POST', p: '/twin/api/query', s: '' }),
    );
    expect(b.mode).toBe('hybrid');
    expect(b.tag_filter.all).toEqual(['tag-name']);
  });

  it('also matches /twin/api/query/stream with tag_filter', () => {
    const b = JSON.parse(
      requestBodyFor({ m: 'POST', p: '/twin/api/query/stream', s: '' }),
    );
    expect(b.query).toMatch(/indexed knowledge base/);
    expect(b.tag_filter.all).toEqual(['tag-name']);
  });

  it('keeps tag_filter on /twin/api/query/data', () => {
    const b = JSON.parse(
      requestBodyFor({ m: 'POST', p: '/twin/api/query/data', s: '' }),
    );
    expect(b.query).toMatch(/indexed knowledge base/);
    expect(b.chunk_top_k).toBe(20);
    expect(b.tag_filter.all).toEqual(['tag-name']);
  });

  it('keeps tag_filter on the native /query/data path too', () => {
    const b = JSON.parse(
      requestBodyFor({ m: 'POST', p: '/query/data', s: '' }),
    );
    expect(b.tag_filter.all).toEqual(['tag-name']);
  });
});

describe('Helpers — responsesFor', () => {
  it('uses operator-readable descriptions when an endpoint omits responses', () => {
    expect(
      responsesFor({ m: 'POST', p: '/whatever', s: 'Do work' }, true),
    ).toEqual([
      { code: '200', desc: 'Request completed successfully.' },
      {
        code: '422',
        desc: 'The request body or parameters failed validation.',
      },
      {
        code: '401',
        desc: 'Authentication credentials are missing, invalid, or expired.',
      },
    ]);
  });
});

describe('Helpers — curlFor', () => {
  it('emits a GET curl without -d / -H Content-Type / -d body', () => {
    const c = curlFor(
      { m: 'GET', p: '/health', s: '' },
      '',
      '',
      'https://example.com',
    );
    expect(c).toMatch(/^curl -X GET 'https:\/\/example\.com\/health'/);
    expect(c).not.toMatch(/-d /);
    expect(c).not.toMatch(/Content-Type/);
  });

  it('emits a POST curl with Content-Type, Bearer slice and -d body', () => {
    const c = curlFor(
      { m: 'POST', p: '/query', s: '' },
      '{"q":"x"}',
      'eyJabc123long',
      'https://example.com',
    );
    expect(c).toMatch(/-X POST/);
    expect(c).toMatch(/Content-Type: application\/json/);
    expect(c).toMatch(/Authorization: Bearer eyJabc…/);
    expect(c).toMatch(/-d '\{"q":"x"\}'/);
  });
});
