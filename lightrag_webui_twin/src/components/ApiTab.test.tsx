/**
 * Unit tests for ApiTab and its helper utilities.
 *
 * Covers: groups render, filter narrows, endpoint expand, Try it out flow
 * (mock success + unauthorized), authorize dialog flow, curl/mock helpers.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  ApiTab,
  curlFor,
  mockResponseFor,
  mockUnauthorized,
  requestBodyFor,
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
    expect(screen.getByText('LightRAG Server API')).toBeInTheDocument();
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
  it('expanding an endpoint reveals Parameters/Request body/Responses sections', async () => {
    render(<ApiTab {...defaultProps()} />);
    const row = screen.getByTestId('endpoint-GET-/health');
    await userEvent.click(within(row).getByRole('button'));
    expect(within(row).getByText('Parameters')).toBeInTheDocument();
    expect(within(row).getByText('Responses')).toBeInTheDocument();
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

describe('Helpers — mockResponseFor / mockUnauthorized', () => {
  it('mockUnauthorized returns 401 with bearer hint', () => {
    const r = mockUnauthorized();
    expect(r.status).toBe(401);
    expect(JSON.parse(r.body).detail).toMatch(/Bearer token/);
  });

  it('mockResponseFor native /query returns sources but no Twin tag_filter echo', () => {
    const r = mockResponseFor(
      { m: 'POST', p: '/query', s: '' },
      '{}',
      200,
    );
    expect(r.status).toBe(200);
    const body = JSON.parse(r.body);
    expect(body.sources).toHaveLength(2);
    expect(body.mode).toBe('hybrid');
    // Native /query samples stay upstream-shaped; Twin routes carry tag_filter.
    expect(body).not.toHaveProperty('tag_filter');
  });

  it('mockResponseFor /query/stream also omits tag_filter in the response body', () => {
    const r = mockResponseFor(
      { m: 'POST', p: '/twin/api/query/stream', s: '' },
      '{}',
      200,
    );
    const body = JSON.parse(r.body);
    expect(body).not.toHaveProperty('tag_filter');
  });

  it('mockResponseFor /query/data returns structured retrieval data with metadata.tag_filter', () => {
    const r = mockResponseFor(
      { m: 'POST', p: '/twin/api/query/data', s: '' },
      '{}',
      200,
    );
    expect(r.status).toBe(200);
    const body = JSON.parse(r.body);
    expect(body.status).toBe('success');
    expect(body.data.chunks).toHaveLength(1);
    expect(body.data.chunks[0].score).toBe(0.91);
    // /query/data is the one endpoint that legitimately echoes
    // tag_filter in metadata (audit C2 wired it via TAGGED_WITH).
    expect(body.metadata.tag_filter.all).toEqual(['tag-name']);
  });

  it('mockResponseFor /documents (GET) returns the production list envelope', () => {
    const r = mockResponseFor(
      { m: 'GET', p: '/documents', s: '' },
      '',
      200,
    );
    const body = JSON.parse(r.body);
    expect(body.total).toBe(1);
    expect(body.page).toBe(1);
    expect(body.page_size).toBe(50);
    expect(body.status_counts.completed).toBe(1);
    expect(body.next_cursor).toBeNull();
    expect(body.items[0].source).toBe('source-document.pdf');
  });
});
