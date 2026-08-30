/**
 * Unit tests for AboutSection — Settings → About.
 *
 * The property under test is the two-tier payload: an admin sees the
 * deployment shape, a non-admin must get a valid shorter card rather than
 * empty rows or a crash. Also covers the two failure modes the panel is
 * actually opened in — a backend error, and a reachable-but-degraded
 * Memgraph.
 */

import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { setupServer } from 'msw/node';
import { http, HttpResponse } from 'msw';
import { AboutSection } from './AboutSection';
import { handlers } from '../../mocks/handlers';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
afterEach(() => server.resetHandlers());

function renderSection() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <AboutSection />
    </QueryClientProvider>,
  );
}

/** Override the default (admin) fixture with an arbitrary payload. */
function serveAbout(body: unknown, status = 200) {
  server.use(
    http.get('*/twin/api/system/about', () =>
      HttpResponse.json(body as never, { status }),
    ),
  );
}

const NON_ADMIN = {
  twin: '1.1.0',
  lightrag: { native: '1.4.9.11', composite: '1.4.9.11+memgraph-1.1.0' },
  admin: false,
  memgraph: null,
  runtime: null,
  storage: null,
  overlay: null,
};

describe('AboutSection', () => {
  it('renders the full card for an admin', async () => {
    renderSection();
    await waitFor(() =>
      expect(screen.getByTestId('settings-about')).toBeInTheDocument(),
    );

    expect(await screen.findByText('3.12.0')).toBeInTheDocument();
    expect(screen.getByText('1.4.9.11')).toBeInTheDocument();
    expect(screen.getByText('1.4.9.11+memgraph-1.1.0')).toBeInTheDocument();
    expect(screen.getByText('MemgraphKVStorage')).toBeInTheDocument();
    expect(screen.getByText(/CPython/)).toBeInTheDocument();
    // Admin card must not show the "sign in as admin" hint.
    expect(
      screen.queryByTestId('settings-about-reduced'),
    ).not.toBeInTheDocument();
  });

  it('renders a valid shorter card for a non-admin', async () => {
    serveAbout(NON_ADMIN);
    renderSection();

    expect(await screen.findByText('1.4.9.11')).toBeInTheDocument();
    expect(
      screen.getByTestId('settings-about-reduced'),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        'Deployment details are available to administrators only.',
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/sign in/i)).not.toBeInTheDocument();
    // No empty deployment rows leaked.
    expect(screen.queryByText('Memgraph')).not.toBeInTheDocument();
    expect(screen.queryByText('Runtime')).not.toBeInTheDocument();
    expect(screen.queryByText('Storage backends')).not.toBeInTheDocument();
  });

  it('surfaces an unreachable Memgraph instead of a blank version', async () => {
    serveAbout({
      ...NON_ADMIN,
      admin: true,
      memgraph: {
        reachable: false,
        version: null,
        mage: null,
        procedures: null,
        error: 'ServiceUnavailable',
      },
    });
    renderSection();

    expect(
      await screen.findByText(/unreachable \(ServiceUnavailable\)/),
    ).toBeInTheDocument();
    // An unreachable server tells us nothing about MAGE. Claiming "floor
    // tier" here would send the operator debugging the wrong thing.
    expect(screen.queryByText(/floor tier/)).not.toBeInTheDocument();
    expect(screen.getByText(/unknown/)).toBeInTheDocument();
  });

  it('distinguishes an unresolved MAGE probe from a confirmed floor tier', async () => {
    serveAbout({
      ...NON_ADMIN,
      admin: true,
      memgraph: {
        reachable: true,
        version: '3.12.0',
        mage: null,
        procedures: null,
        error: null,
      },
    });
    renderSection();

    expect(
      await screen.findByText(/unknown — capability probe unavailable/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/floor tier/)).not.toBeInTheDocument();
  });

  it('reports a confirmed floor tier on a core-only instance', async () => {
    renderSection(); // default fixture: reachable, core procedures, no MAGE
    expect(
      await screen.findByText('not available — floor tier'),
    ).toBeInTheDocument();
  });

  it.each([
    [true, 'available — configured override'],
    [false, 'not available — configured floor tier'],
  ])(
    'renders a resolved %s override without inventing a procedure count',
    async (mage, label) => {
      serveAbout({
        ...NON_ADMIN,
        admin: true,
        memgraph: {
          reachable: true,
          version: '3.12.0',
          mage,
          procedures: null,
          error: null,
        },
      });
      renderSection();

      expect(await screen.findByText(label)).toBeInTheDocument();
      expect(screen.queryByText(/0 procedures/)).not.toBeInTheDocument();
    },
  );

  it('shows an error message when the backend fails', async () => {
    serveAbout({ detail: 'boom' }, 500);
    renderSection();

    await waitFor(() =>
      expect(screen.getByTestId('settings-about-error')).toBeInTheDocument(),
    );
  });

  it('copies the payload as JSON', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText },
      configurable: true,
    });

    renderSection();
    const button = await screen.findByTestId('settings-about-copy');
    await userEvent.click(button);

    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    const payload = JSON.parse(writeText.mock.calls[0][0] as string);
    expect(payload.twin).toBe('1.1.0');
    expect(payload.memgraph.version).toBe('3.12.0');
    expect(await screen.findByText('Copied')).toBeInTheDocument();
  });

  it('reports a denied clipboard instead of failing silently', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: vi.fn().mockRejectedValue(new Error('denied')) },
      configurable: true,
    });

    renderSection();
    await userEvent.click(await screen.findByTestId('settings-about-copy'));

    // The operator is mid-ticket: a silent no-op would mean an empty paste.
    expect(await screen.findByText('Copy failed')).toBeInTheDocument();
  });
});
