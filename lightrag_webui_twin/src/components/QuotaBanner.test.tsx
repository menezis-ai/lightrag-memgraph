/**
 * Unit tests for QuotaBanner + useIngestionDisabled.
 *
 * Drives MSW directly via the ``setMockQuotaState`` knob, mirroring
 * the API keys test pattern.
 */

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { setupServer } from 'msw/node';
import { QuotaBanner } from './QuotaBanner';
import { handlers, resetDocumentsState, setMockQuotaState } from '../mocks/handlers';
import { useIngestionDisabled } from '../hooks/useIngestionDisabled';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
afterAll(() => server.close());
beforeEach(() => resetDocumentsState());
afterEach(() => {
  server.resetHandlers();
  resetDocumentsState();
});

function renderBanner(tone?: 'compact' | 'block') {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      <QuotaBanner tone={tone} />
    </QueryClientProvider>,
  );
}

function renderHookFlag() {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  function Probe() {
    const disabled = useIngestionDisabled();
    return <div data-testid="probe">{String(disabled)}</div>;
  }
  return render(
    <QueryClientProvider client={client}>
      <Probe />
    </QueryClientProvider>,
  );
}

describe('QuotaBanner', () => {
  it('renders nothing in OK state', async () => {
    setMockQuotaState({ status: 'ok' });
    renderBanner();
    // Give the query time to resolve; banner should never appear
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId('quota-banner-warning')).toBeNull();
    expect(screen.queryByTestId('quota-banner-blocked')).toBeNull();
  });

  it('renders nothing when Memgraph and fallback expose no limit', async () => {
    setMockQuotaState({
      configured: false,
      status: 'ok',
      limit_bytes: null,
      used_pct: null,
    });
    renderBanner();
    await new Promise((r) => setTimeout(r, 50));
    expect(screen.queryByTestId('quota-banner-warning')).toBeNull();
    expect(screen.queryByTestId('quota-banner-blocked')).toBeNull();
  });

  it('renders amber warning banner with percentage at 85%+', async () => {
    setMockQuotaState({
      status: 'warning',
      used_bytes: Math.floor(0.9 * 2 * 1024 * 1024 * 1024),
      limit_bytes: 2 * 1024 * 1024 * 1024,
    });
    renderBanner();
    const banner = await screen.findByTestId('quota-banner-warning');
    expect(banner).toHaveAttribute('role', 'status');
    expect(banner.textContent).toMatch(/Storage at 90%/);
    expect(banner.textContent).toMatch(/1\.80 GiB \/ 2\.00 GiB/);
  });

  it('renders red blocking banner with role=alert at 100%+', async () => {
    setMockQuotaState({
      status: 'blocked',
      used_bytes: 2 * 1024 * 1024 * 1024,
      limit_bytes: 2 * 1024 * 1024 * 1024,
    });
    renderBanner();
    const banner = await screen.findByTestId('quota-banner-blocked');
    expect(banner).toHaveAttribute('role', 'alert');
    expect(banner.textContent).toMatch(/Memgraph instance quota reached/i);
    expect(banner.textContent).toMatch(/ingestion disabled/i);
  });

  it('honours the compact tone class for inline use', async () => {
    setMockQuotaState({
      status: 'warning',
      used_bytes: Math.floor(0.9 * 2 * 1024 * 1024 * 1024),
      limit_bytes: 2 * 1024 * 1024 * 1024,
    });
    renderBanner('compact');
    const banner = await screen.findByTestId('quota-banner-warning');
    expect(banner.className).toContain('quota-banner-compact');
  });
});

describe('useIngestionDisabled', () => {
  it('returns false when status is ok', async () => {
    setMockQuotaState({ status: 'ok' });
    renderHookFlag();
    await waitFor(() => {
      expect(screen.getByTestId('probe').textContent).toBe('false');
    });
  });

  it('returns false when status is warning', async () => {
    setMockQuotaState({
      status: 'warning',
      used_bytes: Math.floor(0.9 * 2 * 1024 * 1024 * 1024),
      limit_bytes: 2 * 1024 * 1024 * 1024,
    });
    renderHookFlag();
    await waitFor(() => {
      expect(screen.getByTestId('probe').textContent).toBe('false');
    });
  });

  it('returns true when status is blocked', async () => {
    setMockQuotaState({
      status: 'blocked',
      used_bytes: 2 * 1024 * 1024 * 1024,
      limit_bytes: 2 * 1024 * 1024 * 1024,
    });
    renderHookFlag();
    await waitFor(() => {
      expect(screen.getByTestId('probe').textContent).toBe('true');
    });
  });
});
