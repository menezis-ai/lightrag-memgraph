/**
 * Unit tests for ApiKeysSection — Settings → API keys UI.
 *
 * Drives the full mutation loop through MSW handlers:
 *   - empty state
 *   - create → one-time-reveal modal exposes full_value
 *   - copy-to-clipboard handler is called with the secret
 *   - dismiss removes the secret from the DOM
 *   - revoke double-confirm (first click sets state, second click DELETEs)
 *   - revoked rows display the revoked pill and hide the revoke button
 */

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { setupServer } from 'msw/node';
import { ApiKeysSection } from './ApiKeysSection';
import { handlers, resetDocumentsState } from '../../mocks/handlers';

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterAll(() => server.close());
beforeEach(() => resetDocumentsState());
afterEach(() => {
  server.resetHandlers();
  resetDocumentsState();
});

function renderSection(copyToClipboard?: (v: string) => Promise<void>) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      <ApiKeysSection copyToClipboard={copyToClipboard} />
    </QueryClientProvider>,
  );
}

describe('ApiKeysSection', () => {
  it('shows the empty state when no keys exist', async () => {
    renderSection();
    await waitFor(() =>
      expect(screen.getByTestId('settings-api-keys-empty')).toBeInTheDocument(),
    );
    expect(
      screen.queryByTestId('settings-api-keys-table'),
    ).not.toBeInTheDocument();
  });

  it('create flow surfaces full_value exactly once, then drops it on dismiss', async () => {
    const user = userEvent.setup();
    renderSection();
    await screen.findByTestId('settings-api-keys-empty');

    await user.click(screen.getByTestId('settings-api-keys-create-btn'));
    const backdrop = await screen.findByTestId(
      'settings-api-keys-create-backdrop',
    );
    const nameInput = within(backdrop).getByTestId(
      'settings-api-keys-create-name',
    );
    await user.type(nameInput, 'ingestion-bot');
    await user.click(
      within(backdrop).getByTestId('settings-api-keys-create-submit'),
    );

    const reveal = await screen.findByTestId(
      'settings-api-keys-reveal-backdrop',
    );
    const value = within(reveal).getByTestId(
      'settings-api-keys-reveal-value',
    );
    expect(value.textContent).toMatch(/^twk_/);
    expect(within(reveal).getByTestId('settings-api-keys-reveal-name'))
      .toHaveTextContent('ingestion-bot');

    await user.click(
      within(reveal).getByTestId('settings-api-keys-reveal-dismiss'),
    );
    await waitFor(() =>
      expect(
        screen.queryByTestId('settings-api-keys-reveal-backdrop'),
      ).not.toBeInTheDocument(),
    );

    // List now contains the key but the full value is gone from the DOM.
    const table = await screen.findByTestId('settings-api-keys-table');
    expect(within(table).getByText('ingestion-bot')).toBeInTheDocument();
    expect(screen.queryByText(value.textContent ?? '!!!')).not.toBeInTheDocument();
  });

  it('copy-to-clipboard hook receives the full secret', async () => {
    const copyMock = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    renderSection(copyMock);
    await screen.findByTestId('settings-api-keys-empty');

    await user.click(screen.getByTestId('settings-api-keys-create-btn'));
    await user.type(
      await screen.findByTestId('settings-api-keys-create-name'),
      'copy-test',
    );
    await user.click(screen.getByTestId('settings-api-keys-create-submit'));
    const reveal = await screen.findByTestId(
      'settings-api-keys-reveal-backdrop',
    );

    const value = within(reveal)
      .getByTestId('settings-api-keys-reveal-value')
      .textContent;
    await user.click(
      within(reveal).getByTestId('settings-api-keys-reveal-copy'),
    );

    expect(copyMock).toHaveBeenCalledTimes(1);
    expect(copyMock).toHaveBeenCalledWith(value);
    expect(
      await within(reveal).findByText(/Copied/i),
    ).toBeInTheDocument();
  });

  it('rejects blank name client-side without round-tripping', async () => {
    const user = userEvent.setup();
    renderSection();
    await screen.findByTestId('settings-api-keys-empty');

    await user.click(screen.getByTestId('settings-api-keys-create-btn'));
    const submit = await screen.findByTestId(
      'settings-api-keys-create-submit',
    );
    // Empty input → submit disabled.
    expect(submit).toBeDisabled();

    // Typing then deleting still keeps disabled.
    const input = screen.getByTestId('settings-api-keys-create-name');
    await user.type(input, 'x');
    expect(submit).not.toBeDisabled();
    await user.clear(input);
    expect(submit).toBeDisabled();
  });

  it('revoke requires a second confirm click', async () => {
    const user = userEvent.setup();
    renderSection();
    await screen.findByTestId('settings-api-keys-empty');

    // Seed a key.
    await user.click(screen.getByTestId('settings-api-keys-create-btn'));
    await user.type(
      await screen.findByTestId('settings-api-keys-create-name'),
      'to-revoke',
    );
    await user.click(screen.getByTestId('settings-api-keys-create-submit'));
    const reveal = await screen.findByTestId(
      'settings-api-keys-reveal-backdrop',
    );
    await user.click(
      within(reveal).getByTestId('settings-api-keys-reveal-dismiss'),
    );

    const table = await screen.findByTestId('settings-api-keys-table');
    const row = within(table).getByText('to-revoke').closest('tr');
    expect(row).not.toBeNull();
    const rowId = row!.getAttribute('data-testid')!.replace(
      'settings-api-keys-row-',
      '',
    );

    // First click → confirm state appears.
    await user.click(
      screen.getByTestId(`settings-api-keys-revoke-${rowId}`),
    );
    const confirmBtn = await screen.findByTestId(
      `settings-api-keys-revoke-confirm-${rowId}`,
    );
    expect(confirmBtn).toBeInTheDocument();

    // Cancel restores the original button.
    await user.click(
      screen.getByTestId(`settings-api-keys-revoke-cancel-${rowId}`),
    );
    expect(
      screen.getByTestId(`settings-api-keys-revoke-${rowId}`),
    ).toBeInTheDocument();

    // Confirm path: re-open, confirm, row is marked revoked.
    await user.click(
      screen.getByTestId(`settings-api-keys-revoke-${rowId}`),
    );
    await user.click(
      await screen.findByTestId(
        `settings-api-keys-revoke-confirm-${rowId}`,
      ),
    );

    await waitFor(() => {
      const updatedRow = screen.getByTestId(
        `settings-api-keys-row-${rowId}`,
      );
      expect(within(updatedRow).getByText('revoked')).toBeInTheDocument();
    });
    expect(
      screen.queryByTestId(`settings-api-keys-revoke-${rowId}`),
    ).not.toBeInTheDocument();
  });
});
