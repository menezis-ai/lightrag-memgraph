/**
 * Unit tests for VisionSection — Settings → Vision UI.
 *
 * Drives the full mutation loop through the MSW handlers:
 *   - renders GET values (min OCR chars + drop-class chips)
 *   - env-default provenance hint, flipped to runtime after a save
 *   - admin-only procedure-ingestion toggle
 *   - editing + save sends the right PUT body
 *   - chip add (input + Enter) / remove (chip ✕)
 *   - 403 from PUT surfaces the "Admin scope required" toast
 *   - non-admin users get disabled inputs + the read-only badge
 */

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { VisionSection } from './VisionSection';
import { handlers, resetDocumentsState } from '../../mocks/handlers';
import type { AuthenticatedUser, TwinRuntimeConfig } from '../../types/auth';

const server = setupServer(...handlers);

const adminUser: AuthenticatedUser = {
  sso_subject: 'steward@example.test',
  email: 'steward@example.test',
  name: 'Steward',
  palier: {
    level: 3,
    label: 'Steward',
    scopes: ['twin:read', 'twin:write', 'twin:approve'],
  },
  folders: ['cib'],
  idp: 'keycloak',
  idp_realm: 'twin-cib',
  sub: 'steward-1',
  session_expires: '2026-06-04T23:59:00Z',
  gateway_scopes: ['read:documents', 'admin:folders'],
};

const readonlyUser: AuthenticatedUser = {
  ...adminUser,
  gateway_scopes: ['read:documents'],
};

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
afterAll(() => server.close());

beforeEach(() => {
  resetDocumentsState();
  window.__twinE2eRuntimeConfig = undefined;
  window.__twinConfig = undefined;
});

afterEach(() => {
  server.resetHandlers();
  resetDocumentsState();
  window.__twinE2eRuntimeConfig = undefined;
  window.__twinConfig = undefined;
});

function renderSection(
  props: Partial<Parameters<typeof VisionSection>[0]> = {},
) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={client}>
      <VisionSection {...props} />
    </QueryClientProvider>,
  );
}

describe('VisionSection — rendering', () => {
  it('renders the GET values: min OCR chars input + drop-class chips', async () => {
    renderSection();
    const input = await screen.findByTestId('settings-vision-min-ocr');
    expect(input).toHaveValue(20);
    const chips = screen.getByTestId('settings-vision-classes');
    expect(within(chips).getByText('invalid')).toBeInTheDocument();
    expect(within(chips).getByText('logo')).toBeInTheDocument();
    expect(within(chips).getByText('signature')).toBeInTheDocument();
    expect(
      screen.getByTestId('settings-vision-procedure-toggle'),
    ).toHaveAttribute('aria-checked', 'false');
  });

  it('shows the env-default provenance hint when nothing was ever saved', async () => {
    renderSection();
    const hint = await screen.findByTestId('settings-vision-provenance-env');
    expect(hint).toHaveTextContent(/Defaults from deployment environment/);
    expect(
      screen.queryByTestId('settings-vision-provenance-runtime'),
    ).not.toBeInTheDocument();
  });

  it('stays clean when an older backend omits the procedure flag', async () => {
    server.use(
      http.get('*/twin/api/settings/vision', () =>
        HttpResponse.json({
          min_ocr_chars: 20,
          drop_classes: ['invalid', 'logo', 'signature'],
          source: 'runtime',
          updated_at: null,
          updated_by: null,
        }),
      ),
    );

    renderSection({ user: adminUser });
    await screen.findByTestId('settings-vision-min-ocr');

    await waitFor(() =>
      expect(screen.getByTestId('settings-vision-save')).toBeDisabled(),
    );
    expect(
      screen.getByTestId('settings-vision-procedure-toggle'),
    ).toHaveAttribute('aria-checked', 'false');
  });

  it('disables inputs and shows the read-only badge for non-admin users', async () => {
    renderSection({ user: readonlyUser });
    const input = await screen.findByTestId('settings-vision-min-ocr');
    expect(input).toBeDisabled();
    expect(
      screen.getByTestId('settings-vision-readonly-badge'),
    ).toHaveTextContent(/Read-only — admin scope required/);
    expect(
      screen.queryByTestId('settings-vision-class-input'),
    ).not.toBeInTheDocument();
    expect(screen.getByTestId('settings-vision-save')).toBeDisabled();
    expect(
      screen.getByTestId('settings-vision-procedure-toggle'),
    ).toBeDisabled();
  });
});

describe('VisionSection — editing + save', () => {
  it('sends the edited values as the PUT body and flips provenance to runtime', async () => {
    const requests: unknown[] = [];
    server.events.on('request:start', ({ request }) => {
      if (request.method === 'PUT' && request.url.includes('/settings/vision')) {
        void request
          .clone()
          .json()
          .then((body) => requests.push(body));
      }
    });

    const user = userEvent.setup();
    const onToast = vi.fn();
    renderSection({ user: adminUser, onToast });

    const input = await screen.findByTestId('settings-vision-min-ocr');
    await user.clear(input);
    await user.type(input, '120');
    await waitFor(() => expect(input).toHaveValue(120));

    // Remove one class, add another.
    await user.click(
      screen.getByRole('button', { name: 'Remove drop class logo' }),
    );
    const classInput = screen.getByTestId('settings-vision-class-input');
    // Repo doctrine: split type + Enter into two calls (slow-CI race).
    await user.type(classInput, 'screenshot');
    await waitFor(() => expect(classInput).toHaveValue('screenshot'));
    await user.type(classInput, '{Enter}');
    await waitFor(() =>
      expect(
        within(screen.getByTestId('settings-vision-classes')).getByText(
          'screenshot',
        ),
      ).toBeInTheDocument(),
    );
    await user.click(screen.getByTestId('settings-vision-procedure-toggle'));

    await user.click(screen.getByTestId('settings-vision-save'));

    await waitFor(() =>
      expect(
        screen.getByTestId('settings-vision-provenance-runtime'),
      ).toBeInTheDocument(),
    );
    expect(requests).toHaveLength(1);
    expect(requests[0]).toEqual({
      min_ocr_chars: 120,
      drop_classes: ['invalid', 'signature', 'screenshot'],
      procedure_enabled: true,
    });
    expect(onToast).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'done', title: 'Vision settings saved' }),
    );
    // Refetch lands the canonicalized (sorted) list back into the chips.
    const chips = screen.getByTestId('settings-vision-classes');
    expect(within(chips).queryByText('logo')).not.toBeInTheDocument();
    expect(within(chips).getByText('screenshot')).toBeInTheDocument();
    expect(
      screen.getByTestId('settings-vision-procedure-toggle'),
    ).toHaveAttribute('aria-checked', 'true');
  });

  it('rejects an invalid drop class client-side without round-tripping', async () => {
    const user = userEvent.setup();
    renderSection({ user: adminUser });
    await screen.findByTestId('settings-vision-min-ocr');

    const classInput = screen.getByTestId('settings-vision-class-input');
    await user.type(classInput, 'Bad Slug!');
    await waitFor(() => expect(classInput).toHaveValue('Bad Slug!'));
    await user.type(classInput, '{Enter}');
    expect(
      await screen.findByTestId('settings-vision-class-error'),
    ).toHaveTextContent(/Invalid class/);
    // Not added as a chip.
    expect(
      within(screen.getByTestId('settings-vision-classes')).queryByText(
        'bad slug!',
      ),
    ).not.toBeInTheDocument();
  });

  it('flags an out-of-range min OCR chars value and disables Save', async () => {
    const user = userEvent.setup();
    renderSection({ user: adminUser });
    const input = await screen.findByTestId('settings-vision-min-ocr');
    await user.clear(input);
    await user.type(input, '100001');
    expect(
      await screen.findByTestId('settings-vision-min-ocr-invalid'),
    ).toBeInTheDocument();
    expect(screen.getByTestId('settings-vision-save')).toBeDisabled();
  });

  it('surfaces an Admin scope required toast when the backend returns 403', async () => {
    // Backend gate simulation: the MSW handler denies mutations when the
    // runtime-config user lacks the admin:folders gateway scope (same
    // mechanism as the folder-admin handlers).
    window.__twinE2eRuntimeConfig = {
      apiBaseUrl: '/twin/api',
      lightragBaseUrl: '',
      idpLogoutUrl: '',
      debugUser: readonlyUser,
    } satisfies TwinRuntimeConfig;

    const user = userEvent.setup();
    const onToast = vi.fn();
    // UI believes the user is admin (inputs enabled) — the backend says no.
    renderSection({ user: adminUser, onToast });

    const input = await screen.findByTestId('settings-vision-min-ocr');
    await user.clear(input);
    await user.type(input, '55');
    await user.click(screen.getByTestId('settings-vision-save'));

    await waitFor(() =>
      expect(onToast).toHaveBeenCalledWith(
        expect.objectContaining({
          kind: 'error',
          title: 'Admin scope required',
        }),
      ),
    );
    // Provenance stays env-default — nothing was persisted.
    expect(
      screen.getByTestId('settings-vision-provenance-env'),
    ).toBeInTheDocument();
  });
});
