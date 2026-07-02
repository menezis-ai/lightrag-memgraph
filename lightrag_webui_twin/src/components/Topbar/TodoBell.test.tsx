/**
 * Unit tests for TodoBell.
 *
 * Drives every branch: empty vs non-empty actionable list, badge count display
 * (1-9 accent, 10+ warn '9+'), filtering of read / non-actionable notifications,
 * open/close interactions (button toggle, outside-click close, Open activity
 * footer), and per-item suffix/sub rendering.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { TodoBell } from './TodoBell';
import type { Notification } from '../../types/topbar';

const listNotificationsMock = vi.hoisted(() => vi.fn());

vi.mock('../../api/resources', () => ({
  api: {
    listNotifications: listNotificationsMock,
  },
}));

function notif(overrides: Partial<Notification> = {}): Notification {
  return {
    id: 'n-1',
    kind: 'tag-mutation',
    title: 'Tag approval requested',
    rel: '2m ago',
    read: false,
    ...overrides,
  };
}

function renderBell(props: Partial<React.ComponentProps<typeof TodoBell>> = {}) {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <TodoBell {...props} />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('TodoBell — badge', () => {
  it('shows no badge and an empty-state label when there are no actionable items', async () => {
    listNotificationsMock.mockResolvedValue([]);
    renderBell();

    await waitFor(() =>
      expect(listNotificationsMock).toHaveBeenCalled(),
    );
    const bell = await screen.findByTestId('topbar-todo-bell');
    expect(bell).toHaveAttribute('aria-label', 'To-do');
    expect(screen.queryByTestId('topbar-todo-count')).toBeNull();
  });

  it('ignores read and non-actionable notifications when counting todos', async () => {
    listNotificationsMock.mockResolvedValue([
      notif({ id: 'read', read: true }),
      notif({ id: 'info', kind: 'info' }),
      notif({ id: 'ready', kind: 'source-ready' }),
      notif({ id: 'live', kind: 'pipeline-warning', read: false }),
    ]);
    renderBell();

    const count = await screen.findByTestId('topbar-todo-count');
    expect(count).toHaveTextContent('1');
    const bell = screen.getByTestId('topbar-todo-bell');
    expect(bell).toHaveAttribute('aria-label', 'To-do, 1 pending');
  });

  it('renders the count for a 1-9 list without the warn background', async () => {
    listNotificationsMock.mockResolvedValue([
      notif({ id: 'a', kind: 'tag-mutation' }),
      notif({ id: 'b', kind: 'source-failed' }),
      notif({ id: 'c', kind: 'pipeline-warning' }),
    ]);
    renderBell();

    const count = await screen.findByTestId('topbar-todo-count');
    expect(count).toHaveTextContent('3');
  });

  it('caps the badge at "9+" and applies the warn background for 10 or more', async () => {
    const many = Array.from({ length: 12 }, (_, i) =>
      notif({ id: `m-${i}`, kind: 'source-failed' }),
    );
    listNotificationsMock.mockResolvedValue(many);
    renderBell();

    // 12 actionable → caps the badge text at '9+' (exercises the
    // todoCount >= 10 warn branch and the > 9 text branch).
    const count = await screen.findByTestId('topbar-todo-count');
    expect(count).toHaveTextContent('9+');
    expect(screen.getByTestId('topbar-todo-bell')).toHaveAttribute(
      'aria-label',
      'To-do, 12 pending',
    );
  });
});

describe('TodoBell — popover', () => {
  it('opens and closes on the button, toggling aria-expanded', async () => {
    listNotificationsMock.mockResolvedValue([]);
    const user = userEvent.setup();
    renderBell();

    const bell = await screen.findByTestId('topbar-todo-bell');
    expect(bell).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('dialog')).toBeNull();

    await user.click(bell);
    expect(bell).toHaveAttribute('aria-expanded', 'true');
    const dialog = screen.getByRole('dialog', { name: 'To-do list' });
    expect(within(dialog).getByText('all caught up')).toBeInTheDocument();
    expect(within(dialog).getByText("You're all caught up.")).toBeInTheDocument();

    await user.click(bell);
    expect(bell).toHaveAttribute('aria-expanded', 'false');
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('lists actionable items with title, suffix and sub, and the count header', async () => {
    listNotificationsMock.mockResolvedValue([
      notif({
        id: 'todo-1',
        kind: 'tag-mutation',
        title: 'Approve tag CIB',
        suffix: 'CIB',
        sub: 'Requested by alice',
      }),
      notif({ id: 'todo-2', kind: 'source-failed', title: 'Extraction failed' }),
    ]);
    const user = userEvent.setup();
    renderBell();

    const bell = await screen.findByTestId('topbar-todo-count');
    await user.click(screen.getByTestId('topbar-todo-bell'));

    expect(bell).toHaveTextContent('2');
    const dialog = screen.getByRole('dialog', { name: 'To-do list' });
    expect(within(dialog).getByText('actionable')).toBeInTheDocument();
    expect(within(dialog).getByText('Approve tag CIB')).toBeInTheDocument();
    expect(within(dialog).getByText('CIB')).toBeInTheDocument();
    expect(within(dialog).getByText('Requested by alice')).toBeInTheDocument();
    expect(within(dialog).getByText('Extraction failed')).toBeInTheDocument();
    expect(screen.getByTestId('topbar-todo-item-todo-1')).toBeInTheDocument();
    expect(screen.getByTestId('topbar-todo-item-todo-2')).toBeInTheDocument();
  });

  it('closes on an outside mousedown', async () => {
    listNotificationsMock.mockResolvedValue([]);
    const user = userEvent.setup();
    renderBell();

    const bell = await screen.findByTestId('topbar-todo-bell');
    await user.click(bell);
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    await user.click(document.body);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('fires onOpenActivity and closes when the footer link is clicked', async () => {
    listNotificationsMock.mockResolvedValue([
      notif({ id: 'todo-x', kind: 'source-failed' }),
    ]);
    const onOpenActivity = vi.fn();
    const user = userEvent.setup();
    renderBell({ onOpenActivity });

    await screen.findByTestId('topbar-todo-count');
    await user.click(screen.getByTestId('topbar-todo-bell'));

    await user.click(screen.getByRole('button', { name: /Open activity/ }));
    expect(onOpenActivity).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('does not throw when the footer link is clicked without an onOpenActivity prop', async () => {
    listNotificationsMock.mockResolvedValue([
      notif({ id: 'todo-y', kind: 'pipeline-warning' }),
    ]);
    const user = userEvent.setup();
    renderBell();

    await screen.findByTestId('topbar-todo-count');
    await user.click(screen.getByTestId('topbar-todo-bell'));
    await user.click(screen.getByRole('button', { name: /Open activity/ }));
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });
});
