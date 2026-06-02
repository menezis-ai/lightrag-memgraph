/**
 * Unit tests for Topbar.
 *
 * Behaviors under test:
 *   - tabs render and click invokes onTab
 *   - active tab gets the .active class
 *   - workspace pill opens/closes the menu; pick invokes onSwitchWorkspace
 *   - Escape closes both popovers
 *   - notifications bell opens the popover; unread count shows badge
 *   - empty notifications -> "all caught up"
 *   - theme button toggle -> onTheme called
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Topbar } from './Topbar';
import type { Notification, Workspace } from '../types/topbar';

const sampleWorkspaces: Workspace[] = [
  {
    id: 'cib',
    kb: 'CIB KB',
    visibility: 'private',
    sources: 247,
    role: 'admin / steward',
    current: true,
  },
  {
    id: 'payments',
    kb: 'Payments KB',
    visibility: 'internal',
    sources: 1318,
    role: 'reader',
    current: false,
  },
];

const sampleNotifs: Notification[] = [
  {
    id: 'n1',
    kind: 'tag-mutation',
    title: 'Tag',
    tagname: 'rman',
    suffix: 'applied',
    sub: 'oracle.pdf',
    rel: '12m ago',
    read: false,
  },
];

function baseProps() {
  return {
    tab: 'documents',
    onTab: vi.fn(),
    theme: 'light' as const,
    onTheme: vi.fn(),
    workspace: 'cib',
    kbName: 'CIB KB',
    onSwitchWorkspace: vi.fn(),
    workspaces: sampleWorkspaces,
    notifications: sampleNotifs,
    unreadCount: 1,
    onMarkAllRead: vi.fn(),
    onClearNotifications: vi.fn(),
    onOpenActivity: vi.fn(),
  };
}

describe('Topbar — tabs', () => {
  it('renders the default tabs in the canonical order without API', () => {
    render(<Topbar {...baseProps()} />);
    // Doctrine product (2026-05-31) — Documents · Tags · Retrieval · Graph ·
    // Activity · Settings, in this exact order. API is moved into Settings
    // and MUST NOT appear in the topbar nav.
    const tabBtns = Array.from(
      document.querySelectorAll('.tabs button.tab'),
    ) as HTMLButtonElement[];
    expect(tabBtns.map((b) => b.textContent)).toEqual([
      'Documents',
      'Tags',
      'Retrieval',
      'Graph',
      'Activity',
      'Settings',
    ]);
    expect(screen.queryByRole('button', { name: 'API' })).toBeNull();
  });

  it('marks the active tab', () => {
    render(<Topbar {...baseProps()} tab="retrieval" />);
    const active = screen.getByRole('button', { name: 'Retrieval' });
    expect(active.className).toContain('active');
  });

  it('invokes onTab when a tab is clicked', async () => {
    const p = baseProps();
    render(<Topbar {...p} />);
    await userEvent.click(screen.getByRole('button', { name: 'Tags' }));
    expect(p.onTab).toHaveBeenCalledWith('tags');
  });

  it('honors custom tabs prop', () => {
    render(
      <Topbar
        {...baseProps()}
        tabs={[{ id: 'only', label: 'Only' }]}
      />,
    );
    expect(screen.getByRole('button', { name: 'Only' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Documents' })).toBeNull();
  });
});

describe('Topbar — space switcher', () => {
  it('opens the space menu on pill click', async () => {
    render(<Topbar {...baseProps()} />);
    await userEvent.click(screen.getByTitle('Switch space'));
    expect(screen.getByRole('menu', { name: 'Switch space' })).toBeInTheDocument();
  });

  it('invokes onSwitchWorkspace when a non-active row is picked', async () => {
    const p = baseProps();
    render(<Topbar {...p} />);
    await userEvent.click(screen.getByTitle('Switch space'));
    // The "payments" row should be enabled (not current).
    const payments = screen.getByRole('menuitemradio', { checked: false });
    await userEvent.click(payments);
    expect(p.onSwitchWorkspace).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'payments' }),
    );
  });

  it('disables the row matching the current space', async () => {
    render(<Topbar {...baseProps()} />);
    await userEvent.click(screen.getByTitle('Switch space'));
    const current = screen.getByRole('menuitemradio', { checked: true });
    expect(current).toBeDisabled();
  });

  it('closes the menu when Escape is pressed', async () => {
    render(<Topbar {...baseProps()} />);
    await userEvent.click(screen.getByTitle('Switch space'));
    expect(screen.queryByRole('menu')).toBeInTheDocument();
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByRole('menu')).toBeNull();
  });
});

describe('Topbar — notifications', () => {
  it('shows the unread badge when unreadCount > 0', () => {
    render(<Topbar {...baseProps()} unreadCount={3} />);
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('caps the badge at "9+"', () => {
    render(<Topbar {...baseProps()} unreadCount={42} />);
    expect(screen.getByText('9+')).toBeInTheDocument();
  });

  it('opens the popover on bell click', async () => {
    render(<Topbar {...baseProps()} />);
    await userEvent.click(
      screen.getByRole('button', { name: /Notifications, 1 unread/ }),
    );
    expect(
      screen.getByRole('dialog', { name: 'Notifications' }),
    ).toBeInTheDocument();
  });

  it('shows "all caught up" when notifications is empty', async () => {
    render(<Topbar {...baseProps()} notifications={[]} unreadCount={0} />);
    await userEvent.click(screen.getByRole('button', { name: /Notifications$/ }));
    expect(screen.getByText("You're all caught up.")).toBeInTheDocument();
  });
});

describe('Topbar — theme toggle', () => {
  it('calls onTheme when the theme button is clicked', async () => {
    const p = baseProps();
    render(<Topbar {...p} />);
    await userEvent.click(screen.getByRole('button', { name: 'Theme' }));
    expect(p.onTheme).toHaveBeenCalled();
  });
});
