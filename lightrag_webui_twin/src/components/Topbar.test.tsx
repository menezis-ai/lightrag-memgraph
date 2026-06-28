/**
 * Unit tests for Topbar.
 *
 * Behaviors under test:
 *   - tabs render and click invokes onTab
 *   - active tab gets the .active class
 *   - folder pill opens/closes the menu; pick invokes onSwitchFolder
 *   - Escape closes both popovers
 *   - notifications bell opens the popover; unread count shows badge
 *   - empty notifications -> "all caught up"
 *   - theme button toggle -> onTheme called
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Topbar } from './Topbar';
import type { Notification, Folder } from '../types/topbar';

const sampleFolders: Folder[] = [
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
    folder: 'cib',
    kbName: 'CIB KB',
    onSwitchFolder: vi.fn(),
    folders: sampleFolders,
    notifications: sampleNotifs,
    unreadCount: 1,
    onMarkAllRead: vi.fn(),
    onClearNotifications: vi.fn(),
    onOpenActivity: vi.fn(),
    onManageFolders: vi.fn(),
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

  it('opens Documents when the brand is clicked', async () => {
    const p = baseProps();
    render(<Topbar {...p} tab="graph" />);
    await userEvent.click(screen.getByRole('button', { name: 'Open Documents' }));
    expect(p.onTab).toHaveBeenCalledWith('documents');
  });

  it('uses the Twin KMS product name in the brand', () => {
    render(<Topbar {...baseProps()} />);
    expect(document.querySelector('.brand-name')).toHaveTextContent('Twin KMS');
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

describe('Topbar — folder switcher', () => {
  it('opens the folder menu on pill click', async () => {
    render(<Topbar {...baseProps()} />);
    await userEvent.click(screen.getByTitle('Switch folder'));
    expect(screen.getByRole('menu', { name: 'Switch folder' })).toBeInTheDocument();
  });

  it('invokes onSwitchFolder when a non-active row is picked', async () => {
    const p = baseProps();
    render(<Topbar {...p} />);
    await userEvent.click(screen.getByTitle('Switch folder'));
    // The "payments" row should be enabled (not current).
    const payments = screen.getByRole('menuitemradio', { checked: false });
    await userEvent.click(payments);
    expect(p.onSwitchFolder).toHaveBeenCalledWith(
      expect.objectContaining({ id: 'payments' }),
    );
  });

  it('disables the row matching the current folder', async () => {
    render(<Topbar {...baseProps()} />);
    await userEvent.click(screen.getByTitle('Switch folder'));
    const current = screen.getByRole('menuitemradio', { checked: true });
    expect(current).toBeDisabled();
  });

  it('closes the menu when Escape is pressed', async () => {
    render(<Topbar {...baseProps()} />);
    await userEvent.click(screen.getByTitle('Switch folder'));
    expect(screen.queryByRole('menu')).toBeInTheDocument();
    await userEvent.keyboard('{Escape}');
    expect(screen.queryByRole('menu')).toBeNull();
  });

  it('routes the manage folders action to the host', async () => {
    const p = baseProps();
    render(<Topbar {...p} />);
    await userEvent.click(screen.getByTitle('Switch folder'));
    await userEvent.click(screen.getByRole('button', { name: /Manage folders/i }));
    expect(p.onManageFolders).toHaveBeenCalledTimes(1);
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

  it('uses an anchored popover element instead of a native dialog', async () => {
    render(<Topbar {...baseProps()} />);
    await userEvent.click(
      screen.getByRole('button', { name: /Notifications, 1 unread/ }),
    );
    expect(
      screen.getByRole('dialog', { name: 'Notifications' }).tagName,
    ).toBe('DIV');
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
