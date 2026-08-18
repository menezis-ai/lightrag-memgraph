import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FolderSwitcher } from './FolderSwitcher';
import { FOLDER_FIXTURES } from '../../fixtures';

describe('FolderSwitcher', () => {
  it('renders the active folder and opens a menu on click', async () => {
    const onPick = vi.fn();
    render(
      <FolderSwitcher
        active="cib"
        folders={FOLDER_FIXTURES}
        onPick={onPick}
      />,
    );
    expect(screen.getByTestId('topbar-folder-switcher')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('topbar-folder-switcher'));
    expect(screen.getByTestId('topbar-folder-menu')).toBeInTheDocument();
  });

  it('emits onPick(id) when a non-active folder is clicked', async () => {
    const onPick = vi.fn();
    render(
      <FolderSwitcher
        active="cib"
        folders={FOLDER_FIXTURES}
        onPick={onPick}
      />,
    );
    await userEvent.click(screen.getByTestId('topbar-folder-switcher'));
    const nonActive = FOLDER_FIXTURES.find((folder) => folder.id !== 'cib');
    if (!nonActive) throw new Error('expected fixture diversity');
    await userEvent.click(
      screen.getByTestId(`topbar-folder-pick-${nonActive.id}`),
    );
    expect(onPick).toHaveBeenCalledWith(nonActive.id);
  });
});
