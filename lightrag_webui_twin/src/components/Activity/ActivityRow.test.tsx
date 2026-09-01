import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ACTIVITY_FIXTURES } from '../../fixtures';
import type { ActivityEvent } from '../../types/activity';
import { ActivityRow } from './ActivityRow';

describe('ActivityRow', () => {
  it('projects selection, scope and the row activation callback', async () => {
    const onClick = vi.fn();
    render(
      <ActivityRow
        event={ACTIVITY_FIXTURES[0]}
        relativeLabel="5m ago"
        folder="sandbox"
        selected
        onClick={onClick}
      />,
    );

    const row = screen.getByRole('button');
    expect(row).toHaveAttribute('aria-current', 'true');
    expect(screen.getByTestId('activity-row-folder')).toHaveTextContent('sandbox');
    await userEvent.click(row);
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('humanizes a backend event kind unknown to the static map', () => {
    render(
      <ActivityRow
        event={
          {
            ...ACTIVITY_FIXTURES[0],
            kind: 'future-policy-event',
          } as unknown as ActivityEvent
        }
        relativeLabel="now"
        folder="main"
        selected={false}
        onClick={() => {}}
      />,
    );
    expect(screen.getByText('Future policy event')).toBeInTheDocument();
  });
});
