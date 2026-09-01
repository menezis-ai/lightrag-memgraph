import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ACTIVITY_FIXTURES } from '../../fixtures';
import { ActivityDetail } from './ActivityDetail';

function renderDetail(index: number) {
  const onNavigate = vi.fn();
  const onPushToast = vi.fn();
  const event = ACTIVITY_FIXTURES[index];
  render(
    <ActivityDetail
      event={event}
      relativeLabel="5m ago"
      folder="sandbox"
      onNavigate={onNavigate}
      onPushToast={onPushToast}
    />,
  );
  return { event, onNavigate, onPushToast };
}

describe('ActivityDetail', () => {
  it('projects query context into host-controlled retrieval navigation', async () => {
    const { event, onNavigate } = renderDetail(0);
    await userEvent.click(screen.getByRole('button', { name: /Re-run query/ }));
    expect(onNavigate).toHaveBeenCalledWith('retrieval', {
      q: event.target.label,
      mode: 'hybrid',
    });
  });

  it('keeps failed-source replay honest about the batch endpoint', async () => {
    const { onPushToast } = renderDetail(3);
    await userEvent.click(screen.getByRole('button', { name: /Replay ingestion/ }));
    expect(onPushToast).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: 'propagating',
        sub: expect.stringContaining('POST /documents/reprocess_failed'),
      }),
    );
  });
});
