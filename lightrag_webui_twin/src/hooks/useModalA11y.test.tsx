/**
 * Unit tests for useModalA11y.
 *
 * Behaviors under test:
 *   - Focuses the first input on open (after the 30ms defer)
 *   - Escape calls onClose
 *   - Tab from the last item cycles to the first
 *   - Shift-Tab from the first item cycles to the last
 *   - Closing restores the previously focused element
 */

import { describe, expect, it, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useRef } from 'react';
import { useModalA11y } from './useModalA11y';

function Modal({
  open,
  onClose,
}: {
  open: boolean;
  onClose?: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useModalA11y({ open, onClose, ref });
  return (
    <div role="dialog" aria-modal="true" ref={ref}>
      <input data-testid="first" placeholder="first" />
      <button data-testid="ok">OK</button>
      <button data-testid="cancel">Cancel</button>
    </div>
  );
}

describe('useModalA11y', () => {
  it('focuses the first input on open after the 30ms defer', () => {
    vi.useFakeTimers();
    render(<Modal open />);
    act(() => {
      vi.advanceTimersByTime(50);
    });
    expect(document.activeElement?.getAttribute('data-testid')).toBe('first');
    vi.useRealTimers();
  });

  it('does not steal focus when the user focuses inside the modal before the defer fires', () => {
    vi.useFakeTimers();
    const { getByTestId } = render(<Modal open />);
    getByTestId('cancel').focus();
    act(() => {
      vi.advanceTimersByTime(50);
    });
    expect(document.activeElement?.getAttribute('data-testid')).toBe('cancel');
    vi.useRealTimers();
  });

  it('Escape inside the modal calls onClose', () => {
    const onClose = vi.fn();
    render(<Modal open onClose={onClose} />);
    const dialog = document.querySelector('[role="dialog"]') as HTMLElement;
    dialog.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(onClose).toHaveBeenCalled();
  });

  it('cycles Tab forward from last to first', async () => {
    vi.useFakeTimers();
    render(<Modal open />);
    act(() => {
      vi.advanceTimersByTime(50);
    });
    vi.useRealTimers();

    const user = userEvent.setup();
    // tab forward twice -> first -> ok -> cancel
    await user.tab();
    await user.tab();
    expect(document.activeElement?.getAttribute('data-testid')).toBe('cancel');
    // one more should cycle back to first
    await user.tab();
    expect(document.activeElement?.getAttribute('data-testid')).toBe('first');
  });

  it('cycles Shift+Tab from first back to last', async () => {
    vi.useFakeTimers();
    render(<Modal open />);
    act(() => {
      vi.advanceTimersByTime(50);
    });
    vi.useRealTimers();

    const user = userEvent.setup();
    // ensure focus on first
    expect(document.activeElement?.getAttribute('data-testid')).toBe('first');
    await user.tab({ shift: true });
    expect(document.activeElement?.getAttribute('data-testid')).toBe('cancel');
  });

  it('restores previously focused element on close', () => {
    vi.useFakeTimers();

    const outsideButton = document.createElement('button');
    outsideButton.setAttribute('data-testid', 'opener');
    document.body.appendChild(outsideButton);
    outsideButton.focus();
    expect(document.activeElement).toBe(outsideButton);

    const { rerender, unmount } = render(<Modal open />);
    act(() => {
      vi.advanceTimersByTime(50);
    });
    expect(document.activeElement?.getAttribute('data-testid')).toBe('first');

    // Close -> on cleanup, previouslyFocused is refocused.
    rerender(<Modal open={false} />);
    unmount();

    expect(document.activeElement).toBe(outsideButton);
    document.body.removeChild(outsideButton);
    vi.useRealTimers();
  });
});
