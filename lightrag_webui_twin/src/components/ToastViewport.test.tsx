/**
 * Unit tests for ToastViewport.
 *
 * Behaviors under test:
 *   - renders all toasts when count <= TOAST_MAX_VISIBLE
 *   - clips to last N and shows "+N more" when overflow
 *   - "+N more" click dismisses the hidden ones
 *   - done toast with `undo` shows the Undo button and calls onUndo
 *   - error toast shows the X (dismiss) button and calls onDismiss
 *   - polite ARIA region is populated for non-error, assertive for error
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ToastViewport } from './ToastViewport';
import { TOAST_AUTO_DISMISS_MS } from '../types/toast';
import type { Toast } from '../types/toast';

const propagating: Toast = {
  id: 't1',
  kind: 'propagating',
  title: 'Applying tag',
  tagname: 'rman',
};

const done: Toast = {
  id: 't2',
  kind: 'done',
  title: 'Tag applied',
  tagname: 'rman',
  sub: '418 chunks',
  undo: { tag: 'rman', docId: 'd1' },
};

const error: Toast = {
  id: 't3',
  kind: 'error',
  title: 'Failed',
  sub: 'unsupported MIME',
};

describe('ToastViewport — rendering', () => {
  it('renders all toasts when count fits visible cap', () => {
    render(<ToastViewport toasts={[propagating, done]} onUndo={() => {}} onDismiss={() => {}} />);
    expect(screen.getByText('Applying tag')).toBeInTheDocument();
    expect(screen.getByText('Tag applied')).toBeInTheDocument();
    expect(screen.queryByText(/more/)).toBeNull();
  });

  it('clips and shows "+N more" when count exceeds TOAST_MAX_VISIBLE', () => {
    const many: Toast[] = Array.from({ length: 5 }, (_, i) => ({
      id: `m${i}`,
      kind: 'propagating',
      title: `Task ${i}`,
    }));
    render(<ToastViewport toasts={many} onUndo={() => {}} onDismiss={() => {}} />);
    expect(screen.getByText('+2 more')).toBeInTheDocument();
    // Last 3 visible — scope to the visible viewport, since ARIA live regions
    // also include the last toast text and would cause duplicate matches.
    const viewport = document.querySelector('.toast-viewport') as HTMLElement;
    expect(viewport.textContent).toContain('Task 2');
    expect(viewport.textContent).toContain('Task 3');
    expect(viewport.textContent).toContain('Task 4');
    expect(viewport.textContent).not.toContain('Task 0');
    expect(viewport.textContent).not.toContain('Task 1');
  });
});

describe('ToastViewport — interactions', () => {
  it('+N more click dismisses the hidden toasts', async () => {
    const onDismiss = vi.fn();
    const many: Toast[] = Array.from({ length: 5 }, (_, i) => ({
      id: `m${i}`,
      kind: 'propagating',
      title: `T${i}`,
    }));
    render(<ToastViewport toasts={many} onUndo={() => {}} onDismiss={onDismiss} />);
    await userEvent.click(screen.getByText('+2 more'));
    expect(onDismiss).toHaveBeenCalledTimes(2);
    expect(onDismiss).toHaveBeenCalledWith(many[0]);
    expect(onDismiss).toHaveBeenCalledWith(many[1]);
  });

  it('done toast with undo renders Undo and propagates onUndo', async () => {
    const onUndo = vi.fn();
    render(<ToastViewport toasts={[done]} onUndo={onUndo} onDismiss={() => {}} />);
    await userEvent.click(screen.getByRole('button', { name: 'Undo' }));
    expect(onUndo).toHaveBeenCalledWith(done);
  });

  it('uses the real toast auto-dismiss duration for the undo progress bar', () => {
    render(<ToastViewport toasts={[done]} onUndo={() => {}} onDismiss={() => {}} />);
    const progress = document.querySelector('.undo-progress') as HTMLElement;
    expect(progress).toBeInTheDocument();
    expect(progress.style.getPropertyValue('--toast-undo-progress-ms')).toBe(
      `${TOAST_AUTO_DISMISS_MS}ms`,
    );
  });

  it('error toast renders Dismiss and propagates onDismiss', async () => {
    const onDismiss = vi.fn();
    render(<ToastViewport toasts={[error]} onUndo={() => {}} onDismiss={onDismiss} />);
    await userEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(onDismiss).toHaveBeenCalledWith(error);
  });

  it('done toast without undo does not render Undo', () => {
    const noUndo: Toast = { id: 't4', kind: 'done', title: 'Done' };
    render(<ToastViewport toasts={[noUndo]} onUndo={() => {}} onDismiss={() => {}} />);
    expect(screen.queryByRole('button', { name: 'Undo' })).toBeNull();
  });

  it('propagating toast also renders Dismiss and propagates onDismiss', async () => {
    const onDismiss = vi.fn();
    render(
      <ToastViewport toasts={[propagating]} onUndo={() => {}} onDismiss={onDismiss} />,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(onDismiss).toHaveBeenCalledWith(propagating);
  });

  it('done toast also renders Dismiss (even with Undo) and propagates onDismiss', async () => {
    const onDismiss = vi.fn();
    render(
      <ToastViewport toasts={[done]} onUndo={() => {}} onDismiss={onDismiss} />,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Dismiss' }));
    expect(onDismiss).toHaveBeenCalledWith(done);
  });
});

describe('ToastViewport — ARIA announcements', () => {
  it('uses the polite live region for non-error toasts', () => {
    render(<ToastViewport toasts={[done]} onUndo={() => {}} onDismiss={() => {}} />);
    const polite = document.querySelector('[role="status"]');
    expect(polite?.textContent).toMatch(/Tag applied rman/);
  });

  it('uses the assertive live region for error toasts', () => {
    render(<ToastViewport toasts={[error]} onUndo={() => {}} onDismiss={() => {}} />);
    const assertive = document.querySelector('[role="alert"]');
    expect(assertive?.textContent).toMatch(/Failed\. unsupported MIME/);
  });

  it('announces simultaneous errors in one assertive update', () => {
    render(
      <ToastViewport
        toasts={[
          { id: 'e1', kind: 'error', title: 'Upload failed', sub: 'a.pdf' },
          { id: 'e2', kind: 'error', title: 'Upload failed', sub: 'b.pdf' },
          { id: 'e3', kind: 'error', title: 'Upload failed', sub: 'c.pdf' },
        ]}
        onUndo={() => {}}
        onDismiss={() => {}}
      />,
    );
    const assertive = document.querySelector('[role="alert"]');
    expect(assertive?.textContent).toContain('a.pdf');
    expect(assertive?.textContent).toContain('b.pdf');
    expect(assertive?.textContent).toContain('c.pdf');
  });
});
