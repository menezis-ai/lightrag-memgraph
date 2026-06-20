/**
 * Focus trap + Escape + restore-previous-focus for modal dialogs.
 *
 * Ported from Desktop/UI/a11y.jsx. Usage:
 *   const ref = useRef<HTMLDivElement>(null);
 *   useModalA11y({ open, onClose, ref });
 *   return <div role="dialog" aria-modal="true" ref={ref}>...</div>
 *
 * Behaviors:
 *   - On open: focuses the first input/textarea/select inside the modal,
 *     falling back to any focusable, falling back to the container itself.
 *   - Tab/Shift-Tab is trapped inside the modal.
 *   - Escape calls onClose.
 *   - On close: restores focus to whatever was focused before open.
 */

import { useEffect, useRef, type RefObject } from 'react';

const FOCUSABLE_SELECTOR = [
  'input:not([disabled]):not([type=hidden])',
  'textarea:not([disabled])',
  'select:not([disabled])',
  'button:not([disabled])',
  '[href]',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

export interface UseModalA11yOptions {
  open: boolean;
  onClose?: () => void;
  ref: RefObject<HTMLElement | null>;
}

export function useModalA11y({ open, onClose, ref }: UseModalA11yOptions): void {
  // Keep onClose in a ref so an unstable (inline) onClose from the caller does
  // NOT re-run this effect on every render. Re-running mid-typing fired the
  // cleanup's focus-restore + re-autofocus, yanking focus out of a non-first
  // input on each keystroke (BUG-04: "Apply tags to all" lost focus per char).
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!open || !ref.current) return;
    const node = ref.current;
    const previouslyFocused = document.activeElement as HTMLElement | null;

    const focusable = (): HTMLElement[] =>
      Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
        (el) => !el.hasAttribute('aria-hidden') && el.offsetParent !== null,
      );

    const items = focusable();
    const first =
      items.find((el) =>
        ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName),
      ) ??
      items[0] ??
      node;

    let focusTimer: ReturnType<typeof setTimeout> | null = null;
    if (typeof (first as HTMLElement).focus === 'function') {
      // Defer to let layout settle (modal mount animation, etc.)
      focusTimer = setTimeout(() => {
        const active = document.activeElement;
        if (active instanceof HTMLElement && node.contains(active)) return;
        (first as HTMLElement).focus();
      }, 30);
    }

    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCloseRef.current?.();
        return;
      }
      if (e.key === 'Tab') {
        const current = focusable();
        if (current.length === 0) {
          e.preventDefault();
          return;
        }
        const idx = current.indexOf(document.activeElement as HTMLElement);
        if (e.shiftKey) {
          if (idx <= 0) {
            e.preventDefault();
            current[current.length - 1].focus();
          }
        } else {
          if (idx === current.length - 1 || idx === -1) {
            e.preventDefault();
            current[0].focus();
          }
        }
      }
    };

    node.addEventListener('keydown', onKey);
    return () => {
      if (focusTimer !== null) clearTimeout(focusTimer);
      node.removeEventListener('keydown', onKey);
      const fallback = document.querySelector<HTMLElement>(
        '[data-focus-fallback="app-main"]',
      );
      const target =
        previouslyFocused?.isConnected === true ? previouslyFocused : fallback;
      if (target && typeof target.focus === 'function') {
        try {
          target.focus({ preventScroll: true });
        } catch {
          /* element no longer in DOM */
        }
      }
    };
  }, [open, ref]);
}
