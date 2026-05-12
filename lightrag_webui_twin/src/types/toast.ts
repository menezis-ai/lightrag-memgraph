/**
 * Toast types — runtime notifications for the toast viewport.
 *
 * Kinds:
 *   propagating  – work in progress (spinner)
 *   done         – completed; may carry an `undo` payload
 *   error        – failure; manually dismissible
 */

export type ToastKind = 'propagating' | 'done' | 'error';

export interface Toast {
  id: string;
  kind: ToastKind;
  title: string;
  tagname?: string;
  titleSuffix?: string;
  sub?: string;
  /** Arbitrary payload passed back to the host on Undo click. */
  undo?: unknown;
}

export const TOAST_MAX_VISIBLE = 3;
