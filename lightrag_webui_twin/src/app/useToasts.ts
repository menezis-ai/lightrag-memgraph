import { useCallback, useState } from 'react';
import { TOAST_AUTO_DISMISS_MS, type Toast } from '../types/toast';

export interface RetagUndoPayload {
  targets: readonly string[];
  adds: readonly string[];
  removes: readonly string[];
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

export function asRetagUndoPayload(value: unknown): RetagUndoPayload | null {
  if (!value || typeof value !== 'object') return null;
  const payload = value as Record<string, unknown>;
  if (
    !isStringArray(payload.targets) ||
    !isStringArray(payload.adds) ||
    !isStringArray(payload.removes)
  ) {
    return null;
  }
  return {
    targets: payload.targets,
    adds: payload.adds,
    removes: payload.removes,
  };
}

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const pushToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = `tst_${Date.now()}_${Math.random().toString(16).slice(2, 6)}`;
    setToasts((current) => [...current, { id, ...toast }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((item) => item.id !== id));
    }, TOAST_AUTO_DISMISS_MS);
  }, []);

  const dismissToast = useCallback((toast: Toast) => {
    setToasts((current) => current.filter((item) => item.id !== toast.id));
  }, []);

  return {
    toasts,
    setToasts,
    pushToast,
    dismissToast,
  };
}
