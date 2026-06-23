import { getTwinRuntimeConfig } from '../api/client';
import type { Theme } from '../types/topbar';

export const THEME_STORAGE_KEY = 'twin.ui.theme.v1';
export const FOLDER_STORAGE_KEY = 'twin.ui.folder.v1';

export function readUiPreference(key: string): string | null {
  if (globalThis.window === undefined) return null;
  try {
    return globalThis.localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeUiPreference(key: string, value: string): void {
  if (globalThis.window === undefined) return;
  try {
    globalThis.localStorage.setItem(key, value);
  } catch {
    // Browsers can reject localStorage in private/restricted modes. The UI
    // still works for the current session; only refresh persistence is lost.
  }
}

function isTheme(value: string | null): value is Theme {
  return value === 'light' || value === 'dark';
}

export function getInitialTheme(): Theme {
  const stored = readUiPreference(THEME_STORAGE_KEY);
  return isTheme(stored) ? stored : 'light';
}

function getConfiguredDefaultFolderId(): string {
  const cfg = getTwinRuntimeConfig();
  return cfg.defaultFolderId || cfg.folders?.[0]?.id || 'default';
}

export function getInitialFolderId(): string {
  const cfg = getTwinRuntimeConfig();
  const fallback = getConfiguredDefaultFolderId();
  const stored = readUiPreference(FOLDER_STORAGE_KEY);
  if (!stored) return fallback;
  if (cfg.folders && !cfg.folders.some((folder) => folder.id === stored)) {
    return fallback;
  }
  return stored;
}
