import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FOLDER_STORAGE_KEY, getInitialFolderId } from './uiPreferences';

interface RuntimeConfig {
  defaultFolderId?: string;
  folders?: ReadonlyArray<{ id: string; label: string }>;
}

const runtimeConfig = {
  current: {
    defaultFolderId: 'default',
    folders: [{ id: 'default', label: 'Default KB' }] as RuntimeConfig['folders'],
  } as RuntimeConfig,
};

vi.mock('../api/client', () => ({
  getTwinRuntimeConfig: () => runtimeConfig.current,
}));

beforeEach(() => {
  globalThis.localStorage.clear();
  runtimeConfig.current = {
    defaultFolderId: 'default',
    folders: [{ id: 'default', label: 'Default KB' }],
  };
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('uiPreferences — getInitialFolderId', () => {
  it('falls back to configured default when no stored folder exists', () => {
    runtimeConfig.current = {
      defaultFolderId: 'finance',
      folders: [{ id: 'finance', label: 'Finance' }],
    };
    expect(getInitialFolderId()).toBe('finance');
  });

  it('keeps a stored folder when it is still configured for this runtime', () => {
    globalThis.localStorage.setItem(FOLDER_STORAGE_KEY, 'finance');
    runtimeConfig.current = {
      defaultFolderId: 'default',
      folders: [
        { id: 'default', label: 'Default KB' },
        { id: 'finance', label: 'Finance' },
      ],
    };
    expect(getInitialFolderId()).toBe('finance');
  });

  it('keeps a stored folder when runtimeConfig.folders omits it and validation must wait for runtime/live data', () => {
    globalThis.localStorage.setItem(FOLDER_STORAGE_KEY, 'ghost');
    runtimeConfig.current = {
      defaultFolderId: 'default',
      folders: [{ id: 'default', label: 'Default KB' }],
    };
    expect(getInitialFolderId()).toBe('ghost');
  });

  it('keeps the stored folder when runtime folders config is missing', () => {
    globalThis.localStorage.setItem(FOLDER_STORAGE_KEY, 'tests');
    runtimeConfig.current = {
      defaultFolderId: 'default',
      folders: [],
    };
    expect(getInitialFolderId()).toBe('default');

    runtimeConfig.current = { folders: undefined };
    expect(getInitialFolderId()).toBe('tests');
  });

  it('keeps the stored folder even when no default is configured', () => {
    globalThis.localStorage.setItem(FOLDER_STORAGE_KEY, 'ghost');
    runtimeConfig.current = {
      folders: [{ id: 'finance', label: 'Finance' }, { id: 'ops', label: 'Ops' }],
    };
    expect(getInitialFolderId()).toBe('ghost');
  });
});
