import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});

// happy-dom 20.x on Bun ships without a localStorage / sessionStorage
// implementation; provide a minimal in-memory mock so components that
// persist state (RetrievalTab threads, etc.) can run in tests.
function makeStorage(): Storage {
  const store = new Map<string, string>();
  return {
    get length() { return store.size; },
    clear: () => { store.clear(); },
    getItem: (k) => (store.has(k) ? store.get(k)! : null),
    setItem: (k, v) => { store.set(k, String(v)); },
    removeItem: (k) => { store.delete(k); },
    key: (i) => Array.from(store.keys())[i] ?? null,
  };
}

if (typeof window !== 'undefined' && !window.localStorage) {
  Object.defineProperty(window, 'localStorage', {
    value: makeStorage(),
    writable: true,
  });
}
if (typeof window !== 'undefined' && !window.sessionStorage) {
  Object.defineProperty(window, 'sessionStorage', {
    value: makeStorage(),
    writable: true,
  });
}
