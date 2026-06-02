/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'happy-dom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    // Vitest defaults to *.test.ts AND *.spec.ts. e2e/ uses Playwright's
    // own .spec.ts files which aren't Vitest-compatible (different
    // global API, no @playwright/test injection in Vitest). Exclude.
    exclude: ['**/node_modules/**', '**/dist/**', 'e2e/**'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      exclude: ['node_modules/', 'src/test/', '**/*.d.ts', 'dist/', 'e2e/'],
    },
  },
});
