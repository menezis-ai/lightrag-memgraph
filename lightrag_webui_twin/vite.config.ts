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
    // Vitest defaults to 5s. That is ample locally — the slowest interaction
    // tests run in ~300ms — but the bunker docker pool runs many jobs at once
    // and the same suite has been observed at 353s wall clock where it takes a
    // fraction of that here. A ~16x squeeze turns userEvent-driven tests into
    // spurious timeouts: TagsTab "Reject modal requires a non-empty reason"
    // failed that way on 2026-07-25 while passing in 301ms locally, and it is
    // one of a family (see the userEvent pitfalls in project_webui_fork).
    // Raised to cover contention, not to accommodate slow tests. A genuine
    // hang now costs 20s instead of 5s to surface — an acceptable trade for
    // not fabricating red builds out of runner load.
    testTimeout: 20000,
    hookTimeout: 20000,
    coverage: {
      provider: 'v8',
      // lcov feeds SonarQube (sonar.javascript.lcov.reportPaths); text/html
      // for local inspection.
      reporter: ['text', 'html', 'lcov'],
      exclude: ['node_modules/', 'src/test/', '**/*.d.ts', 'dist/', 'e2e/'],
    },
  },
});
