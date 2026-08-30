/**
 * Local WebUI mutation-testing baseline.  This deliberately uses Stryker's
 * built-in command runner to invoke the existing Vitest CLI: it keeps both
 * `bun run test:mutation` and `npm run test:mutation` on the same locally
 * installed toolchain, with no package-manager fetch or browser/backend run.
 *
 * `coverageAnalysis: 'off'` is intentional.  The command runner has no
 * per-test coverage protocol, so the five dedicated test files in
 * `commandRunner.command` are the explicit contract for every mutant.  Do not
 * widen this to the React tree
 * until a Vitest-runner baseline proves stable.
 *
 * @type {import('@stryker-mutator/api').StrykerOptions}
 */
export default {
  // Keep the first campaign to deterministic, IO-free logic that carries
  // operator and wire-contract risk.  React components, hooks and clients are
  // deliberately outside this baseline.
  mutate: [
    'src/lib/permissions.ts',
    'src/lib/docStatus.ts',
    'src/lib/uploadPaths.ts',
    'src/lib/policyRejection.ts',
    'src/utils/documents.ts',
  ],
  // Vitest already transpiles TypeScript at test time.  The checker separately
  // classifies type-invalid mutants before they reach that runner.  Stryker's
  // command runner does not accept `testFiles`; the command below is therefore
  // the single, explicit test-selection contract for this bounded baseline.
  testRunner: 'command',
  commandRunner: {
    command:
      'node ./node_modules/vitest/vitest.mjs run --config vite.config.ts src/lib/permissions.test.ts src/lib/docStatus.test.ts src/lib/uploadPaths.test.ts src/lib/policyRejection.test.ts src/utils/documents.test.ts',
  },
  checkers: ['typescript'],
  tsconfigFile: 'tsconfig.app.json',
  coverageAnalysis: 'off',

  // Do not copy or exercise browser journeys, MSW state, fixture payloads, or
  // build artefacts.  The selected unit tests have no fetch, backend, or
  // Memgraph dependency.
  ignorePatterns: [
    '/e2e/**',
    '/src/mocks/**',
    '/src/fixtures/**',
    '/dist/**',
    '/coverage/**',
    '/reports/**',
    '/playwright-report/**',
    '/test-results/**',
    '**/*.d.ts',
  ],

  reporters: ['clear-text', 'html', 'json'],
  htmlReporter: { fileName: 'reports/mutation/stryker.html' },
  jsonReporter: { fileName: 'reports/mutation/stryker.json' },
  thresholds: { high: 80, low: 60, break: null },

  // Bounded local audit: never a CI gate, never incremental, and leave no
  // stale sandbox after an interrupted run.
  concurrency: 2,
  timeoutFactor: 3,
  timeoutMS: 10_000,
  dryRunTimeoutMinutes: 2,
  incremental: false,
  cleanTempDir: 'always',
};
