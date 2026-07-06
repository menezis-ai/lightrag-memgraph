/**
 * Micro-benchmark: AddSourceModal client-side pre-submit upload gate.
 *
 * Run as a script:
 *   node tests/benchmarks/upload_client_ready_delay.mjs
 *
 * The old UI did not start the real multipart POST when a file was selected.
 * It first advanced a simulated progress bar by 4% every 320ms, making every
 * valid file wait 8 seconds before the Add button could submit it. This
 * benchmark models that gate deterministically without sleeping.
 */

import { performance } from 'node:perf_hooks';

const FILE_BYTES = 24 * 1024;
const ITERATIONS = 10_000;
const OLD_PROGRESS_STEP_PERCENT = 4;
const OLD_TICK_MS = 320;

function oldReadyDelayMs() {
  let progress = 0;
  let ticks = 0;
  while (progress < 100) {
    progress = Math.min(100, progress + OLD_PROGRESS_STEP_PERCENT);
    ticks += 1;
  }
  return ticks * OLD_TICK_MS;
}

function newReadyDelayMs() {
  return 0;
}

function measure(label, fn) {
  const start = performance.now();
  let totalDelayMs = 0;
  for (let i = 0; i < ITERATIONS; i += 1) {
    totalDelayMs += fn();
  }
  const wallMs = performance.now() - start;
  const meanMs = totalDelayMs / ITERATIONS;
  const clientGatedKiBPerS =
    meanMs === 0 ? Infinity : FILE_BYTES / 1024 / (meanMs / 1000);
  const reqPerS = meanMs === 0 ? Infinity : 1000 / meanMs;
  return {
    label,
    iterations: ITERATIONS,
    modeled_mean_ms: meanMs,
    modeled_req_per_s: reqPerS,
    modeled_kib_per_s_for_24kib: clientGatedKiBPerS,
    benchmark_wall_ms: wallMs,
  };
}

const baseline = measure('baseline_fake_progress_gate', oldReadyDelayMs);
const optimized = measure('optimized_immediate_ready', newReadyDelayMs);
const latencyGain =
  ((baseline.modeled_mean_ms - optimized.modeled_mean_ms) /
    baseline.modeled_mean_ms) *
  100;

for (const row of [baseline, optimized]) {
  console.log(row);
}
console.log();
console.log('## SUMMARY');
console.log(
  `client gate: ${baseline.modeled_mean_ms.toFixed(3)}ms -> ` +
    `${optimized.modeled_mean_ms.toFixed(3)}ms (${latencyGain.toFixed(1)}% faster)`,
);
console.log(
  `24KiB apparent throughput: ${baseline.modeled_kib_per_s_for_24kib.toFixed(1)} KiB/s -> network-bound`,
);
console.log(
  `readiness throughput: ${baseline.modeled_req_per_s.toFixed(3)} files/s -> network-bound`,
);
