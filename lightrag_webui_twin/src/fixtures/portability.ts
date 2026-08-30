import type {
  PortabilityDryRunReport,
  PortabilityJob,
} from '../types/portability';

export const PORTABILITY_REPORT_HASH = 'a'.repeat(64);

export const PORTABILITY_DRY_RUN_FIXTURE: PortabilityDryRunReport = {
  report_hash: PORTABILITY_REPORT_HASH,
  blocking: [],
  compat: [
    { dimension: 'format', ok: true, reason: 'twin-kb-bundle/1 supported' },
    {
      dimension: 'embedding',
      ok: true,
      reason: 'all three probe cosines must be >= 0.999',
    },
    { dimension: 'classification', ok: true, reason: 'C2 is within C2' },
  ],
  classification: {
    source_max: 'C2',
    target_ceiling: 'C2',
    unknown_present: false,
  },
  stats: {
    counts: {
      documents: 24,
      chunks: 312,
      entities: 86,
      relations: 141,
      folders: 2,
      tags: 17,
    },
  },
  folders: {
    effective_mapping: { staging: 'production', shared: 'shared' },
  },
};

export function makePortabilityJob(
  kind: 'export' | 'import',
  id: string,
): PortabilityJob {
  const now = '2026-08-26T13:30:00Z';
  return {
    id,
    kind,
    workspace: 'base',
    status: kind === 'export' ? 'running' : 'dry-running',
    created_at: now,
    updated_at: now,
    actor: 'operator.demo',
    options: {},
    result: null,
    report: null,
    validation: null,
    error: null,
    download_available: false,
  };
}
