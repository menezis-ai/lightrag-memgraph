/**
 * Typed procedure-bundle fixtures — contract template for the
 * `/twin/api/procedures` endpoints (`server/procedure_routes.py`).
 *
 * Two seeds:
 *   - `proc-1` — pending review, 2 schematics fully described (blind +
 *     informed + divergence: page 1 coherent, page 2 divergent) so the
 *     review modal exercises both divergence renderings.
 *   - `proc-2` — failed processing with an error + partial results.
 */

import type { ProcedureBundle } from '../types/procedure';

/** Minimal valid 1x1 PNG (base64) — enough for `<img src="data:image/png;base64,...">`. */
export const TINY_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

export const PROCEDURE_BUNDLE_FIXTURES: readonly ProcedureBundle[] = [
  {
    id: 'proc-1',
    file_name: 'oracle-failover-procedure.pdf',
    state: 'pending',
    reason: 'procedure detected: schematic-heavy layout',
    source: 'detected',
    original_path: '/inputs/oracle-failover-procedure.pdf',
    track_id: 'track_proc_1',
    folder: 'default',
    content_hash: 'hash-proc-1',
    full_text: 'Failover procedure for the Oracle demo estate.',
    schematics_total: 2,
    classification: {
      class_id: 'C2',
      class_name: 'C2 Confidentiel',
      reason: null,
    },
    operator_classification: 'C2',
    duplicate_requests: [],
    created_at: '2026-07-18T09:00:00Z',
    updated_at: '2026-07-18T09:05:00Z',
    schematics: [
      {
        page: 1,
        png_base64: TINY_PNG_BASE64,
        blind: {
          title: 'Failover decision tree (blind)',
          description:
            'A decision tree starting from a primary outage alert, branching on data-guard sync state.',
          tasks: [
            {
              id: 'T1',
              title: 'Confirm outage',
              responsible: 'DBA on-call',
              actors: 'DBA, monitoring',
              inputs: 'OEM alert',
              outputs: 'Outage ticket',
              conditions: 'Alert older than 5 minutes',
              links: 'Runbook §2',
            },
          ],
        },
        informed: {
          title: 'Failover decision tree',
          description:
            'Decision tree for switching the demo primary to the secondary site, gated on Data Guard lag.',
          tasks: [
            {
              id: 'T1',
              title: 'Confirm outage',
              responsible: 'DBA on-call',
              actors: 'DBA, monitoring',
              inputs: 'OEM alert',
              outputs: 'Outage ticket',
              conditions: 'Alert older than 5 minutes',
              links: 'Runbook §2',
            },
            {
              id: 'T2',
              title: 'Check Data Guard lag',
              responsible: 'DBA on-call',
              actors: 'DBA',
              inputs: 'v$dataguard_stats',
              outputs: 'Lag report',
              conditions: 'Lag below 30 seconds',
              links: 'Runbook §3',
            },
          ],
        },
        divergence: {
          coherent: true,
          divergences: [],
          summary: 'Blind and informed passes agree on the decision flow.',
        },
        error: null,
      },
      {
        page: 3,
        png_base64: TINY_PNG_BASE64,
        blind: {
          title: 'Switchback sequence (blind)',
          description: 'A four-step sequence returning traffic to the primary.',
          tasks: [],
        },
        informed: {
          title: 'Switchback sequence',
          description:
            'Five-step switchback: the informed pass adds the mandatory ISAB validation gate.',
          tasks: [
            {
              id: 'T1',
              title: 'ISAB validation gate',
              responsible: 'ISAB duty officer',
              actors: 'ISAB, DBA',
              inputs: 'Switchback request',
              outputs: 'Go / no-go',
              conditions: 'Change window approved',
              links: 'Runbook §7',
            },
          ],
        },
        divergence: {
          coherent: false,
          divergences: [
            'Blind pass sees 4 steps; informed pass documents 5 (ISAB gate missing from the diagram).',
            'Blind pass omits the change-window condition.',
          ],
          summary: 'The diagram omits the ISAB validation gate described in the text.',
        },
        error: null,
      },
    ],
  },
  {
    id: 'proc-2',
    file_name: 'network-segmentation-procedure.pdf',
    state: 'failed',
    reason: 'vision pass failed on page 2 (LLM timeout)',
    source: 'forced',
    original_path: '/inputs/network-segmentation-procedure.pdf',
    track_id: 'track_proc_2',
    folder: null,
    content_hash: 'hash-proc-2',
    full_text: 'Segmentation procedure for the edge network.',
    schematics_total: 2,
    classification: null,
    operator_classification: null,
    duplicate_requests: [],
    created_at: '2026-07-17T15:00:00Z',
    updated_at: '2026-07-17T15:20:00Z',
    schematics: [
      {
        page: 1,
        png_base64: TINY_PNG_BASE64,
        blind: {
          title: 'Zone map (blind)',
          description: 'Three network zones with unidirectional flows.',
          tasks: [],
        },
        informed: {
          title: 'Zone map',
          description: 'Edge, DMZ and core zones with the allowed flow matrix.',
          tasks: [],
        },
        divergence: {
          coherent: true,
          divergences: [],
          summary: 'Passes agree.',
        },
        error: null,
      },
      {
        page: 2,
        png_base64: null,
        blind: null,
        informed: null,
        divergence: null,
        error: 'vision LLM timeout after 3 attempts',
      },
    ],
  },
];
