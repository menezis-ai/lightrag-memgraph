/** Settings → Portability — canonical export / inspect / approve / apply / validate. */

import { useMemo, useRef, useState } from 'react';
import { Icon } from '../Icon';
import {
  useApplyPortabilityImport,
  useApprovePortabilityImport,
  useCancelPortabilityImport,
  useDownloadPortabilityExport,
  usePortabilityExportJob,
  usePortabilityImportJob,
  useStartPortabilityExport,
  useStartPortabilityImport,
  useValidatePortabilityImport,
} from '../../api/queries';
import { canManageFolders } from '../../lib/permissions';
import { userErrorMessage } from '../../lib/errorMessages';
import type { AuthenticatedUser } from '../../types/auth';
import type { PortabilityJob, PortabilityJobStatus } from '../../types/portability';
import type { Toast } from '../../types/toast';

export interface PortabilitySectionProps {
  user?: AuthenticatedUser | null;
  onToast?: (toast: Omit<Toast, 'id'>) => void;
}

const JOB_STORAGE_KEYS = {
  export: 'twin.portability.export-job.v1',
  import: 'twin.portability.import-job.v1',
} as const;

const MAX_BUNDLE_BYTES = 100 * 1024 * 1024;

function readPersistedJobId(kind: keyof typeof JOB_STORAGE_KEYS): string | null {
  if (globalThis.window === undefined) return null;
  try {
    const value = globalThis.sessionStorage.getItem(JOB_STORAGE_KEYS[kind]);
    const prefix = kind === 'export' ? 'exp_' : 'imp_';
    if (value?.startsWith(prefix) && /^[A-Za-z0-9_-]{5,132}$/.test(value)) {
      return value;
    }
    if (value) globalThis.sessionStorage.removeItem(JOB_STORAGE_KEYS[kind]);
  } catch {
    // Session storage may be unavailable in privacy-restricted browsers.
  }
  return null;
}

function persistJobId(kind: keyof typeof JOB_STORAGE_KEYS, id: string): void {
  try {
    globalThis.sessionStorage.setItem(JOB_STORAGE_KEYS[kind], id);
  } catch {
    // Polling still works for the current mount when storage is unavailable.
  }
}

function statusLabel(status: PortabilityJob['status']): string {
  return status.replaceAll('-', ' ');
}

function activeStatus(status: PortabilityJob['status']): boolean {
  return [
    'queued',
    'uploading',
    'running',
    'dry-running',
    'applying',
    'validating',
  ].includes(status);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const mib = bytes / (1024 * 1024);
  if (mib >= 1) return `${mib.toFixed(mib >= 10 ? 0 : 1)} MiB`;
  return `${(bytes / 1024).toFixed(0)} KiB`;
}

/**
 * A blank field means "no folder is renamed" — the canonical importer treats a
 * missing mapping and an empty object identically, so the placeholder can stay
 * visible instead of the operator having to read past a literal `{}`.
 */
function parseFolderMap(raw: string): Record<string, string> {
  if (!raw.trim()) return {};
  const parsed: unknown = JSON.parse(raw);
  if (
    parsed === null ||
    Array.isArray(parsed) ||
    typeof parsed !== 'object' ||
    !Object.entries(parsed).every(
      ([source, target]) => source.trim() && typeof target === 'string' && target.trim(),
    )
  ) {
    throw new Error('Folder mapping must be a JSON object of source and target ids.');
  }
  return parsed as Record<string, string>;
}

export function PortabilitySection({
  user = null,
  onToast,
}: PortabilitySectionProps = {}) {
  const canManage = canManageFolders(user);
  const [exportId, setExportId] = useState<string | null>(() =>
    readPersistedJobId('export'),
  );
  const [importId, setImportId] = useState<string | null>(() =>
    readPersistedJobId('import'),
  );
  const [bundle, setBundle] = useState<File | null>(null);
  const [folderMap, setFolderMap] = useState('');
  const [folderMapError, setFolderMapError] = useState<string | null>(null);
  const [includeActivity, setIncludeActivity] = useState(false);
  const [includeProcedures, setIncludeProcedures] = useState(false);
  const [allowUnverified, setAllowUnverified] = useState(false);
  const [drag, setDrag] = useState(false);
  const bundleInputRef = useRef<HTMLInputElement>(null);

  const startExport = useStartPortabilityExport();
  const exportQuery = usePortabilityExportJob(exportId);
  const downloadExport = useDownloadPortabilityExport();
  const startImport = useStartPortabilityImport();
  const importQuery = usePortabilityImportJob(importId);
  const approveImport = useApprovePortabilityImport();
  const applyImport = useApplyPortabilityImport();
  const validateImport = useValidatePortabilityImport();
  const cancelImport = useCancelPortabilityImport();

  const exportJob = exportQuery.data ?? startExport.data ?? null;
  const importJob = importQuery.data ?? startImport.data ?? null;
  const report = importJob?.report ?? null;
  const mutationError = useMemo(
    () =>
      startExport.error ??
      downloadExport.error ??
      startImport.error ??
      approveImport.error ??
      applyImport.error ??
      validateImport.error ??
      cancelImport.error ??
      exportQuery.error ??
      importQuery.error,
    [
      applyImport.error,
      approveImport.error,
      cancelImport.error,
      downloadExport.error,
      exportQuery.error,
      importQuery.error,
      startExport.error,
      startImport.error,
      validateImport.error,
    ],
  );

  const notifyFailure = (title: string, error: unknown) =>
    onToast?.({ kind: 'error', title, sub: userErrorMessage(error) });

  const pickBundle = (file: File | null | undefined): void => {
    setBundle(file ?? null);
    setFolderMapError(null);
  };

  const beginImport = (): void => {
    if (!bundle) return;
    let mapping: Record<string, string>;
    try {
      mapping = parseFolderMap(folderMap);
      setFolderMapError(null);
    } catch (error) {
      setFolderMapError(
        error instanceof Error ? error.message : userErrorMessage(error),
      );
      return;
    }
    startImport.mutate(
      { bundle, folderMap: mapping, allowUnverified },
      {
        onSuccess: (job) => {
          persistJobId('import', job.id);
          setImportId(job.id);
        },
        onError: (error) => notifyFailure('Could not start import', error),
      },
    );
  };

  const download = (): void => {
    if (!exportJob) return;
    downloadExport.mutate(exportJob.id, {
      onSuccess: (blob) => {
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `twin-kb-${exportJob.workspace}-${exportJob.id}.tar.gz`;
        document.body.append(anchor);
        anchor.click();
        setTimeout(() => {
          URL.revokeObjectURL(url);
          anchor.remove();
        }, 0);
      },
      onError: (error) => notifyFailure('Could not download export', error),
    });
  };

  if (!canManage) {
    return (
      <div className="settings-section" data-testid="settings-portability">
        <h3>Portability</h3>
        <span className="env-badge" data-testid="portability-admin-required">
          <Icon name="lock" size={10} /> Admin scope required
        </span>
      </div>
    );
  }

  const oversized = bundle !== null && bundle.size > MAX_BUNDLE_BYTES;

  return (
    <div className="settings-section settings-portability" data-testid="settings-portability">
      <h3>Portability</h3>
      <p className="muted">
        Move a complete KB through the canonical export → dry-run → approval →
        apply → validation workflow. Use this surface only during a maintenance
        window; bundles above 100 MiB go through the CLI.
      </p>

      <WorkflowRail exportJob={exportJob} importJob={importJob} />

      {mutationError && (
        <div className="error-banner" role="alert" data-testid="portability-error">
          {userErrorMessage(mutationError)}
        </div>
      )}

      <div className="portability-grid">
        <section className="set-card" aria-labelledby="portability-export-title">
          <div className="set-card-h" id="portability-export-title">
            <Icon name="cloud-upload" size={15} /> Export this KB
          </div>
          <p className="muted">
            Produces an integrity-checked tar.gz archive. Activity and procedure
            files remain excluded unless explicitly selected.
          </p>

          <OptionSwitch
            label="Activity ledger"
            hint="Include the append-only audit trail in the archive."
            checked={includeActivity}
            onChange={setIncludeActivity}
            testId="portability-include-activity"
          />
          <OptionSwitch
            label="Procedures and schematics"
            hint="Include parked procedure bundles and their extracted assets."
            checked={includeProcedures}
            onChange={setIncludeProcedures}
            testId="portability-include-procedures"
          />

          <div className="portability-actions">
            <button
              type="button"
              className="ghost-btn primary"
              disabled={startExport.isPending || Boolean(exportJob && activeStatus(exportJob.status))}
              onClick={() =>
                startExport.mutate(
                  {
                    include_activity: includeActivity,
                    include_procedures: includeProcedures,
                  },
                  {
                    onSuccess: (job) => {
                      persistJobId('export', job.id);
                      setExportId(job.id);
                    },
                    onError: (error) => notifyFailure('Could not start export', error),
                  },
                )
              }
              data-testid="portability-export-start"
            >
              <Icon name="cloud-upload" size={12} />
              {startExport.isPending ? 'Starting…' : 'Create export'}
            </button>
            {exportJob?.download_available && (
              <button
                type="button"
                className="ghost-btn"
                onClick={download}
                disabled={downloadExport.isPending}
                data-testid="portability-export-download"
              >
                <Icon name="file-text" size={12} /> Download archive
              </button>
            )}
          </div>
          {exportJob && <JobStatus job={exportJob} />}
        </section>

        <section className="set-card" aria-labelledby="portability-import-title">
          <div className="set-card-h" id="portability-import-title">
            <Icon name="folder" size={15} /> Import a KB bundle
          </div>
          <p className="muted">
            Uploading only produces a dry-run report. Nothing is written to this
            KB until you approve that report and apply it.
          </p>

          <input
            ref={bundleInputRef}
            type="file"
            accept=".tar.gz,application/gzip"
            style={{ display: 'none' }}
            onChange={(event) => {
              pickBundle(event.target.files?.[0]);
              // Reset so re-picking the same file re-fires change.
              event.target.value = '';
            }}
            data-testid="portability-import-file"
          />
          <button
            type="button"
            className={drag ? 'dropzone portability-dropzone drag' : 'dropzone portability-dropzone'}
            aria-label="Choose a canonical KB bundle, or drop one here"
            onClick={() => bundleInputRef.current?.click()}
            onDragOver={(event) => {
              event.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDrag(false);
              pickBundle(event.dataTransfer.files?.[0]);
            }}
            data-testid="portability-import-dropzone"
          >
            <Icon name="cloud-upload" size={24} color="var(--color-text-secondary)" />
            <div className="title">
              {bundle ? bundle.name : 'Drop a bundle here or click to browse'}
            </div>
            <div className="sub">
              {bundle ? (
                <span data-testid="portability-import-filesize">
                  {formatBytes(bundle.size)}
                </span>
              ) : (
                <span>Canonical twin-kb-bundle tar.gz · max 100 MiB</span>
              )}
            </div>
          </button>
          {oversized && (
            <div className="settings-error" role="alert" data-testid="portability-import-oversized">
              This bundle exceeds the 100 MiB browser limit — import it with{' '}
              <code>python -m twindb_lightrag_memgraph.portability</code> instead.
            </div>
          )}

          <label className="settings-field portability-folder-field">
            <span>
              Folder mapping <em>(optional JSON)</em>
            </span>
            <textarea
              value={folderMap}
              onChange={(event) => {
                setFolderMap(event.target.value);
                setFolderMapError(null);
              }}
              rows={3}
              spellCheck={false}
              placeholder='{"staging":"production"}'
              data-testid="portability-folder-map"
            />
            <span className="portability-field-hint">
              Leave empty to keep every folder id from the bundle.
            </span>
          </label>
          {folderMapError && <div className="settings-error" role="alert">{folderMapError}</div>}

          <div className="portability-danger" data-testid="portability-danger-zone">
            <Icon name="alert-triangle" size={13} />
            <div className="portability-danger-copy">
              <span className="portability-danger-title">
                Allow an unverified bundle
              </span>
              {/* There is no signature check to skip. compat.py:321 reads
                  `manifest.consistency.status == "verified" or allow_unverified`,
                  and the per-file sha256 + size verification in bundle.py runs
                  unconditionally either way. */}
              <span className="muted" id="portability-unverified-help">
                Allows an export whose pre/post consistency check was
                unverified. Integrity checks still apply. Never for production.
              </span>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={allowUnverified}
              aria-label="Allow an unverified bundle"
              // Without this, the one destructive control on the surface
              // announces its label and nothing else — "Never for production"
              // would never be heard by someone tabbing onto it.
              aria-describedby="portability-unverified-help"
              className={`switch settings-switch portability-danger-switch${allowUnverified ? ' on' : ''}`}
              onClick={() => setAllowUnverified((allowed) => !allowed)}
              data-testid="portability-allow-unverified"
            />
          </div>

          <div className="portability-actions">
            <button
              type="button"
              className="ghost-btn primary"
              disabled={
                !bundle ||
                oversized ||
                startImport.isPending ||
                Boolean(importJob && !PORTABILITY_DONE.has(importJob.status))
              }
              onClick={beginImport}
              data-testid="portability-import-start"
            >
              <Icon name="folder" size={12} />
              {startImport.isPending ? 'Uploading…' : 'Upload and dry-run'}
            </button>
          </div>
          {importJob && <JobStatus job={importJob} />}
        </section>
      </div>

      {report && importJob && (
        <DryRunReport
          job={importJob}
          onApprove={() =>
            approveImport.mutate(
              { jobId: importJob.id, reportHash: report.report_hash },
              { onError: (error) => notifyFailure('Could not approve import', error) },
            )
          }
          onApply={() =>
            applyImport.mutate(importJob.id, {
              onError: (error) => notifyFailure('Could not apply import', error),
            })
          }
          onValidate={() =>
            validateImport.mutate(importJob.id, {
              onError: (error) => notifyFailure('Could not validate import', error),
            })
          }
          onCancel={() => cancelImport.mutate(importJob.id)}
          pending={
            approveImport.isPending ||
            applyImport.isPending ||
            validateImport.isPending ||
            cancelImport.isPending
          }
        />
      )}
    </div>
  );
}

const PORTABILITY_DONE = new Set([
  'failed',
  'cancelled',
  'validated',
  'validation-failed',
]);

interface OptionSwitchProps {
  label: string;
  hint: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  testId: string;
}

function OptionSwitch({
  label,
  hint,
  checked,
  onChange,
  testId,
}: Readonly<OptionSwitchProps>) {
  // Tabbing straight onto the control otherwise announces the label alone —
  // the visible explanation beside it is never read. Same id +
  // aria-describedby pairing as VisionSection's procedure toggle.
  const hintId = `${testId}-help`;
  return (
    <div className="settings-toggle-row portability-toggle-row">
      <div className="settings-toggle-copy">
        <span className="field-label">{label}</span>
        <span className="muted" id={hintId}>
          {hint}
        </span>
      </div>
      <div className="settings-toggle-control">
        <button
          type="button"
          role="switch"
          aria-checked={checked}
          aria-label={label}
          aria-describedby={hintId}
          className={`switch settings-switch${checked ? ' on' : ''}`}
          onClick={() => onChange(!checked)}
          data-testid={testId}
        />
      </div>
    </div>
  );
}

/*
 * The five workflow stages, in the order the section's own prose announces
 * them. Stage 0 (Export) runs on the SOURCE instance and is driven by the
 * export job; stages 1-4 run here, on the target.
 *
 * What a stage REACHED comes from the job's durable fields, never from the
 * current status alone — a cancelled or failed run keeps the history it
 * earned. Deriving from status made `cancelled` and `failed` collapse the
 * whole rail to "not started" while the dry-run report was still on screen,
 * saying two contradictory things at once.
 *
 * The contract, from server/portability_jobs.py:
 *   report                 written when the dry-run completes  -> COMPLETED
 *   approved_report_hash   written on the approve transition   -> COMPLETED
 *   applied_by             written ENTERING `applying` (:540)  -> ATTEMPTED
 *   validated_by           written ENTERING `validating` (:558)-> ATTEMPTED
 * The two `*_by` fields prove an attempt, not a success, so completion of
 * apply/validate is read from the status the attempt reached.
 *
 * cancel() refuses {applying, applied, validating} (:573), so a cancelled job
 * never carries `applied_by` — no ambiguity between the two terminal states.
 */
const WORKFLOW_STAGES = [
  'Export',
  'Dry-run',
  'Approve',
  'Apply',
  'Validate',
] as const;

/** Statuses only reachable once apply itself finished writing. */
const APPLY_COMPLETED: ReadonlySet<PortabilityJobStatus> = new Set([
  'applied',
  'validating',
  'validated',
  'validation-failed',
  'completed',
]);

/** The dry-run is still producing its report. */
const DRY_RUN_ACTIVE: ReadonlySet<PortabilityJobStatus> = new Set([
  'queued',
  'uploading',
  'running',
  'dry-running',
]);

type StageState = 'pending' | 'active' | 'done' | 'failed';

/** Announced to assistive tech; the internal token itself would not be. */
const STAGE_STATE_LABEL: Record<StageState, string> = {
  pending: 'not started',
  active: 'in progress',
  done: 'completed',
  failed: 'failed',
};

function exportStageState(job: PortabilityJob | null): StageState {
  if (!job) return 'pending';
  if (job.status === 'failed') return 'failed';
  if (job.download_available || job.status === 'completed') return 'done';
  return activeStatus(job.status) ? 'active' : 'pending';
}

function dryRunStageState(job: PortabilityJob, failed: boolean): StageState {
  if (job.report) return 'done';
  if (failed) return 'failed';
  return DRY_RUN_ACTIVE.has(job.status) ? 'active' : 'pending';
}

function approveStageState(job: PortabilityJob): StageState {
  // A blocking report never leaves `awaiting-approval`, so approval has no
  // failure state of its own — the blockers panel owns that message.
  if (job.approved_report_hash) return 'done';
  return job.status === 'awaiting-approval' ? 'active' : 'pending';
}

// "active" means this is the stage the run is sitting on — whether the server
// is working (`applying`) or the operator still has to press the button
// (`approved`), exactly as Approve is active while `awaiting-approval`.
function applyStageState(job: PortabilityJob, failed: boolean): StageState {
  if (APPLY_COMPLETED.has(job.status)) return 'done';
  if (job.status === 'approved' || job.status === 'applying') return 'active';
  if (job.applied_by && failed) return 'failed';
  return 'pending';
}

function validateStageState(job: PortabilityJob, failed: boolean): StageState {
  if (job.status === 'validated' || job.status === 'completed') return 'done';
  if (job.status === 'validation-failed') return 'failed';
  if (job.status === 'applied' || job.status === 'validating') return 'active';
  if (job.validated_by && failed) return 'failed';
  return 'pending';
}

function importStageStates(job: PortabilityJob | null): StageState[] {
  if (!job) return ['pending', 'pending', 'pending', 'pending'];
  const failed = job.status === 'failed';
  return [
    dryRunStageState(job, failed),
    approveStageState(job),
    applyStageState(job, failed),
    validateStageState(job, failed),
  ];
}

function WorkflowRail({
  exportJob,
  importJob,
}: Readonly<{ exportJob: PortabilityJob | null; importJob: PortabilityJob | null }>) {
  const states: StageState[] = [
    exportStageState(exportJob),
    ...importStageStates(importJob),
  ];
  const haltedStatus =
    importJob && (importJob.status === 'failed' || importJob.status === 'cancelled')
      ? importJob.status
      : null;

  return (
    <ol
      className="portability-rail"
      aria-label="Portability workflow progress"
      data-testid="portability-rail"
      data-halted={haltedStatus ?? undefined}
    >
      {WORKFLOW_STAGES.map((stage, index) => {
        const state = states[index];
        return (
          <li
            key={stage}
            className={`portability-rail-step is-${state}`}
            data-state={state}
            data-testid={`portability-rail-${stage.toLowerCase()}`}
            aria-current={state === 'active' ? 'step' : undefined}
          >
            <span className="portability-rail-bullet" aria-hidden="true">
              {state === 'done' && <Icon name="check" size={10} />}
              {state === 'failed' && <Icon name="x" size={10} />}
            </span>
            <span className="portability-rail-label">{stage}</span>
            {/* Colour, the bullet glyph and data-state are all invisible to a
                screen reader, and aria-current only marks the active step —
                pending, done and failed were indistinguishable. axe cannot
                catch a state that is simply never announced. */}
            <span className="sr-only"> — {STAGE_STATE_LABEL[state]}</span>
          </li>
        );
      })}
      {haltedStatus && (
        <li className="portability-rail-halted" data-testid="portability-rail-halted">
          run {statusLabel(haltedStatus)}
        </li>
      )}
    </ol>
  );
}

function JobStatus({ job }: Readonly<{ job: PortabilityJob }>) {
  return (
    <div className="portability-status" data-testid={`portability-${job.kind}-status`}>
      {activeStatus(job.status) && <Icon name="loader-2" size={12} className="spin" />}
      <span className={`env-badge${job.status === 'validated' || job.status === 'completed' ? ' active' : ''}`}>
        {statusLabel(job.status)}
      </span>
      <code>{job.id}</code>
      {job.error && <span className="settings-error">{job.error}</span>}
    </div>
  );
}

interface DryRunReportProps {
  job: PortabilityJob;
  onApprove: () => void;
  onApply: () => void;
  onValidate: () => void;
  onCancel: () => void;
  pending: boolean;
}

function DryRunReport({
  job,
  onApprove,
  onApply,
  onValidate,
  onCancel,
  pending,
}: Readonly<DryRunReportProps>) {
  const report = job.report;
  if (!report) return null;
  const blockers = report.blocking;
  const counts = Object.entries(report.stats?.counts ?? {});
  const folderMapping = Object.entries(report.folders?.effective_mapping ?? {});

  return (
    <section className="set-card portability-report" data-testid="portability-report">
      <div className="set-card-h">
        Dry-run report
        <span className={`env-badge${blockers.length === 0 ? ' active' : ''}`}>
          {blockers.length === 0 ? 'ready for approval' : `${blockers.length} blocking`}
        </span>
      </div>

      <div className="portability-report-grid">
        <div>
          <h4>Compatibility</h4>
          <ul className="portability-check-list">
            {report.compat.map((check) => (
              <li key={check.dimension}>
                <Icon name={check.ok ? 'circle-check' : 'alert-triangle'} size={13} />
                <span><strong>{check.dimension.replaceAll('_', ' ')}</strong>{check.reason ? ` — ${check.reason}` : ''}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4>Classification</h4>
          <dl className="set-dl portability-dl">
            <dt>Bundle</dt><dd>{report.classification.source_max ?? 'none detected'}</dd>
            <dt>Target ceiling</dt><dd>{report.classification.target_ceiling ?? 'not configured'}</dd>
            <dt>Unknown labels</dt><dd>{report.classification.unknown_present ? 'yes — blocked' : 'no'}</dd>
          </dl>
        </div>
        {counts.length > 0 && (
          <div>
            <h4>Content</h4>
            <dl className="set-dl portability-dl">
              {/* <div>, not <span>: a <dl> may only directly contain dt/dd/div
                  (axe definition-list + dlitem, both serious). Invisible either
                  way — .portability-dl-row is display: contents. */}
              {counts.map(([label, value]) => (
                <div className="portability-dl-row" key={label}>
                  <dt>{label}</dt><dd>{value.toLocaleString('en-US')}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
        {folderMapping.length > 0 && (
          <div>
            <h4>Folder mapping</h4>
            <dl className="set-dl portability-dl">
              {folderMapping.map(([source, target]) => (
                <div className="portability-dl-row" key={source}>
                  <dt>{source}</dt><dd>→ {target}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </div>

      {blockers.length > 0 && (
        <div className="portability-blockers" role="alert">
          {blockers.map((blocker) => (
            <div key={blocker.code}>
              <Icon name="alert-triangle" size={13} />
              <span><strong>{blocker.code}</strong> — {blocker.message}</span>
            </div>
          ))}
        </div>
      )}

      {job.validation && (
        <div
          className={job.validation.ok === true ? 'portability-validation ok' : 'portability-validation failed'}
          data-testid="portability-validation"
        >
          <Icon name={job.validation.ok === true ? 'circle-check' : 'alert-triangle'} size={14} />
          {job.validation.ok === true
            ? 'Validation passed — normalized target state matches the bundle.'
            : `Validation failed — ${String((job.validation.problems as unknown[] | undefined)?.length ?? 0)} problem(s).`}
        </div>
      )}

      <div className="portability-actions">
        {job.status === 'awaiting-approval' && (
          <button
            type="button"
            className="ghost-btn primary"
            disabled={pending || blockers.length > 0}
            onClick={onApprove}
            data-testid="portability-approve"
          >
            Approve this report
          </button>
        )}
        {job.status === 'approved' && (
          <button
            type="button"
            className="ghost-btn primary"
            disabled={pending || blockers.length > 0}
            onClick={onApply}
            data-testid="portability-apply"
          >
            Apply import
          </button>
        )}
        {job.status === 'applied' && (
          <button
            type="button"
            className="ghost-btn primary"
            disabled={pending}
            onClick={onValidate}
            data-testid="portability-validate"
          >
            Validate target
          </button>
        )}
        {(job.status === 'awaiting-approval' || job.status === 'approved') && (
          <button type="button" className="ghost-btn danger" disabled={pending} onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </section>
  );
}
