/** Settings → Portability — canonical export / inspect / approve / apply / validate. */

import { useMemo, useState } from 'react';
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
import type { PortabilityJob } from '../../types/portability';
import type { Toast } from '../../types/toast';

export interface PortabilitySectionProps {
  user?: AuthenticatedUser | null;
  onToast?: (toast: Omit<Toast, 'id'>) => void;
}

const JOB_STORAGE_KEYS = {
  export: 'twin.portability.export-job.v1',
  import: 'twin.portability.import-job.v1',
} as const;

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

function parseFolderMap(raw: string): Record<string, string> {
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
  const [folderMap, setFolderMap] = useState('{}');
  const [folderMapError, setFolderMapError] = useState<string | null>(null);
  const [includeActivity, setIncludeActivity] = useState(false);
  const [includeProcedures, setIncludeProcedures] = useState(false);
  const [allowUnverified, setAllowUnverified] = useState(false);

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

  return (
    <div className="settings-section settings-portability" data-testid="settings-portability">
      <h3>Portability</h3>
      <p className="muted">
        Move a complete KB through the canonical export → dry-run → approval →
        apply → validation workflow. Use this surface only during a maintenance
        window; bundles above 100 MiB go through the CLI.
      </p>

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
          <label className="portability-check">
            <input
              type="checkbox"
              checked={includeActivity}
              onChange={(event) => setIncludeActivity(event.target.checked)}
            />
            Include Activity ledger
          </label>
          <label className="portability-check">
            <input
              type="checkbox"
              checked={includeProcedures}
              onChange={(event) => setIncludeProcedures(event.target.checked)}
            />
            Include procedures and schematics
          </label>
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
          <label className="settings-field">
            <span>Canonical tar.gz bundle (max 100 MiB)</span>
            <input
              type="file"
              accept=".tar.gz,application/gzip"
              onChange={(event) => setBundle(event.target.files?.[0] ?? null)}
              data-testid="portability-import-file"
            />
          </label>
          <label className="settings-field">
            <span>Folder mapping (JSON)</span>
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
          </label>
          {folderMapError && <div className="settings-error" role="alert">{folderMapError}</div>}
          <label className="portability-check portability-check-warning">
            <input
              type="checkbox"
              checked={allowUnverified}
              onChange={(event) => setAllowUnverified(event.target.checked)}
            />
            Allow an unverified bundle (never for production)
          </label>
          <div className="portability-actions">
            <button
              type="button"
              className="ghost-btn primary"
              disabled={!bundle || startImport.isPending || Boolean(importJob && !PORTABILITY_DONE.has(importJob.status))}
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
              {counts.map(([label, value]) => (
                <span className="portability-dl-row" key={label}>
                  <dt>{label}</dt><dd>{value.toLocaleString('en-US')}</dd>
                </span>
              ))}
            </dl>
          </div>
        )}
        {folderMapping.length > 0 && (
          <div>
            <h4>Folder mapping</h4>
            <dl className="set-dl portability-dl">
              {folderMapping.map(([source, target]) => (
                <span className="portability-dl-row" key={source}>
                  <dt>{source}</dt><dd>→ {target}</dd>
                </span>
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
