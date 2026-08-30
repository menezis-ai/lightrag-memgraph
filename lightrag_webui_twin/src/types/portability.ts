/** Wire contract for Settings → Portability admin jobs. */

export type PortabilityJobStatus =
  | 'queued'
  | 'uploading'
  | 'running'
  | 'dry-running'
  | 'awaiting-approval'
  | 'approved'
  | 'applying'
  | 'applied'
  | 'validating'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'validated'
  | 'validation-failed';

export interface PortabilityBlockingFinding {
  code: string;
  message: string;
  [key: string]: unknown;
}

export interface PortabilityCompatibilityCheck {
  dimension: string;
  ok: boolean;
  reason?: string;
  source?: unknown;
  target?: unknown;
}

export interface PortabilityDryRunReport {
  report_hash: string;
  blocking: readonly PortabilityBlockingFinding[];
  compat: readonly PortabilityCompatibilityCheck[];
  classification: {
    source_max?: string | null;
    target_ceiling?: string | null;
    unknown_present?: boolean;
  };
  stats?: {
    counts?: Record<string, number>;
    stores?: Record<string, number>;
  };
  folders?: {
    effective_mapping?: Record<string, string>;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

export interface PortabilityJob {
  id: string;
  kind: 'export' | 'import';
  workspace: string;
  status: PortabilityJobStatus;
  created_at: string;
  updated_at: string;
  actor: string;
  approved_report_hash?: string | null;
  approved_by?: string | null;
  applied_by?: string | null;
  validated_by?: string | null;
  cancelled_by?: string | null;
  options: Record<string, unknown>;
  result: Record<string, unknown> | null;
  report: PortabilityDryRunReport | null;
  validation: Record<string, unknown> | null;
  error: string | null;
  download_available: boolean;
}

export interface PortabilityExportInput {
  workspace?: string;
  include_activity?: boolean;
  include_procedures?: boolean;
  force?: boolean;
}

export interface PortabilityImportInput {
  bundle: File;
  workspace?: string;
  folderMap?: Record<string, string>;
  allowUnverified?: boolean;
}
