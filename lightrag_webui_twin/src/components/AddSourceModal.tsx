/**
 * AddSourceModal — drop files, paste URLs, optionally apply tags.
 *
 * Ported from Desktop/UI/modals.jsx. Behavior delta vs the proto:
 *   - tag catalog + formatCategories injected via props (no window.* reads)
 *   - submit emits an AddSourceAction; the host owns the toast lifecycle
 *   - useModalA11y attached via the hook port from S1
 *
 * The upload progress animation (setInterval bumping `progress` by 4% every
 * 320ms) is preserved — for tests we keep timers real-time and rely on
 * Vitest fake timers when we want to observe transitions.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon } from './Icon';
import { TagChip } from './TagChip';
import { useModalA11y } from '../hooks/useModalA11y';
import type { FormatCategory } from '../types/format';
import type { TagEntry } from '../types/tag';
import { tagMatchesQuery, tagSuggestionComparator } from '../utils/tags';

const MAX_FILE_MB = 50;
const SUPPORTED_EXTENSIONS = new Set([
  'csv',
  'doc',
  'docx',
  'htm',
  'html',
  'json',
  'log',
  'md',
  'eml',
  'msg',
  'pdf',
  'ppt',
  'pptx',
  'rtf',
  'text',
  'txt',
  'xls',
  'xlsx',
  'xml',
  'yaml',
  'yml',
]);
const SUPPORTED_MIME_PREFIXES = ['text/'];
const SUPPORTED_MIME_TYPES = new Set([
  'application/json',
  'application/msword',
  'application/pdf',
  'application/rtf',
  'application/vnd.ms-outlook',
  'application/vnd.ms-excel',
  'application/vnd.ms-powerpoint',
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'application/xml',
  'message/rfc822',
]);

export type FileUploadState = 'uploading' | 'uploaded' | 'error';

/**
 * Operator-set MIP sensitivity classification (BNP C1..C4). Empty selection
 * means "no MIP" — the backend keeps any embedded label / its default. The
 * backend treats a set value as a floor-raiser (can raise above the embedded
 * label, never lower it). The C-code → business-name map: C1=Public,
 * C2=Internal, C3=Confidential, C4=Secret.
 */
export type UploadClassification = 'C1' | 'C2' | 'C3' | 'C4';

/** Per-file upload options. Classification-only — the LightRAG/RAG1.5 engine
 *  toggle is NOT exposed (RAG 1.5 connector isn't live). */
export interface FileUploadOptions {
  name: string;
  classification?: UploadClassification;
}

export interface FileUpload {
  name: string;
  /** Megabytes (1 decimal place). Retained for backwards compat with
   *  fixtures; display uses `sizeBytes` when present so files smaller
   *  than 0.1 MB show in KB instead of "0 MB". */
  size: number;
  /** Raw byte count from the picked File. Optional so existing test
   *  fixtures keep working. */
  sizeBytes?: number;
  state: FileUploadState;
  progress?: number;
  uploaded?: number;
  error?: string;
  classification?: UploadClassification;
}

/** Format a byte count as "B / KB / MB" depending on magnitude. */
// eslint-disable-next-line react-refresh/only-export-components -- pure helper, tests import it directly.
export function formatFileSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function displaySize(f: FileUpload): string {
  if (typeof f.sizeBytes === 'number') return formatFileSize(f.sizeBytes);
  return `${f.size} MB`;
}

function displayUploadedOverTotal(f: FileUpload): string {
  if (typeof f.sizeBytes === 'number') {
    const uploadedBytes = ((f.progress ?? 0) / 100) * f.sizeBytes;
    return `${formatFileSize(uploadedBytes)} / ${formatFileSize(f.sizeBytes)}`;
  }
  return `${(f.uploaded ?? 0).toFixed(1)} / ${f.size} MB`;
}

function addSourceButtonLabel(ready: number): string {
  return `Add ${ready} source${ready === 1 ? '' : 's'}`;
}

function fileClassification(f: FileUpload): UploadClassification | '' {
  return f.classification ?? '';
}

/** C-code → operator-facing option label. Business names per the BNP MIP
 *  taxonomy (C1=Public … C4=Secret). */
const CLASSIFICATION_OPTIONS: readonly {
  value: UploadClassification;
  label: string;
}[] = [
  { value: 'C1', label: 'C1 · Public' },
  { value: 'C2', label: 'C2 · Internal' },
  { value: 'C3', label: 'C3 · Confidential' },
  { value: 'C4', label: 'C4 · Secret' },
];

export type LinkedSourceType = 'confluence' | 'sharepoint' | 'url';

export interface LinkedSource {
  url: string;
  type: LinkedSourceType;
}

export interface AddSourceAction {
  files: readonly FileUpload[];
  /**
   * Raw File objects keyed by display name, captured when the user picks
   * or drops files. The host uses these to POST multipart/form-data to
   * LightRAG's ``/documents/upload``. May be empty in tests that only
   * exercise UI flows (initialFiles paths).
  */
  rawFiles: readonly File[];
  /** Per-file options (classification-only), aligned with `rawFiles` order —
   *  both are derived from the `uploaded`-state files in the same sequence. */
  fileOptions: readonly FileUploadOptions[];
  urls: readonly LinkedSource[];
  tags: readonly string[];
  /** Files in `uploaded` state + all URLs. Mirrors the proto's `ready` count. */
  readyCount: number;
}

export interface AddSourceModalProps {
  open: boolean;
  tagCatalog: readonly TagEntry[];
  formatCategories: readonly FormatCategory[];
  /** Initial mock state — used in dev preview. Empty arrays in real use. */
  initialFiles?: readonly FileUpload[];
  initialUrls?: readonly LinkedSource[];
  onClose: () => void;
  onSubmit: (action: AddSourceAction) => void;
  /** Server upload in progress (host's upload mutation pending). While true
   *  the modal stays open and close (X / backdrop / Escape) + submit are
   *  disabled so the operator can't dismiss an in-flight upload. */
  submitting?: boolean;
}

function fileExtension(name: string): string {
  const idx = name.lastIndexOf('.');
  return idx >= 0 ? name.slice(idx + 1).toLowerCase() : '';
}

function unsupportedFileType(file: File): boolean {
  const ext = fileExtension(file.name);
  if (ext && SUPPORTED_EXTENSIONS.has(ext)) return false;
  if (file.type && SUPPORTED_MIME_TYPES.has(file.type)) return false;
  return !SUPPORTED_MIME_PREFIXES.some((prefix) => file.type.startsWith(prefix));
}

function validateFile(file: File): string | null {
  const errors: string[] = [];
  if (file.size > MAX_FILE_MB * 1024 * 1024) errors.push('Exceeds 50 MB');
  if (unsupportedFileType(file)) errors.push('unsupported type');
  return errors.length ? errors.join(' · ') : null;
}

function fileUploadFromFile(file: File, rawFiles: Map<string, File>): FileUpload {
  const error = validateFile(file);
  const base = {
    name: file.name,
    size: Number((file.size / (1024 * 1024)).toFixed(1)),
    sizeBytes: file.size,
  };
  if (error) {
    rawFiles.delete(file.name);
    return { ...base, state: 'error', error };
  }
  rawFiles.set(file.name, file);
  return {
    ...base,
    state: 'uploading',
    progress: 0,
    uploaded: 0,
  };
}

function advanceUploadProgress(file: FileUpload): FileUpload {
  if (file.state !== 'uploading') return file;
  const progress = Math.min(100, (file.progress ?? 0) + 4);
  if (progress >= 100) {
    return { ...file, state: 'uploaded', progress: 100 };
  }
  return {
    ...file,
    progress,
    uploaded: (progress / 100) * file.size,
  };
}

function uploadCounts(files: readonly FileUpload[], urlsCount: number) {
  const readyFiles = files.filter((f) => f.state === 'uploaded').length;
  const uploading = files.filter((f) => f.state === 'uploading').length;
  const errors = files.filter((f) => f.state === 'error').length;
  return {
    uploading,
    errors,
    ready: readyFiles + urlsCount,
  };
}

function UploadFormatTooltip({
  show,
  formatCategories,
}: Readonly<{
  show: boolean;
  formatCategories: readonly FormatCategory[];
}>) {
  if (!show) return null;
  return (
    <div
      className="tooltip"
      role="tooltip"
      style={{
        left: '50%',
        transform: 'translateX(-50%)',
        top: 'calc(100% + 8px)',
        textAlign: 'left',
      }}
    >
      {formatCategories.map((c) => (
        <div
          key={c.cat}
          className={c.future ? 'tooltip-row future' : 'tooltip-row'}
        >
          <span className="cat">{c.cat}</span>
          <span className="fmts">{c.fmts}</span>
        </div>
      ))}
    </div>
  );
}

function UploadStatusText({
  uploading,
  errors,
  ready,
  fileCount,
}: Readonly<{
  uploading: number;
  errors: number;
  ready: number;
  fileCount: number;
}>) {
  if (uploading > 0) return <>Uploading {uploading} of {fileCount} files…</>;
  if (errors > 0) {
    return (
      <>
        {errors} error{errors > 1 ? 's' : ''} · {ready} ready
      </>
    );
  }
  return ready > 0 ? <>{ready} ready to ingest</> : null;
}

export function AddSourceModal({
  open,
  tagCatalog,
  formatCategories,
  initialFiles = [],
  initialUrls = [],
  onClose,
  onSubmit,
  submitting = false,
}: Readonly<AddSourceModalProps>) {
  const tagSuggListId = 'addsource-tag-suggestions';
  const modalRef = useRef<HTMLDialogElement>(null);
  // While an upload is in flight, neutralise close so X / backdrop / Escape
  // can't dismiss the modal mid-upload (matches LightRAG's native UX).
  const guardedClose = submitting ? () => {} : onClose;
  useModalA11y({ open, onClose: guardedClose, ref: modalRef });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<readonly FileUpload[]>(initialFiles);
  // Raw File objects parallel to `files` — kept out of state shape
  // proper because they're not serializable + the test fixtures only
  // care about the metadata. Keyed by display name so removeFile can
  // drop both metadata + binary in one operation.
  const rawFilesRef = useRef<Map<string, File>>(new Map());

  const appendDroppedFiles = (incoming: FileList | null): void => {
    const incomingArr = Array.from(incoming || []);
    if (incomingArr.length === 0) return;
    const dropped = incomingArr.map((file) =>
      fileUploadFromFile(file, rawFilesRef.current),
    );
    setFiles((current) => [...current, ...dropped]);
  };
  // Linked sources are gated until the RAG 1.5 connector lands — see the
  // "Coming soon" badge in the JSX. We keep `urls` in state (initialised
  // from props for tests) so the submit pipeline still emits it.
  const [urls] = useState<readonly LinkedSource[]>(initialUrls);
  const [tags, setTags] = useState<readonly string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [drag, setDrag] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);
  const [tagSuggestionIndex, setTagSuggestionIndex] = useState(0);

  // Animate the uploading file's progress (4%/tick @ 320ms — same as proto).
  useEffect(() => {
    if (!open) return;
    const tick = setInterval(() => {
      setFiles((fs) => fs.map(advanceUploadProgress));
    }, 320);
    return () => clearInterval(tick);
  }, [open]);

  const tagSugg = useMemo(() => {
    return tagCatalog
      .filter((t) => !tags.includes(t.tag))
      .filter((t) => tagMatchesQuery(t, tagInput))
      .sort(tagSuggestionComparator(tagInput))
      .slice(0, 4);
  }, [tagCatalog, tags, tagInput]);
  /* eslint-disable react-hooks/set-state-in-effect -- intentional reset of the highlighted tag-suggestion index when the filter query changes. */
  useEffect(() => {
    setTagSuggestionIndex(0);
  }, [tagInput]);
  /* eslint-enable react-hooks/set-state-in-effect */

  if (!open) return null;

  const removeFile = (n: string) => {
    rawFilesRef.current.delete(n);
    setFiles(files.filter((f) => f.name !== n));
  };
  const updateFileOptions = (name: string, patch: Partial<FileUploadOptions>) => {
    setFiles((current) =>
      current.map((f) => (f.name === name ? { ...f, ...patch } : f)),
    );
  };
  const addTag = (t: string) => {
    if (t && !tags.includes(t)) setTags([...tags, t]);
    setTagInput('');
    setTagSuggestionIndex(0);
  };
  const removeTag = (t: string) => setTags(tags.filter((x) => x !== t));
  const activeTagSuggestion =
    tagSugg.length === 0
      ? undefined
      : tagSugg[Math.min(tagSuggestionIndex, tagSugg.length - 1)];
  const activeTagSuggestionId = activeTagSuggestion
    ? `addsource-tag-suggestion-${activeTagSuggestion.tag}`
    : undefined;

  const { uploading, errors, ready } = uploadCounts(files, urls.length);

  const submit = () => {
    if (ready === 0) return;
    // Collect raw File objects in the same order as `files` so callers
    // (host App) can correlate progress feedback per file.
    const uploadedFiles = files.filter((f) => f.state === 'uploaded');
    const rawFiles = uploadedFiles
      .map((f) => rawFilesRef.current.get(f.name))
      .filter((f): f is File => f !== undefined);
    // Per-file options aligned with `uploadedFiles` order — classification-only.
    const fileOptions = uploadedFiles.map((f) => {
      const classification = fileClassification(f);
      return {
        name: f.name,
        ...(classification ? { classification } : {}),
      };
    });
    onSubmit({ files, rawFiles, fileOptions, urls, tags, readyCount: ready });
    // Do NOT close here: the host keeps the modal open during the upload
    // (submitting=true) and closes it when the mutation settles, so the
    // operator sees progress and can't dismiss an in-flight upload.
  };

  return (
    <div
      className="modal-backdrop"
    >
      <button
        type="button"
        className="modal-backdrop-dismiss"
        onClick={guardedClose}
        aria-label="Close add source dialog"
        data-testid="addsource-backdrop"
      />
      <dialog
        open
        ref={modalRef}
        className="modal"
        style={{ width: 480 }}
        aria-modal="true"
        aria-labelledby="addsource-title"
        tabIndex={-1}
      >
        <div className="modal-header">
          <div>
            <h2 id="addsource-title">Add source</h2>
          </div>
          <button
            type="button"
            className="icon-btn"
            onClick={guardedClose}
            disabled={submitting}
            aria-label="Close dialog"
            title={submitting ? 'Upload in progress…' : undefined}
          >
            <Icon name="x" size={18} />
          </button>
        </div>

        <div className="modal-body">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            style={{ display: 'none' }}
            data-testid="addsource-file-input"
            onChange={(e) => {
              appendDroppedFiles(e.target.files);
              // Reset so re-picking the same file re-fires change.
              e.target.value = '';
            }}
          />
          <button
            type="button"
            className={drag ? 'dropzone drag' : 'dropzone'}
            aria-label="Drop files or click to browse"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              appendDroppedFiles(e.dataTransfer.files);
            }}
          >
            <Icon name="cloud-upload" size={28} color="var(--color-text-secondary)" />
            <div className="title">Drop files here or click to browse</div>
            <div className="sub">
              <span>PDF, DOCX, MD, TXT and 35+ formats</span>
              <span style={{ color: 'var(--color-text-tertiary)' }}>
                · max 50 MB per file
              </span>
            </div>
          </button>
          <div className="dropzone-info-row">
            <button
              type="button"
              className="dropzone-info"
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              onFocus={() => setShowTooltip(true)}
              onBlur={() => setShowTooltip(false)}
              aria-label="Show supported formats"
            >
              <Icon name="info-circle" size={13} />
              <span>Supported formats</span>
              <UploadFormatTooltip
                show={showTooltip}
                formatCategories={formatCategories}
              />
            </button>
          </div>

          {files.length > 0 && (
            <div>
              <div className="section-label">
                <span>
                  Files{' '}
                  <span
                    style={{
                      color: 'var(--color-text-tertiary)',
                      fontWeight: 400,
                      fontSize: 11,
                    }}
                  >
                    ({files.length} added)
                  </span>
                </span>
              </div>
              <div className="file-list" style={{ marginTop: 6 }}>
                {files.map((f) => (
                  <div
                    key={f.name}
                    className={f.state === 'error' ? 'file-row error' : 'file-row'}
                  >
                    <Icon name="file-text" size={15} className="file-icon" />
                    <div className="info">
                      <div className="row1">
                        <span className="name">{f.name}</span>
                        <span className="size">{displaySize(f)}</span>
                      </div>
                      {f.state === 'uploading' && (
                        <div className="row2">
                          <div className="bar">
                            <div
                              className="bar-fill"
                              style={{ width: `${f.progress ?? 0}%` }}
                            />
                          </div>
                          <span style={{ fontFamily: 'var(--font-mono)' }}>
                            {f.progress ?? 0}% · {displayUploadedOverTotal(f)}
                          </span>
                        </div>
                      )}
                      {f.state === 'error' && (
                        <div className="row2">
                          <Icon name="alert-triangle" size={12} /> {f.error}
                        </div>
                      )}
                    </div>
                    {f.state !== 'error' && (
                      <label
                        className="file-classification-label"
                        title="MIP classification (operator value raises the embedded label, never lowers it)"
                      >
                        <span>C</span>
                        <select
                          className="file-classification-control"
                          value={fileClassification(f)}
                          onChange={(e) =>
                            updateFileOptions(f.name, {
                              classification:
                                e.target.value === ''
                                  ? undefined
                                  : (e.target.value as UploadClassification),
                            })
                          }
                          aria-label={`Classification for ${f.name}`}
                          data-testid={`addsource-classification-${f.name}`}
                        >
                          <option value="">no MIP</option>
                          {CLASSIFICATION_OPTIONS.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    )}
                    {f.state === 'uploaded' && (
                      <Icon name="circle-check" size={16} className="ok" />
                    )}
                    <button
                      type="button"
                      className="x-btn"
                      onClick={() => removeFile(f.name)}
                      aria-label={`Remove ${f.name}`}
                    >
                      <Icon name="x" size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="or-divider">
            <span>OR</span>
          </div>

          <div>
            <div className="section-label">
              <span>Linked sources</span>
              <span
                className="coming-soon"
                title="Waiting on the RAG 1.5 API to be exposed"
              >
                Coming soon
              </span>
            </div>
            <div
              className="linked-box disabled"
              aria-disabled="true"
              style={{ marginTop: 6 }}
            >
              <input
                className="url-input"
                value=""
                onChange={() => {}}
                disabled
                placeholder="Available once the RAG 1.5 API is wired"
                aria-label="URL input (disabled — coming soon)"
                tabIndex={-1}
              />
            </div>
            <div className="helper-note" style={{ marginTop: 4 }}>
              Confluence / SharePoint linking will use the RAG 1.5 connector
              (upstream RAG team). Endpoint is not yet available — drop
              files in the box above in the meantime.
            </div>
          </div>

          <div>
            <div className="section-label">
              <span>Apply tags to all</span>
              <span className="optional">optional</span>
            </div>
            <div
              className="chip-input"
              style={{ marginTop: 6, background: 'var(--color-background-primary)' }}
            >
              {tags.map((t) => (
                <TagChip key={t} tag={t} removable onRemove={removeTag} />
              ))}
              <input
                value={tagInput}
                onChange={(e) => {
                  setTagInput(e.target.value);
                  setTagSuggestionIndex(0);
                }}
                onKeyDownCapture={(e) => {
                  if (e.key === 'Escape' && tagInput) {
                    e.stopPropagation();
                    setTagInput('');
                    setTagSuggestionIndex(0);
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === 'ArrowDown') {
                    if (tagSugg.length === 0) return;
                    e.preventDefault();
                    setTagSuggestionIndex((idx) => (idx + 1) % tagSugg.length);
                    return;
                  }
                  if (e.key === 'ArrowUp') {
                    if (tagSugg.length === 0) return;
                    e.preventDefault();
                    setTagSuggestionIndex(
                      (idx) => (idx - 1 + tagSugg.length) % tagSugg.length,
                    );
                    return;
                  }
                  if (e.key === 'Enter' && activeTagSuggestion) {
                    e.preventDefault();
                    addTag(activeTagSuggestion.tag);
                  }
                }}
                placeholder={tags.length ? '' : 'Search tags…'}
                aria-label="Tag input"
                aria-autocomplete="list"
                aria-expanded={tagSugg.length > 0}
                aria-controls={tagSuggListId}
                aria-activedescendant={activeTagSuggestionId}
                style={{ fontSize: 12 }}
              />
            </div>
            {tagInput && tagSugg.length > 0 && (
              <div
                id={tagSuggListId}
                role="listbox"
                aria-label="Tag suggestions"
                className="autocomplete modal-autocomplete"
                style={{ marginTop: 4 }}
              >
                {tagSugg.map((s, i) => (
                  <button
                    type="button"
                    key={s.tag}
                    id={`addsource-tag-suggestion-${s.tag}`}
                    role="option"
                    aria-selected={i === tagSuggestionIndex}
                    className={`autocomplete-row${
                      i === tagSuggestionIndex ? ' focus' : ''
                    }`}
                    onMouseEnter={() => setTagSuggestionIndex(i)}
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => addTag(s.tag)}
                    data-testid={`tag-sugg-${s.tag}`}
                  >
                    <div className="row1">
                      <span style={{ fontSize: 12 }}>{s.tag}</span>
                      <span className="badge">{s.category}</span>
                    </div>
                    <div className="def">{s.def}</div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer">
          <div className="status">
            <UploadStatusText
              uploading={uploading}
              errors={errors}
              ready={ready}
              fileCount={files.length}
            />
          </div>
          <div className="actions">
            <button
              type="button"
              className="btn"
              onClick={guardedClose}
              disabled={submitting}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn primary"
              disabled={ready === 0 || submitting}
              onClick={submit}
            >
              {submitting ? 'Uploading…' : addSourceButtonLabel(ready)}
            </button>
          </div>
        </div>
      </dialog>
    </div>
  );
}
