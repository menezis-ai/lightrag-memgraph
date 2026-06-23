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

  if (!open) return null;

  const removeFile = (n: string) => {
    rawFilesRef.current.delete(n);
    setFiles(files.filter((f) => f.name !== n));
  };
  const addTag = (t: string) => {
    if (t && !tags.includes(t)) setTags([...tags, t]);
    setTagInput('');
  };
  const removeTag = (t: string) => setTags(tags.filter((x) => x !== t));

  const { uploading, errors, ready } = uploadCounts(files, urls.length);

  const submit = () => {
    if (ready === 0) return;
    // Collect raw File objects in the same order as `files` so callers
    // (host App) can correlate progress feedback per file.
    const rawFiles = files
      .filter((f) => f.state === 'uploaded')
      .map((f) => rawFilesRef.current.get(f.name))
      .filter((f): f is File => f !== undefined);
    onSubmit({ files, rawFiles, urls, tags, readyCount: ready });
    // Do NOT close here: the host keeps the modal open during the upload
    // (submitting=true) and closes it when the mutation settles, so the
    // operator sees progress and can't dismiss an in-flight upload.
  };

  return (
    <div
      className="modal-backdrop"
      onClick={(e) => {
        if (e.currentTarget === e.target) guardedClose();
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape') guardedClose();
      }}
      data-testid="addsource-backdrop"
    >
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
              PDF, DOCX, MD, TXT and 35+ formats
              <span
                className="info"
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                style={{ position: 'relative' }}
              >
                <Icon name="info-circle" size={13} />
                <UploadFormatTooltip
                  show={showTooltip}
                  formatCategories={formatCategories}
                />
              </span>
              <span style={{ color: 'var(--color-text-tertiary)' }}>
                {' '}· max 50 MB per file
              </span>
            </div>
          </button>

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
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDownCapture={(e) => {
                  if (e.key === 'Escape') {
                    e.stopPropagation();
                    setTagInput('');
                  }
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && tagSugg[0]) {
                    addTag(tagSugg[0].tag);
                  }
                }}
                placeholder={tags.length ? '' : 'Search tags…'}
                aria-label="Tag input"
                style={{ fontSize: 12 }}
              />
            </div>
            {tagInput && tagSugg.length > 0 && (
              <ul
                className="autocomplete modal-autocomplete"
                style={{ marginTop: 4 }}
              >
                {tagSugg.map((s, i) => (
                  <li
                    key={s.tag}
                    className={`autocomplete-row${i === 0 ? ' focus' : ''}`}
                    onMouseDown={() => addTag(s.tag)}
                    aria-selected={i === 0}
                    data-testid={`tag-sugg-${s.tag}`}
                  >
                    <div className="row1">
                      <span style={{ fontSize: 12 }}>{s.tag}</span>
                      <span className="badge">{s.category}</span>
                    </div>
                    <div className="def">{s.def}</div>
                  </li>
                ))}
              </ul>
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
              {submitting
                ? 'Uploading…'
                : `Add ${ready} source${ready === 1 ? '' : 's'}`}
            </button>
          </div>
        </div>
      </dialog>
    </div>
  );
}
