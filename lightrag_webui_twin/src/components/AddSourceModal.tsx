/**
 * AddSourceModal — drop files, paste URLs, optionally apply tags.
 *
 * Ported from Desktop/UI/modals.jsx. Behavior delta vs the proto:
 *   - thesaurus + formatCategories injected via props (no window.* reads)
 *   - submit emits an AddSourceAction; the host owns the toast lifecycle
 *   - useModalA11y attached via the hook port from S1
 *
 * The upload progress animation (setInterval bumping `progress` by 4% every
 * 320ms) is preserved — for tests we keep timers real-time and rely on
 * Vitest fake timers when we want to observe transitions.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { Icon, SourceIcon } from './Icon';
import { TagChip } from './TagChip';
import { useModalA11y } from '../hooks/useModalA11y';
import type { FormatCategory } from '../types/format';
import type { ThesaurusEntry } from '../types/thesaurus';

export type FileUploadState = 'uploading' | 'uploaded' | 'error';

export interface FileUpload {
  name: string;
  /** Megabytes (1 decimal place from proto). */
  size: number;
  state: FileUploadState;
  progress?: number;
  uploaded?: number;
  error?: string;
}

export type LinkedSourceType = 'confluence' | 'sharepoint' | 'url';

export interface LinkedSource {
  url: string;
  type: LinkedSourceType;
}

export interface AddSourceAction {
  files: readonly FileUpload[];
  urls: readonly LinkedSource[];
  tags: readonly string[];
  /** Files in `uploaded` state + all URLs. Mirrors the proto's `ready` count. */
  readyCount: number;
}

export interface AddSourceModalProps {
  open: boolean;
  thesaurus: readonly ThesaurusEntry[];
  formatCategories: readonly FormatCategory[];
  /** Initial mock state — used in dev preview. Empty arrays in real use. */
  initialFiles?: readonly FileUpload[];
  initialUrls?: readonly LinkedSource[];
  onClose: () => void;
  onSubmit: (action: AddSourceAction) => void;
}

function detectLinkedType(url: string): LinkedSourceType {
  if (/confluence/i.test(url)) return 'confluence';
  if (/sharepoint/i.test(url)) return 'sharepoint';
  return 'url';
}

export function AddSourceModal({
  open,
  thesaurus,
  formatCategories,
  initialFiles = [],
  initialUrls = [],
  onClose,
  onSubmit,
}: AddSourceModalProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  useModalA11y({ open, onClose, ref: modalRef });

  const [files, setFiles] = useState<readonly FileUpload[]>(initialFiles);
  const [urls, setUrls] = useState<readonly LinkedSource[]>(initialUrls);
  const [urlInput, setUrlInput] = useState('');
  const [tags, setTags] = useState<readonly string[]>([]);
  const [tagInput, setTagInput] = useState('');
  const [drag, setDrag] = useState(false);
  const [showTooltip, setShowTooltip] = useState(false);

  // Animate the uploading file's progress (4%/tick @ 320ms — same as proto).
  useEffect(() => {
    if (!open) return;
    const tick = setInterval(() => {
      setFiles((fs) =>
        fs.map((f) => {
          if (f.state !== 'uploading') return f;
          const np = Math.min(100, (f.progress ?? 0) + 4);
          if (np >= 100) {
            return { ...f, state: 'uploaded', progress: 100 };
          }
          return {
            ...f,
            progress: np,
            uploaded: (np / 100) * f.size,
          };
        }),
      );
    }, 320);
    return () => clearInterval(tick);
  }, [open]);

  const tagSugg = useMemo(() => {
    return thesaurus
      .filter((t) => !tags.includes(t.tag))
      .filter(
        (t) => !tagInput || t.tag.includes(tagInput.toLowerCase()),
      )
      .slice(0, 4);
  }, [thesaurus, tags, tagInput]);

  if (!open) return null;

  const addUrls = (val: string) => {
    const parts = val
      .split(/[\s,]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    setUrls([...urls, ...parts.map((u) => ({ url: u, type: detectLinkedType(u) }))]);
    setUrlInput('');
  };
  const removeUrl = (u: string) => setUrls(urls.filter((x) => x.url !== u));
  const removeFile = (n: string) =>
    setFiles(files.filter((f) => f.name !== n));
  const addTag = (t: string) => {
    if (t && !tags.includes(t)) setTags([...tags, t]);
    setTagInput('');
  };
  const removeTag = (t: string) => setTags(tags.filter((x) => x !== t));

  const readyFiles = files.filter((f) => f.state === 'uploaded').length;
  const uploading = files.filter((f) => f.state === 'uploading').length;
  const errors = files.filter((f) => f.state === 'error').length;
  const ready = readyFiles + urls.length;

  const submit = () => {
    if (ready === 0) return;
    onSubmit({ files, urls, tags, readyCount: ready });
    onClose();
  };

  return (
    <div className="modal-backdrop" onClick={onClose} data-testid="addsource-backdrop">
      <div
        ref={modalRef}
        className="modal"
        style={{ width: 480 }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="addsource-title"
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <div>
            <h2 id="addsource-title">Add source</h2>
          </div>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Close dialog"
          >
            <Icon name="x" size={18} />
          </button>
        </div>

        <div className="modal-body">
          <div
            className={drag ? 'dropzone drag' : 'dropzone'}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              const dropped = Array.from(e.dataTransfer.files || []).map<FileUpload>(
                (f) => ({
                  name: f.name,
                  size: Number((f.size / (1024 * 1024)).toFixed(1)),
                  state: 'uploading',
                  progress: 0,
                  uploaded: 0,
                }),
              );
              setFiles([...files, ...dropped]);
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
                {showTooltip && (
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
                )}
              </span>
              <span style={{ color: 'var(--color-text-tertiary)' }}>
                {' '}· max 50 MB per file
              </span>
            </div>
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
                        <span className="size">{f.size} MB</span>
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
                            {f.progress ?? 0}% · {(f.uploaded ?? 0).toFixed(1)} /{' '}
                            {f.size} MB
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
            </div>
            <div className="linked-box" style={{ marginTop: 6 }}>
              {urls.map((u) => (
                <span key={u.url} className="url-chip">
                  <span className="ico">
                    <SourceIcon type={u.type} size={12} />
                  </span>
                  <span>{u.url}</span>
                  <button
                    type="button"
                    className="x-btn"
                    onClick={() => removeUrl(u.url)}
                    aria-label={`Remove ${u.url}`}
                  >
                    <Icon name="x" size={10} />
                  </button>
                </span>
              ))}
              <input
                className="url-input"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') addUrls(urlInput);
                }}
                placeholder={
                  urls.length
                    ? 'Paste another URL — Enter to add'
                    : 'Paste Confluence or SharePoint URL — Enter to add'
                }
                aria-label="URL input"
              />
            </div>
            <div className="helper-note" style={{ marginTop: 4 }}>
              Auto-detects Confluence, SharePoint, or generic URL · paste
              multiple separated by space or comma
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
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && tagSugg[0]) addTag(tagSugg[0].tag);
                }}
                placeholder={tags.length ? '' : 'Search tags from thesaurus…'}
                aria-label="Tag input"
                style={{ fontSize: 12 }}
              />
            </div>
            {tagInput && tagSugg.length > 0 && (
              <div className="autocomplete" style={{ marginTop: 4 }}>
                {tagSugg.map((s, i) => (
                  <div
                    key={s.tag}
                    className={`autocomplete-row${i === 0 ? ' focus' : ''}`}
                    onMouseDown={() => addTag(s.tag)}
                    data-testid={`tag-sugg-${s.tag}`}
                  >
                    <div className="row1">
                      <span style={{ fontSize: 12 }}>{s.tag}</span>
                      <span className="badge">{s.category}</span>
                    </div>
                    <div className="def">{s.def}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer">
          <div className="status">
            {uploading > 0 &&
              `Uploading ${uploading} of ${files.length} files…`}
            {uploading === 0 && errors > 0 &&
              `${errors} error${errors > 1 ? 's' : ''} · ${ready} ready`}
            {uploading === 0 && errors === 0 && ready > 0 &&
              `${ready} ready to ingest`}
          </div>
          <div className="actions">
            <button type="button" className="btn" onClick={onClose}>
              Cancel
            </button>
            <button
              type="button"
              className="btn primary"
              disabled={ready === 0}
              onClick={submit}
            >
              Add {ready} source{ready === 1 ? '' : 's'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
