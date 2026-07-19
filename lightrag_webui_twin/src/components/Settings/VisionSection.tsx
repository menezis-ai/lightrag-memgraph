/**
 * Settings → Vision (image-ingestion curation knobs).
 *
 * Surfaces the two operator-tunable settings of the image pipeline:
 *
 *   - min OCR chars : RapidOCR pre-filter threshold. Images with less
 *     OCR text are refused without a vision-LLM call (0 = caption
 *     everything).
 *   - drop classes  : image classifications refused *after* the vision
 *     LLM call (env defaults: invalid / logo / signature).
 *
 * Backend contract (`server/vision_settings_routes.py`):
 *   GET /twin/api/settings/vision → effective values + provenance
 *     (`source: 'runtime' | 'env-default'`).
 *   PUT /twin/api/settings/vision → persist (admin-gated, 403 otherwise).
 *
 * Admin gating mirrors FoldersAdminSection: the section stays visible
 * for every authenticated operator (GET is open) but inputs + Save are
 * disabled without the `admin:folders` gateway scope, and an unexpected
 * backend 403 surfaces as an "Admin scope required" toast.
 */

import { useEffect, useState } from 'react';
import { Icon } from '../Icon';
import { ApiError } from '../../api/client';
import { useUpdateVisionSettings, useVisionSettings } from '../../api/queries';
import { userErrorMessage } from '../../lib/errorMessages';
import { canManageFolders } from '../../lib/permissions';
import type { AuthenticatedUser } from '../../types/auth';
import type { Toast } from '../../types/toast';
import { relativeTime } from '../../utils/relativeTime';

/** Mirrors the backend `_CLASS_PATTERN` (lowercase slugs). */
const DROP_CLASS_RE = /^[a-z0-9][a-z0-9 _-]{0,39}$/;
const MIN_OCR_CHARS_MAX = 100_000;
const MAX_DROP_CLASSES = 20;

export interface VisionSectionProps {
  user?: AuthenticatedUser | null;
  onToast?: (toast: Omit<Toast, 'id'>) => void;
}

function sameList(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}

export function VisionSection({
  user = null,
  onToast,
}: VisionSectionProps = {}) {
  const { data, isLoading, isError, error, refetch } = useVisionSettings();
  const updateMutation = useUpdateVisionSettings();
  const canManage = canManageFolders(user);

  const [minOcrChars, setMinOcrChars] = useState('0');
  const [dropClasses, setDropClasses] = useState<readonly string[]>([]);
  const [newClass, setNewClass] = useState('');
  const [classError, setClassError] = useState<string | null>(null);

  // Re-sync the draft with the server value after a (re)fetch — same
  // pattern as FolderRow. A successful save refetches and lands the
  // canonicalized (sorted, deduped) values back into the inputs.
  /* eslint-disable react-hooks/set-state-in-effect -- intentional re-sync with the new server-side value. */
  useEffect(() => {
    if (!data) return;
    setMinOcrChars(String(data.min_ocr_chars));
    setDropClasses(data.drop_classes);
  }, [data]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const parsedMin = Number.parseInt(minOcrChars, 10);
  const minValid =
    minOcrChars.trim() !== '' &&
    Number.isInteger(parsedMin) &&
    parsedMin >= 0 &&
    parsedMin <= MIN_OCR_CHARS_MAX;
  const dirty =
    data !== undefined &&
    (data.source === 'env-default' ||
      parsedMin !== data.min_ocr_chars ||
      !sameList(dropClasses, data.drop_classes));
  const canSave =
    canManage && minValid && dirty && !updateMutation.isPending && !isLoading;

  const addClass = (): void => {
    const slug = newClass.trim().toLowerCase();
    if (!slug) return;
    if (!DROP_CLASS_RE.test(slug)) {
      setClassError(
        'Invalid class — lowercase letters, digits, space, underscore or dash (max 40 chars).',
      );
      return;
    }
    if (dropClasses.includes(slug)) {
      setClassError(`Class “${slug}” is already listed.`);
      return;
    }
    if (dropClasses.length >= MAX_DROP_CLASSES) {
      setClassError(`At max — ${MAX_DROP_CLASSES} drop classes cap reached.`);
      return;
    }
    setDropClasses([...dropClasses, slug]);
    setNewClass('');
    setClassError(null);
  };

  const removeClass = (slug: string): void => {
    setDropClasses(dropClasses.filter((c) => c !== slug));
    setClassError(null);
  };

  const save = (): void => {
    if (!canSave) return;
    updateMutation.mutate(
      { min_ocr_chars: parsedMin, drop_classes: [...dropClasses] },
      {
        onSuccess: () => {
          onToast?.({
            kind: 'done',
            title: 'Vision settings saved',
            sub: 'The new curation thresholds apply to the next ingested image.',
          });
        },
        onError: (err) => {
          if (err instanceof ApiError && err.status === 403) {
            onToast?.({
              kind: 'error',
              title: 'Admin scope required',
              sub: 'Your account does not have the administration permission required to change vision settings. Contact Twincore Team to request it.',
            });
            return;
          }
          onToast?.({
            kind: 'error',
            title: 'Could not save vision settings',
            sub: errorToMessage(err) ?? 'Unexpected error.',
          });
        },
      },
    );
  };

  return (
    <div className="settings-section" data-testid="settings-vision">
      <h3>Vision</h3>
      <p className="muted">
        Curation knobs of the image-ingestion pipeline. The OCR pre-filter
        refuses low-text images before any vision-LLM call; drop classes
        refuse images by classification after the call. Infrastructure
        wiring (endpoint, model, credentials) stays in the deploy env.
      </p>
      {!canManage && (
        <span className="env-badge" data-testid="settings-vision-readonly-badge">
          <Icon name="lock" size={10} /> Read-only — admin scope required
        </span>
      )}

      {isLoading && (
        <div className="muted" data-testid="settings-vision-loading">
          Loading vision settings…
        </div>
      )}
      {isError && (
        <div
          className="error-banner"
          role="alert"
          data-testid="settings-vision-error"
        >
          Could not load vision settings
          {error === undefined || error === null
            ? ''
            : ` — ${userErrorMessage(error)}`}
          .{' '}
          <button
            type="button"
            className="ghost-btn"
            onClick={() => refetch()}
          >
            Retry
          </button>
        </div>
      )}

      {data && (
        <>
          {data.source === 'env-default' ? (
            <div
              className="muted"
              data-testid="settings-vision-provenance-env"
            >
              <Icon name="info-circle" size={12} /> Defaults from deployment
              environment — no runtime override saved yet.
            </div>
          ) : (
            <div
              className="muted"
              data-testid="settings-vision-provenance-runtime"
            >
              <Icon name="circle-check" size={12} /> Runtime override
              {data.updated_by ? ` by ${data.updated_by}` : ''}
              {data.updated_at
                ? `, ${relativeTime(new Date(data.updated_at).toISOString())}`
                : ''}
              .
            </div>
          )}

          <label className="settings-field">
            <span>
              Min OCR chars <em>(0–{MIN_OCR_CHARS_MAX.toLocaleString('en-US')})</em>
            </span>
            <input
              type="number"
              min={0}
              max={MIN_OCR_CHARS_MAX}
              step={1}
              value={minOcrChars}
              onChange={(e) => setMinOcrChars(e.target.value)}
              disabled={!canManage}
              aria-label="Minimum OCR characters"
              data-testid="settings-vision-min-ocr"
            />
          </label>
          <p className="muted">
            Images whose OCR text is shorter than this are refused without a
            vision-LLM call. 0 = caption everything.
          </p>
          {!minValid && (
            <div
              role="alert"
              className="settings-error"
              data-testid="settings-vision-min-ocr-invalid"
            >
              Enter an integer between 0 and{' '}
              {MIN_OCR_CHARS_MAX.toLocaleString('en-US')}.
            </div>
          )}

          <span className="field-label">Drop classes</span>
          <div className="alias-chips" data-testid="settings-vision-classes">
            {dropClasses.length === 0 && (
              <span className="muted">
                No drop classes — every classified image is kept.
              </span>
            )}
            {dropClasses.map((c) => (
              <span key={c} className="alias-chip">
                <code>{c}</code>
                {canManage && (
                  <button
                    type="button"
                    aria-label={`Remove drop class ${c}`}
                    onClick={() => removeClass(c)}
                  >
                    <Icon name="x" size={10} />
                  </button>
                )}
              </span>
            ))}
          </div>
          {canManage && (
            <label className="settings-field">
              <span>
                Add drop class <em>(lowercase slug, Enter to add)</em>
              </span>
              <input
                type="text"
                value={newClass}
                onChange={(e) => {
                  setNewClass(e.target.value);
                  if (classError) setClassError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addClass();
                  }
                }}
                placeholder="e.g. screenshot"
                aria-label="New drop class"
                data-testid="settings-vision-class-input"
              />
            </label>
          )}
          {classError && (
            <div
              role="alert"
              className="settings-error"
              data-testid="settings-vision-class-error"
            >
              {classError}
            </div>
          )}

          <div className="settings-form-actions">
            <button
              type="button"
              className="primary-btn"
              onClick={save}
              disabled={!canSave}
              data-testid="settings-vision-save"
            >
              <Icon name="check" size={11} />{' '}
              {updateMutation.isPending ? 'Saving…' : 'Save'}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function errorToMessage(err: unknown): string | null {
  if (err === null || err === undefined) return null;
  if (err instanceof ApiError) {
    const detail = (err.body as { detail?: string } | undefined)?.detail;
    return (
      (typeof detail === 'string' ? detail : null) ||
      userErrorMessage(err, { action: 'saving the vision settings' })
    );
  }
  if (typeof err === 'string') return err;
  return userErrorMessage(err, { action: 'saving the vision settings' });
}
