/**
 * Settings → API keys section.
 *
 * Lists per-operator API keys (one row per key, revoked included so the
 * audit trail stays visible). Two write actions:
 *
 *   - Create: opens a small form modal. On success, the modal swaps to
 *     a one-time-reveal panel showing the full value with copy-to-clip
 *     and a loud warning that the value will never be shown again.
 *   - Revoke: row button with a double-confirm pattern (one click sets
 *     the row into confirm state; the confirm click issues DELETE).
 *
 * The static infrastructure key (env `LIGHTRAG_API_KEY`) is **not**
 * managed here — it remains a deploy-env secret, intentionally
 * invisible from the UI.
 */

import { useState } from 'react';
import { Icon } from '../Icon';
import { ApiError } from '../../api/client';
import {
  useApiKeys,
  useCreateApiKey,
  useRevokeApiKey,
} from '../../api/queries';
import type { ApiKeyCreated, ApiKeyPublic } from '../../types/apiKey';
import { relativeTime } from '../../utils/relativeTime';

function formatMs(ms: number | null | undefined): string {
  if (!ms) return 'never';
  return relativeTime(new Date(ms).toISOString());
}

function isApiError(err: unknown): err is ApiError {
  return err instanceof ApiError;
}

function errorMessage(err: unknown, fallback: string): string {
  if (isApiError(err) && typeof err.body === 'object' && err.body) {
    const detail = (err.body as { detail?: string }).detail;
    if (detail) return detail;
  }
  if (err instanceof Error) return err.message;
  return fallback;
}

export interface ApiKeysSectionProps {
  /** Hook seam — tests inject deterministic copy behaviour. */
  copyToClipboard?: (value: string) => Promise<void>;
}

export function ApiKeysSection({
  copyToClipboard,
}: ApiKeysSectionProps = {}) {
  const { data, isLoading, isError, error, refetch } = useApiKeys();
  const createMutation = useCreateApiKey();
  const revokeMutation = useRevokeApiKey();

  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState('');
  const [createError, setCreateError] = useState<string | null>(null);
  const [revealed, setRevealed] = useState<ApiKeyCreated | null>(null);
  const [confirmingRevoke, setConfirmingRevoke] = useState<string | null>(null);
  const [revokeError, setRevokeError] = useState<string | null>(null);
  const [copyState, setCopyState] = useState<'idle' | 'ok' | 'err'>('idle');

  const keys: readonly ApiKeyPublic[] = data ?? [];

  const openCreate = (): void => {
    setNewName('');
    setCreateError(null);
    setCreateOpen(true);
  };

  const closeCreate = (): void => {
    setCreateOpen(false);
    setNewName('');
    setCreateError(null);
  };

  const submitCreate = async (): Promise<void> => {
    const name = newName.trim();
    if (!name) {
      setCreateError('Name is required.');
      return;
    }
    if (name.length > 120) {
      setCreateError('Name is too long (max 120 characters).');
      return;
    }
    try {
      const created = await createMutation.mutateAsync({ name });
      setCreateOpen(false);
      setRevealed(created);
      setCopyState('idle');
    } catch (err) {
      setCreateError(errorMessage(err, 'Could not create the API key.'));
    }
  };

  const dismissReveal = (): void => {
    setRevealed(null);
    setCopyState('idle');
  };

  const copyValue = async (): Promise<void> => {
    if (!revealed) return;
    const copier =
      copyToClipboard ??
      (async (v: string) => {
        await navigator.clipboard.writeText(v);
      });
    try {
      await copier(revealed.full_value);
      setCopyState('ok');
    } catch {
      setCopyState('err');
    }
  };

  const startRevoke = (id: string): void => {
    setConfirmingRevoke(id);
    setRevokeError(null);
  };

  const cancelRevoke = (): void => {
    setConfirmingRevoke(null);
    setRevokeError(null);
  };

  const confirmRevoke = async (id: string): Promise<void> => {
    try {
      await revokeMutation.mutateAsync(id);
      setConfirmingRevoke(null);
    } catch (err) {
      setRevokeError(errorMessage(err, 'Could not revoke the key.'));
    }
  };

  return (
    <div
      className="settings-section settings-api-keys"
      data-testid="settings-api-keys"
    >
      <h3>API keys</h3>
      <p className="muted">
        Per-operator API keys for programmatic access to this folder. The
        full value is shown <strong>once</strong> at creation — copy it
        immediately. The static infrastructure key set via the deploy
        environment is not listed here.
      </p>

      <div className="settings-api-keys-actions">
        <button
          type="button"
          className="primary-btn"
          onClick={openCreate}
          disabled={createMutation.isPending}
          data-testid="settings-api-keys-create-btn"
        >
          <Icon name="plus" size={12} /> Create new key
        </button>
      </div>

      {isLoading && (
        <div className="muted" data-testid="settings-api-keys-loading">
          Loading API keys…
        </div>
      )}
      {isError && (
        <div
          className="error-banner"
          role="alert"
          data-testid="settings-api-keys-error"
        >
          Could not load API keys
          {error instanceof Error ? ` — ${error.message}` : ''}.{' '}
          <button
            type="button"
            className="ghost-btn"
            onClick={() => refetch()}
          >
            Retry
          </button>
        </div>
      )}
      {!isLoading && !isError && keys.length === 0 && (
        <div className="muted" data-testid="settings-api-keys-empty">
          No API keys yet. Create one to allow programmatic access.
        </div>
      )}
      {keys.length > 0 && (
        <table
          className="settings-api-keys-table"
          data-testid="settings-api-keys-table"
        >
          <thead>
            <tr>
              <th>Name</th>
              <th>Prefix</th>
              <th>Created</th>
              <th>Last used</th>
              <th>Status</th>
              <th className="actions" aria-label="Actions"></th>
            </tr>
          </thead>
          <tbody>
            {keys.map((k) => {
              const isRevoked = k.revoked_at !== null;
              const isConfirming = confirmingRevoke === k.id;
              return (
                <tr
                  key={k.id}
                  className={isRevoked ? 'is-revoked' : undefined}
                  data-testid={`settings-api-keys-row-${k.id}`}
                >
                  <td>{k.name}</td>
                  <td
                    className="settings-api-keys-prefix"
                    data-testid={`settings-api-keys-prefix-${k.id}`}
                  >
                    <span className="mono settings-api-keys-prefix-value">
                      {k.prefix}
                    </span>
                  </td>
                  <td title={new Date(k.created_at).toISOString()}>
                    {formatMs(k.created_at)}
                  </td>
                  <td>{formatMs(k.last_used_at)}</td>
                  <td>
                    {isRevoked ? (
                      <span className="status-pill revoked">revoked</span>
                    ) : (
                      <span className="status-pill active">active</span>
                    )}
                  </td>
                  <td className="actions">
                    {!isRevoked && !isConfirming && (
                      <button
                        type="button"
                        className="ghost-btn danger"
                        onClick={() => startRevoke(k.id)}
                        data-testid={`settings-api-keys-revoke-${k.id}`}
                      >
                        Revoke
                      </button>
                    )}
                    {!isRevoked && isConfirming && (
                      <>
                        <button
                          type="button"
                          className="primary-btn danger"
                          onClick={() => confirmRevoke(k.id)}
                          disabled={revokeMutation.isPending}
                          data-testid={`settings-api-keys-revoke-confirm-${k.id}`}
                        >
                          Confirm revoke
                        </button>
                        <button
                          type="button"
                          className="ghost-btn"
                          onClick={cancelRevoke}
                          disabled={revokeMutation.isPending}
                          data-testid={`settings-api-keys-revoke-cancel-${k.id}`}
                        >
                          Cancel
                        </button>
                      </>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {revokeError && (
        <div
          className="error-banner"
          role="alert"
          data-testid="settings-api-keys-revoke-error"
        >
          {revokeError}
        </div>
      )}

      {createOpen && (
        <div
          className="modal-backdrop api-key-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="api-key-create-title"
          data-testid="settings-api-keys-create-backdrop"
        >
          <div className="modal modal-small api-key-modal">
            <div className="modal-header api-key-modal-header">
              <div>
                <h2 id="api-key-create-title">Create API key</h2>
                <p className="ctx">
                  You will see the full value once, then only its prefix.
                </p>
              </div>
            </div>
            <div className="modal-body api-key-modal-body">
              <label className="field-label" htmlFor="api-key-name">
                Name
              </label>
              <input
                id="api-key-name"
                type="text"
                value={newName}
                onChange={(e) => {
                  setNewName(e.target.value);
                  if (createError) setCreateError(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    void submitCreate();
                  }
                }}
                placeholder="ingestion-agent"
                maxLength={120}
                autoFocus
                data-testid="settings-api-keys-create-name"
              />
              {createError && (
                <div
                  className="inline-error"
                  role="alert"
                  data-testid="settings-api-keys-create-error"
                >
                  {createError}
                </div>
              )}
            </div>
            <div className="modal-footer api-key-modal-footer">
              <button
                type="button"
                className="ghost-btn"
                onClick={closeCreate}
                disabled={createMutation.isPending}
                data-testid="settings-api-keys-create-cancel"
              >
                Cancel
              </button>
              <button
                type="button"
                className="primary-btn"
                onClick={() => void submitCreate()}
                disabled={createMutation.isPending || !newName.trim()}
                data-testid="settings-api-keys-create-submit"
              >
                {createMutation.isPending ? 'Creating…' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      )}

      {revealed && (
        <div
          className="modal-backdrop api-key-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-labelledby="api-key-reveal-title"
          data-testid="settings-api-keys-reveal-backdrop"
        >
          <div className="modal modal-small api-key-modal api-key-reveal-modal">
            <div className="modal-header api-key-modal-header">
              <div>
                <h2 id="api-key-reveal-title">
                  <Icon name="lock" size={14} /> Your new API key
                </h2>
                <p className="ctx">Copy the secret before closing this dialog.</p>
              </div>
            </div>
            <div className="modal-body api-key-modal-body">
              <p className="warning-banner" role="alert">
                <strong>This value will not be shown again.</strong> Copy it
                now and store it somewhere safe.
              </p>
              <div className="api-key-reveal-field">
                <label className="field-label">Name</label>
                <div
                  className="api-key-reveal-name"
                  data-testid="settings-api-keys-reveal-name"
                >
                  {revealed.name}
                </div>
              </div>
              <div className="api-key-reveal-field">
                <label className="field-label">Full value</label>
                <div
                  className="mono settings-api-keys-reveal-value"
                  data-testid="settings-api-keys-reveal-value"
                >
                  {revealed.full_value}
                </div>
              </div>
            </div>
            <div className="modal-footer api-key-modal-footer">
              <button
                type="button"
                className="ghost-btn"
                onClick={() => void copyValue()}
                data-testid="settings-api-keys-reveal-copy"
              >
                {copyState === 'ok' && (
                  <>
                    <Icon name="check" size={12} /> Copied
                  </>
                )}
                {copyState === 'err' && 'Copy failed'}
                {copyState === 'idle' && 'Copy to clipboard'}
              </button>
              <button
                type="button"
                className="primary-btn"
                onClick={dismissReveal}
                data-testid="settings-api-keys-reveal-dismiss"
              >
                I've stored the key
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
