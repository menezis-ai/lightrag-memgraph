/**
 * Settings → About section.
 *
 * Read-only runtime identity card, backed by `GET /twin/api/system/about`.
 * Its reason to exist is debugging: answering "which build, against which
 * Memgraph, on which Python" without an SSH session.
 *
 * The payload is two-tier. Every caller served by the backend sees the
 * software versions; the deployment-shape blocks (Memgraph, runtime,
 * storage, overlay) arrive only for admins and are `null` otherwise. This
 * component must therefore render a valid, shorter card for a non-admin —
 * absent blocks are skipped, never shown as empty rows.
 *
 * The copy button is the actual ergonomic payload: it puts the whole JSON
 * on the clipboard so it can be pasted into a ticket verbatim.
 */

import { useState } from 'react';
import { Icon } from '../Icon';
import { useAbout } from '../../api/queries';
import { userErrorMessage } from '../../lib/errorMessages';
import type { MemgraphInfo } from '../../types/systemInfo';

/**
 * MAGE is tri-state. `null` means the capability probe could not answer —
 * rendering that as "floor tier" would assert an absence the backend never
 * established, and send the operator debugging a missing MAGE that may well
 * be installed.
 */
function mageLabel(memgraph: MemgraphInfo): string {
  if (memgraph.mage === null) return 'unknown — capability probe unavailable';
  if (memgraph.procedures === null) {
    return memgraph.mage
      ? 'available — configured override'
      : 'not available — configured floor tier';
  }
  if (!memgraph.mage) return 'not available — floor tier';
  return `available (${memgraph.procedures} procedures)`;
}

function Row({ label, value }: Readonly<{ label: string; value: string }>) {
  return (
    <>
      <dt>{label}</dt>
      <dd className="mono">{value}</dd>
    </>
  );
}

export function AboutSection() {
  const { data, isLoading, isError, error } = useAbout();
  const [copyState, setCopyState] = useState<'idle' | 'ok' | 'err'>('idle');

  const onCopy = async () => {
    if (!data) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      setCopyState('ok');
    } catch {
      // Clipboard can be denied by permissions policy. Say so rather than
      // failing silently — the operator is mid-ticket and needs to know the
      // paste will be empty.
      setCopyState('err');
    }
    window.setTimeout(() => setCopyState('idle'), 2000);
  };

  if (isLoading) {
    return (
      <div className="settings-section" data-testid="settings-about">
        <h3>About</h3>
        <p className="muted">Loading runtime information…</p>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="settings-section" data-testid="settings-about">
        <h3>About</h3>
        <p className="muted" data-testid="settings-about-error">
          {userErrorMessage(error)}
        </p>
      </div>
    );
  }

  const { lightrag, memgraph, runtime, storage, overlay } = data;

  return (
    <div className="settings-section" data-testid="settings-about">
      <h3>About</h3>
      <p className="muted">
        Runtime identity of this instance. Include it when reporting an issue.
      </p>

      <div className="set-card">
        <div className="set-card-h">
          Versions
          <button
            type="button"
            className="ghost-btn"
            onClick={onCopy}
            data-testid="settings-about-copy"
          >
            <Icon name="file-text" size={12} />{' '}
            {copyState === 'ok' && 'Copied'}
            {copyState === 'err' && 'Copy failed'}
            {copyState === 'idle' && 'Copy details'}
          </button>
        </div>
        <dl className="set-dl">
          <Row label="Twin KMS" value={data.twin} />
          <Row label="LightRAG" value={lightrag.native ?? 'unknown'} />
          {lightrag.composite && (
            <Row label="Composite build" value={lightrag.composite} />
          )}
        </dl>
      </div>

      {memgraph && (
        <div className="set-card">
          <div className="set-card-h">Memgraph</div>
          <dl className="set-dl">
            {memgraph.reachable ? (
              <Row label="Version" value={memgraph.version ?? 'unknown'} />
            ) : (
              <Row
                label="Status"
                value={`unreachable${memgraph.error ? ` (${memgraph.error})` : ''}`}
              />
            )}
            <Row label="MAGE tier" value={mageLabel(memgraph)} />
          </dl>
        </div>
      )}

      {runtime && (
        <div className="set-card">
          <div className="set-card-h">Runtime</div>
          <dl className="set-dl">
            <Row
              label="Python"
              value={`${runtime.python} (${runtime.implementation})`}
            />
            <Row label="Platform" value={runtime.platform} />
          </dl>
        </div>
      )}

      {storage && Object.keys(storage).length > 0 && (
        <div className="set-card">
          <div className="set-card-h">Storage backends</div>
          <dl className="set-dl">
            {Object.entries(storage).map(([slot, cls]) => (
              <Row key={slot} label={slot} value={cls} />
            ))}
          </dl>
        </div>
      )}

      {overlay && (
        <div className="set-card">
          <div className="set-card-h">
            Overlay{' '}
            <span className="env-badge">
              <Icon name="lock" size={10} /> env-controlled
            </span>
          </div>
          <dl className="set-dl">
            {Object.entries(overlay).map(([flag, on]) => (
              <Row key={flag} label={flag} value={on ? 'on' : 'off'} />
            ))}
          </dl>
        </div>
      )}

      {!data.admin && (
        <p className="muted" data-testid="settings-about-reduced">
          Deployment details are available to administrators only.
        </p>
      )}
    </div>
  );
}
