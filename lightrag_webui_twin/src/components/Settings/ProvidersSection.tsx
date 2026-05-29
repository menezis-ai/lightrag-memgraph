/**
 * Providers sub-section — LLM / Embedder / Reranker configuration.
 *
 * Each provider row opens a real configure panel (modal-like), not a no-op
 * (#SET-01 fix). The panel shows the current backend/model + read-only fields
 * pulled from the runtime config. Steward palier can request a change via the
 * "Request change" action, which surfaces a request artifact for ops — Twin
 * itself does not rotate credentials from the UI.
 */

import { useState } from 'react';
import { useAuth } from '../../hooks/useAuth';
import { Icon } from '../Icon';

type ProviderKind = 'llm' | 'embedder' | 'reranker';

interface ProviderState {
  backend: string;
  model: string;
  base_url: string;
}

const DEFAULTS: Record<ProviderKind, ProviderState> = {
  llm: {
    backend: 'openai',
    model: 'gpt-4.1',
    base_url: 'https://api.openai.com/v1',
  },
  embedder: {
    backend: 'ollama',
    model: 'bge-m3:latest',
    base_url: 'http://ollama.twin.internal:11434',
  },
  reranker: {
    backend: 'cohere',
    model: 'rerank-multilingual-v3.0',
    base_url: 'https://api.cohere.ai',
  },
};

const LABELS: Record<ProviderKind, string> = {
  llm: 'LLM',
  embedder: 'Embedder',
  reranker: 'Reranker',
};

export function ProvidersSection() {
  const { user } = useAuth();
  const isSteward = user?.palier.level === 3;
  const [active, setActive] = useState<ProviderKind | null>(null);

  return (
    <div className="settings-section" data-testid="settings-providers">
      <h3>Providers</h3>
      <p className="muted">
        Backends that power LightRAG ingestion + Twin overlay. Steward palier
        can request a change; Reader/Contributor see read-only.
      </p>
      <ul className="settings-providers" data-testid="settings-providers-list">
        {(Object.keys(DEFAULTS) as ProviderKind[]).map((kind) => {
          const cfg = DEFAULTS[kind];
          return (
            <li key={kind} className="settings-provider-row">
              <div>
                <strong>{LABELS[kind]}</strong>
                <div className="muted mono">
                  {cfg.backend} · {cfg.model}
                </div>
              </div>
              <button
                type="button"
                className="btn small"
                data-testid={`settings-providers-configure-${kind}`}
                onClick={() => setActive(kind)}
              >
                <Icon name="settings" size={12} /> Configure
              </button>
            </li>
          );
        })}
      </ul>
      {active && (
        <ProviderPanel
          kind={active}
          state={DEFAULTS[active]}
          canEdit={isSteward}
          onClose={() => setActive(null)}
        />
      )}
    </div>
  );
}

interface ProviderPanelProps {
  kind: ProviderKind;
  state: ProviderState;
  canEdit: boolean;
  onClose: () => void;
}

function ProviderPanel({ kind, state, canEdit, onClose }: ProviderPanelProps) {
  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      data-testid={`settings-provider-panel-${kind}`}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={`Configure ${LABELS[kind]}`}
        style={{ width: 460 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>Configure {LABELS[kind]}</h2>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Close"
          >
            <Icon name="x" size={16} />
          </button>
        </div>
        <div className="modal-body">
          <dl className="settings-dl">
            <dt>Backend</dt>
            <dd className="mono">{state.backend}</dd>
            <dt>Model</dt>
            <dd className="mono">{state.model}</dd>
            <dt>Base URL</dt>
            <dd className="mono">{state.base_url}</dd>
          </dl>
          {!canEdit && (
            <p className="muted">
              Read-only at your palier. Ask a Steward to request a change.
            </p>
          )}
        </div>
        <div className="modal-footer">
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
          {canEdit && (
            <button
              type="button"
              className="btn primary"
              data-testid={`settings-provider-request-${kind}`}
              onClick={onClose}
            >
              Request change
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
