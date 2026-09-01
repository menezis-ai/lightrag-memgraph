import { useRef, useState } from 'react';
import { useModalA11y } from '../../hooks/useModalA11y';
import { Icon } from '../Icon';

export interface AuthorizeDialogProps {
  token: string;
  onSave: (token: string) => void;
  onLogout: () => void;
  onClose: () => void;
}

export function AuthorizeDialog({ token, onSave, onLogout, onClose }: Readonly<AuthorizeDialogProps>) {
  const [value, setValue] = useState(token);
  const [revokeArmed, setRevokeArmed] = useState(false);
  const ref = useRef<HTMLDialogElement>(null);
  useModalA11y({ open: true, onClose, ref });
  return (
    <div className="modal-backdrop">
      <button type="button" className="modal-backdrop-dismiss" onClick={onClose} aria-label="Close authorize dialog" data-testid="authorize-backdrop" />
      <dialog open className="modal small" aria-modal="true" aria-labelledby="auth-title" ref={ref}>
        <div className="modal-header">
          <div style={{ flex: 1 }}>
            <h2 id="auth-title">Authorize</h2>
            <div className="ctx"><Icon name="lock" size={12} /> Bearer (HTTP, scheme: bearer)</div>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="x" size={18} /></button>
        </div>
        <div className="modal-body">
          <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
            Paste a bearer token (from <code>POST /login</code>) or an API key
            (Settings → API keys). It&apos;s attached to every request from &quot;Try it out&quot;.
          </p>
          <label className="field-label" htmlFor="auth-token-input" style={{ display: 'block', marginBottom: 6 }}>Value</label>
          <input
            id="auth-token-input" type="password" autoFocus value={value}
            onChange={(event) => { setValue(event.target.value); setRevokeArmed(false); }}
            placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9…"
            style={{ width: '100%', fontFamily: 'var(--font-mono)', fontSize: 12, padding: '8px 10px', borderRadius: 4, border: '0.5px solid var(--color-border-secondary)', background: 'var(--color-background-secondary)', color: 'var(--color-text-primary)' }}
          />
          <div style={{ marginTop: 14, fontSize: 11, color: 'var(--color-text-tertiary)' }}>
            Endpoints marked with a lock require authentication; admin endpoints additionally require an administrator identity.
          </div>
        </div>
        <div className="modal-footer">
          {token && (
            <button className="ghost-btn" onClick={() => { if (!revokeArmed) { setRevokeArmed(true); return; } onLogout(); }} aria-describedby={revokeArmed ? 'auth-revoke-confirm' : undefined}>
              {revokeArmed ? 'Confirm revoke token' : 'Revoke token'}
            </button>
          )}
          {revokeArmed && <span id="auth-revoke-confirm" className="muted" style={{ fontSize: 11 }}>Click again to remove the bearer token from this session.</span>}
          <button className="ghost-btn" onClick={onClose} style={{ marginLeft: 'auto' }}>Close</button>
          <button className="primary-btn" onClick={() => onSave(value.trim())} disabled={!value.trim()}>Authorize</button>
        </div>
      </dialog>
    </div>
  );
}
