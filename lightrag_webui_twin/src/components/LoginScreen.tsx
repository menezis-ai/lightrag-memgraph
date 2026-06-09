import { useState } from 'react';
import { Icon } from './Icon';

export interface LoginScreenProps {
  checking?: boolean;
  error?: string | null;
  onLogin: (username: string, password: string) => Promise<void>;
}

export function LoginScreen({
  checking = false,
  error = null,
  onLogin,
}: LoginScreenProps) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [pending, setPending] = useState(false);
  const canSubmit = username.trim().length > 0 && password.length > 0 && !pending;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setPending(true);
    try {
      await onLogin(username.trim(), password);
    } finally {
      setPending(false);
    }
  };

  return (
    <main className="login-shell" data-testid="login-screen">
      <section className="login-panel" aria-labelledby="login-title">
        <div className="login-brand">
          <div className="brand-mark">TR</div>
          <div>
            <h1 id="login-title">Twin</h1>
            <p>Knowledge console</p>
          </div>
        </div>

        {checking ? (
          <div className="login-checking" data-testid="login-checking">
            <Icon name="loader-2" size={16} /> Checking session
          </div>
        ) : (
          <form className="login-form" onSubmit={submit}>
            <label>
              <span>Username</span>
              <input
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                autoFocus
                data-testid="login-username"
              />
            </label>
            <label>
              <span>Password</span>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
                data-testid="login-password"
              />
            </label>
            {error && (
              <div className="login-error" role="alert" data-testid="login-error">
                {error}
              </div>
            )}
            <button
              type="submit"
              className="login-submit"
              disabled={!canSubmit}
              data-testid="login-submit"
            >
              {pending ? 'Signing in' : 'Sign in'}
            </button>
          </form>
        )}
      </section>
    </main>
  );
}
