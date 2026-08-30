import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LoginScreen } from './LoginScreen';

describe('LoginScreen', () => {
  it('submits credentials after both fields are filled', async () => {
    const onLogin = vi.fn().mockResolvedValue(undefined);
    render(<LoginScreen onLogin={onLogin} />);

    const submit = screen.getByTestId('login-submit');
    expect(submit).toBeDisabled();

    await userEvent.type(screen.getByTestId('login-username'), ' twinadmin ');
    await userEvent.type(screen.getByTestId('login-password'), 'secret');
    await userEvent.click(submit);

    expect(onLogin).toHaveBeenCalledWith('twinadmin', 'secret');
  });

  it('uses the Twin KMS product name in the login brand', () => {
    render(<LoginScreen onLogin={vi.fn()} />);
    expect(
      screen.getByRole('heading', { name: 'Twin KMS' }),
    ).toBeInTheDocument();
  });

  it('renders the session checking state', () => {
    render(<LoginScreen checking onLogin={vi.fn()} />);

    expect(screen.getByTestId('login-checking')).toHaveTextContent(
      'Checking session',
    );
    expect(screen.queryByTestId('login-submit')).toBeNull();
  });

  it('consumes a rejected login while preserving the hook-owned error and resetting pending', async () => {
    const onLogin = vi.fn().mockRejectedValue(new Error('invalid credentials'));
    render(
      <LoginScreen
        error="Incorrect username or password."
        onLogin={onLogin}
      />,
    );

    await userEvent.type(screen.getByTestId('login-username'), 'twinadmin');
    await userEvent.type(screen.getByTestId('login-password'), 'wrong');
    await userEvent.click(screen.getByTestId('login-submit'));

    await waitFor(() => expect(screen.getByTestId('login-submit')).toBeEnabled());
    expect(screen.getByTestId('login-submit')).toHaveTextContent('Sign in');
    expect(screen.getByTestId('login-error')).toHaveTextContent(
      'Incorrect username or password.',
    );
  });
});
