import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
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

  it('uses the TwinRAG product name in the login brand', () => {
    render(<LoginScreen onLogin={vi.fn()} />);
    expect(
      screen.getByRole('heading', { name: 'TwinRAG' }),
    ).toBeInTheDocument();
  });

  it('renders the session checking state', () => {
    render(<LoginScreen checking onLogin={vi.fn()} />);

    expect(screen.getByTestId('login-checking')).toHaveTextContent(
      'Checking session',
    );
    expect(screen.queryByTestId('login-submit')).toBeNull();
  });
});
