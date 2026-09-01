import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { AuthorizeDialog } from './AuthorizeDialog';

describe('AuthorizeDialog', () => {
  it('normalizes and saves a non-empty token', async () => {
    const onSave = vi.fn();
    render(
      <AuthorizeDialog
        token=""
        onSave={onSave}
        onLogout={() => {}}
        onClose={() => {}}
      />,
    );

    await userEvent.type(screen.getByLabelText('Value'), '  bearer-value  ');
    await userEvent.click(screen.getByRole('button', { name: 'Authorize' }));

    expect(onSave).toHaveBeenCalledWith('bearer-value');
  });

  it('requires a second explicit click before revoking a session token', async () => {
    const onLogout = vi.fn();
    render(
      <AuthorizeDialog
        token="bearer-value"
        onSave={() => {}}
        onLogout={onLogout}
        onClose={() => {}}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Revoke token' }));
    expect(onLogout).not.toHaveBeenCalled();
    await userEvent.click(
      screen.getByRole('button', { name: 'Confirm revoke token' }),
    );
    expect(onLogout).toHaveBeenCalledOnce();
  });
});
