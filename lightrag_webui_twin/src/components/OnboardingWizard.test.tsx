/**
 * Unit tests for OnboardingWizard + useOnboarding state machine.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { renderHook, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { OnboardingWizard } from './OnboardingWizard';
import { useOnboarding } from '../hooks/useOnboarding';

beforeEach(() => {
  window.localStorage.clear();
});

describe('useOnboarding', () => {
  it('starts at welcome with 5 undone tasks', () => {
    const { result } = renderHook(() => useOnboarding());
    expect(result.current.state.step).toBe('welcome');
    expect(result.current.state.tasks).toHaveLength(5);
    expect(result.current.state.tasks.every((t) => !t.done)).toBe(true);
  });

  it('next() walks the 6 steps in order', () => {
    const { result } = renderHook(() => useOnboarding());
    const order = [
      'welcome',
      'kb-empty',
      'checklist',
      'first-source',
      'first-query',
      'completion',
    ] as const;
    order.forEach((expected, i) => {
      expect(result.current.state.step).toBe(expected);
      if (i < order.length - 1) act(() => result.current.next());
    });
  });

  it('toggleTask flips a task done flag', () => {
    const { result } = renderHook(() => useOnboarding());
    act(() => result.current.toggleTask('add-source'));
    expect(
      result.current.state.tasks.find((t) => t.id === 'add-source')?.done,
    ).toBe(true);
  });

  it('persists state through localStorage between hook instances', () => {
    const a = renderHook(() => useOnboarding());
    act(() => a.result.current.next());
    act(() => a.result.current.next());
    const b = renderHook(() => useOnboarding());
    expect(b.result.current.state.step).toBe('checklist');
  });

  it('dismiss flips the dismissed flag', () => {
    const { result } = renderHook(() => useOnboarding());
    act(() => result.current.dismiss());
    expect(result.current.state.dismissed).toBe(true);
  });
});

describe('OnboardingWizard', () => {
  it('renders nothing when closed', () => {
    const { container } = render(<OnboardingWizard open={false} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the welcome step when open', () => {
    render(<OnboardingWizard open />);
    expect(screen.getByTestId('onboarding-step-welcome')).toBeInTheDocument();
  });

  it('Next walks forward through steps; Back walks back', async () => {
    render(<OnboardingWizard open />);
    await userEvent.click(screen.getByTestId('onboarding-next'));
    expect(screen.getByTestId('onboarding-step-kb-empty')).toBeInTheDocument();
    await userEvent.click(screen.getByTestId('onboarding-prev'));
    expect(screen.getByTestId('onboarding-step-welcome')).toBeInTheDocument();
  });

  it('Add Source CTA triggers the callback and dismisses the wizard', async () => {
    const onAddSource = vi.fn();
    render(<OnboardingWizard open onAddSource={onAddSource} />);
    // Walk to the first-source step (welcome → kb-empty → checklist → first-source)
    await userEvent.click(screen.getByTestId('onboarding-next'));
    await userEvent.click(screen.getByTestId('onboarding-next'));
    await userEvent.click(screen.getByTestId('onboarding-next'));
    await userEvent.click(screen.getByTestId('onboarding-add-source'));
    expect(onAddSource).toHaveBeenCalledTimes(1);
  });

  it('Skip dismisses the wizard', async () => {
    const onDone = vi.fn();
    render(<OnboardingWizard open onDone={onDone} />);
    await userEvent.click(screen.getByTestId('onboarding-skip'));
    expect(onDone).toHaveBeenCalled();
  });
});
