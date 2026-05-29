/**
 * useOnboarding — local persistent state machine for the 6-step wizard.
 *
 * State is persisted in localStorage under `twin.onboarding.v1` so an operator
 * doesn't get the welcome modal every refresh once they've started. The hook
 * exposes:
 *   - step: current step ('welcome' | 'kb-empty' | 'checklist' | 'first-source'
 *           | 'first-query' | 'completion')
 *   - dismissed: true after user clicks "Done" or "Skip"
 *   - tasks: checklist of 5 first-touch tasks
 *
 * The wizard is intentionally non-blocking: it overlays the app shell but the
 * user can dismiss it at any point. The data model lives entirely on the
 * client — there is no backend endpoint for onboarding (yet).
 */

import { useCallback, useEffect, useState } from 'react';

export type OnboardingStep =
  | 'welcome'
  | 'kb-empty'
  | 'checklist'
  | 'first-source'
  | 'first-query'
  | 'completion';

export interface OnboardingTask {
  id: string;
  label: string;
  done: boolean;
}

export interface OnboardingState {
  step: OnboardingStep;
  dismissed: boolean;
  tasks: OnboardingTask[];
}

const STORAGE_KEY = 'twin.onboarding.v1';

const DEFAULT_TASKS: OnboardingTask[] = [
  { id: 'add-source', label: 'Add your first source', done: false },
  { id: 'first-query', label: 'Run your first retrieval query', done: false },
  { id: 'apply-tag', label: 'Apply a tag to a document', done: false },
  { id: 'view-graph', label: 'Open the Graph tab and explore entities', done: false },
  { id: 'check-audit', label: 'Check the audit trail for your actions', done: false },
];

const INITIAL_STATE: OnboardingState = {
  step: 'welcome',
  dismissed: false,
  tasks: DEFAULT_TASKS,
};

function read(): OnboardingState {
  if (typeof window === 'undefined') return INITIAL_STATE;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return INITIAL_STATE;
    const parsed = JSON.parse(raw) as Partial<OnboardingState>;
    return {
      step: parsed.step ?? 'welcome',
      dismissed: parsed.dismissed ?? false,
      tasks: parsed.tasks ?? DEFAULT_TASKS,
    };
  } catch {
    return INITIAL_STATE;
  }
}

function write(state: OnboardingState): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // ignore quota errors
  }
}

const STEP_ORDER: readonly OnboardingStep[] = [
  'welcome',
  'kb-empty',
  'checklist',
  'first-source',
  'first-query',
  'completion',
];

export interface UseOnboardingResult {
  state: OnboardingState;
  next: () => void;
  prev: () => void;
  goTo: (step: OnboardingStep) => void;
  toggleTask: (taskId: string) => void;
  dismiss: () => void;
  reset: () => void;
}

export function useOnboarding(): UseOnboardingResult {
  const [state, setState] = useState<OnboardingState>(() => read());

  useEffect(() => {
    write(state);
  }, [state]);

  const next = useCallback(() => {
    setState((s) => {
      const i = STEP_ORDER.indexOf(s.step);
      const nextStep = STEP_ORDER[Math.min(STEP_ORDER.length - 1, i + 1)];
      return { ...s, step: nextStep };
    });
  }, []);

  const prev = useCallback(() => {
    setState((s) => {
      const i = STEP_ORDER.indexOf(s.step);
      const prevStep = STEP_ORDER[Math.max(0, i - 1)];
      return { ...s, step: prevStep };
    });
  }, []);

  const goTo = useCallback((step: OnboardingStep) => {
    setState((s) => ({ ...s, step }));
  }, []);

  const toggleTask = useCallback((taskId: string) => {
    setState((s) => ({
      ...s,
      tasks: s.tasks.map((t) =>
        t.id === taskId ? { ...t, done: !t.done } : t,
      ),
    }));
  }, []);

  const dismiss = useCallback(() => {
    setState((s) => ({ ...s, dismissed: true }));
  }, []);

  const reset = useCallback(() => {
    setState(INITIAL_STATE);
  }, []);

  return { state, next, prev, goTo, toggleTask, dismiss, reset };
}
