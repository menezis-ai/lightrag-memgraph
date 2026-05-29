/**
 * OnboardingWizard — 6-step first-touch flow for new operators.
 *
 * Steps:
 *   1. welcome       — splash with the Twin name + one-line value-prop
 *   2. kb-empty      — explanation of an empty KB + CTA "Add source"
 *   3. checklist     — 5-task checklist persisted via useOnboarding
 *   4. first-source  — guided source upload pointer (hands off to AddSource modal)
 *   5. first-query   — guided query pointer (hands off to RetrievalTab)
 *   6. completion    — congrats + "Done" closes the wizard for good
 *
 * The wizard is a modal overlay. Skip is always available on every step. The
 * host (App.tsx) decides when to show it: `useOnboarding().state.dismissed`
 * + "show wizard if it's the user's first visit to the WebUI".
 */

import { useOnboarding } from '../hooks/useOnboarding';
import { Icon } from './Icon';

export interface OnboardingWizardProps {
  open: boolean;
  onAddSource?: () => void;
  onGoToRetrieval?: () => void;
  onDone?: () => void;
}

export function OnboardingWizard({
  open,
  onAddSource,
  onGoToRetrieval,
  onDone,
}: OnboardingWizardProps) {
  const { state, next, prev, toggleTask, dismiss } = useOnboarding();

  if (!open || state.dismissed) return null;

  const close = () => {
    dismiss();
    onDone?.();
  };

  return (
    <div
      className="modal-backdrop"
      onClick={close}
      data-testid="onboarding-backdrop"
    >
      <div
        className="modal onboarding"
        role="dialog"
        aria-modal="true"
        aria-label="Onboarding"
        style={{ width: 540 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-header">
          <h2>Welcome to Twin</h2>
          <button
            type="button"
            className="icon-btn"
            onClick={close}
            aria-label="Close"
          >
            <Icon name="x" size={16} />
          </button>
        </div>

        <div
          className="modal-body"
          data-testid={`onboarding-step-${state.step}`}
        >
          {state.step === 'welcome' && (
            <div>
              <p>
                Twin is your <strong>knowledge management hub</strong>: it
                unifies retrieval, lineage, and audit across your sources.
              </p>
              <p className="muted">
                This 6-step tour takes ~2 minutes. You can skip and revisit
                later from Settings.
              </p>
            </div>
          )}

          {state.step === 'kb-empty' && (
            <div>
              <h3>Your knowledge base is empty</h3>
              <p>
                Twin doesn't ingest anything until you tell it to. Add your
                first source to see retrieval, tagging, and lineage light up.
              </p>
            </div>
          )}

          {state.step === 'checklist' && (
            <div>
              <h3>5-task starter checklist</h3>
              <ul className="onboarding-checklist">
                {state.tasks.map((task) => (
                  <li key={task.id}>
                    <label>
                      <input
                        type="checkbox"
                        checked={task.done}
                        onChange={() => toggleTask(task.id)}
                        data-testid={`onboarding-task-${task.id}`}
                      />
                      <span
                        style={{
                          textDecoration: task.done ? 'line-through' : 'none',
                          opacity: task.done ? 0.6 : 1,
                        }}
                      >
                        {task.label}
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {state.step === 'first-source' && (
            <div>
              <h3>Add your first source</h3>
              <p>
                Twin ingests files, Confluence pages, SharePoint paths, and
                URLs. The Add Source modal walks you through tagging on
                upload.
              </p>
              <button
                type="button"
                className="btn primary"
                onClick={() => {
                  dismiss();
                  onAddSource?.();
                }}
                data-testid="onboarding-add-source"
              >
                <Icon name="cloud-upload" size={13} /> Open Add Source
              </button>
            </div>
          )}

          {state.step === 'first-query' && (
            <div>
              <h3>Run your first retrieval query</h3>
              <p>
                The Retrieval tab is the operator surface for Twin's hybrid
                RAG. Try a question on a source you've just uploaded.
              </p>
              <button
                type="button"
                className="btn primary"
                onClick={() => {
                  dismiss();
                  onGoToRetrieval?.();
                }}
                data-testid="onboarding-go-retrieval"
              >
                <Icon name="search" size={13} /> Open Retrieval
              </button>
            </div>
          )}

          {state.step === 'completion' && (
            <div>
              <h3>You're set</h3>
              <p>
                Twin will keep the audit trail running in the background. Visit
                the Activity tab to see what happens after every action.
              </p>
              <p className="muted">
                You can replay this tour from Settings → Profile.
              </p>
            </div>
          )}
        </div>

        <div className="modal-footer onboarding-footer">
          <button
            type="button"
            className="link-btn"
            onClick={close}
            data-testid="onboarding-skip"
          >
            Skip
          </button>
          <div className="actions">
            {state.step !== 'welcome' && (
              <button
                type="button"
                className="btn"
                onClick={prev}
                data-testid="onboarding-prev"
              >
                Back
              </button>
            )}
            {state.step !== 'completion' ? (
              <button
                type="button"
                className="btn primary"
                onClick={next}
                data-testid="onboarding-next"
              >
                Next
              </button>
            ) : (
              <button
                type="button"
                className="btn primary"
                onClick={close}
                data-testid="onboarding-done"
              >
                Done
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
