import { useEffect, useMemo, useState, type FormEvent } from 'react';
import {
  useCreateLinkedSource,
  useDisableLinkedSource,
  useLinkedSources,
  usePreviewLinkedSource,
  useUpdateLinkedSource,
} from '../api/queries';
import type {
  CatalogPreview,
  LinkedSourceCreateInput,
  LinkedSourcePatchInput,
  RagDocType,
  RagLinkedSource,
} from '../types/linkedSource';
import type { Toast } from '../types/toast';
import {
  DISABLE_REASON,
  describeRagScope,
  inferRagSourceType,
} from '../lib/linkedSources';
import { logTechnicalError, userErrorMessage } from '../lib/errorMessages';

const COLUMNS: readonly { key: RagDocType; label: string }[] = [
  { key: 'di', label: 'DI' },
  { key: 'de', label: 'DE' },
  { key: 'sats', label: 'SATS' },
  { key: 'general', label: 'Confluence Space' },
];

type Draft = {
  linkId?: string;
  expectedVersion?: number;
  url: string;
  docType: RagDocType;
  publicChoice: '' | 'true' | 'false';
};

type PendingMutation =
  | { kind: 'create'; body: LinkedSourceCreateInput; preview: CatalogPreview }
  | {
      kind: 'patch';
      id: string;
      body: LinkedSourcePatchInput;
      preview: CatalogPreview;
    }
  | {
      kind: 'disable';
      id: string;
      expectedVersion: number;
      preview: CatalogPreview;
    };

const EMPTY_DRAFT: Draft = {
  url: '',
  docType: 'di',
  publicChoice: '',
};

function SourceCard({
  link,
  onEdit,
  onDisable,
  busy,
}: Readonly<{
  link: RagLinkedSource;
  onEdit: (link: RagLinkedSource) => void;
  onDisable: (link: RagLinkedSource) => void;
  busy: boolean;
}>) {
  const sourceType =
    link.source_type === 'sharepoint' || link.source_type === 'confluence'
      ? link.source_type
      : inferRagSourceType(link.url);
  const inactive = link.status === 'disabled' || link.status === 'deleted';
  return (
    <article className={`rag-source-card${inactive ? ' is-disabled' : ''}`}>
      <div className="rag-source-card-head">
        <span className={`rag-source-kind ${sourceType ?? ''}`}>
          {sourceType ?? 'unknown'}
        </span>
        <span className="rag-source-status">{link.status}</span>
      </div>
      <a href={link.url} target="_blank" rel="noreferrer" title={link.url}>
        {link.title || link.url}
      </a>
      <div className="rag-source-meta">
        {describeRagScope(link.url, link.doc_type) ?? 'Scope unavailable'} ·{' '}
        {link.public ? 'Public' : 'Restricted'}
      </div>
      <div className="rag-source-batch">Next nightly batch (~J+1)</div>
      {!inactive && (
        <div className="rag-source-actions">
          <button
            type="button"
            className="ghost-btn small"
            onClick={() => onEdit(link)}
            disabled={busy}
          >
            Edit
          </button>
          <button
            type="button"
            className="ghost-btn small danger"
            onClick={() => onDisable(link)}
            disabled={busy}
          >
            Disable
          </button>
        </div>
      )}
    </article>
  );
}

export interface SourcesRagTabProps {
  activeFolder: string;
  onToast?: (toast: Omit<Toast, 'id'>) => void;
}

export function SourcesRagTab({
  activeFolder,
  onToast,
}: Readonly<SourcesRagTabProps>) {
  const sources = useLinkedSources({ enabled: true, folderKey: activeFolder });
  const preview = usePreviewLinkedSource();
  const create = useCreateLinkedSource();
  const update = useUpdateLinkedSource();
  const disable = useDisableLinkedSource();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [pending, setPending] = useState<PendingMutation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const snapshot = sources.data;
  const application = snapshot?.application ?? null;
  const links = snapshot?.links;
  const byColumn = useMemo(
    () =>
      new Map(
        COLUMNS.map(({ key }) => [
          key,
          (links ?? []).filter((link) => link.doc_type === key),
        ]),
      ),
    [links],
  );
  const busy =
    preview.isPending || create.isPending || update.isPending || disable.isPending;
  const detectedSource = draft ? inferRagSourceType(draft.url) : null;
  const derivedScope = draft
    ? describeRagScope(draft.url, draft.docType)
    : null;

  useEffect(() => {
    if (sources.isError) {
      logTechnicalError('SourcesRagTab list', sources.error);
    }
  }, [sources.error, sources.isError]);

  const reportError = (caught: unknown, action: string) => {
    logTechnicalError('SourcesRagTab mutation', caught);
    setError(userErrorMessage(caught, { action }));
  };

  const editLink = (link: RagLinkedSource) => {
    setPending(null);
    setError(null);
    setDraft({
      linkId: link.id,
      expectedVersion: link.row_version,
      url: link.url,
      docType: link.doc_type,
      publicChoice: link.public ? 'true' : 'false',
    });
  };

  const requestPreview = async (event: FormEvent) => {
    event.preventDefault();
    if (!draft || draft.publicChoice === '') {
      setError('Choose Public or Restricted before previewing.');
      return;
    }
    setError(null);
    try {
      if (draft.linkId && draft.expectedVersion !== undefined) {
        const body: LinkedSourcePatchInput = {
          doc_type: draft.docType,
          public: draft.publicChoice === 'true',
          expected_version: draft.expectedVersion,
        };
        const result = await preview.mutateAsync({
          operation: 'patch',
          target_id: draft.linkId,
          body: { ...body },
        });
        setPending({ kind: 'patch', id: draft.linkId, body, preview: result });
      } else {
        const body: LinkedSourceCreateInput = {
          url: draft.url.trim(),
          doc_type: draft.docType,
          public: draft.publicChoice === 'true',
          status: 'active',
        };
        const result = await preview.mutateAsync({
          operation: 'create',
          body: { ...body },
        });
        setPending({ kind: 'create', body, preview: result });
      }
    } catch (caught) {
      reportError(caught, 'previewing the RAG source change');
    }
  };

  const requestDisable = async (link: RagLinkedSource) => {
    setDraft(null);
    setPending(null);
    setError(null);
    try {
      const result = await preview.mutateAsync({
        operation: 'transition',
        target_id: link.id,
        action: 'disable',
        body: {
          expected_version: link.row_version,
          reason: DISABLE_REASON,
        },
      });
      setPending({
        kind: 'disable',
        id: link.id,
        expectedVersion: link.row_version,
        preview: result,
      });
    } catch (caught) {
      reportError(caught, 'previewing the RAG source disable action');
    }
  };

  const confirmMutation = async () => {
    if (!pending) return;
    setError(null);
    try {
      if (pending.kind === 'create') await create.mutateAsync(pending.body);
      if (pending.kind === 'patch') {
        await update.mutateAsync({ id: pending.id, body: pending.body });
      }
      if (pending.kind === 'disable') {
        await disable.mutateAsync({
          id: pending.id,
          expectedVersion: pending.expectedVersion,
        });
      }
      const verb =
        pending.kind === 'create'
          ? 'declared'
          : pending.kind === 'patch'
            ? 'updated'
            : 'disabled';
      onToast?.({
        kind: 'done',
        title: `RAG source ${verb}`,
        sub: 'The change will reach the next nightly batch (~J+1).',
      });
      setPending(null);
      setDraft(null);
    } catch (caught) {
      const action =
        pending.kind === 'create'
          ? 'declaring the RAG source'
          : pending.kind === 'patch'
            ? 'updating the RAG source'
            : 'disabling the RAG source';
      reportError(caught, action);
    }
  };

  return (
    <section className="rag-sources" aria-labelledby="rag-sources-title">
      <header className="rag-sources-header">
        <div>
          <h1 id="rag-sources-title">Sources RAG</h1>
          <p>Folder {activeFolder} · declarations for the nightly RAG 1.5 batch</p>
        </div>
        <button
          type="button"
          className="primary-btn"
          onClick={() => {
            setPending(null);
            setError(null);
            setDraft({ ...EMPTY_DRAFT });
          }}
          disabled={!application || busy}
        >
          Add linked source
        </button>
      </header>

      <div className="rag-rules" role="note" aria-label="Linked source rules">
        <strong>ABSOLUTELY NO CONFIDENTIAL DOCUMENTS.</strong> Production documents
        only. Confluence pages or SharePoint documents only; SharePoint accepts
        docx, pdf, doc and docm. The pipeline technical user or Azure application
        must have READ permission. Link quality and maintenance remain the entity’s
        responsibility.
      </div>

      {sources.isLoading && <div className="rag-state">Loading catalogue…</div>}
      {sources.isError && (
        <div className="rag-state error" role="alert">
          {userErrorMessage(sources.error, { action: 'loading RAG sources' })}
        </div>
      )}
      {!sources.isLoading && !sources.isError && !application && (
        <div className="rag-state" role="status">
          This KB is not bound to an active application profile. Ask the catalogue
          administrator to bind its AUID before declaring sources.
        </div>
      )}

      {application && (
        <div className="rag-grid-wrap">
          <table className="rag-grid">
            <thead>
              <tr>
                <th>AUID</th>
                <th>Business Application</th>
                <th>Classification</th>
                {COLUMNS.map((column) => (
                  <th key={column.key}>{column.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="rag-app-id">{application.auid}</td>
                <td>
                  <strong>{application.business_app}</strong>
                  <span className="rag-app-owner">
                    {application.product_owner || 'Owner not assigned'}
                  </span>
                </td>
                <td>
                  <span className="rag-classification">
                    {application.classification}
                  </span>
                </td>
                {COLUMNS.map((column) => (
                  <td key={column.key} className="rag-source-cell">
                    {(byColumn.get(column.key) ?? []).map((link) => (
                      <SourceCard
                        key={link.id}
                        link={link}
                        onEdit={editLink}
                        onDisable={(item) => void requestDisable(item)}
                        busy={busy}
                      />
                    ))}
                    {(byColumn.get(column.key) ?? []).length === 0 && (
                      <span className="rag-cell-empty">No source</span>
                    )}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {draft && (
        <form className="rag-editor" onSubmit={(event) => void requestPreview(event)}>
          <div className="rag-editor-title">
            <strong>{draft.linkId ? 'Edit linked source' : 'Declare a source'}</strong>
            <button
              type="button"
              className="ghost-btn small"
              onClick={() => {
                setDraft(null);
                setPending(null);
                setError(null);
              }}
            >
              Close
            </button>
          </div>
          <label>
            URL
            <input
              type="url"
              required
              value={draft.url}
              readOnly={Boolean(draft.linkId)}
              placeholder="https://confluence… or https://…sharepoint…/file.pdf"
              onChange={(event) => {
                setDraft({ ...draft, url: event.target.value });
                setPending(null);
              }}
            />
          </label>
          <label>
            CrossPoint column
            <select
              value={draft.docType}
              onChange={(event) => {
                setDraft({
                  ...draft,
                  docType: event.target.value as RagDocType,
                });
                setPending(null);
              }}
            >
              {COLUMNS.map((column) => (
                <option key={column.key} value={column.key}>
                  {column.label}
                </option>
              ))}
            </select>
          </label>
          <fieldset>
            <legend>Visibility (required)</legend>
            <label>
              <input
                type="radio"
                name="rag-source-public"
                value="false"
                checked={draft.publicChoice === 'false'}
                onChange={() => {
                  setDraft({ ...draft, publicChoice: 'false' });
                  setPending(null);
                }}
              />
              Restricted to the entity
            </label>
            <label>
              <input
                type="radio"
                name="rag-source-public"
                value="true"
                checked={draft.publicChoice === 'true'}
                onChange={() => {
                  setDraft({ ...draft, publicChoice: 'true' });
                  setPending(null);
                }}
              />
              Public
            </label>
          </fieldset>
          {detectedSource && derivedScope && (
            <div className="rag-derived-scope" aria-live="polite">
              <span>Detected source: {detectedSource}</span>
              <strong>{derivedScope}</strong>
              <span>The scope is derived from the URL and cannot be edited.</span>
            </div>
          )}
          <button type="submit" className="primary-btn" disabled={busy}>
            Preview change
          </button>
        </form>
      )}

      {pending && (
        <aside className="rag-preview" aria-label="Mutation preview">
          <div>
            <strong>Catalogue preview · {pending.kind}</strong>
            <span>
              {pending.preview.application_count} application ·{' '}
              {pending.preview.link_count} links ·{' '}
              {pending.preview.unchanged ? 'unchanged' : 'snapshot will change'}
            </span>
            {!pending.preview.verdict.safe && (
              <span className="rag-preview-warning">
                Approval required: {pending.preview.verdict.reasons.join(', ')}
              </span>
            )}
          </div>
          <div className="rag-preview-actions">
            <button
              type="button"
              className="ghost-btn"
              onClick={() => setPending(null)}
              disabled={busy}
            >
              Cancel
            </button>
            <button
              type="button"
              className="primary-btn"
              onClick={() => void confirmMutation()}
              disabled={busy}
            >
              Confirm change
            </button>
          </div>
        </aside>
      )}

      {error && (
        <div className="rag-state error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
}
