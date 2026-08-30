import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ApiError } from '../api/client';
import type { RagLinkedSource } from '../types/linkedSource';

const mocks = vi.hoisted(() => ({
  preview: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  disable: vi.fn(),
}));

const baseLink = vi.hoisted(() => ({
  id: '11111111-1111-4111-8111-111111111111',
  auid: 'AP11121',
  url: 'https://knowledge.example.com/pages/viewpage.action?pageId=42',
  url_raw: 'https://knowledge.example.com/pages/viewpage.action?pageId=42',
  source_type: 'confluence',
  resource_kind: 'page',
  resource_id: '42',
  doc_type: 'de',
  public: false,
  title: 'Move2Cloud',
  language: 'en',
  tags: [],
  status: 'active',
  kb_instance_id: '22222222-2222-4222-8222-222222222222',
  folder_id: 'default',
  declared_by: 'instance:mock',
  declared_at: '2026-08-19T08:00:00Z',
  last_validated_at: null,
  row_version: 3,
  updated_at: '2026-08-19T08:00:00Z',
}) satisfies RagLinkedSource);

vi.mock('../api/queries', () => ({
  useLinkedSources: () => ({
    data: {
      application: {
        auid: 'AP11121',
        business_app: 'CTCK',
        classification: 'C1',
        product_owner: 'Demo Steward',
        product_owner_uid: 'demo.steward',
        entity_code: 'PF',
        status: 'active',
        description: '',
        tags: [],
        row_version: 1,
        updated_at: '2026-08-19T08:00:00Z',
      },
      links: [
        baseLink,
        {
          ...baseLink,
          id: '33333333-3333-4333-8333-333333333333',
          url: 'https://tenant.sharepoint.com/sites/pf/Shared/guide.pdf',
          url_raw: 'https://tenant.sharepoint.com/sites/pf/Shared/guide.pdf',
          source_type: 'sharepoint',
          resource_kind: 'document',
          resource_id: 'guide.pdf',
          title: 'Guide',
        },
      ],
    },
    isLoading: false,
    isError: false,
    error: null,
  }),
  usePreviewLinkedSource: () => ({
    mutateAsync: mocks.preview,
    isPending: false,
  }),
  useCreateLinkedSource: () => ({
    mutateAsync: mocks.create,
    isPending: false,
  }),
  useUpdateLinkedSource: () => ({
    mutateAsync: mocks.update,
    isPending: false,
  }),
  useDisableLinkedSource: () => ({
    mutateAsync: mocks.disable,
    isPending: false,
  }),
}));

import { describeRagScope, inferRagSourceType } from '../lib/linkedSources';
import { SourcesRagTab } from './SourcesRagTab';

const previewResult = {
  snapshot_id: 'candidate',
  unchanged: false,
  application_count: 1,
  link_count: 3,
  diff: {},
  verdict: { safe: true, reasons: [] },
  published_snapshot_id: 'published',
};

beforeEach(() => {
  vi.clearAllMocks();
  mocks.preview.mockResolvedValue(previewResult);
  mocks.create.mockResolvedValue({});
  mocks.update.mockResolvedValue({});
  mocks.disable.mockResolvedValue({});
});

describe('SourcesRagTab', () => {
  it('renders the CrossPoint columns, rules and multiple links in one cell', () => {
    render(<SourcesRagTab activeFolder="default" />);

    for (const header of [
      'AUID',
      'Business Application',
      'Classification',
      'DI',
      'DE',
      'SATS',
      'Confluence Space',
    ]) {
      expect(screen.getByRole('columnheader', { name: header })).toBeInTheDocument();
    }
    expect(screen.getByText('AP11121')).toBeInTheDocument();
    expect(screen.getByText('ABSOLUTELY NO CONFIDENTIAL DOCUMENTS.')).toBeInTheDocument();
    const deCell = (
      screen.getByRole('columnheader', { name: 'DE' }) as HTMLTableCellElement
    ).cellIndex;
    const dataCell = screen.getAllByRole('cell')[deCell];
    expect(within(dataCell).getAllByText(/Next nightly batch/)).toHaveLength(2);
  });

  it('derives source type and immutable scope from URL and column', () => {
    expect(inferRagSourceType('')).toBeNull();
    expect(
      inferRagSourceType(
        'https://tenant.sharepoint.com/sites/pf/Shared/Documents/a.pdf',
      ),
    ).toBe('sharepoint');
    expect(
      describeRagScope('https://knowledge.example.com/spaces/PF', 'general'),
    ).toBe('Entire space, recursive');
    expect(
      describeRagScope(
        'https://knowledge.example.com/pages/viewpage.action?pageId=42',
        'general',
      ),
    ).toBe('Root page and descendants, recursive');
  });

  it('requires an explicit visibility and previews before creating', async () => {
    const user = userEvent.setup();
    render(<SourcesRagTab activeFolder="default" />);

    await user.click(screen.getByRole('button', { name: 'Add linked source' }));
    expect(screen.queryByText(/Detected source:/)).toBeNull();
    const radios = screen.getAllByRole('radio');
    expect(radios).toHaveLength(2);
    expect(radios.every((radio) => !(radio as HTMLInputElement).checked)).toBe(true);

    await user.type(
      screen.getByRole('textbox', { name: 'URL' }),
      'https://knowledge.example.com/pages/viewpage.action?pageId=99',
    );
    await user.click(screen.getByRole('button', { name: 'Preview change' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Choose Public or Restricted');
    expect(mocks.preview).not.toHaveBeenCalled();

    await user.click(screen.getByRole('radio', { name: 'Restricted to the entity' }));
    await user.click(screen.getByRole('button', { name: 'Preview change' }));
    expect(mocks.preview).toHaveBeenCalledWith({
      operation: 'create',
      body: {
        url: 'https://knowledge.example.com/pages/viewpage.action?pageId=99',
        doc_type: 'di',
        public: false,
        status: 'active',
      },
    });
    expect(await screen.findByText(/Catalogue preview · create/)).toBeInTheDocument();
    expect(mocks.create).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Confirm change' }));
    expect(mocks.create).toHaveBeenCalledOnce();
  });

  it('previews the optimistic version before disabling', async () => {
    const user = userEvent.setup();
    render(<SourcesRagTab activeFolder="default" />);
    await user.click(screen.getAllByRole('button', { name: 'Disable' })[0]);

    expect(mocks.preview).toHaveBeenCalledWith({
      operation: 'transition',
      target_id: baseLink.id,
      action: 'disable',
      body: {
        expected_version: 3,
        reason: 'disabled from the Twin Sources RAG grid',
      },
    });
    await user.click(await screen.findByRole('button', { name: 'Confirm change' }));
    expect(mocks.disable).toHaveBeenCalledWith({
      id: baseLink.id,
      expectedVersion: 3,
    });
  });

  it('shows structured conflict guidance without leaking the technical request', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    mocks.preview.mockRejectedValueOnce(
      new ApiError(
        `PATCH /twin/api/linked-sources/${baseLink.id} → 409 Conflict`,
        409,
        {
          detail: {
            message: 'row_version mismatch — reload and retry',
            current_version: 4,
          },
        },
      ),
    );
    const user = userEvent.setup();
    render(<SourcesRagTab activeFolder="default" />);

    await user.click(screen.getAllByRole('button', { name: 'Edit' })[0]);
    await user.click(screen.getByRole('button', { name: 'Preview change' }));

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('row_version mismatch — reload and retry');
    expect(alert).not.toHaveTextContent('PATCH /twin/api');
    expect(warn).toHaveBeenCalled();
    warn.mockRestore();
  });
});
