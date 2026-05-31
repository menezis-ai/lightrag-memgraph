/**
 * Unit tests for ReadSourceModal (Bucket B2 of the prototype port).
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ReadSourceModal } from './ReadSourceModal';
import type { Document } from '../types/document';

function makeDoc(overrides: Partial<Document> = {}): Document {
  return {
    doc_id: 'rs-1',
    track_id: null,
    type: 'file',
    file_path: 'sample.pdf',
    content_summary: 'A sample document',
    content_length: 2048,
    status: 'PROCESSED',
    chunks_count: 12,
    created_at: '2026-05-29T14:00:00Z',
    updated_at: '2026-05-29T14:00:00Z',
    error_msg: null,
    metadata: {},
    tags: ['rman'],
    workspace: 'cib',
    visibility: 'private',
    extracted_text: 'Hello extracted world',
    ...overrides,
  };
}

describe('ReadSourceModal — visibility', () => {
  it('returns null when no doc is selected', () => {
    const { container } = render(
      <ReadSourceModal doc={null} onClose={() => {}} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders the file path, chunks count and extracted KB', () => {
    render(<ReadSourceModal doc={makeDoc()} onClose={() => {}} />);
    expect(screen.getByTestId('read-source-modal')).toBeInTheDocument();
    expect(screen.getByText('sample.pdf')).toBeInTheDocument();
    expect(screen.getByText('12 chunks indexed')).toBeInTheDocument();
    expect(screen.getByText('2.0 KB extracted')).toBeInTheDocument();
  });

  it('renders the extracted_text inside the <pre>', () => {
    render(<ReadSourceModal doc={makeDoc()} onClose={() => {}} />);
    expect(screen.getByText('Hello extracted world')).toBeInTheDocument();
  });

  it('falls back to a placeholder when extracted_text is absent', () => {
    render(
      <ReadSourceModal
        doc={makeDoc({ extracted_text: undefined })}
        onClose={() => {}}
      />,
    );
    expect(
      screen.getByText(/Extracted text preview is not available/),
    ).toBeInTheDocument();
  });
});

describe('ReadSourceModal — status pills', () => {
  it('shows "awaiting reviewer sign-off" for pending-review', () => {
    render(
      <ReadSourceModal
        doc={makeDoc({
          review: {
            state: 'pending-review',
            requested_by: 'x',
            requested_at: '2026-05-20',
            justification: 'because',
          },
        })}
        onClose={() => {}}
      />,
    );
    expect(screen.getByTestId('rs-pill-pending')).toBeInTheDocument();
  });

  it('shows "modified — awaiting re-validation" for modified', () => {
    render(
      <ReadSourceModal
        doc={makeDoc({
          review: {
            state: 'modified',
            update: {
              requested_by: 'x',
              edited_rel: '1h ago',
              detected_at: '2026-05-26',
              chunks_indexed: 12,
              summary_diff: 'changed',
            },
          },
        })}
        onClose={() => {}}
      />,
    );
    expect(screen.getByTestId('rs-pill-modified')).toBeInTheDocument();
  });
});

describe('ReadSourceModal — interactions', () => {
  it('calls onClose when the close button is clicked', async () => {
    const onClose = vi.fn();
    const user = (await import('@testing-library/user-event')).default;
    render(<ReadSourceModal doc={makeDoc()} onClose={onClose} />);
    await user.setup().click(screen.getByLabelText('Close'));
    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when Escape is pressed', () => {
    const onClose = vi.fn();
    render(<ReadSourceModal doc={makeDoc()} onClose={onClose} />);
    document.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }),
    );
    expect(onClose).toHaveBeenCalled();
  });
});
