import { describe, expect, it, vi } from 'vitest';
import { ACTIVITY_FIXTURES } from '../../fixtures';
import { exportActivityCsv } from './activityExport';

describe('exportActivityCsv', () => {
  it('creates a dated CSV download and schedules URL cleanup', () => {
    vi.useFakeTimers();
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    const createObjectURL = vi.fn(() => 'blob:activity-csv');
    const revokeObjectURL = vi.fn();
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;
    try {
      exportActivityCsv(ACTIVITY_FIXTURES.slice(0, 2), '7d');
      expect(createObjectURL).toHaveBeenCalledOnce();
      const anchor = document.querySelector<HTMLAnchorElement>(
        'a[download^="twin-rag-activity-7d-"]',
      );
      expect(anchor).not.toBeNull();
      vi.runAllTimers();
      expect(revokeObjectURL).toHaveBeenCalledWith('blob:activity-csv');
      expect(anchor).not.toBeInTheDocument();
    } finally {
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
      vi.useRealTimers();
    }
  });
});
