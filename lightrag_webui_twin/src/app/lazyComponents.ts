import { lazy } from 'react';

// Keep the default Documents surface eager, but split secondary tabs and
// modal bodies out of the entry bundle so first paint does less JS work.
export const ActivityTab = lazy(() =>
  import('../components/ActivityTab').then(({ ActivityTab }) => ({
    default: ActivityTab,
  })),
);
export const AddSourceModal = lazy(() =>
  import('../components/AddSourceModal').then(({ AddSourceModal }) => ({
    default: AddSourceModal,
  })),
);
export const GraphTab = lazy(() =>
  import('../components/GraphTab').then(({ GraphTab }) => ({
    default: GraphTab,
  })),
);
export const ReadSourceModal = lazy(() =>
  import('../components/ReadSourceModal').then(({ ReadSourceModal }) => ({
    default: ReadSourceModal,
  })),
);
export const RetagModal = lazy(() =>
  import('../components/RetagModal').then(({ RetagModal }) => ({
    default: RetagModal,
  })),
);
export const RetrievalTab = lazy(() =>
  import('../components/RetrievalTab').then(({ RetrievalTab }) => ({
    default: RetrievalTab,
  })),
);
export const SettingsTab = lazy(() =>
  import('../components/SettingsTab').then(({ SettingsTab }) => ({
    default: SettingsTab,
  })),
);
export const TagsTab = lazy(() =>
  import('../components/TagsTab').then(({ TagsTab }) => ({ default: TagsTab })),
);
