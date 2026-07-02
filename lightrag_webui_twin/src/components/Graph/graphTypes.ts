import type { GraphEntity, GraphEntityType, GraphRelation } from '../../types/graph';
import type { Toast } from '../../types/toast';

export type MatchMode = 'any' | 'all';

export interface GraphTabProps {
  entities: readonly GraphEntity[];
  relations: readonly GraphRelation[];
  /** Optional color override; defaults to the package palette. */
  colors?: Record<GraphEntityType, string>;
  /** doc_id → file name, so the document filter shows human-readable labels. */
  docLabels?: Readonly<Record<string, string>>;
  /**
   * doc_id → document tags. Graph entities inherit the tags of their
   * source_docs for the tag filter — LightRAG-ingested entities carry no
   * twin_tags_json of their own, so without this join the filter is empty
   * in production (doc-level tag propagation, tagging roadmap phase 1).
   */
  docTags?: Readonly<Record<string, readonly string[]>>;
  /** Canonical tag catalog from /tags, used by the graph tag picker. */
  tagCatalog?: readonly string[];
  /** Active folder label for the header; the segment is hidden when unset. */
  folderLabel?: string;
  /** Host-controlled tab navigation. */
  onNavigate?: (tab: string, params?: Record<string, string>) => void;
  /**
   * Host-owned toast pusher (App.tsx → ``pushToast``). Used to surface
   * the half-success case on entity creation (HTTP 500 = projection
   * failed, write committed — see ``mapCreateEntityError``).
   */
  onToast?: (toast: Omit<Toast, 'id'>) => void;
}
