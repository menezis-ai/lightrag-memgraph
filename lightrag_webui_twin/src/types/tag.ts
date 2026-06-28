/**
 * Tag catalog governance types.
 *
 * Contract template for backend phase 1:
 *   GET /tags?folder=  -> TagEntry[]
 *   GET /tags/categories  -> TagCategory[]
 *   POST /tags/{name}     -> CRUD
 *   POST /tags/request    -> queue Tier-3 proposal
 *
 * The tag catalog is a 3-tier taxonomy:
 *   1 = Trunk   — gov-validated, cross-folder
 *   2 = Branch  — dept-scoped, steward-approved
 *   3 = Leaf    — user-proposed, lightweight review
 *   "requested" = not yet a tag, awaiting palier-3 acceptance.
 */

export type TagTier = 1 | 2 | 3 | 'requested';

export type TagStatus =
  | 'active'
  | 'pending-promotion'
  | 'pending-review'
  | 'deprecated'
  | 'rejected';

export interface TagRelated {
  tag: string;
  /** Co-occurrence strength, 0..1. */
  strength: number;
}

export interface TagAudit {
  by: string;
  at: string;
  /** Optional verb describing the edit (e.g. "added alias ora"). */
  action?: string;
}

export interface TagEntry {
  tag: string;
  tier: TagTier;
  category: string;
  status: TagStatus;
  def: string;
  long_description?: string;
  aliases: readonly string[];
  deprecates: readonly string[];
  sources_count: number;
  chunks_count: number;
  query_freq_30d: number;
  created: TagAudit;
  last_edit: TagAudit;
  related: readonly TagRelated[];
  examples: readonly string[];
  /** Only present when tier === "requested". */
  requested_by?: string;
  requested_at?: string;
  justification?: string;
  reject_reason?: string;
}

export interface TagCategory {
  id: string;
  label: string;
  color: string;
}

export type TagActionKind =
  | 'edit'
  | 'suggest'
  | 'synonyms'
  | 'deprecate'
  | 'delete'
  | 'reject'
  | 'edit-approve'
  | 'request';

export interface TagAction {
  kind: TagActionKind;
  tag?: TagEntry;
}

export interface TagCurrentUser {
  name: string;
  /** 1 = read-only, 2 = suggest, 3 = full steward. */
  palier: 1 | 2 | 3;
  role: string;
}

export const TAG_STATUS_FILTERS = [
  'all',
  'active',
  'pending-promotion',
  'deprecated',
  'rejected',
] as const;
export type TagStatusFilter = (typeof TAG_STATUS_FILTERS)[number];
