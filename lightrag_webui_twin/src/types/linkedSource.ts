export type RagDocType = 'di' | 'de' | 'sats' | 'general';
export type RagSourceType = 'confluence' | 'sharepoint';

export interface CatalogApplication {
  auid: string;
  business_app: string;
  classification: 'C1' | 'C2' | string;
  product_owner: string | null;
  product_owner_uid: string | null;
  entity_code: string;
  status: string;
  description: string;
  tags: readonly string[];
  row_version: number;
  updated_at: string;
}

export interface RagLinkedSource {
  id: string;
  auid: string;
  url: string;
  url_raw: string;
  source_type: RagSourceType | string;
  resource_kind: 'page' | 'space' | 'document' | string | null;
  resource_id: string | null;
  doc_type: RagDocType;
  public: boolean;
  title: string | null;
  language: string | null;
  tags: readonly string[];
  status: string;
  kb_instance_id: string | null;
  folder_id: string | null;
  declared_by: string;
  declared_at: string;
  last_validated_at: string | null;
  row_version: number;
  updated_at: string;
}

export interface LinkedSourcesSnapshot {
  application: CatalogApplication | null;
  links: readonly RagLinkedSource[];
}

export interface LinkedSourceCreateInput {
  url: string;
  doc_type: RagDocType;
  public: boolean;
  title?: string;
  language?: string;
  tags?: readonly string[];
  status?: 'draft' | 'active';
}

export interface LinkedSourcePatchInput {
  doc_type?: RagDocType;
  public?: boolean;
  title?: string | null;
  language?: string | null;
  tags?: readonly string[];
  expected_version: number;
}

export interface LinkedSourcePreviewInput {
  operation: 'create' | 'patch' | 'transition';
  target_id?: string;
  action?: 'suspend' | 'activate' | 'disable';
  body: Record<string, unknown>;
}

export interface CatalogPreview {
  snapshot_id: string;
  unchanged: boolean;
  application_count: number;
  link_count: number;
  diff: Record<string, unknown>;
  verdict: { safe: boolean; reasons: readonly string[] };
  published_snapshot_id: string | null;
}

export interface LinkedSourceMutation {
  link: RagLinkedSource;
  revision: {
    id: number;
    state: string;
    snapshot_id: string;
  } | null;
}
