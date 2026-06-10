/**
 * Knowledge graph types — entities + relations extracted by LightRAG.
 *
 * Read-only teaser; `:MENTIONED_IN` traversal + tag-filtered graph reasoning
 * remain Twin Graph tier features.
 *
 * Contract template for backend phase 1:
 *   GET /graph/entities?folder=&type=  -> GraphEntity[]
 *   GET /graph/relations?folder=       -> GraphRelation[]
 * Positions (x, y) are precomputed server-side or fallback to a static layout
 * — no force simulation runs in the browser.
 */

export type GraphEntityType =
  | 'PRODUCT'
  | 'TECHNOLOGY'
  | 'CONCEPT'
  | 'ORG'
  | 'PERSON'
  | 'LOCATION';

export interface GraphEntity {
  id: string;
  name: string;
  type: GraphEntityType;
  /** Precomputed layout x (SVG coords). */
  x: number;
  /** Precomputed layout y (SVG coords). */
  y: number;
  /** Total chunk mentions, drives node radius. */
  mentions: number;
  /** Distinct source count. */
  sources: number;
  /** Parent document ids or source labels that mention this entity. */
  source_docs?: readonly string[];
  summary: string;
  /** Free-form attribute tags carried on the node itself.
   *  Decoupled from the governed WebuiTag taxonomy (doc-level relations) —
   *  this is node-attribute metadata, mirrors the JSX maquette EntityEditor
   *  behavior. Will migrate to [:TAGGED_WITH] once the graph taxonomy lands. */
  tags?: readonly string[];
  /** Custom key/value metadata. Persisted as node properties in Memgraph. */
  properties?: Readonly<Record<string, string>>;
}

export interface GraphRelation {
  id: string;
  source: string;
  target: string;
  label: string;
  /** Confidence / strength 0..1. ≥0.75 renders as "strong" edge. */
  strength: number;
  /** Custom key/value metadata. Persisted as edge properties in Memgraph. */
  properties?: Readonly<Record<string, string>>;
}

export interface GraphEntityPatch {
  name?: string;
  type?: GraphEntityType;
  summary?: string;
  tags?: readonly string[];
  properties?: Readonly<Record<string, string>>;
}

export interface GraphRelationPatch {
  label?: string;
  strength?: number;
  properties?: Readonly<Record<string, string>>;
}

export const GRAPH_TYPE_LABEL: Record<GraphEntityType, string> = {
  PRODUCT: 'Product',
  TECHNOLOGY: 'Technology',
  CONCEPT: 'Concept',
  ORG: 'Org',
  PERSON: 'Person',
  LOCATION: 'Location',
};

export const GRAPH_TYPE_COLORS: Record<GraphEntityType, string> = {
  PRODUCT: '#3871B4',
  TECHNOLOGY: '#6A4FB6',
  CONCEPT: '#8A5C0E',
  ORG: '#1F8A7A',
  PERSON: '#B03060',
  LOCATION: '#2C3E50',
};
