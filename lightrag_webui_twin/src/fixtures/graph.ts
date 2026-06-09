/**
 * Knowledge graph fixtures — mirrors `Desktop/UI/data.js` MOCK_GRAPH_*.
 *
 * Layout is hardcoded (no in-browser force simulation). Types match
 * LightRAG's default extraction prompt (PRODUCT, ORG, PERSON, TECHNOLOGY,
 * CONCEPT, LOCATION).
 */

import type { GraphEntity, GraphRelation } from '../types/graph';

export const GRAPH_ENTITY_FIXTURES: readonly GraphEntity[] = [
  // Oracle cluster
  { id: 'e_oracle', name: 'Oracle Database', type: 'PRODUCT', x: 240, y: 200, mentions: 412, sources: 47, summary: 'Relational database engine; primary OLTP backing store for CIB workloads.' },
  { id: 'e_rman', name: 'RMAN', type: 'TECHNOLOGY', x: 130, y: 290, mentions: 318, sources: 31, summary: 'Oracle Recovery Manager — supported backup/restore toolchain.' },
  { id: 'e_archlog', name: 'Archive Log', type: 'CONCEPT', x: 80, y: 160, mentions: 142, sources: 24, summary: 'Redo log archive used for PITR and standby replication.' },
  { id: 'e_rhel', name: 'RHEL 9', type: 'PRODUCT', x: 340, y: 320, mentions: 198, sources: 38, summary: 'Red Hat Enterprise Linux 9 — certified OS for Oracle 19c+.' },
  { id: 'e_pga', name: 'PGA tuning', type: 'CONCEPT', x: 380, y: 130, mentions: 64, sources: 9, summary: 'Program Global Area sizing for OLTP workload concurrency.' },
  // Virt
  { id: 'e_vmware', name: 'VMware vSphere 8', type: 'PRODUCT', x: 540, y: 240, mentions: 287, sources: 22, summary: 'Hypervisor stack; banking-grade configuration baseline.' },
  { id: 'e_esxi', name: 'ESXi host', type: 'PRODUCT', x: 640, y: 320, mentions: 122, sources: 14, summary: 'Bare-metal hypervisor node.' },
  { id: 'e_vmotion', name: 'vMotion', type: 'TECHNOLOGY', x: 700, y: 200, mentions: 58, sources: 9, summary: 'Live migration of running VMs across ESXi hosts.' },
  // Memgraph / RAG
  { id: 'e_memgraph', name: 'Memgraph', type: 'PRODUCT', x: 470, y: 480, mentions: 156, sources: 19, summary: 'Graph DB backing LightRAG entity/relation storage.' },
  { id: 'e_mage', name: 'MAGE 3.8', type: 'PRODUCT', x: 560, y: 560, mentions: 84, sources: 7, summary: 'Memgraph Algorithm Extensions Engine — vector_search modules.' },
  { id: 'e_cypher', name: 'Cypher', type: 'TECHNOLOGY', x: 360, y: 540, mentions: 109, sources: 12, summary: 'Graph query language used for pre-filter retrieval.' },
  { id: 'e_lightrag', name: 'LightRAG', type: 'PRODUCT', x: 250, y: 460, mentions: 274, sources: 28, summary: 'Open-source retrieval framework forked into Twin RAG.' },
  // Payment / SWIFT
  { id: 'e_swift', name: 'SWIFT', type: 'ORG', x: 820, y: 130, mentions: 198, sources: 17, summary: 'Society for Worldwide Interbank Financial Telecommunication.' },
  { id: 'e_iso20022', name: 'ISO 20022', type: 'CONCEPT', x: 880, y: 230, mentions: 142, sources: 14, summary: 'XML messaging standard for financial transactions; SWIFT migration target.' },
  { id: 'e_cft', name: 'CFT', type: 'PRODUCT', x: 780, y: 330, mentions: 92, sources: 11, summary: 'Cross File Transfer middleware — Axway product.' },
  // People / process
  { id: 'e_marc', name: 'Marc Berthier', type: 'PERSON', x: 100, y: 420, mentions: 28, sources: 12, summary: 'DBA — primary author on Oracle restart procedures.' },
  { id: 'e_claire', name: 'Claire Benoit', type: 'PERSON', x: 160, y: 580, mentions: 41, sources: 18, summary: 'KB Admin / Tier 3 steward for CIB workspace.' },
  // Location
  { id: 'e_paris', name: 'DC Paris', type: 'LOCATION', x: 700, y: 440, mentions: 37, sources: 8, summary: 'Primary datacenter; active site of the dual-DC topology.' },
  { id: 'e_aubervil', name: 'DC Aubervilliers', type: 'LOCATION', x: 820, y: 520, mentions: 31, sources: 7, summary: 'Secondary datacenter; standby site.' },
];

export const GRAPH_RELATION_FIXTURES: readonly GraphRelation[] = [
  // Oracle cluster
  { id: 'r_01', source: 'e_rman', target: 'e_oracle', label: 'BACKS_UP', strength: 0.92 },
  { id: 'r_02', source: 'e_rman', target: 'e_archlog', label: 'MANAGES', strength: 0.74 },
  { id: 'r_03', source: 'e_oracle', target: 'e_rhel', label: 'RUNS_ON', strength: 0.88 },
  { id: 'r_04', source: 'e_oracle', target: 'e_pga', label: 'TUNED_VIA', strength: 0.61 },
  { id: 'r_05', source: 'e_archlog', target: 'e_oracle', label: 'GENERATED_BY', strength: 0.55 },
  // Virt
  { id: 'r_06', source: 'e_esxi', target: 'e_vmware', label: 'PART_OF', strength: 0.9 },
  { id: 'r_07', source: 'e_vmotion', target: 'e_vmware', label: 'FEATURE_OF', strength: 0.78 },
  { id: 'r_08', source: 'e_oracle', target: 'e_vmware', label: 'HOSTED_ON', strength: 0.66 },
  { id: 'r_09', source: 'e_esxi', target: 'e_paris', label: 'DEPLOYED_AT', strength: 0.7 },
  { id: 'r_10', source: 'e_esxi', target: 'e_aubervil', label: 'DEPLOYED_AT', strength: 0.62 },
  // RAG / graph
  { id: 'r_11', source: 'e_lightrag', target: 'e_memgraph', label: 'USES', strength: 0.89 },
  { id: 'r_12', source: 'e_lightrag', target: 'e_cypher', label: 'QUERIES_WITH', strength: 0.71 },
  { id: 'r_13', source: 'e_memgraph', target: 'e_mage', label: 'EXTENDED_BY', strength: 0.83 },
  { id: 'r_14', source: 'e_memgraph', target: 'e_cypher', label: 'SPEAKS', strength: 0.8 },
  // Payment
  { id: 'r_15', source: 'e_swift', target: 'e_iso20022', label: 'MIGRATING_TO', strength: 0.85 },
  { id: 'r_16', source: 'e_cft', target: 'e_swift', label: 'TRANSPORTS_FOR', strength: 0.64 },
  { id: 'r_17', source: 'e_iso20022', target: 'e_oracle', label: 'PERSISTED_IN', strength: 0.42 },
  // People
  { id: 'r_18', source: 'e_marc', target: 'e_oracle', label: 'AUTHORED_ON', strength: 0.79 },
  { id: 'r_19', source: 'e_marc', target: 'e_rman', label: 'AUTHORED_ON', strength: 0.82 },
  { id: 'r_20', source: 'e_claire', target: 'e_lightrag', label: 'ADMINISTERS', strength: 0.68 },
  { id: 'r_21', source: 'e_claire', target: 'e_rman', label: 'TAGGED', strength: 0.55 },
];

/**
 * Per-entity tag map for the Graph rail's "Filter by tag" picker (B4).
 * Keys are entity ids; values are the tags that entity is associated with.
 * Mirrors the prototype's GRAPH_ENTITY_TAGS (~/Downloads/prototype/data.js).
 */
export const GRAPH_ENTITY_TAGS: Record<string, readonly string[]> = {
  e_rhel: ['rhel9', 'production'],
  e_pga: ['oracle'],
  e_archlog: ['rman', 'oracle'],
  e_vmware: ['vmware', 'production'],
  e_esxi: ['vmware'],
  e_vmotion: ['vmware'],
  e_mage: ['memgraph'],
  e_lightrag: ['graphrag', 'memgraph'],
  e_cypher: ['memgraph'],
  e_swift: ['swift'],
  e_iso20022: ['iso20022', 'swift'],
  e_cft: ['cft', 'network'],
  e_marc: ['oracle', 'rman'],
  e_claire: ['governance'],
  e_paris: ['production'],
  e_aubervil: ['production'],
};

/**
 * Reverse index: doc_id → entity_ids it sourced. Used by the MSW
 * bulk-delete handler so a doc removal cascades the same way the real
 * Memgraph delete does (entities whose only source was the deleted doc
 * vanish). Hand-curated rather than derived from file_path matching —
 * the latter is fragile and was the reason the e2e bulk-delete spec
 * couldn't detect a cascade-on-graph regression before 2026-06-08.
 */
export const DOC_TO_GRAPH_ENTITIES: Record<string, readonly string[]> = {
  d1: ['e_oracle', 'e_rman', 'e_rhel', 'e_marc'],
  d2: ['e_oracle', 'e_pga'],
  d3: ['e_oracle', 'e_rman', 'e_archlog'],
  d4: ['e_memgraph', 'e_mage', 'e_lightrag', 'e_cypher'],
  d5: ['e_vmware', 'e_esxi', 'e_vmotion'],
  d6: ['e_cft'],
  d7: ['e_swift', 'e_iso20022'],
};

/**
 * Per-entity source-doc map for the Graph rail's "Filter by document" picker.
 * Values are file_path strings matching DOCUMENT_FIXTURES.
 */
export const GRAPH_ENTITY_DOCS: Record<string, readonly string[]> = {
  e_oracle: ['oracle-restart-procedure.pdf', '/cib/runbooks/oracle-pga-tuning'],
  e_rman: ['oracle-restart-procedure.pdf', '/cib/runbooks/rman-restore-cookbook'],
  e_rhel: ['oracle-restart-procedure.pdf', 'rhel9-kernel-tuning.pdf'],
  e_pga: ['/cib/runbooks/oracle-pga-tuning'],
  e_archlog: ['/cib/runbooks/rman-restore-cookbook'],
  e_vmware: ['vmware-best-practices-2026.pdf'],
  e_esxi: ['vmware-best-practices-2026.pdf'],
  e_vmotion: ['vmware-best-practices-2026.pdf'],
  e_memgraph: ['memgraph-mage-3.8-release-notes.md'],
  e_mage: ['memgraph-mage-3.8-release-notes.md'],
  e_lightrag: ['memgraph-mage-3.8-release-notes.md'],
  e_cypher: ['memgraph-mage-3.8-release-notes.md'],
  e_swift: ['swift-iso20022-migration.pdf'],
  e_iso20022: ['swift-iso20022-migration.pdf'],
  e_cft: ['cft-network-architecture.docx'],
  e_marc: ['oracle-restart-procedure.pdf'],
  e_claire: ['/cib/governance/tagging-charter'],
};
