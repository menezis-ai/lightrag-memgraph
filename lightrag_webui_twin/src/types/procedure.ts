/**
 * Procedure approval-bundle types — TS mirror of the backend contract in
 * `src/twindb_lightrag_memgraph/server/procedure_routes.py` (PR #384/#385).
 *
 * Procedure PDFs are PARKED as bundles instead of being indexed; a human
 * reviews the double vision pass (blind vs informed schematic descriptions +
 * divergence report) and approves / rejects / reroutes. Two projections:
 *
 *   - `ProcedureBundleSummary` — the folder-bound list row returned by
 *     `GET /twin/api/procedures`. NO paths, NO PNGs, NO full text (a bundle
 *     visible through a folder must not leak another folder's context).
 *   - `ProcedureBundle` — the full admin-only detail returned by
 *     `GET /twin/api/procedures/{id}`, PNGs included.
 */

export type BundleState =
  | 'processing'
  | 'pending'
  | 'failed'
  | 'approved'
  | 'rejected'
  | 'rerouted';

/** MIP classification detected on the parked original (subset the list
 *  projection exposes — mirrors `BundleSummary.classification`). */
export interface BundleClassification {
  class_id: string | null;
  class_name: string | null;
  reason: string | null;
  [key: string]: unknown;
}

/** One task extracted from a schematic description pass. All fields are
 *  plain strings (the vision LLM emits prose, not structured refs). */
export interface ProcedureTask {
  id: string;
  title: string;
  responsible: string;
  actors: string;
  inputs: string;
  outputs: string;
  conditions: string;
  links: string;
}

/** One vision pass over a schematic (blind = image only; informed =
 *  image + full document text). */
export interface PassPayload {
  title: string;
  description: string;
  tasks: readonly ProcedureTask[];
}

/** Blind-vs-informed comparison verdict for one schematic. */
export interface DivergenceReport {
  coherent: boolean;
  divergences: readonly string[];
  summary: string;
}

/** One schematic page of the bundle (admin detail only). */
export interface SchematicEntry {
  page: number;
  png_base64: string | null;
  blind: PassPayload | null;
  informed: PassPayload | null;
  divergence: DivergenceReport | null;
  error: string | null;
}

/** Folder-bound list projection (`GET /twin/api/procedures`). */
export interface ProcedureBundleSummary {
  id: string;
  file_name: string;
  state: BundleState | (string & {});
  reason: string;
  source: string;
  /** Opaque ingestion track id — lets the upload flow reconcile a PARKED
   *  upload against this queue (a parked doc never lands in /documents). */
  track_id: string | null;
  schematics_total: number;
  schematics_described: number;
  classification: BundleClassification | null;
  operator_classification: string | null;
  created_at: string | null;
  updated_at: string | null;
}

/** A duplicate upload request recorded against an already-parked bundle
 *  (same content hash, possibly another folder). */
export interface ProcedureDuplicateRequest {
  folder: string | null;
  operator_classification?: string | null;
  [key: string]: unknown;
}

/** Full bundle (`GET /twin/api/procedures/{id}` — admin only). */
export interface ProcedureBundle {
  id: string;
  file_name: string;
  state: BundleState | (string & {});
  reason: string;
  source: string;
  original_path: string;
  track_id: string | null;
  folder: string | null;
  content_hash: string | null;
  full_text: string;
  schematics: readonly SchematicEntry[];
  schematics_total: number;
  classification: BundleClassification | null;
  operator_classification: string | null;
  duplicate_requests: readonly ProcedureDuplicateRequest[];
  created_at: string | null;
  updated_at: string | null;
  [key: string]: unknown;
}

/** Folders that requested this bundle (primary first). Mirrors the backend
 *  `_procedure.bundle_folders`: scan-created bundles (folder null, no
 *  operator duplicate request) return []. */
export function bundleFolders(bundle: ProcedureBundle): readonly string[] {
  const folders: string[] = [];
  if (bundle.folder) folders.push(bundle.folder);
  for (const request of bundle.duplicate_requests ?? []) {
    const folder = request?.folder;
    if (folder && !folders.includes(folder)) folders.push(folder);
  }
  return folders;
}
