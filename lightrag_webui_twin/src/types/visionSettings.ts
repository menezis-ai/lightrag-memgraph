/**
 * Vision-ingestion settings minted via Settings → Vision.
 *
 * Wire format mirrors `server/vision_settings_routes.py:VisionSettingsPublic`.
 * Scope is deliberately limited to curation plus the procedure-ingestion
 * activation flag. Infrastructure wiring (endpoint URL, API key, model,
 * timeouts) stays env-only and never surfaces in the UI.
 */

export interface VisionSettings {
  /**
   * RapidOCR pre-filter threshold: images with less OCR text than this are
   * refused without a vision-LLM call. 0 captions every image. 0..100000.
   */
  min_ocr_chars: number;
  /**
   * Image classifications refused after the vision-LLM call. Lowercase
   * slugs, max 20 entries. Env defaults: invalid / logo / signature.
   */
  drop_classes: readonly string[];
  /**
   * Admin-controlled intent for new procedure PDFs to enter the review
   * workflow. Existing parked bundles remain reviewable when false.
   */
  procedure_enabled: boolean;
}

/**
 * GET/PUT response: effective values + provenance. `source ===
 * 'env-default'` means nothing was ever saved — the values shown come from
 * the deployment environment.
 */
export interface VisionSettingsPublic extends VisionSettings {
  /** Whether the deployment prerequisites can currently run the profile. */
  procedure_available: boolean;
  source: 'runtime' | 'env-default';
  /** Milliseconds since epoch, or null when never saved. */
  updated_at: number | null;
  updated_by: string | null;
}
