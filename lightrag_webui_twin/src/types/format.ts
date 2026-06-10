/**
 * Allowed input formats grouped by category for the Add Source UI.
 * `future: true` lights up a "coming soon" pill instead of accepting the
 * format right now.
 */

export interface FormatCategory {
  cat: string;
  fmts: string;
  future?: boolean;
}
