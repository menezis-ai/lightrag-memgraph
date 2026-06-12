/**
 * Controlled tag vocabulary, surfaced in autocomplete (Retag, AddSource,
 * tag filters) and in the Tags admin page.
 *
 * `category` groups tags for the grouped autocomplete UI; `def` is the
 * tooltip definition.
 */

export type ThesaurusCategory = string;

export interface ThesaurusEntry {
  tag: string;
  category: ThesaurusCategory;
  def: string;
}
