/**
 * Centralized fixture exports.
 *
 * Single import point for tests and (later) MSW handlers:
 *   import { DOCUMENT_FIXTURES, FOLDER_FIXTURES, NOTIFICATION_FIXTURES } from '../fixtures';
 */

export { DOCUMENT_FIXTURES } from './documents';
export { FOLDER_FIXTURES } from './folders';
export { NOTIFICATION_FIXTURES } from './notifications';
export { THESAURUS_FIXTURES } from './thesaurus';
export { FORMAT_CATEGORY_FIXTURES } from './formatCategories';
export {
  ANSWER_TOKENS_FIXTURE,
  RETRIEVAL_SOURCES_FIXTURE,
  THREAD_FIXTURES,
  makeSampleThreads,
} from './retrieval';
export { ACTIVITY_FIXTURES, ACTIVITY_NOW_MS } from './activity';
export { API_VERSION, OPENAPI_GROUPS } from './api';
export {
  DOC_TO_GRAPH_ENTITIES,
  GRAPH_ENTITY_DOCS,
  GRAPH_ENTITY_FIXTURES,
  GRAPH_RELATION_FIXTURES,
} from './graph';
export { TAG_CATEGORY_FIXTURES, TAG_FIXTURES } from './tags';
export { PROCEDURE_BUNDLE_FIXTURES, TINY_PNG_BASE64 } from './procedures';
