/**
 * OpenAPI tab fixtures — mirrors `Desktop/UI/api.jsx` static endpoint table.
 *
 * In phase 1 these will be replaced by a fetch against `/openapi.json` on the
 * configured server.
 */

import type { OpenApiGroup, OpenApiServer } from '../types/api';

export const API_VERSION = 'v1.4.12/0279';

export const OPENAPI_GROUPS: readonly OpenApiGroup[] = [
  {
    id: 'documents',
    name: 'documents',
    desc: 'Source ingestion, listing and lifecycle.',
    endpoints: [
      { m: 'POST', p: '/documents/upload', s: 'Upload Document' },
      { m: 'POST', p: '/documents/text', s: 'Insert Text' },
      { m: 'POST', p: '/documents/texts', s: 'Insert Texts' },
      { m: 'POST', p: '/documents/scan', s: 'Scan For New Documents' },
      { m: 'GET', p: '/documents', s: 'List Documents' },
      { m: 'GET', p: '/documents/pipeline_status', s: 'Get Pipeline Status' },
      { m: 'DELETE', p: '/documents', s: 'Clear Documents' },
      { m: 'DELETE', p: '/documents/delete_document', s: 'Delete Document' },
      { m: 'POST', p: '/documents/clear_cache', s: 'Clear Cache' },
    ],
  },
  {
    id: 'query',
    name: 'query',
    desc: 'Retrieval + LLM synthesis endpoints.',
    endpoints: [
      { m: 'POST', p: '/query', s: 'Query Text' },
      { m: 'POST', p: '/query/stream', s: 'Query Text Stream' },
    ],
  },
  {
    id: 'graph',
    name: 'graph',
    desc: 'Knowledge-graph CRUD and label browsing.',
    endpoints: [
      { m: 'GET', p: '/graph/label/list', s: 'Get Graph Labels' },
      { m: 'GET', p: '/graph/label/popular', s: 'Get Popular Labels' },
      { m: 'GET', p: '/graph/label/search', s: 'Search Labels' },
      { m: 'GET', p: '/graphs', s: 'Get Knowledge Graph' },
      { m: 'GET', p: '/graph/entity/exists', s: 'Check Entity Exists' },
      { m: 'POST', p: '/graph/entity/edit', s: 'Update Entity' },
      { m: 'POST', p: '/graph/relation/edit', s: 'Update Relation' },
      { m: 'POST', p: '/graph/entity/create', s: 'Create Entity' },
      { m: 'POST', p: '/graph/relation/create', s: 'Create Relation' },
    ],
  },
  {
    id: 'ollama',
    name: 'ollama',
    desc: 'Drop-in Ollama-compatible chat & generate surface.',
    endpoints: [
      { m: 'GET', p: '/api/version', s: 'Get Version' },
      { m: 'GET', p: '/api/tags', s: 'Get Tags' },
      { m: 'GET', p: '/api/ps', s: 'Get Running Models' },
      { m: 'POST', p: '/api/generate', s: 'Generate' },
      { m: 'POST', p: '/api/chat', s: 'Chat' },
    ],
  },
  {
    id: 'default',
    name: 'default',
    desc: 'Auth, health and root.',
    endpoints: [
      { m: 'GET', p: '/', s: 'Redirect To Webui' },
      { m: 'GET', p: '/auth-status', s: 'Get Auth Status' },
      { m: 'POST', p: '/login', s: 'Login' },
      { m: 'GET', p: '/health', s: 'Get system health and configuration status' },
    ],
  },
];

export const API_SERVERS: readonly OpenApiServer[] = [
  { id: 'prod', label: 'https://cib-kb.twin.internal — production' },
  { id: 'stg', label: 'https://cib-kb.stg.twin.internal — staging' },
];

export const API_BASE_URL: Record<string, string> = {
  prod: 'https://cib-kb.twin.internal',
  stg: 'https://cib-kb.stg.twin.internal',
};
