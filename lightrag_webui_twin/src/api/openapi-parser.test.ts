/**
 * Tests for the OpenAPI 3.1 → Twin ApiTab format converter.
 *
 * Mock-kill F2 — the parser is what keeps the Twin API tab ISO with the
 * LightRAG WebUI by construction (single source of truth = `/openapi.json`).
 */

import { describe, expect, it } from 'vitest';
import { parseOpenApiSpec } from './openapi-parser';

describe('parseOpenApiSpec', () => {
  it('groups operations by their first tag and orders groups alphabetically', () => {
    const spec = {
      info: { version: '1.4.12+memgraph-1.0.0' },
      paths: {
        '/documents': {
          get: { tags: ['documents'], summary: 'List documents' },
        },
        '/query': { post: { tags: ['query'], summary: 'Query text' } },
        '/health': { get: { tags: ['default'], summary: 'Health' } },
      },
    };
    const result = parseOpenApiSpec(spec);
    expect(result.version).toBe('1.4.12+memgraph-1.0.0');
    const groupIds = result.groups.map((g) => g.id);
    expect(groupIds).toEqual(['default', 'documents', 'query']);
  });

  it('falls back to "default" tag when an operation has no tags', () => {
    const spec = {
      info: { version: 'x' },
      paths: { '/foo': { get: { summary: 'Foo' } } },
    };
    const result = parseOpenApiSpec(spec);
    expect(result.groups).toHaveLength(1);
    expect(result.groups[0].id).toBe('default');
    expect(result.groups[0].endpoints[0]).toEqual({
      m: 'GET',
      p: '/foo',
      s: 'Foo',
    });
  });

  it('honors tag descriptions when the spec declares them', () => {
    const spec = {
      info: { version: 'x' },
      tags: [{ name: 'documents', description: 'Source ingestion + lifecycle' }],
      paths: { '/documents': { get: { tags: ['documents'], summary: 'List' } } },
    };
    const result = parseOpenApiSpec(spec);
    expect(result.groups[0].desc).toBe('Source ingestion + lifecycle');
  });

  it('falls back to operationId when summary is empty', () => {
    const spec = {
      info: { version: 'x' },
      paths: {
        '/foo': {
          post: { tags: ['x'], operationId: 'create_foo' },
        },
      },
    };
    const result = parseOpenApiSpec(spec);
    expect(result.groups[0].endpoints[0].s).toBe('create_foo');
  });

  it('falls back to "METHOD path" when neither summary nor operationId is set', () => {
    const spec = {
      info: { version: 'x' },
      paths: { '/foo': { delete: { tags: ['x'] } } },
    };
    const result = parseOpenApiSpec(spec);
    expect(result.groups[0].endpoints[0].s).toBe('DELETE /foo');
  });

  it('ignores non-HTTP-method keys (parameters, summary on path item)', () => {
    const spec = {
      info: { version: 'x' },
      paths: {
        '/foo': {
          summary: 'Path summary — must not appear as an endpoint',
          parameters: [{ name: 'q', in: 'query' }],
          get: { tags: ['x'], summary: 'Real op' },
        },
      },
    };
    const result = parseOpenApiSpec(spec);
    expect(result.groups[0].endpoints).toHaveLength(1);
    expect(result.groups[0].endpoints[0].s).toBe('Real op');
  });

  it('returns a safe empty result for malformed input', () => {
    expect(parseOpenApiSpec(null).groups).toEqual([]);
    expect(parseOpenApiSpec('not a spec').groups).toEqual([]);
    expect(parseOpenApiSpec({}).version).toBe('unknown');
  });
});
