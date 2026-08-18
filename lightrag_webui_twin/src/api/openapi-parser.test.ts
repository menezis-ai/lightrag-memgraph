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
      hasBody: false,
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

  it('extracts the operation description', () => {
    const spec = {
      paths: {
        '/foo': {
          get: { tags: ['x'], summary: 'S', description: 'Long form.\n' },
        },
      },
    };
    expect(parseOpenApiSpec(spec).groups[0].endpoints[0].desc).toBe(
      'Long form.',
    );
  });

  it('extracts parameters with type, required flag, description and example', () => {
    const spec = {
      paths: {
        '/docs/{id}': {
          get: {
            tags: ['x'],
            parameters: [
              {
                name: 'id',
                in: 'path',
                required: true,
                description: 'Document id.',
                schema: { type: 'string', examples: ['doc-1'] },
              },
              {
                name: 'limit',
                in: 'query',
                schema: { anyOf: [{ type: 'integer' }, { type: 'null' }] },
              },
            ],
          },
        },
      },
    };
    const [ep] = parseOpenApiSpec(spec).groups[0].endpoints;
    expect(ep.params).toEqual([
      {
        name: 'id',
        in: 'path',
        type: 'string',
        required: true,
        desc: 'Document id.',
        example: 'doc-1',
      },
      {
        name: 'limit',
        in: 'query',
        type: 'integer',
        required: false,
        desc: '',
        example: undefined,
      },
    ]);
  });

  it('merges path-level parameters into every operation', () => {
    const spec = {
      paths: {
        '/foo': {
          parameters: [{ name: 'shared', in: 'query', schema: { type: 'string' } }],
          get: { tags: ['x'] },
        },
      },
    };
    const [ep] = parseOpenApiSpec(spec).groups[0].endpoints;
    expect(ep.params?.map((p) => p.name)).toEqual(['shared']);
  });

  it('resolves $ref against components (parameters and schemas)', () => {
    const spec = {
      components: {
        parameters: {
          Folder: {
            name: 'X-Twin-Folder',
            in: 'header',
            description: 'Folder scope.',
            schema: { type: 'string' },
          },
        },
        schemas: {
          Body: {
            type: 'object',
            required: ['name'],
            properties: { name: { type: 'string', examples: ['a'] } },
          },
        },
      },
      paths: {
        '/foo': {
          post: {
            tags: ['x'],
            parameters: [{ $ref: '#/components/parameters/Folder' }],
            requestBody: {
              content: {
                'application/json': {
                  schema: { $ref: '#/components/schemas/Body' },
                },
              },
            },
          },
        },
      },
    };
    const [ep] = parseOpenApiSpec(spec).groups[0].endpoints;
    expect(ep.params?.[0]).toMatchObject({
      name: 'X-Twin-Folder',
      in: 'header',
      desc: 'Folder scope.',
    });
    expect(JSON.parse(ep.bodyExample ?? '{}')).toEqual({ name: 'a' });
    expect(ep.hasBody).toBe(true);
  });

  it('prefers a declared body example over the schema skeleton', () => {
    const spec = {
      paths: {
        '/foo': {
          post: {
            tags: ['x'],
            requestBody: {
              content: {
                'application/json': {
                  schema: { type: 'object', properties: { a: { type: 'string' } } },
                  example: { a: 'from-example' },
                },
              },
            },
          },
        },
      },
    };
    const [ep] = parseOpenApiSpec(spec).groups[0].endpoints;
    expect(JSON.parse(ep.bodyExample ?? '{}')).toEqual({ a: 'from-example' });
  });

  it('builds a placeholder for required, exampleless body fields', () => {
    const spec = {
      paths: {
        '/foo': {
          post: {
            tags: ['x'],
            requestBody: {
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    required: ['name', 'tags'],
                    properties: {
                      name: { type: 'string' },
                      tags: { type: 'array', items: { type: 'string' } },
                      optional: { type: 'string' },
                    },
                  },
                },
              },
            },
          },
        },
      },
    };
    const [ep] = parseOpenApiSpec(spec).groups[0].endpoints;
    expect(JSON.parse(ep.bodyExample ?? '{}')).toEqual({ name: '', tags: [] });
  });

  it('builds type-compatible placeholders for required exampleless fields', () => {
    const spec = {
      paths: {
        '/foo': {
          put: {
            tags: ['x'],
            requestBody: {
              content: {
                'application/json': {
                  schema: {
                    type: 'object',
                    required: ['count', 'enabled', 'name', 'mode'],
                    properties: {
                      count: { type: 'integer', minimum: 0 },
                      enabled: { type: 'boolean' },
                      name: { type: 'string' },
                      mode: { enum: ['a', 'b'] },
                    },
                  },
                },
              },
            },
          },
        },
      },
    };
    const [ep] = parseOpenApiSpec(spec).groups[0].endpoints;
    expect(JSON.parse(ep.bodyExample ?? '{}')).toEqual({
      count: 0,
      enabled: false,
      name: '',
      mode: 'a',
    });
  });

  it('derives the per-operation secured flag from security', () => {
    const spec = {
      paths: {
        '/locked': { get: { tags: ['x'], security: [{ HTTPBearer: [] }] } },
        '/open': { get: { tags: ['x'], security: [] } },
        '/optional': {
          get: { tags: ['x'], security: [{ HTTPBearer: [] }, {}] },
        },
        '/silent': { get: { tags: ['x'] } },
      },
    };
    const eps = Object.fromEntries(
      parseOpenApiSpec(spec)
        .groups.flatMap((g) => g.endpoints)
        .map((e) => [e.p, e.secured]),
    );
    expect(eps['/locked']).toBe(true);
    expect(eps['/open']).toBe(false);
    // An empty requirement means anonymous access is allowed.
    expect(eps['/optional']).toBe(false);
    // No security field at all: the spec says nothing.
    expect(eps['/silent']).toBeUndefined();
  });

  it('extracts responses sorted by status code', () => {
    const spec = {
      paths: {
        '/foo': {
          delete: {
            tags: ['x'],
            responses: {
              '404': { description: 'Not found' },
              '204': { description: 'Deleted' },
            },
          },
        },
      },
    };
    const [ep] = parseOpenApiSpec(spec).groups[0].endpoints;
    expect(ep.responses).toEqual([
      { code: '204', desc: 'Deleted' },
      { code: '404', desc: 'Not found' },
    ]);
  });
});
