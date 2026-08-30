import { expect, test, type APIRequestContext } from '@playwright/test';

import {
  coverRoute,
  discoverTwinRoutes,
  isClientErrorStatus,
  requireEphemeralBackendTarget,
  selectWrongMethod,
  type CoverageRoute,
} from './api-coverage';

const cfg = {
  authToken: 'test-token',
  expectAuth: false,
  defaultFolder: 'default',
  credentialTier: 'infrastructure-root' as const,
};

function route(overrides: Partial<CoverageRoute> = {}): CoverageRoute {
  return {
    method: 'GET',
    path: '/twin/api/example',
    hasBody: false,
    isPublic: false,
    isAdminOnly: false,
    preLlmOnly: false,
    ...overrides,
  };
}

function contextReturning(...statuses: number[]): APIRequestContext {
  let call = 0;
  return {
    fetch: async () => {
      const status = statuses[call++];
      if (status === undefined) throw new Error(`Unexpected fetch call ${call}`);
      return {
        status: () => status,
        text: async () => '',
      };
    },
  } as unknown as APIRequestContext;
}

test.describe('adversarial API reachability policy', () => {
  test('discovery records every supported method on a shared path', () => {
    const routes = discoverTwinRoutes({
      paths: {
        '/twin/api/folders': {
          get: { responses: {} },
          post: { requestBody: {}, responses: {} },
          parameters: [],
        },
        '/outside': { get: { responses: {} } },
      },
    });

    expect(routes).toHaveLength(2);
    expect(routes.map(({ method }) => method)).toEqual(['GET', 'POST']);
    expect(routes[0].supportedMethods).toEqual(['GET', 'POST']);
    expect(routes[1].hasBody).toBe(true);
  });

  test('wrong-method selection cannot choose another declared operation', () => {
    expect(
      selectWrongMethod(route({ supportedMethods: ['GET', 'POST', 'PUT', 'PATCH'] })),
    ).toBe('DELETE');
    expect(() =>
      selectWrongMethod(
        route({ supportedMethods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'] }),
      ),
    ).toThrow(/No unsupported HTTP method/);
  });

  test('only 4xx statuses satisfy a promised client rejection', () => {
    expect(isClientErrorStatus(399)).toBe(false);
    expect(isClientErrorStatus(400)).toBe(true);
    expect(isClientErrorStatus(499)).toBe(true);
    expect(isClientErrorStatus(500)).toBe(false);
  });

  test('coverRoute rejects a wrong-method 2xx', async () => {
    const ctx = contextReturning(200, 200);
    await expect(coverRoute(ctx, route(), cfg)).rejects.toThrow();
  });

  test('coverRoute rejects an unknown-id 2xx', async () => {
    const ctx = contextReturning(404, 405, 200);
    await expect(
      coverRoute(ctx, route({ path: '/twin/api/example/{id}' }), cfg),
    ).rejects.toThrow();
  });

  test('coverRoute rejects success for guaranteed-invalid hostile input', async () => {
    const ctx = contextReturning(200);
    await expect(coverRoute(ctx, route({ method: 'POST', hasBody: true }), cfg)).rejects.toThrow();
  });

  test('coverRoute keeps malformed JSON rejection strict', async () => {
    const ctx = contextReturning(422, 405, 200);
    await expect(coverRoute(ctx, route({ method: 'POST', hasBody: true }), cfg)).rejects.toThrow();
  });

  test('a configured backend requires the exact ephemeral opt-in', () => {
    expect(() => requireEphemeralBackendTarget('', undefined)).not.toThrow();
    expect(() => requireEphemeralBackendTarget('http://127.0.0.1:9621', 'true')).not.toThrow();
    expect(() => requireEphemeralBackendTarget('http://127.0.0.1:9621', undefined)).toThrow(
      /REAL_E2E_EPHEMERAL=true/,
    );
    expect(() => requireEphemeralBackendTarget('http://127.0.0.1:9621', '1')).toThrow(
      /REAL_E2E_EPHEMERAL=true/,
    );
  });
});
