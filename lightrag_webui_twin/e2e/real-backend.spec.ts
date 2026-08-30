import { expect, test, type Page } from './fixtures';
import { openTab } from './helpers';

const backendUrl = process.env.REAL_BACKEND_URL?.replace(/\/$/, '') ?? '';
const authToken = process.env.REAL_BACKEND_AUTH_TOKEN ?? process.env.VITE_AUTH_TOKEN;
const defaultFolder = process.env.REAL_BACKEND_FOLDER ?? 'default';
const expectAuth = process.env.REAL_E2E_EXPECT_AUTH === 'true';
const mutationDocId = process.env.REAL_E2E_MUTATION_DOC_ID;
const configuredRetagTag = process.env.REAL_E2E_RETAG_TAG;
const bulkDeleteDocId = process.env.REAL_E2E_BULK_DELETE_DOC_ID;
const uploadForDeleteSetting = process.env.REAL_E2E_UPLOAD_FOR_DELETE;
const uploadForDelete = uploadForDeleteSetting === 'true';

function assertRealCiFixtures(): void {
  // The generic MSW gate deliberately loads this file without a backend URL;
  // keep that whole-file skip clean. Once CI names a real target, however,
  // missing non-LLM fixtures are configuration errors, not test skips.
  if (process.env.CI !== 'true' || !backendUrl) {
    return;
  }

  const missing: string[] = [];
  if (!process.env.REAL_BACKEND_FOLDER) missing.push('REAL_BACKEND_FOLDER');
  if (!authToken) missing.push('REAL_BACKEND_AUTH_TOKEN or VITE_AUTH_TOKEN');
  if (!expectAuth) missing.push('REAL_E2E_EXPECT_AUTH=true');
  if (!mutationDocId) missing.push('REAL_E2E_MUTATION_DOC_ID');
  if (!configuredRetagTag) missing.push('REAL_E2E_RETAG_TAG');
  if (!['true', 'false'].includes(uploadForDeleteSetting ?? '')) {
    missing.push('REAL_E2E_UPLOAD_FOR_DELETE=true|false');
  }
  if (!uploadForDelete && !bulkDeleteDocId) {
    missing.push('REAL_E2E_BULK_DELETE_DOC_ID');
  }

  if (missing.length > 0) {
    throw new Error(`Real-backend CI fixtures missing: ${missing.join(', ')}`);
  }
}

assertRealCiFixtures();

interface DocumentListEnvelope {
  items: {
    doc_id: string;
    file_path: string;
    status: string;
    tags?: string[];
    metadata?: Record<string, unknown>;
  }[];
  total: number;
}

interface DocumentMetadata {
  tags: string[];
  folder: string;
  metadata?: Record<string, unknown>;
  classification?: unknown;
}

interface ActivityEnvelope {
  items: {
    kind: string;
    summary?: string;
    meta?: Record<string, unknown>;
    target?: {
      id?: string;
      label?: string;
    };
  }[];
  total: number;
}

interface TrackStatusEnvelope {
  documents: {
    id: string;
    status: string;
    file_path: string;
  }[];
}

interface TagEntry {
  tag: string;
  status?: string;
}

interface GraphEntity {
  id: string;
  name: string;
  type: string;
  x: number;
  y: number;
  mentions: number;
  sources: number;
  summary: string;
}

interface GraphRelation {
  id: string;
  source: string;
  target: string;
  label: string;
  strength: number;
}

function authHeaders(): Record<string, string> {
  return {
    Accept: 'application/json',
    'X-Twin-Folder': defaultFolder,
    ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
  };
}

function nativeUploadHeaders(): Record<string, string> {
  return {
    Accept: 'application/json',
    'X-Twin-Folder': defaultFolder,
    ...(authToken ? { 'X-API-Key': authToken } : {}),
  };
}

function noAuthHeaders(): Record<string, string> {
  return {
    Accept: 'application/json',
    'X-Twin-Folder': defaultFolder,
  };
}

async function fetchFromBrowser<T>(
  page: Page,
  path: string,
  init: {
    method?: string;
    body?: unknown;
    headers?: Record<string, string>;
  } = {},
): Promise<{ status: number; ok: boolean; body: T; contentType: string | null }> {
  return page.evaluate(
    async ({ base, path: requestPath, headers, method, body }) => {
      const res = await fetch(`${base}${requestPath}`, {
        method,
        headers: {
          ...headers,
          ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        credentials: 'include',
      });
      const text = await res.text();
      let parsed: unknown = text;
      try {
        parsed = text ? JSON.parse(text) : null;
      } catch {
        // Keep raw text for clearer assertion output.
      }
      return {
        status: res.status,
        ok: res.ok,
        body: parsed,
        contentType: res.headers.get('content-type'),
      };
    },
    {
      base: backendUrl,
      path,
      headers: init.headers ?? authHeaders(),
      method: init.method ?? 'GET',
      body: init.body,
    },
  ) as Promise<{ status: number; ok: boolean; body: T; contentType: string | null }>;
}

async function uploadTextDocument(page: Page): Promise<string> {
  const filename = `real-e2e-delete-${Date.now()}.txt`;
  const result = await page.evaluate(
    async ({ base, headers, filename: uploadedFilename }) => {
      const form = new FormData();
      form.append(
        'file',
        new File(
          [
            `Real backend e2e disposable document ${uploadedFilename}.\n` +
              'This file is created only to verify bulk-delete cleanup.\n',
          ],
          uploadedFilename,
          { type: 'text/plain' },
        ),
      );
      const res = await fetch(`${base}/documents/upload`, {
        method: 'POST',
        headers,
        body: form,
        credentials: 'include',
      });
      const text = await res.text();
      let parsed: unknown = text;
      try {
        parsed = text ? JSON.parse(text) : null;
      } catch {
        // keep raw body
      }
      return { status: res.status, ok: res.ok, body: parsed };
    },
    {
      base: backendUrl,
      headers: nativeUploadHeaders(),
      filename,
    },
  );

  expect(result.ok, JSON.stringify(result.body)).toBe(true);
  const body = result.body as { track_id?: string };
  expect(body.track_id, JSON.stringify(body)).toBeTruthy();

  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    const track = await fetchFromBrowser<TrackStatusEnvelope>(
      page,
      `/documents/track_status/${encodeURIComponent(body.track_id!)}`,
      { headers: nativeUploadHeaders() },
    );
    expect(track.ok, JSON.stringify(track.body)).toBe(true);
    const doc = track.body.documents.find((item) => item.file_path.includes(filename));
    if (doc?.id) {
      return doc.id;
    }
    await page.waitForTimeout(2_000);
  }

  throw new Error(`Timed out waiting for uploaded test document ${filename}`);
}

async function firstDocumentId(page: Page): Promise<string | null> {
  const documents = await fetchFromBrowser<DocumentListEnvelope>(page, '/documents');
  expect(documents.ok, JSON.stringify(documents.body)).toBe(true);
  return documents.body.items[0]?.doc_id ?? null;
}

async function resolveRetagTag(page: Page): Promise<string | null> {
  if (configuredRetagTag) {
    return configuredRetagTag;
  }
  const tags = await fetchFromBrowser<TagEntry[]>(page, '/twin/api/tags');
  expect(tags.ok, JSON.stringify(tags.body)).toBe(true);
  return tags.body.find((tag) => tag.status === 'active')?.tag ?? tags.body[0]?.tag ?? null;
}

test.describe('real backend smoke', () => {
  test.skip(!backendUrl, 'Set REAL_BACKEND_URL to run real-backend e2e smoke.');

  test.beforeEach(async ({ page }) => {
    await page.addInitScript(
      ({ apiBaseUrl, lightragBaseUrl, folder }) => {
        for (let i = window.localStorage.length - 1; i >= 0; i -= 1) {
          const k = window.localStorage.key(i);
          if (k && k.startsWith('twin-rag.threads.v')) {
            window.localStorage.removeItem(k);
          }
        }
        window.__twinE2eRuntimeConfig = {
          apiBaseUrl,
          lightragBaseUrl,
          idpLogoutUrl: 'https://idp.example.invalid/logout',
          defaultFolderId: folder,
          maxFolders: 5,
          // The real server always injects this (registry
          // _build_runtime_config); the e2e override must mirror it or the
          // procedure review section silently disappears from the app.
          procedureReviewEnabled: true,
          folders: [
            {
              id: folder,
              label: folder,
              kind: 'primary',
              description: 'Real backend e2e smoke folder',
            },
          ],
          debugUser: {
            sso_subject: 'real-e2e@local',
            email: 'real-e2e@local',
            name: 'Real E2E',
            palier: {
              level: 3,
              label: 'Steward',
              scopes: ['twin:read', 'twin:write', 'twin:approve'],
            },
            folders: [folder],
            idp: 'e2e',
            idp_realm: 'real-backend',
            sub: 'real-e2e',
            session_expires: '2099-12-31T23:59:00Z',
            // Deliberately privileged-looking fixture: /auth-status is the
            // authority, so the static key (no RBAC claims) must still project
            // a Reader and keep admin affordances hidden.
            gateway_scopes: [
              'read:documents',
              'read:query',
              'read:activity',
              'admin:folders',
            ],
          },
        };
      },
      {
        apiBaseUrl: `${backendUrl}/twin/api`,
        lightragBaseUrl: backendUrl,
        folder: defaultFolder,
      },
    );
  });

  test('browser app boots with MSW disabled and reaches real read endpoints', async ({
    page,
  }) => {
    await page.goto('/');
    await expect(
      page.getByRole('button', { name: 'Documents', exact: true }),
    ).toBeVisible();

    const workerRegistrations = await page.evaluate(async () => {
      if (!('serviceWorker' in navigator)) {
        return [];
      }
      return (await navigator.serviceWorker.getRegistrations()).map(
        (registration) => registration.active?.scriptURL ?? '',
      );
    });
    expect(workerRegistrations.filter((url) => url.includes('mockServiceWorker'))).toEqual(
      [],
    );

    const health = await fetchFromBrowser<{ status: string }>(page, '/health');
    expect(health.ok, JSON.stringify(health.body)).toBe(true);
    expect(health.body).toHaveProperty('status');

    const documents = await fetchFromBrowser<DocumentListEnvelope>(page, '/documents');
    expect(documents.ok, JSON.stringify(documents.body)).toBe(true);
    expect(Array.isArray(documents.body.items)).toBe(true);
    expect(typeof documents.body.total).toBe('number');

    const folders = await fetchFromBrowser<unknown[]>(page, '/twin/api/folders');
    expect(folders.ok, JSON.stringify(folders.body)).toBe(true);
    expect(Array.isArray(folders.body)).toBe(true);

    const graphEntities = await fetchFromBrowser<unknown[]>(
      page,
      '/twin/api/graph/entities',
    );
    expect(graphEntities.ok, JSON.stringify(graphEntities.body)).toBe(true);
    expect(Array.isArray(graphEntities.body)).toBe(true);
  });

  test('focus mode consumes the real graph API and cached one-hop topology', async ({
    page,
  }) => {
    await page.goto('/');

    const suffix = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const createdEntityIds: string[] = [];
    let createdRelationId: string | null = null;

    try {
      const createEntity = async (name: string) => {
        const response = await fetchFromBrowser<GraphEntity>(
          page,
          '/twin/api/graph/entities',
          {
            method: 'POST',
            body: {
              name,
              type: 'TECHNOLOGY',
              summary: `Real graph focus contract ${suffix}`,
            },
          },
        );
        expect(response.status, JSON.stringify(response.body)).toBe(201);
        expect(response.body).toMatchObject({
          name,
          type: 'TECHNOLOGY',
        });
        expect(typeof response.body.id).toBe('string');
        expect(typeof response.body.x).toBe('number');
        expect(typeof response.body.y).toBe('number');
        expect(typeof response.body.mentions).toBe('number');
        expect(typeof response.body.sources).toBe('number');
        createdEntityIds.push(response.body.id);
        return response.body;
      };

      const selected = await createEntity(`Focus selected ${suffix}`);
      const neighbor = await createEntity(`Focus neighbor ${suffix}`);
      const unrelated = await createEntity(`Focus unrelated ${suffix}`);

      const relation = await fetchFromBrowser<GraphRelation>(
        page,
        '/twin/api/graph/relations',
        {
          method: 'POST',
          body: {
            source: selected.id,
            target: neighbor.id,
            label: 'FOCUS_NEIGHBOR',
            strength: 0.9,
          },
        },
      );
      expect(relation.status, JSON.stringify(relation.body)).toBe(201);
      expect(relation.body).toMatchObject({
        source: selected.id,
        target: neighbor.id,
        label: 'FOCUS_NEIGHBOR',
      });
      createdRelationId = relation.body.id;

      // This is the #87 end-to-end Activity contract: a real graph mutation
      // produces a record found through the durable target id, then the
      // browser renders that exact record in its Activity journey.
      const activityPath = `/twin/api/activity?${new URLSearchParams({
        'resource.id': selected.id,
        kind: 'graph-entity-edited',
      }).toString()}`;
      const activity = await fetchFromBrowser<ActivityEnvelope>(page, activityPath);
      expect(activity.ok, JSON.stringify(activity.body)).toBe(true);
      expect(
        activity.body.items.some(
          (event) =>
            event.target?.id === selected.id && event.meta?.operation === 'create',
        ),
      ).toBe(true);

      await openTab(page, 'Activity');
      await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible();
      const activitySearch = page.getByLabel('Search events');
      const filteredActivity = page.waitForResponse((response) => {
        const url = new URL(response.url());
        return (
          response.request().method() === 'GET' &&
          url.pathname.endsWith('/twin/api/activity') &&
          url.searchParams.get('q') === selected.name
        );
      });
      await activitySearch.fill(selected.name);
      await filteredActivity;
      const refreshedActivity = page.waitForResponse((response) => {
        const url = new URL(response.url());
        return (
          response.request().method() === 'GET' &&
          url.pathname.endsWith('/twin/api/activity') &&
          url.searchParams.get('q') === selected.name
        );
      });
      await page.getByRole('button', { name: 'Refresh' }).click();
      await refreshedActivity;
      await expect(page.getByRole('heading', { name: selected.name })).toBeVisible();
      await expect(page.getByRole('complementary')).toContainText('created');

      await page
        .getByRole('navigation')
        .getByRole('button', { name: 'Graph', exact: true })
        .click();

      await expect(page.getByTestId(`kg-node-${selected.id}`)).toBeVisible();
      await expect(page.getByTestId(`kg-node-${neighbor.id}`)).toBeVisible();
      await expect(page.getByTestId(`kg-node-${unrelated.id}`)).toBeVisible();

      const cache = await page.evaluate(
        ({ folder }) => {
          const client = (
            window as Window & {
              __TWIN_E2E_QUERY_CLIENT?: {
                getQueryData: (key: readonly string[]) => unknown;
              };
            }
          ).__TWIN_E2E_QUERY_CLIENT;
          return {
            entities: client?.getQueryData(['graph-entities', folder]),
            relations: client?.getQueryData(['graph-relations', folder]),
          };
        },
        { folder: defaultFolder },
      );
      expect(cache.entities).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ id: selected.id }),
          expect.objectContaining({ id: neighbor.id }),
          expect.objectContaining({ id: unrelated.id }),
        ]),
      );
      expect(cache.relations).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: relation.body.id,
            source: selected.id,
            target: neighbor.id,
          }),
        ]),
      );

      await page.getByTestId(`kg-node-${selected.id}`).click();
      await page.getByTestId('kg-focus-mode').click();

      await expect(page.getByTestId('kg-focus-mode')).toHaveAttribute(
        'aria-pressed',
        'true',
      );
      await expect(page.getByTestId(`kg-node-${selected.id}`)).toBeVisible();
      await expect(page.getByTestId(`kg-node-${neighbor.id}`)).toBeVisible();
      await expect(page.getByTestId(`kg-node-${unrelated.id}`)).toBeHidden();

      // Exercise the complete doctrine chain after the initial cache fill:
      // UI mutation -> real DELETE -> query invalidation/refetch -> cache + DOM.
      await page.getByTestId(`kg-node-${neighbor.id}`).click();
      await page.getByTestId('kg-entity-delete').click();

      const deleteResponse = page.waitForResponse((response) => {
        const url = new URL(response.url());
        return (
          response.request().method() === 'DELETE' &&
          url.pathname.endsWith(
            `/twin/api/graph/entities/${encodeURIComponent(neighbor.id)}`,
          )
        );
      });
      const entitiesRefetch = page.waitForResponse((response) => {
        const url = new URL(response.url());
        return (
          response.request().method() === 'GET' &&
          url.pathname === '/twin/api/graph/entities'
        );
      });
      const relationsRefetch = page.waitForResponse((response) => {
        const url = new URL(response.url());
        return (
          response.request().method() === 'GET' &&
          url.pathname === '/twin/api/graph/relations'
        );
      });

      await page.getByTestId('kg-entity-delete').click();
      const [deleted, entitiesReloaded, relationsReloaded] = await Promise.all([
        deleteResponse,
        entitiesRefetch,
        relationsRefetch,
      ]);
      expect(deleted.ok()).toBe(true);
      expect(entitiesReloaded.ok()).toBe(true);
      expect(relationsReloaded.ok()).toBe(true);

      await expect(page.getByTestId(`kg-node-${neighbor.id}`)).toBeHidden();
      await expect(page.getByTestId(`kg-node-${selected.id}`)).toBeVisible();
      await expect
        .poll(() =>
          page.evaluate(
            ({ folder, entityId, relationId }) => {
              const client = (
                window as Window & {
                  __TWIN_E2E_QUERY_CLIENT?: {
                    getQueryData: (key: readonly string[]) => unknown;
                  };
                }
              ).__TWIN_E2E_QUERY_CLIENT;
              const entities =
                (client?.getQueryData([
                  'graph-entities',
                  folder,
                ]) as GraphEntity[]) ?? [];
              const relations =
                (client?.getQueryData([
                  'graph-relations',
                  folder,
                ]) as GraphRelation[]) ?? [];
              return {
                entityPresent: entities.some((entity) => entity.id === entityId),
                relationPresent: relations.some(
                  (candidate) => candidate.id === relationId,
                ),
              };
            },
            {
              folder: defaultFolder,
              entityId: neighbor.id,
              relationId: relation.body.id,
            },
          ),
        )
        .toEqual({ entityPresent: false, relationPresent: false });

      // The test mutation, not finally cleanup, owns these deletions.
      createdEntityIds.splice(createdEntityIds.indexOf(neighbor.id), 1);
      createdRelationId = null;
    } finally {
      if (createdRelationId) {
        await fetchFromBrowser<unknown>(
          page,
          `/twin/api/graph/relations/${encodeURIComponent(createdRelationId)}`,
          { method: 'DELETE' },
        );
      }
      for (const entityId of createdEntityIds.reverse()) {
        await fetchFromBrowser<unknown>(
          page,
          `/twin/api/graph/entities/${encodeURIComponent(entityId)}`,
          { method: 'DELETE' },
        );
      }
    }
  });

  test('auth failures reject missing and invalid bearer tokens when production auth is enabled', async ({
    page,
  }) => {
    test.skip(
      !expectAuth,
      'Set REAL_E2E_EXPECT_AUTH=true when the target backend should enforce auth.',
    );

    await page.goto('/');

    const missing = await fetchFromBrowser<unknown>(page, '/twin/api/folders', {
      headers: noAuthHeaders(),
    });
    expect(missing.status, JSON.stringify(missing.body)).toBe(401);

    const invalid = await fetchFromBrowser<unknown>(page, '/twin/api/folders', {
      headers: {
        ...noAuthHeaders(),
        Authorization: 'Bearer real-e2e-invalid-token',
      },
    });
    expect(invalid.status, JSON.stringify(invalid.body)).toBe(401);
  });

  test('document metadata endpoint returns overlay fields for a real document', async ({
    page,
  }) => {
    await page.goto('/');

    const docId = mutationDocId ?? (await firstDocumentId(page));
    test.skip(!docId, 'No document available for metadata coverage.');

    const metadata = await fetchFromBrowser<DocumentMetadata>(
      page,
      `/twin/api/documents/${encodeURIComponent(docId)}/metadata`,
    );
    expect(metadata.ok, JSON.stringify(metadata.body)).toBe(true);
    expect(Array.isArray(metadata.body.tags)).toBe(true);
    expect(metadata.body.folder).toBe(defaultFolder);
    expect(metadata.body).toHaveProperty('metadata');
  });

  test('retag mutation persists to metadata and emits activity', async ({ page }) => {
    test.skip(
      !mutationDocId,
      'Set REAL_E2E_MUTATION_DOC_ID to run non-destructive retag mutation coverage.',
    );

    await page.goto('/');
    const retagTag = await resolveRetagTag(page);
    if (!retagTag) {
      test.skip(
        true,
        'No existing tag available; set REAL_E2E_RETAG_TAG to run retag coverage.',
      );
      return;
    }

    const before = await fetchFromBrowser<DocumentMetadata>(
      page,
      `/twin/api/documents/${encodeURIComponent(mutationDocId!)}/metadata`,
    );
    expect(before.ok, JSON.stringify(before.body)).toBe(true);
    const hadTag = before.body.tags.includes(retagTag);
    let addedTag = false;

    try {
      const add = await fetchFromBrowser<{ updated: number; failed: string[] }>(
        page,
        '/twin/api/documents/_bulk-retag',
        {
          method: 'POST',
          body: {
            targets: [mutationDocId],
            adds: [retagTag],
            removes: [],
            actor: 'real-e2e',
          },
        },
      );
      expect(add.ok, JSON.stringify(add.body)).toBe(true);
      expect(add.body.failed).toEqual([]);
      expect(add.body.updated).toBe(1);
      addedTag = true;

      const afterAdd = await fetchFromBrowser<DocumentMetadata>(
        page,
        `/twin/api/documents/${encodeURIComponent(mutationDocId!)}/metadata`,
      );
      expect(afterAdd.ok, JSON.stringify(afterAdd.body)).toBe(true);
      expect(afterAdd.body.tags).toContain(retagTag);

      const activity = await fetchFromBrowser<ActivityEnvelope>(
        page,
        '/twin/api/activity',
      );
      expect(activity.ok, JSON.stringify(activity.body)).toBe(true);
      expect(
        activity.body.items.some(
          (event) =>
            event.kind === 'doc-retagged' && event.meta?.doc_id === mutationDocId,
        ),
      ).toBe(true);
    } finally {
      if (!hadTag && addedTag) {
        const cleanup = await fetchFromBrowser<{ updated: number; failed: string[] }>(
          page,
          '/twin/api/documents/_bulk-retag',
          {
            method: 'POST',
            body: {
              targets: [mutationDocId],
              adds: [],
              removes: [retagTag],
              actor: 'real-e2e-cleanup',
            },
          },
        );
        expect(cleanup.ok, JSON.stringify(cleanup.body)).toBe(true);
        expect(cleanup.body.failed).toEqual([]);
      }
    }

    const restored = await fetchFromBrowser<DocumentMetadata>(
      page,
      `/twin/api/documents/${encodeURIComponent(mutationDocId!)}/metadata`,
    );
    expect(restored.ok, JSON.stringify(restored.body)).toBe(true);
    if (!hadTag) {
      expect(restored.body.tags).not.toContain(retagTag);
    } else {
      expect(restored.body.tags).toContain(retagTag);
    }
  });

  test('bulk delete removes an explicitly disposable document', async ({ page }) => {
    test.skip(
      !bulkDeleteDocId && !uploadForDelete,
      'Set REAL_E2E_BULK_DELETE_DOC_ID or REAL_E2E_UPLOAD_FOR_DELETE=true to run bulk-delete coverage.',
    );

    await page.goto('/');

    const targetDocId = uploadForDelete ? await uploadTextDocument(page) : bulkDeleteDocId!;
    const result = await fetchFromBrowser<{ deleted: number; failed: string[] }>(
      page,
      '/twin/api/documents/bulk-delete',
      {
        method: 'POST',
        body: { doc_ids: [targetDocId], actor: 'real-e2e' },
      },
    );
    expect(result.ok, JSON.stringify(result.body)).toBe(true);
    expect(result.body.deleted).toBe(1);
    expect(result.body.failed).toEqual([]);

    const metadata = await fetchFromBrowser<unknown>(
      page,
      `/twin/api/documents/${encodeURIComponent(targetDocId)}/metadata`,
    );
    expect(metadata.status, JSON.stringify(metadata.body)).toBe(404);
  });

  test('optional retrieval UI journey reaches the real backend', async ({ page }) => {
    test.skip(
      process.env.REAL_E2E_QUERY !== 'true',
      'Set REAL_E2E_QUERY=true to run the potentially LLM-backed Twin query journey.',
    );

    await page.goto('/');
    await page.getByRole('button', { name: 'Retrieval', exact: true }).click();
    await page.getByLabel('Query input').fill('health check');

    const [response] = await Promise.all([
      page.waitForResponse(
        (res) =>
          res.url() === `${backendUrl}/twin/api/query` &&
          res.request().method() === 'POST',
        { timeout: 120_000 },
      ),
      page.getByRole('button', { name: 'Send' }).click(),
    ]);

    const body = (await response.json()) as {
      response?: unknown;
      sources?: unknown;
    };
    expect(response.ok(), JSON.stringify(body)).toBe(true);
    expect(typeof body.response).toBe('string');
    expect(Array.isArray(body.sources)).toBe(true);
    await expect(page.locator('.msg-user')).toContainText('health check');
    await expect(page.locator('.msg-assistant')).toBeVisible();
  });

  test('procedure-typed upload parks a real failed bundle; credential-only UI stays read-only and server reject persists', async ({
    page,
  }) => {
    // Upload + park + server rejection + reload does not fit the default 30s budget.
    test.setTimeout(120_000);
    // Composition test for the approval workflow against the REAL seam +
    // fcntl-locked procedure store. The CI backend installs neither
    // [procedure] nor [vision] and has no model credentials — which is the
    // point: an explicit `X-Twin-Doc-Type: procedure` request must route to
    // the profile anyway and park an actionable FAILED bundle. A silent
    // standard enqueue here would be an approval bypass (and would start
    // indexing against the intentionally disabled LLM credentials).
    await page.goto('/');
    await expect(
      page.getByRole('button', { name: 'Documents', exact: true }),
    ).toBeVisible();

    const filename = `real-e2e-procedure-${Date.now()}.pdf`;

    await page.getByRole('button', { name: 'Add source' }).click();
    await page.getByTestId('addsource-file-input').setInputFiles({
      name: filename,
      mimeType: 'application/pdf',
      buffer: Buffer.from(`%PDF-1.4 real e2e parked procedure ${filename}`),
    });
    await page.getByTestId('addsource-doc-type').selectOption('procedure');
    await page.getByRole('button', { name: 'Add 1 source' }).click();

    // The parked bundle surfaces through the real /twin/api/procedures.
    // (`processing` is skipped: a deps-less park writes `failed` directly,
    // but a local run with real vision deps transits through processing.)
    let bundleId: string | null = null;
    const parkDeadline = Date.now() + 60_000;
    while (Date.now() < parkDeadline) {
      const list = await fetchFromBrowser<
        { id: string; file_name: string; state: string }[]
      >(page, '/twin/api/procedures');
      expect(list.ok, JSON.stringify(list.body)).toBe(true);
      const bundle = list.body.find(
        (b) => b.file_name === filename && b.state !== 'processing',
      );
      if (bundle) {
        expect(bundle.state, JSON.stringify(bundle)).toBe('failed');
        bundleId = bundle.id;
        break;
      }
      await page.waitForTimeout(2_000);
    }
    expect(bundleId, 'parked bundle never appeared in /twin/api/procedures').toBeTruthy();

    // The upload reconciliation resolves it into a review card without a
    // manual refresh, with the failed pill.
    const card = page.getByTestId(`pending-proc-${bundleId}`);
    await expect(card).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId(`pending-proc-state-${bundleId}`)).toContainText(
      'Procedure failed',
    );

    // The backend is protected by a static root key. It can authorize admin
    // routes, but /auth-status has no RBAC claims to project: the frontend must
    // therefore stay Reader instead of trusting the injected debug Steward.
    await expect(
      page.getByTestId(`pending-proc-review-${bundleId}`),
    ).toHaveCount(0);
    await expect(
      page.getByTestId(`pending-proc-reject-${bundleId}`),
    ).toHaveCount(0);

    // Preserve the real store composition coverage through the authoritative
    // server seam. The bearer remains server-authorized even though the UI
    // correctly refuses to manufacture a Steward identity from it.
    const detail = await fetchFromBrowser<{ file_name: string }>(
      page,
      `/twin/api/procedures/${bundleId}`,
    );
    expect(detail.ok, JSON.stringify(detail.body)).toBe(true);
    expect(detail.body.file_name).toBe(filename);

    const rejected = await fetchFromBrowser<{ state: string }>(
      page,
      `/twin/api/procedures/${bundleId}/reject`,
      { method: 'POST', body: { comment: 'real e2e reject' } },
    );
    expect(rejected.ok, JSON.stringify(rejected.body)).toBe(true);
    expect(rejected.body.state).toBe('rejected');

    // Full reload: the rejected state survives — file-backed store, not
    // client cache or optimistic UI.
    await page.reload();
    await expect(page.getByTestId(`pending-proc-state-${bundleId}`)).toContainText(
      'Procedure rejected',
      { timeout: 15_000 },
    );

    // The seam never enqueued: no document exists for the parked file. The
    // shim filters server-side BEFORE paginating, so the filtered `total`
    // covers the whole corpus — a first-page-only scan could miss a
    // wrongly-created document beyond the page size on a populated backend.
    const documents = await fetchFromBrowser<DocumentListEnvelope>(
      page,
      `/documents?q=${encodeURIComponent(filename)}`,
    );
    expect(documents.ok, JSON.stringify(documents.body)).toBe(true);
    expect(documents.body.total, JSON.stringify(documents.body.items)).toBe(0);
    expect(documents.body.items).toEqual([]);

    // And the store itself reports healthy after the whole journey.
    const health = await fetchFromBrowser<{
      degraded: boolean;
      quarantine_files: string[];
    }>(page, '/twin/api/procedures/store/health');
    expect(health.ok, JSON.stringify(health.body)).toBe(true);
    expect(health.body.degraded).toBe(false);
  });

  test('vision settings surface mirrors the real backend readiness contract', async ({
    page,
  }) => {
    await page.goto('/');
    await expect(
      page.getByRole('button', { name: 'Documents', exact: true }),
    ).toBeVisible();

    const settings = await fetchFromBrowser<{
      min_ocr_chars: number;
      drop_classes: string[];
      procedure_enabled: boolean;
      procedure_available: boolean;
    }>(page, '/twin/api/settings/vision');
    expect(settings.ok, JSON.stringify(settings.body)).toBe(true);

    // The Settings > Vision toggle renders the REAL backend state, not a
    // fixture: enabled mirrors procedure_enabled, readiness mirrors
    // procedure_available.
    await openTab(page, 'Settings');
    await page.getByTestId('settings-rail-vision').click();
    await expect(page.getByTestId('settings-vision')).toBeVisible();
    const toggle = page.getByTestId('settings-vision-procedure-toggle');
    await expect(toggle).toHaveAttribute(
      'aria-checked',
      String(settings.body.procedure_enabled),
    );

    if (!settings.body.procedure_available) {
      await expect(
        page.getByTestId('settings-vision-procedure-unavailable'),
      ).toBeVisible();
      if (settings.body.procedure_enabled) {
        // Enabled BEFORE the prerequisites went away: the toggle must stay
        // operable so an admin can turn it off, and the backend only 409s
        // the false→true transition — a PUT keeping true is a legitimate
        // no-op, so no 409 is expected here.
        await expect(toggle).toBeEnabled();
      } else {
        // CI backend (no [procedure]/[vision] deps, toggle off): enabling is
        // refused end to end — locked toggle in the UI, 409 from the API on
        // the false→true attempt. The PUT echoes the current stored values
        // so a (correctly) rejected request cannot mutate anything even if
        // the guard regressed.
        await expect(toggle).toBeDisabled();
        const put = await fetchFromBrowser<unknown>(
          page,
          '/twin/api/settings/vision',
          {
            method: 'PUT',
            body: {
              min_ocr_chars: settings.body.min_ocr_chars,
              drop_classes: settings.body.drop_classes,
              procedure_enabled: true,
            },
          },
        );
        expect(put.status, JSON.stringify(put.body)).toBe(409);
      }
    }
  });
});
