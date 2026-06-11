import { expect, test, type Page } from '@playwright/test';

const backendUrl = process.env.REAL_BACKEND_URL?.replace(/\/$/, '') ?? '';
const authToken = process.env.REAL_BACKEND_AUTH_TOKEN ?? process.env.VITE_AUTH_TOKEN;
const defaultFolder = process.env.REAL_BACKEND_FOLDER ?? 'default';
const expectAuth = process.env.REAL_E2E_EXPECT_AUTH === 'true';
const mutationDocId = process.env.REAL_E2E_MUTATION_DOC_ID;
const configuredRetagTag = process.env.REAL_E2E_RETAG_TAG;
const bulkDeleteDocId = process.env.REAL_E2E_BULK_DELETE_DOC_ID;
const uploadForDelete = process.env.REAL_E2E_UPLOAD_FOR_DELETE === 'true';

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

function authHeaders(): Record<string, string> {
  return {
    Accept: 'application/json',
    'X-Twin-Folder': defaultFolder,
    ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
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
      headers: authHeaders(),
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
        window.localStorage.setItem(
          'twin.onboarding.v1',
          JSON.stringify({ step: 'completion', dismissed: true, tasks: [] }),
        );
        window.localStorage.removeItem('twin-rag.threads.v3');
        window.__twinE2eRuntimeConfig = {
          apiBaseUrl,
          lightragBaseUrl,
          idpLogoutUrl: 'https://idp.example.invalid/logout',
          defaultFolderId: folder,
          maxFolders: 5,
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
            gateway_scopes: ['read:documents', 'read:query', 'read:activity'],
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

    const targetDocId = bulkDeleteDocId ?? (await uploadTextDocument(page));
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
});
