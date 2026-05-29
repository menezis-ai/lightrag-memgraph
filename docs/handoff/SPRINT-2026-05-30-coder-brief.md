# Brief sprint coder — React port maquette JSX (29/05 17h → 01/06 12h)

**Destinataire** : agent codeur (Claude Code, Codex, ou humain)
**Auteur** : Julien (via Claude Opus 4.7) — synthèse des décisions PO/architecture jusqu'au 2026-05-29 16h
**Deadline** : démo lundi 2026-06-01 14h00 avec Fabrice + Manu + Louis HORVAT + Eric
**Branche cible** : `feat/webui-port-weekend-29-05` (à créer depuis `stable/0.6.x` post-#141)
**Tracker Forgejo** : [#146 M0.6 Weekend React port sprint](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/146)

---

## 1. Contexte stratégique (lire absolument)

Tu vas porter une maquette JSX (`maquette-deploy/patches/site-overlay/*.jsx`, déployée sur https://maquette.sigilum.fr/) vers le **fork React TS existant** `lightrag_webui_twin/` (Vite + Bun + React 19 + TypeScript strict + Tailwind v3 + TanStack Query + Vitest).

**Pourquoi ce port** : décision Anas 2026-05-29 (5e validation maquette). Au lieu de remettre le JSX à l'équipe Eureka (Chafi/Yazid) qui le réécrirait en CRA + MUI + Redux (stack déprécié 2023), on porte nous-mêmes sur stack moderne. Position défendable techniquement, voir [`docs/presentations/pitch-fabrice-2026-06-01.md`](../presentations/pitch-fabrice-2026-06-01.md) Slide 4 sub-section "Stack technique".

**Scope critique — LightRAG only** :
- ✅ File upload + LightRAG ingestion native
- ✅ Twin overlay : tags, audit/activity, workspaces, notifications, MyAccess pill, graph viewer
- ❌ **Pas de Crosspoint / RAG 1.5 / Confluence / SharePoint connectors** (deferred, ConnectionsTab #147 hors sprint)

---

## 2. Architecture cible

```
LightRAG runtime (1.4.9.11)
  ├─ Backend RAG : storage Memgraph (notre patch register() v1.1.0)
  ├─ API native LightRAG : /documents, /query, /insert, /health, /pipeline_status
  ├─ Mount /webui → notre lightrag_webui_twin/dist (via register replace_ui=True)
  └─ Mount /twin/api → notre sub-app Twin (via register mount_server=True)
       └─ Twin overlay : /tags, /audit-events, /activity, /notifications,
                         /workspaces, /graph/*, /auth/logout
```

**Le patch register() est mergé** sur `stable/0.6.x` (PR #141, commit principal). Ton boulot ne touche que `lightrag_webui_twin/`.

**Anti-patterns à éviter** :
- ❌ Hardcoder un mock backend en JS (ServiceWorker en prod = #109 à virer)
- ❌ Réimplémenter un RAG côté front
- ❌ Stocker des docs en localStorage (sauf state UI éphémère type "active tab")
- ❌ Schémas incompatibles avec LightRAG (utilise `track_id`, `doc_id`, `file_path`, pas `id` inventé)

---

## 3. Inventaire — déjà fait sur `lightrag_webui_twin/`

15 composants TSX existent :
- ActivityTab, AddSourceModal, ApiTab, DocumentsTab, GraphTab, Icon
- RetagModal, RetrievalTab, StatusBadge, TagActionModal, TagChip, TagsTab
- ToastViewport, Topbar, TweaksPanel
- App.tsx, main.tsx
- Hooks : `useModalA11y`, `useUrlParam`
- 10 fixtures typés + types + api/queries + mocks/handlers + Vitest setup
- 229 tests verts (lance `cd lightrag_webui_twin && bun run test:run`)

---

## 4. Sprint — 6 issues à fermer (ordre d'attaque imposé)

### Étape 0 — Foundation path alignment LightRAG natif vs Twin overlay (~1h)

**Issue dédiée Forgejo** (à créer par Julien post-brief, numéro à inscrire ici une fois ouverte : `#TBD`). Étape **bloquante** — doit être mergée avant toute autre étape du sprint sinon le reste des composants utilisera les mauvais paths API et les fixtures dérivantes du schéma LightRAG.

**Fichiers à toucher** : `lightrag_webui_twin/src/api/resources.ts`, `lightrag_webui_twin/src/mocks/handlers.ts`, `lightrag_webui_twin/src/types/document.ts`.

**Diff conceptuel** :

```typescript
// AVANT (mélange tout)
apiFetch('/documents', ...)
apiFetch('/tags', ...)
apiFetch('/activity', ...)

// APRÈS
// LightRAG natif (PAS de prefix /twin/api)
apiFetch('/documents', ...)
apiFetch('/query', ...)
apiFetch('/health', ...)

// Twin overlay (préfixe /twin/api)
apiFetch('/twin/api/tags', ...)
apiFetch('/twin/api/audit-events', ...)
apiFetch('/twin/api/notifications', ...)
apiFetch('/twin/api/workspaces', ...)
apiFetch('/twin/api/graph/entities', ...)
apiFetch('/twin/api/graph/relations', ...)
```

**Schéma Document à aligner** sur LightRAG `DocStatus`. Lire `lightrag/api/routers/document.py` upstream — les champs attendus :

```typescript
export interface Document {
  doc_id: string;              // PAS "id"
  track_id: string | null;
  file_path: string;
  content_summary: string;
  content_length: number;
  status: 'PENDING' | 'PROCESSING' | 'PROCESSED' | 'FAILED';
  chunks_count: number | null;
  created_at: string;
  updated_at: string;
  error_msg: string | null;
  metadata: Record<string, unknown>;
  // Champs additionnels Twin overlay (servis par /twin/api/documents/{id}/metadata)
  tags: string[];
  workspace: string;
  review?: { state: 'pending-review' | 'approved' | 'rejected'; requested_by: string; requested_at: string; justification: string; };
}
```

**MSW handlers** : mettre à jour `lightrag_webui_twin/src/mocks/handlers.ts` pour servir les paths préfixés `/twin/api/*` côté Twin overlay. Garder `/documents` etc. pour LightRAG natif.

### Étape 1 — #110 useAuth hook + injectable apiBaseUrl (~2h)

Critique : tous les autres composants en dépendent.

**Fichiers** :
- `lightrag_webui_twin/src/hooks/useAuth.ts` (nouveau)
- `lightrag_webui_twin/src/hooks/useAuth.test.tsx`
- `lightrag_webui_twin/src/api/client.ts` (refactor)
- `lightrag_webui_twin/index.html` (ajout `<script>window.__twinConfig = ...</script>`)
- `lightrag_webui_twin/src/types/auth.ts` (nouveau)

**Spec** :

```typescript
// src/types/auth.ts
export interface Palier {
  level: 1 | 2 | 3;
  label: 'Reader' | 'Contributor' | 'Steward';
  scopes: readonly string[];
}

export interface AuthenticatedUser {
  email: string;
  name: string;
  palier: Palier;
  workspaces: readonly string[]; // workspace IDs accessibles via MyAccess claim
  sso_subject: string;
}

// src/hooks/useAuth.ts
export function useAuth(): {
  user: AuthenticatedUser | null;
  isAuthenticated: boolean;
  signout: () => Promise<void>;
};

// signout doit :
// 1. await fetch('/twin/api/auth/logout', { method: 'POST' })
// 2. clear le queryClient (toutes les queries cached)
// 3. window.location.href = `${IDP_LOGOUT_URL}?redirect_uri=${window.location.origin}`
// 4. PAS de toast (la redirection se charge de la confirmation visuelle)
```

**Runtime config injectable** :

```html
<!-- index.html — le backend Twin (sub-app FastAPI mountée via register(mount_server=True))
     sert ce fichier et fait une substitution Python à la volée AVANT envoi au navigateur.
     PAS de Vite SSG, PAS d'inline statique au build. Le placeholder `__TWIN_CONFIG_JSON__`
     est remplacé à chaque request par les valeurs runtime (env vars BNP + JWT claims). -->
<script>
  window.__twinConfig = __TWIN_CONFIG_JSON__;
</script>
```

**Côté FastAPI (référence pour l'agent backend, hors scope sprint)** :
```python
# server module — extrait conceptuel, ne PAS implémenter dans ce sprint
@app.get("/webui/", response_class=HTMLResponse)
async def serve_webui(request: Request):
    html = (WEBUI_DIST / "index.html").read_text()
    config = {
        "apiBaseUrl": "/twin/api",
        "lightragBaseUrl": "/api",
        "idpLogoutUrl": os.environ["TWIN_IDP_LOGOUT_URL"],
    }
    return html.replace("__TWIN_CONFIG_JSON__", json.dumps(config))
```

**Côté webui (ce sprint)** : tu poses juste le placeholder `__TWIN_CONFIG_JSON__` dans `index.html` et tu lis `window.__twinConfig` dans `useAuth` / `api/client`. En dev (`bun run dev`), Vite ne fait pas la substitution — fallback vers un objet de dev hardcodé dans `src/config/devConfig.ts` (lu UNIQUEMENT si `import.meta.env.DEV === true` ET `window.__twinConfig === '__TWIN_CONFIG_JSON__'`, signe que la substitution n'a pas eu lieu).

**Tests** : `useAuth` lit le JWT depuis cookie HttpOnly (en prod) ou `window.__twinDebugUser` (en dev), expose `palier` et `workspaces`. Couvrir signout flow → vérifier que `queryClient.clear()` est appelé + `window.location.href` set.

### Étape 2 — #102 SettingsTab + sous-sections (~3-4h)

**Décisions PO importantes** (Louis 2026-05-28) :
- ❌ **PAS** de section "Tokens" — interdite. La section s'appelle **"OAuth2 clients"** dans le rail label, et le contenu est ❌ retiré totalement du frontend ET du backend.
- ❌ **PAS** de section "Members" éditable — read-only. La gestion des members vit dans MyAccess.
- ❌ **PAS** de "API generation" — interdite.

**Sections gardées** :
- `Profile` (read-only, lit `useAuth`)
- `Workspace` (env vars read-only)
- `Providers` (LLM, Embedder, Reranker — Configure ouvre vrais panneaux, pas no-op #SET-01)
- `Danger` (Delete Workspace stub)

**Fichiers** :
- `lightrag_webui_twin/src/components/SettingsTab.tsx`
- `lightrag_webui_twin/src/components/Settings/ProfileSection.tsx`
- `lightrag_webui_twin/src/components/Settings/WorkspaceSection.tsx`
- `lightrag_webui_twin/src/components/Settings/ProvidersSection.tsx`
- `lightrag_webui_twin/src/components/Settings/DangerSection.tsx`
- `lightrag_webui_twin/src/components/SettingsTab.test.tsx`

**Source JSX** : `~/Downloads/design_twinrag_backend/settings.jsx` (~700 LOC, ignorer les sections retirées).

### Étape 3 — #103 OnboardingWizard (~3-4h)

Source : `~/Downloads/design_twinrag_backend/onboarding.jsx`. 6 étapes : welcome → kb empty → checklist 5 tasks → first source upload → first query → completion.

**Fichiers** :
- `lightrag_webui_twin/src/components/OnboardingWizard.tsx`
- `lightrag_webui_twin/src/components/Onboarding/Welcome.tsx`, `Checklist.tsx`, `Completion.tsx`
- `lightrag_webui_twin/src/hooks/useOnboarding.ts` (state machine local persisted)
- Tests RTL pour le flow complet

### Étape 4 — #104 Topbar enrichment (~2-3h)

⚠️ **Spec révisée 2026-05-29** (décision PO suite Louis + post-recette Alberto) :
- **PAS de `PalierSwitcher`** — pas de UI dropdown pour changer le palier. Le palier vient du JWT MyAccess, point.
- **PAS de `MyAccessPill`** — supprimé. C'était un gimmick maquette (afficher "Steward / Contributor / Reader" en pill colorée à côté du nom). N'a pas sa place dans une app prod : le palier ne se "porte" pas visuellement comme un badge, il conditionne uniquement les capacités UI (boutons activés/masqués). Si un composant doit faire un gating, il lit `useAuth().user.palier` directement.
- `WorkspaceSwitcher` lit `useAuth().user.workspaces`, click → `setActiveWorkspace(id)` (voir #154 spec workspace switching)
- `TodoBell` consomme `GET /twin/api/notifications` (refresh 20s via TanStack Query refetchInterval)
- `SystemStatusIndicator` consomme `GET /api/health` (LightRAG natif) + `GET /twin/api/health` (overlay) — affiche le pire des deux
- Pour la QA Alberto : 3 comptes démo seed (steward.demo / contrib.demo / reader.demo) — voir #151

**Fichiers** :
- `lightrag_webui_twin/src/components/Topbar.tsx` (enrichir l'existant)
- `lightrag_webui_twin/src/components/Topbar/WorkspaceSwitcher.tsx`
- `lightrag_webui_twin/src/components/Topbar/TodoBell.tsx`
- `lightrag_webui_twin/src/components/Topbar/SystemStatusIndicator.tsx`

### Étape 5 — #105 Documents detail panel (~4-5h)

3 onglets : `Chunks` / `Lineage` / `Audit`.

- **Chunks** : `GET /api/documents/{id}/chunks` (LightRAG natif). Masquer 80% du texte si `classification > internal` (cf. spec compliance Louis).
- **Lineage** : metadata + uploader + source URL
- **Audit** : `GET /twin/api/audit-events?resource.id={doc_id}` filtré

**Footer actions** :
- Retag (ouvre RetagModal existant)
- View raw (gated par classification — montre prompt rappel "Internal max")
- Re-process (POST `/api/documents/{id}/scan`)
- **Delete (cascade)** — voir spec révisée #149 : delete individuel ET multi-sélection

⚠️ Spec révisée #149 : Le delete EVIDEMMENT inclut suppression individuelle. Multi-sélection delete dans la barre d'actions `DocumentsTab` (réintroduire — était en v1, perdu en v2).

### Étape 6 — #106 Pending docs section (~2h)

Section au top de `DocumentsTab` avec docs en `review.state == 'pending-review'`. Actions par card :
- **Approve** — `POST /twin/api/documents/{id}/approve` (mutation TanStack Query)
- **Edit & Approve** — ⚠️ Spec #150 : **doit ouvrir une modale** d'édition (régression v2 où elle ne s'ouvrait plus). À l'approbation, le doc est indexé avec les édits.
- **Reject** — modale avec raison obligatoire (cohérent #TAG-07 du rapport Alberto qui demande la même rigueur sur Reject tag)
- **Simulate change** — preview avant d'approuver

---

## 5. Pattern obligatoire RC-1 (golden path)

Rapport Alberto 2026-05-29 identifie le **pattern RC-1** comme cause racine de 18 bugs S2 sur la maquette JSX : *"UI optimiste sans commit"* — handler émet toast + event sans `await store.write` ni `update state`.

**À ne JAMAIS reproduire dans le port React**. Pattern obligatoire pour CHAQUE mutation :

```typescript
// ✅ BON pattern — TanStack Query useMutation
import { useMutation, useQueryClient } from '@tanstack/react-query';

function MyComponent() {
  const queryClient = useQueryClient();
  const { mutateAsync, isPending } = useMutation({
    mutationFn: (payload: ApplyTagPayload) => api.applyTag(payload),
    onSuccess: (result) => {
      // 1. invalider les queries impactées → refetch automatique
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      queryClient.invalidateQueries({ queryKey: ['activity'] });
      // 2. event métier (optionnel, pour le webhook hub)
      emit('tag.applied', { tag: result.tag, count: result.count });
      // 3. toast APRÈS succès, jamais avant
      toast.success(`${result.tag} applied to ${result.count} sources`);
    },
    onError: (err: ApiError) => {
      toast.error(`Failed to apply tag: ${err.message}`);
    },
  });

  return (
    <button
      disabled={isPending}
      onClick={() => mutateAsync({ tag: 'oracle', sources: selectedIds })}
    >
      {isPending ? 'Applying…' : 'Apply tag'}
    </button>
  );
}
```

```typescript
// ❌ MAUVAIS pattern — c'est exactement le RC-1 d'Alberto
async function handleClick() {
  toast.success('Tag applied to 12 sources'); // toast SANS attendre
  emit('tag.applied'); // event sans write
  // ❌ pas de store update, pas de refetch
}
```

---

## 6. Conventions code et tests

### Stack pinning
- **Bun 1.3.6** (pinné dans `.forgejo/workflows/ci.yml`, ne JAMAIS bump latest sans test CI)
- **React 19** + TypeScript strict (`tsconfig.app.json`)
- **Tailwind v3** — pas Tailwind v4 (cassures breaking dans v4)
- **TanStack Query v5** déjà installé
- **MSW v2** déjà installé pour les mocks
- **Vitest** — pas Jest

### Naming
- Composants : `PascalCase.tsx`, leurs tests `PascalCase.test.tsx` à côté
- Hooks : `useFoo.ts`
- Fixtures : `src/fixtures/<domain>.ts`
- Types : `src/types/<domain>.ts`

### Style
- ESLint config existante, respecter
- Prettier déjà configuré, `bun run format` si besoin
- Pas de classes CSS custom hors `src/styles/`, tout en Tailwind utilities
- Design tokens : `--twin-*` dans `src/styles/tokens.css` (déjà setup pour light + dark)

### Tests
- Convention : un `.test.tsx` par composant TSX, doit couvrir le golden path + 1 edge case minimum
- `@testing-library/react` + `@testing-library/user-event`
- `userEvent.type(input, 'foo{Enter}')` **race sur slow CI** — split en deux calls + `waitFor` (pitfall documenté dans CLAUDE.md)
- ARIA live regions duplique le texte visible — scope `getByText` à un container ou `data-testid`

### Commandes
```bash
cd lightrag_webui_twin
bun install               # installer
bun run dev               # dev server (vite)
bun run typecheck         # tsc strict
bun run test:run          # vitest run (CI)
bun run test              # vitest watch
bun run build             # bundle prod → dist/
```

**Avant chaque commit** : `bun run typecheck && bun run test:run && bun run build` doit passer. CI bloquera sinon.

---

## 7. Conventions Git

### Branche
Pour le sprint : `feat/webui-port-weekend-29-05`, branchée depuis `stable/0.6.x` après merge #141.

```bash
git checkout stable/0.6.x
git pull bunker stable/0.6.x
git checkout -b feat/webui-port-weekend-29-05
```

### Commits
Convention conventionnelle : `<type>(<scope>): <subject>` — exemples :

- `feat(webui): add SettingsTab + sub-sections (closes #102)`
- `feat(webui): add OnboardingWizard 6-step flow (closes #103)`
- `fix(webui): align Document schema with LightRAG DocStatus`
- `refactor(api): split LightRAG native paths from /twin/api/* overlay`
- `test(webui): cover SettingsTab gating by palier`

Co-Authored-By footer obligatoire si tu es agent IA :
```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

### Push
Forgejo `bunker` uniquement (GitHub `origin` archivé). PR vers `stable/0.6.x` :

```bash
git push -u bunker feat/webui-port-weekend-29-05
# puis ouvrir PR via Forgejo UI ou MCP forgejo
```

### CI
Pin CI matrix actuelle : LightRAG `1.4.9.11 / 1.4.11 / 1.4.12` × Memgraph `3.10.1`. **NE PAS** modifier le matrix. Webui-tests job tourne sur Bun 1.3.6 — pin verrouillé.

---

## 8. Pièges connus (lire si tu blockes)

### CSS variables design tokens
Les tokens `--twin-*` (couleurs, espaces, radii) sont dans `src/styles/tokens.css`. `tailwind.config.js` les expose comme utilities (`bg-twin-accent`, `text-twin-green-700`). Ne JAMAIS hardcoder des couleurs hex dans les composants.

### happy-dom + localStorage
happy-dom 20.x sur Bun ne ship pas Storage. `src/test/setup.ts` provisionne un in-memory localStorage. Si tu vois `localStorage is not defined` en test, ajouter au setup.

### MSW handlers
Quand tu ajoutes un nouveau path API, ajouter le handler dans `src/mocks/handlers.ts`, sinon les tests qui appellent ce path échouent en silence (network mock not found).

### useModalA11y autofocus race
Le hook focus le premier input 30ms après mount. Si tu testes typing dans un AUTRE input que le premier, attendre 60ms et appeler `.focus()` explicitement avant `userEvent.type`.

### Race `userEvent.type` avec `{Enter}`
Pas `userEvent.type(input, 'foo{Enter}')` sur slow CI. Split :
```typescript
await userEvent.type(input, 'foo');
await waitFor(() => expect(input).toHaveValue('foo'));
await userEvent.type(input, '{Enter}');
```

### Tests fixtures vs LightRAG real schema
Les fixtures actuelles ont `id: string` qui mélange `doc_id` LightRAG natif et `id` MOCK. À l'étape 0 du sprint, aligner sur `doc_id`. Les composants existants vont casser → fixer.

---

## 9. Critère de done global du sprint

Après tes 6 issues fermées :

```bash
cd lightrag_webui_twin
bun run typecheck    # ✅ 0 erreur TS
bun run test:run     # ✅ 229 + nouveaux tests, 0 fail
bun run build        # ✅ build prod OK
```

Et :
- [ ] `feat/webui-port-weekend-29-05` poussée + PR ouverte vers `stable/0.6.x`
- [ ] PR référence les 6 issues fermées dans la body
- [ ] CI Forgejo verte sur la PR
- [ ] Démo locale fonctionne : `bun run dev` → http://localhost:5173 → tu peux naviguer dans Settings, Onboarding, Topbar, Documents (panel + pending), tous les pattern useMutation en place
- [ ] Pas de régression sur les 229 tests existants

---

## 10. Quoi faire si tu blockes

1. **Premier réflexe** : lire l'issue Forgejo concernée (#102, #103, #104, #105, #106, #110) — elle a les critères d'acceptance détaillés
2. **Deuxième réflexe** : lire le rapport Alberto `docs/audits/TwinRAG - Rapport de recette v2.md` — il documente les bugs JSX à ne PAS reproduire
3. **Doctrine produit** : `~/.claude/projects/-Users-julien-twindb-lightrag-memgraph/memory/MEMORY.md` index toutes les décisions par stakeholder
4. **Schémas LightRAG** : repo upstream `lightrag/api/routers/document.py`, `lightrag/base.py` pour `DocStatus`
5. **Sweden bundle JSX source** : `~/Downloads/design_twinrag_backend/*.jsx` — référence visuelle/logique uniquement, NE PAS copier-coller en JSX, RÉ-ÉCRIRE en TS strict avec pattern useMutation

---

## 11. Refs Forgejo

- Tracker sprint : [#146](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/146)
- Issues sprint : [#102](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/102), [#103](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/103), [#104](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/104), [#105](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/105), [#106](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/106), [#110](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/110)
- Issues décisions PO Alberto : [#148](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/148), [#149](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/149), [#150](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/150), [#151](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/151), [#153](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/153), [#154](http://192.168.1.61:3000/julien/twindb-lightrag-memgraph/issues/154)
- Rapport Alberto : `docs/audits/TwinRAG - Rapport de recette v2.md`
- Pitch deck Fabrice lundi : `docs/presentations/pitch-fabrice-2026-06-01.md`
- Forgejo URL : `http://192.168.1.61:3000/julien/twindb-lightrag-memgraph`
