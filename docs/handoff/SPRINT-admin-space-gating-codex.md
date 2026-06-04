# Sprint handoff — Admin-only Space CRUD (Codex)

**Branche :** `feat/webui-graph-real-memgraph` (en cours, *ne pas* cut nouvelle branche, tout commiter ici jusqu'au merge sur `stable/0.6.x`).
**Auteur du brief :** Claude (cette session).
**Date :** 2026-06-04.
**Travail simultané :** Claude shippe **lots A + B** (backend FastAPI dep + tests pytest) sur la même branche en parallèle. **Codex : lots C + D ci-dessous, tu n'as pas besoin d'attendre A+B pour démarrer C+D — le contrat ci-dessous est gelé.**

---

## Contexte minimal (Couche 3 §3.3)

Les routes `POST/PATCH/DELETE /twin/api/spaces` ont été livrées le 2026-06-04 (commit `173b09f`) sans aucune garde RBAC au-delà du `bind_request_space` router-level. N'importe quel porteur de token IdP valide (ou n'importe quel client en mode IdP-dormant) peut donc créer/modifier/supprimer un Twin Space. `WEBUI-WIRING-PLAN.md` flagge le gap explicitement l. 289 :

> Still open for spaces: JWT/MyAccess admin-only gating for runtime Space CRUD.

Doctrine confirmée par Louis HORVAT 2026-05-28 : **pas de RBAC interne Twin**, tout vient des claims MyAccess transportés par le JWT (cf. `lightrag_webui_twin/src/types/auth.ts:1-9`). Donc le gate doit lire ce que l'IdP nous donne, pas inventer un référentiel parallèle.

---

## Contrat gelé (à respecter pour C+D)

### Côté backend (lot A — implémenté par Claude)

1. Nouvel env : **`TWIN_IDP_ADMIN_GROUPS`** — CSV de groupes MyAccess qui confèrent l'autorité admin. Défaut quand absent : `twin-admin,twin-steward` (cohérent avec la doctrine paliers de Louis : Steward = admin par défaut). Si l'env est défini *vide*, on tombe en `frozenset()` ⇒ personne n'est admin.
2. Dans `IdpConfig` : nouveau champ `admin_groups: frozenset[str]`. `from_env()` parse `TWIN_IDP_ADMIN_GROUPS`.
3. Dans `claims_to_user()` : si `set(groups) ∩ admin_groups ≠ ∅`, injecter `"admin:spaces"` dans `gateway_scopes` (PAS dans `palier.scopes` — cf. convention existante `admin:tags`, `admin:workspace` dans `gateway_scopes`, cf. `lightrag_webui_twin/src/config/devConfig.ts:55-62`).
4. Nouveau dep FastAPI **`require_admin_user(request)`** :
   - **IdP dormant (pas de `TWIN_IDP_JWKS_URL`)** → retourne `None` sans 403. C'est volontaire : dev / OVH standalone / maquette ne doivent pas casser.
   - **IdP actif** → résout l'user via `require_idp_user`. Si `"admin:spaces"` pas dans `user["gateway_scopes"]` ⇒ HTTPException 403 `{"detail": "Admin scope 'admin:spaces' required"}`. Sinon retourne le user dict.
5. Wiring : `Depends(require_admin_user)` collé sur `POST/PATCH/DELETE /spaces` dans `webui_router.py` (lignes 676, 728, 768). `GET /spaces` reste ouvert.

### Côté frontend (lot C — pour Codex)

**Capability check :**

```ts
// lightrag_webui_twin/src/lib/permissions.ts (à créer, ou inliner si tu préfères)
import type { AuthenticatedUser } from '../types/auth';

export function canManageSpaces(user: AuthenticatedUser | null): boolean {
  // Pas d'utilisateur (IdP dormant, dev MSW sans debugUser) → autorisé.
  // Le backend en mode dormant ne 403 pas non plus, on reste cohérents.
  if (!user) return true;
  return user.gateway_scopes.includes('admin:spaces');
}
```

**Gating UI dans `lightrag_webui_twin/src/components/Settings/SpacesAdminSection.tsx` :**

- Recevoir le `user: AuthenticatedUser | null` en prop (ajoute à `SpacesAdminSectionProps`).
- Calculer `canManage = canManageSpaces(user)` une fois en haut du composant.
- **AddSpaceForm** : si `!canManage`, ne pas rendre le formulaire (ou le rendre intégralement disabled — au choix UX). Préférer **ne pas rendre** : l'opérateur sans droits ne doit pas voir un formulaire qui ne sert à rien.
- **Boutons Edit / Delete par ligne** : si `!canManage`, ne pas les rendre.
- **Badge informatif** quand `!canManage` : un petit texte sobre type *« Read-only — admin scope required »* sous le titre de la section. Pas de moralisateur, juste factuel.

**Wiring App.tsx** : la prop `user` doit venir de `useAuth().user`. Vérifie que `App.tsx:993` (`currentUser={CURRENT_USER}`) est cohérent avec ce qu'on passe à SpacesAdminSection — il y a peut-être un mismatch entre `CURRENT_USER` (fixture) et `auth.user` (réel) qui mérite un coup d'œil.

**Toast 403 réseau** : `useCreateSpace`/`useUpdateSpace`/`useDeleteSpace` dans `lightrag_webui_twin/src/api/queries.ts` doivent intercepter le 403 backend et émettre un toast `"Admin scope required"`. Si l'intercept est déjà global (check `client.ts`), ne pas doubler.

### MSW (mocks) — important

Les handlers `lightrag_webui_twin/src/mocks/handlers.ts:626-680` doivent simuler le 403 quand le `debugUser` courant n'a pas `admin:spaces` dans `gateway_scopes`. Sinon les tests RTL en mode `canManage=false` vont obtenir un 201 du mock et la régression passera.

Suggestion : exposer un helper `mockCurrentScopes()` qui lit `window.__twinE2eRuntimeConfig?.debugUser?.gateway_scopes` (déjà utilisé pour d'autres scénarios e2e).

---

## Lot D — Doc + plan (pour Codex)

À mettre à jour quand A+B+C sont landés :

1. **`WEBUI-WIRING-PLAN.md` :**
   - **l. 289** : remplacer « Still open for spaces: JWT/MyAccess admin-only gating for runtime Space CRUD. » par une ligne **Done** datée, mentionnant le commit, l'env `TWIN_IDP_ADMIN_GROUPS` et la scope `admin:spaces`.
   - **l. 1275-1283 (Sequencing recommendation)** : retirer le point 1 « MyAccess admin-only authorization policy » et resserrer la liste à deux items (deployment smoke + retention).
2. **`src/twindb_lightrag_memgraph/server/idp_jwt.py`** — uniquement la docstring d'en-tête (l. 35-58, le tableau d'env vars) : ajouter une ligne pour `TWIN_IDP_ADMIN_GROUPS`. Pas toucher le code (Claude l'a déjà câblé via lot A).
3. **`CLAUDE.md`** racine — section « Server module » : ajouter une phrase à l'item `idp_jwt.py` mentionnant `require_admin_user` + scope `admin:spaces`.
4. **`changelog.md`** — entrée datée `2026-06-04` sous la version courante : *"Admin-only Twin Space CRUD via TWIN_IDP_ADMIN_GROUPS env + `admin:spaces` gateway scope."*

---

## Tests attendus (lot C)

Dans `lightrag_webui_twin/src/components/Settings/SpacesAdminSection.test.tsx` (étend l'existant) :

- `canManage=true` → boutons Add/Edit/Delete présents (couverture actuelle suffit).
- `canManage=false` → `AddSpaceForm` absent, Edit/Delete par ligne absents, badge read-only présent (`data-testid="spaces-admin-readonly-badge"` recommandé).
- `canManage=true` + backend 403 inattendu → toast affiché.

`canManageSpaces()` doit avoir un test unitaire trivial dans `lightrag_webui_twin/src/lib/permissions.test.ts` couvrant les 3 cas (null user, admin scope present, admin scope absent).

---

## Commande de validation finale (Codex, avant push)

```bash
cd /Users/julien/twindb-lightrag-memgraph/lightrag_webui_twin
bun run typecheck && bun run test:run
```

Côté Python (pour confirmer que rien n'a régressé en touchant la docstring `idp_jwt.py`) :

```bash
cd /Users/julien/twindb-lightrag-memgraph
.venv/bin/pytest tests/test_server/test_idp_jwt.py tests/test_server/test_webui_router.py -q
```

---

## Ce que tu ne dois PAS faire

- ❌ Ne pas ajouter de scope dans `palier.scopes` — c'est réservé aux capabilities Twin-internes (`twin:read`, `twin:write`, `twin:approve`). `admin:spaces` est un scope gateway.
- ❌ Ne pas créer de fonction `isAdmin(user)` générique — la doctrine est capability-based (`canManageSpaces`, `canManageTags`, etc.), une fonction par surface.
- ❌ Ne pas casser le mode dev / OVH standalone : `canManage=true` quand `user === null` est non-négociable.
- ❌ Ne pas refaire `bind_request_space` côté frontend — le header `X-Twin-Space` reste géré par `client.ts` indépendamment du gating admin.
- ❌ Ne pas pusher sur GitHub `origin` — push sur `bunker` uniquement (CLAUDE.md projet, section « Git workflow »).

---

## Signal de fin

Quand tu as terminé C+D, ouvre une PR sur `feat/webui-graph-real-memgraph` ou pousse directement si la branche est en flow direct. Ping Julien avec :

- Le hash du/des commits Codex
- Le résultat exact de `bun run test:run` (X passed)
- Confirmation que tu as bien testé `canManage=false` en RTL (pas juste typecheck)
