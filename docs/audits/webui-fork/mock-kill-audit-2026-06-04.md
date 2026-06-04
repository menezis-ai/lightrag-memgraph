# Mock-kill audit — 2026-06-04

> **Statut 2026-06-04 — REMEDIATED.** Findings F1/F2/F3/F5/F6 closed
> by commit `731f0d1` (mock-kill remediation). Finding F4 closed by
> commit `524b2a8` (retrieval streaming + advanced controls + dead-code
> purge). This audit is kept as the historical "before remediation"
> snapshot — the contract it documents is the regression line that
> future contributors must not cross.

**Mandat :** Fabrice 2026-06-01 ([transcription][tr] [1466s]) — *« Et en revanche, c'est fonctionnel. Je ne veux plus de moquer, de machin, de truc. »*
**Deadline implicite :** [1458s] — *« Dans 15 jours, il faut tout ce que tu peux mettre. Et en revanche, c'est fonctionnel. »* → **2026-06-16**.
**Périmètre :** `lightrag_webui_twin/src/` + `src/twindb_lightrag_memgraph/server/` + `src/twindb_lightrag_memgraph/__init__.py`.
**Méthode :** grep systématique sur fixture imports hors test/mocks, gates `VITE_*MSW*`, `webui_seed.*` consumers, branches `if (!onSendQuery)` style, comments `MOCK_*` / `placeholder` / `hardcoded`.

[tr]: /Users/julien/Downloads/Enregistrement%20standard%2060-transcription-20260601T155401.json

---

## Architecture confirmée saine (rappel pour ne pas régresser)

- **MSW gating** (`src/main.tsx` + `src/config/devConfig.ts:75-99`) : MSW boot uniquement si `import.meta.env.DEV` ou `VITE_FORCE_MSW === 'true'`. Build prod non-standalone sans config substituée **throw** loud (testé `useAuth.test.tsx:67`). ✅
- **Couche P1 du plan** (`WEBUI-WIRING-PLAN.md` l. 312-321) déjà marquée Done : les fallback fixtures silencieux côté queries (documents, tags, activity, thesaurus, graph, notifications) ont été tués.
- **Codex C+D Space CRUD** (commit `a62b4b4`) : `SpacesAdminSection` lit le runtime catalog réel + admin gating.

L'audit ci-dessous concerne ce qui **reste après ces protections**.

---

## Findings

### F1 — Settings → Space affiche un Workspace fictif **[S1]**

**Fichiers :** `lightrag_webui_twin/src/components/Settings/WorkspaceSection.tsx:17-83` + fixture source `lightrag_webui_twin/src/fixtures/settings.ts:30-45`.

L'utilisateur ouvre Settings → Space et voit :

| Champ | Valeur affichée (hardcoded fixture) | Réalité |
|---|---|---|
| Space ID | `default` | dépend du runtime catalog + active space |
| Display name | `Default space` | dépend du label défini dans `TWIN_SPACES_JSON` |
| Visibility | `private` (env var `TWIN_INSTANCE_VISIBILITY`) | aucune env var de ce nom n'existe côté serveur |
| Region | `eu-west-3 · dc-paris` | inventé |
| Retention policy | `twin-default-space-retention-v1` + tableau TTL 6 lignes | inventé (cf. plan l. 262-264 — retention sweep PO-gated, pas de policy active) |

**Why S1 :** ment activement même quand l'utilisateur a switché sur un autre space (par ex. `sandbox`). Et la totalité des TTLs affichés est fictive — ça pourrait être pris pour un engagement compliance.

**Remediation proposée :** voir §Action F1 plus bas. TL;DR : tirer ID+display name du runtime config (data déjà présente), dropper les cards Visibility/Region/Retention faute d'endpoint d'ici 2026-06-16.

---

### F2 — Settings → API affiche un OpenAPI figé + URLs serveurs fictives **[S2]**

**Fichiers :** `lightrag_webui_twin/src/components/SettingsTab.tsx:99-103` + fixture source `lightrag_webui_twin/src/fixtures/api.ts`.

Hardcoded :
- `API_VERSION = 'v1.4.12/0279'` — version string statique (faux dès qu'on déploie une autre)
- `OPENAPI_GROUPS` — table d'endpoints (5 groupes, ~30 routes) figée
- `API_SERVERS = [{ id: 'prod', label: 'https://cib-kb.twin.internal — production' }, …]` — URLs **inventées**, pas de DNS BNP correspondant
- `API_BASE_URL = { prod: 'https://cib-kb.twin.internal', stg: 'https://cib-kb.stg.twin.internal' }` — idem

**Why S2 :** visible mais sur tab API explorer, peu utilisé. Ne ment pas activement sur du contenu produit mais affiche des URLs qui n'existent pas si on essaie de les taper en barre d'adresse.

**Backend pertinent :** `server/native_shims.py` projette `GET /openapi.json` (FastAPI complet) → groups statiques. Un endpoint dédié `/twin/api/openapi` retourne `{groups, version}` typed (`OpenApiEnvelope` dans `webui_models.py`). À wirer.

**Remediation proposée :** voir §Action F2.

---

### F3 — GraphTab détails entité : tags+sources toujours vides sur entités Memgraph réelles **[S2]**

**Fichier :** `lightrag_webui_twin/src/components/GraphTab.tsx:32-44`.

```ts
import { GRAPH_ENTITY_DOCS, GRAPH_ENTITY_TAGS } from '../fixtures/graph';

const tagsOf = (e: GraphEntity): readonly string[] =>
  GRAPH_ENTITY_TAGS[e.id] ?? [];
const docsOf = (e: GraphEntity): readonly string[] =>
  GRAPH_ENTITY_DOCS[e.id] ?? [];
```

Les maps `GRAPH_ENTITY_TAGS` et `GRAPH_ENTITY_DOCS` sont indexées par les IDs du prototype (`e_oracle`, `e_rman`, …). Avec entités Memgraph réelles (IDs hashés type `entity-{hash}`), les deux lookups retournent `[]` → le détails-panel affiche systématiquement « 0 tags · 0 sources » pour TOUTES les entités produit.

**Why S2 :** misleading mais sans tromperie active (0 vs un vrai nombre). Symptôme du chantier M12 batch 2 PATCH qui a livré CRUD mais pas l'enrichissement « tags + sources per entity ».

**Remediation :** soit étendre `server/graph_reader.py` pour retourner `tags[]` + `source_doc_ids[]` par entité (LightRAG les a dans les properties `entity.source_id`), soit supprimer ces lookups en attendant. Choix arbitré par F3-décision plus bas.

---

### F4 — RetrievalTab : branche dead code + props fixture inutilement bundlées **[S3]**

**Fichiers :**
- `lightrag_webui_twin/src/components/RetrievalTab.tsx:51-55, 113-114, 290-294`
- `lightrag_webui_twin/src/App.tsx:67,73,937-938`

Le composant accepte `answerTokens` + `answerSources` (props fallback fixture), et `App.tsx` les passe avec `ANSWER_TOKENS_FIXTURE` + `RETRIEVAL_SOURCES_FIXTURE` — mais comme `onSendQuery` + `onStreamQuery` sont **toujours** câblés en parallèle, la branche fallback (l. 290-294) n'est jamais atteinte en runtime App.

**Why S3 :** invisible utilisateur. Pur code mort + ~70KB de tokens fixture bundlés inutilement. À nettoyer pour ne pas tromper un dev qui lit le composant.

**Remediation :** voir §Action F4. Trivial (purge).

---

### F5 — Default `webui_stores="seed"` est un piège prod **[S1 conditionnel]**

**Fichier :** `src/twindb_lightrag_memgraph/__init__.py:54`.

```python
def register(
    ...
    webui_stores: str = "seed",
    ...
)
```

La valeur par défaut bootstrap **tags + activity + notifications + documents + workspaces + thesaurus + graph + OpenAPI groups** depuis `webui_seed.py` en mémoire (cf. `WebuiStore.from_seed()` `webui_router.py:200-211`).

Pour un déploiement BNP prod, l'install runbook doit explicitement passer `webui_stores="memgraph"`. Si oublié → l'opérateur voit une démo de tags/activity/notifications + documents fictifs + 19 entités graph hardcodées (Oracle, RMAN, RHEL, SWIFT, …).

La maquette OVH s'en sert volontairement (vitrine). Mais en prod c'est exactement ce que Fabrice veut tuer.

**Why S1 conditionnel :** dépend de la rigueur du runbook. Pas un bug en soi, mais Fabrice paranoia justifiée.

**Remediation proposée :** voir §Action F5. Warn-loud au boot quand `webui_stores="seed"` ET un signal "prod" est détecté (IdP actif OU `LIGHTRAG_API_KEY` configuré).

---

### F6 — Even in `memgraph` mode, the default Space seeds in-memory demo data **[S2 needs-verification]**

**Fichier :** `src/twindb_lightrag_memgraph/server/webui_router.py:214-228` (`WebuiStore.for_space`).

```python
@classmethod
def for_space(cls, space: str) -> WebuiStore:
    default_space = load_space_catalog().default_space_id
    if space == default_space:
        return cls.from_seed()  # ← full demo, including documents + graph_entities
    return cls(documents=[], ..., tags_seed=[], ...)
```

La **default space** récupère TOUJOURS `from_seed()` — full payload. Seules les spaces *non-default* sont vraiment vides.

Cela coexiste avec le G1 fix (audit l. 252-256) qui s'assure que les **stores Memgraph** (tags + activity + notifications) ne sont pas pollués. Mais les listes in-memory dans `WebuiStore` (`_documents`, `_graph_entities`, `_workspaces`, `_thesaurus`) **sont** seedées même quand `webui_stores="memgraph"`.

**Why S2 needs-verification :** je n'ai pas confirmé dans cette session si `list_documents()` (`webui_router.py:557`) retourne le contenu de `WebuiStore._documents` (in-memory seed) OU les vrais `DocStatus_{workspace}` via `native_shims.py`. Si c'est `native_shims`, alors les `_documents` in-memory sont code mort en mode `memgraph` → S3. Si c'est `WebuiStore._documents`, alors un déploiement memgraph affiche quand même des docs démo sur la default space → S1.

**À vérifier :** lecture sérieuse de `list_documents` + `native_shims.documents_router`. Pas tranché dans ce passage.

---

## Plan d'action

### Action F4 — Purge dead code RetrievalTab **[trivial, je ship dans ce sprint]**

1. Supprimer `answerTokens` + `answerSources` des `RetrievalTabProps`.
2. Supprimer la branche `if (!onSendQuery && !onStreamQuery)` (l. 290-294).
3. Supprimer imports `ANSWER_TOKENS_FIXTURE` + `RETRIEVAL_SOURCES_FIXTURE` de `App.tsx`.
4. Update tests pour ne plus passer les props absentes.
5. **Coordination Codex :** Codex travaille sur le streaming retrieval actuellement — touche probablement RetrievalTab. **Ne PAS shipper F4 maintenant** pour éviter conflit. Le marquer "à faire en aval de la merge Codex retrieval".

### Action F5 — Boot WARN si `webui_stores="seed"` + IdP actif **[trivial, je ship]**

1. Dans `register()` (`__init__.py`), après détection IdP via `_IdpConfig.from_env() is not None` (déjà calculé l. 1172), si `webui_stores == "seed"` ET IdP actif → `logger.warning("DEMO STORES IN PROD: webui_stores='seed' with active IdP — tags/activity/notifications/documents are in-memory fixtures, will not survive restart.")`.
2. Test : nouveau cas dans `tests/test_server/test_app.py` ou ajout à `TestLifespan`.

### Action F1 — WorkspaceSection : remplacer fixture par runtime config + dropper cards fictives **[je ship si tu valides la stratégie]**

Stratégie proposée :
- **Identity card** : tirer `space_id` + `display_name` du `useAuth().runtimeConfig.spaces` croisé avec le space actif (déjà disponible). Drop `visibility` + `visibility_env` + `region` (pas de source).
- **Retention card** : drop entièrement (la doctrine retention est PO-gated cf. plan l. 262-264 → pas de policy active à afficher honnêtement).
- Si tu veux conserver la place visuelle de la retention card pour qu'elle revienne plus tard, on peut afficher un placeholder « Retention policy : configured at install — see install runbook. » avec lien vers `docs/operations/install-runbook.md`.

**Décision à prendre :** drop retention card OU placeholder texte ?

### Action F2 — API tab : wirer fetch /twin/api/openapi + drop API_SERVERS faux **[je ship si tu valides]**

Stratégie :
- Fetch `/twin/api/openapi` au mount de la section → remplir `groups` + `version` depuis backend.
- Supprimer `API_SERVERS` entièrement (URLs `cib-kb.twin.internal` n'existent pas). Si on veut afficher un sélecteur prod/stg, faut un vrai endpoint qui dit "tu es connecté à X".
- Garder la fixture comme **fallback typé** uniquement si le fetch échoue (avec badge erreur visible).

**Décision à prendre :** ship F2 dans ce sprint ou attendre la nouvelle politique parité Fabrice ?

### Action F3 — GraphTab tags/docs : extend backend OU drop **[décision technique]**

Trois options :
- **a)** Étendre `graph_reader.py` pour qu'une entité retourne `tags[]` + `source_doc_ids[]`. Coût : ~2h backend (queries Cypher + mapping) + ~1h frontend (typing). Touche zone Memgraph — peut-être en conflit avec M12 batch 2 si pas encore stable.
- **b)** Supprimer `tagsOf`/`docsOf` du détails-panel. Le panel affiche un peu moins d'info mais zéro mensonge.
- **c)** Ne rien faire d'ici 2026-06-16 et flagger comme « known gap » dans la release notes.

**Décision à prendre :** a, b, ou c ?

### Action F6 — Vérifier `list_documents` et `graph_entities` en mode memgraph **[bloquant pour triage final]**

Je relis `webui_router.py:list_documents` + `native_shims.documents_router` pour confirmer si les `WebuiStore._documents`/`_graph_entities` sont réellement consommés en mode `webui_stores="memgraph"` ou si `native_shims` les shadowe complètement. Selon le résultat, F6 monte en S1 (à fixer) ou descend en S3 (code mort à purger).

**~30min de lecture, je peux le faire maintenant.**

---

## Synthèse exécutive

| Finding | Sévérité | Action proposée | Décision Julien |
|---|---|---|---|
| F1 WorkspaceSection fictif | **S1** | Identity from runtime config + drop visibility/region/retention cards | drop ou placeholder ? |
| F2 API tab fixtures | S2 | Fetch /twin/api/openapi + drop API_SERVERS faux | ship maintenant ou attendre ? |
| F3 GraphTab tags/docs vides | S2 | a/b/c (cf. ci-dessus) | a, b, ou c ? |
| F4 RetrievalTab dead code | S3 | Purge — après merge Codex retrieval | OK séquencer ? |
| F5 default webui_stores=seed | S1 cond. | Boot WARN | ship sans décision (no-op runtime) |
| F6 default space seed leak | S2 needs-verif | Vérification 30min puis triage final | je vérifie ? |

**Ce que je peux shipper sans nouvelle décision :** F5 (warn) + F6 (vérification). Tout le reste demande arbitrage.

**Estimation deadline 2026-06-16 :** F1+F2+F4+F5+F6 livrables sans extension backend lourde. F3 si option (b) sinon ~3h backend.

---

## Annexe — méthodologie

Commandes utilisées pour l'inventory :

```bash
# A. MSW gating
grep -rn "VITE_FORCE_MSW\|VITE_USE_MSW\|installMSW\|enableMocking\|useMSW\|MSW.*fallback\|resolveRuntimeConfig" src/

# B. Fixtures imports hors test/mocks
grep -rn "from.*fixtures/" src/ --include="*.ts" --include="*.tsx" | grep -v "/test/\|/mocks/\|.test."

# C. Markers code (MOCK_/TODO/placeholder)
grep -rnE "(// *MOCK|// *TODO|// *FIXME).*mock|placeholder.*data|hardcoded|stub" src/

# D. Backend webui_seed consumers
grep -nE "webui_seed\.(DOCUMENTS|WORKSPACES|THESAURUS|GRAPH)" src/twindb_lightrag_memgraph/server/

# E. register() default args
grep -nE "webui_stores\s*=|webui_stores:" src/twindb_lightrag_memgraph/__init__.py
```

Coverage exhaustive sur ces 5 axes. Pas exploré : assets CSS, fichiers `.json` statiques, dossier `proto/`.
