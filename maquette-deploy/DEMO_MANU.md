# Démo Manu — 2026-05-22 9h30

Préparation : 17h-22h la veille (post-meeting Kore.ai à 16h).

Manu vient avec **exactement les questions qu'il a posées à Kore.ai
hier**. C'est notre terrain. Le seul risque est de **ne pas répondre
explicitement** aux 5 questions ou de partir dans le polish UI.

## Le verbatim qu'il faut avoir en tête

> *"Your platform is focused on how to create agents efficiently.
> Our need is **before** this step. It's knowledge management.
> Your target is not aligned with our need right now."*
> — Manu à Kore.ai, 2026-05-21 16h

Ça résume la position. Twin RAG = **avant** les agents (gouvernance
du knowledge), pas **sur** les agents (= Kore.ai).

## Flow démo (~20-25 min)

### 1. Open Twin RAG live (1 min)
- Onglet : <https://maquette.sigilum.fr/>
- Cmd+Shift+R pour bypass cache **avant** Manu arrive.
- Si infra fail : passer direct à `compare.html` (slide statique).

### 2. Documents tab — pending review queue (4 min)
**Question 1 Manu : workflow validation + quality gate.**

- Pointer le bandeau jaune "Pending review · 2 documents awaiting your sign-off"
- Cliquer pour expand
- "Voilà ce que Kore.ai n'a pas. Le steward voit la file d'attente directement,
  pas dans un module séparé."
- Pointer les 3 actions : Approve / Edit & approve / Reject
- Cliquer **Approve** sur `cft-vendor-api-spec-draft.pdf`
- "La carte disparait du queue. Elle entre dans le retrieval set actif."
- Faire un Cmd+R (reload). "Et c'est persisté. Reload, ça reste approved."

### 3. Documents tab — audit trail (4 min) ⭐ KILLER
**Question 2 Manu : audit "who/what/when".**

- Cliquer sur le doc qu'on vient d'approuver dans la table principale
- Side panel s'ouvre → onglet "Audit"
- Pointer la timeline :
  - "Requested for review by marc.berthier · seed"
  - "Approved by claire.benoit · UTC timestamp · mutation #N"
- Cliquer "raw payload" sur l'entry mutation
- "Pas dans les logs, pas dans un fichier qu'il faut SSH pour lire.
  Dans l'interface, directement, pour le steward qui pilote."
- **Insister sur le pill SQLite** : "Table mutations, SQLite réel,
  visible côté ops via `docker exec sqlite3`. Pas de magie."

> **Si Manu challenge** : "comment vous savez que c'est pas falsifié ?"
> → "Append-only. AUTO_INCREMENT id, timestamps server-side, immutables
> côté backend. Backup volume Docker comme n'importe quel ledger."

### 4. Topbar — MyAccess pill (3 min) ⭐ KILLER
**Question 3 Manu : inheritance des clearances.**

- Cliquer sur le pill "MyAccess" en haut
- Popover s'ouvre
- "Identity source : MyAccess. C'est la contrainte sécu BNP — pas de
  référentiel concurrent. Twin hérite, ne duplique pas."
- Pointer le KV : Source, user, entity AUID, last sync, refresh cycle
- Lire le paragraphe : "Twin inherits the entity-level access decision
  from MyAccess (ITGP workflow). The Reader / Contributor / Steward
  palier is **local** to Twin."
- "Cache des autorisations dans le graphe Memgraph lui-même. Donc une
  query d'audit du type 'qui a eu accès au doc X entre tel et tel jour'
  se fait au même endroit que la donnée."

> **Honest** : ajouter "preview" — bridge backend pas encore codé. On a
> les 5 questions à poser à Faisal (slack ?) pour finaliser le contrat.
> Vihn a déjà cadré le modèle métier le 2026-05-20.

### 5. Retrieval tab — citations (3 min)
**Question 4 Manu : traçabilité réponse → source.**

- Cliquer Retrieval tab
- Lancer une query "How do I restart Oracle on RHEL 9?" (suggestion clickable)
- Wait pour le stream
- Pointer les chips citations `[1] [2] [3]`
- Hover → tooltip avec source name + score
- "Chaque chunk de la réponse est lié à sa source avec son score.
  Si la réponse est mauvaise, on remonte au chunk, au document,
  au steward qui l'a approuvé. Chaîne complète."

### 6. Activity tab — audit cross-feature (2 min)
**Renforce question 2.**

- Cliquer Activity tab
- Pointer les events en haut (les mutations qu'on vient de faire)
- "Approuvé · cft-vendor-api-spec-draft.pdf · entered active retrieval set"
- "Live polling. Ces lignes viennent de la même table mutations
  qu'on a vue dans l'audit doc."

### 7. Tags tab (2 min)
**Préventif — Manu peut demander la même gouvernance pour les tags.**

- Cliquer Tags tab
- Pointer le bandeau Pending requests (steward queue tags)
- "Même méca que les docs. Tag governance = workflow palier-3."
- Ouvrir une tag card, montrer le panel détail + le bouton More dropdown
  (Deprecate / Delete avec confirmation)

### 8. Reset demo (15s)
- Si Manu veut "voir ce que ça fait à neuf"
- Cliquer le trash icon dans la topbar (côté droit)
- Confirm
- Reload → tout redevient seed JSON

### 9. Slide comparatif (5 min)
- Ouvrir <https://maquette.sigilum.fr/compare.html>
- Suivre les 5 lignes du tableau
- **Finir sur les badges position** : "AI for agents (eux) vs
  knowledge management (nous). C'est pas le même produit."
- Verbatim Manu en haut → "vos mots, pas les miens."

## Trous probables — anticipation

| Question Manu attendue | Réponse |
|---|---|
| "Pourquoi pas Core.AI ?" | C'est exactement Kore.ai, on l'a vu hier. Slide compare répond. |
| "Coût licence ?" | LightRAG = open source (Apache 2). Memgraph = community / enterprise selon scale. Pas de SaaS imposé. |
| "Multi-tenant / cloisonnement ?" | Workspaces. Filtre `workspace_id` au niveau du nœud Memgraph, opacification totale entre TwinGov / CIB / Payments. Cap 64 Go par instance Memgraph licence. |
| "Roadmap quand ?" | Validation doc + audit = livré (aujourd'hui). MyAccess bridge = phase 2 après sync Faisal. Workspace cloisonnement réel = en cours. |
| "Vihn est OK ?" | Oui — sync 26 min le 2026-05-20, support explicite. Lead RAG/KB chez Julien acté. |
| "Sécurité / compliance ?" | Pas de CDN externe (sql.js abandonné pour cette raison hier soir, on est full backend Python + SQLite Docker volume). Code source audité. |

## Ce qu'il ne faut PAS faire

- Ne **pas** présenter ça comme une maquette de design — c'est un produit avec backend réel.
- Ne **pas** s'excuser sur ce qui n'est pas encore wired (Tags governance pas encore en SQLite, etc.). Pointer ce qui marche.
- Ne **pas** dériver sur LightRAG (Vihn a déjà audité, c'est ok). Twin **utilise** LightRAG, ce n'est pas Twin.
- Ne **pas** parler de Sigilum/Erwin Labs si Manu ne le demande pas. Sigilum c'est l'hébergement temporaire de la démo, pas le produit.

## Le pitch en 30 secondes (si Manu interrompt après 3 min)

"Twin RAG = la gouvernance du knowledge **avant** que les agents s'en servent.
Steward review queue, audit trail visible, MyAccess inheritance dans le
graphe. Construit sur LightRAG (que tu as déjà fait auditer par Vihn),
backend FastAPI + SQLite, Docker. Aujourd'hui live à maquette.sigilum.fr.
Kore.ai a admis hier qu'ils n'ont pas ça."

## Live links

- Maquette : <https://maquette.sigilum.fr/>
- Backend health : <https://maquette.sigilum.fr/api/health>
- Mutations live : <https://maquette.sigilum.fr/api/mutations?limit=10>
- Slide comparatif : <https://maquette.sigilum.fr/compare.html>

## Si l'infra OVH lâche

Plan B :
1. Tout est dans `~/twin-maquette/` sur le VPS OVH
2. `ssh erwin '/usr/bin/docker stack services twin-maquette'` pour check
3. Si fail : `docker stack deploy -c stack.yml twin-maquette` pour relancer
4. Si DNS Cloudflare fail : direct IP `37.59.104.111` (HTTP only)
5. Si tout fail : montrer `compare.html` en local + screenshots
